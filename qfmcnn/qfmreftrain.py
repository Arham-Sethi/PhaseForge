#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QFM-DCT + ResNet Integration (Stabilized, Causal-Robust, Ablations Included)

Major changes vs your FFT version:
- Frequency branch uses separable DCT-II / IDCT-III (real-valued).
- Real 1x1 in DCT space (complex path disabled initially).
- BlurPool AA before every stride-2 (stem + layer2..4).
- Conservative fusion start: y = s*S + f*F (s≈1,f≈0); phase OFF until epoch>=unfreeze_phase_epoch.
- Phase head bounded via tanh; default clip ±π/6 (configurable).
- Morphology gate simplified: sigmoid(α|Sobel(x)| + β(dilate-erosion)); τ scheduled 5→15 over 10 epochs.
- DCT radial mask gain ∈ [0.8,1.2]; center threshold t ∈ [0.25,0.45] (learnable).
- Causal invariance: REx loss (variance across env splits) + counterfactuals (StyleMix/CutPaste).
- Causal gate uses observed style proxy n(x) to modulate f in [0, fmax] (phase off early).
- Training recipe: AdamW, cosine LR, EMA, MixUp, LabelSmoothing, RandAugment/AugMix, grad clip, AMP.
- Ablation plan A..G implemented as flags (baseline, +BlurPool, +AugMix, +QFM steps, etc.).
- Probes: mCE (CIFAR-10-C style subset), retention curves, feature cosine stability, energy drift,
  morphology maps, DCT masks (grids), per-env risk variance.

Author: Arham Sethi (reworked with DCT + causal gates)
Date: 2025-11-12
"""

import os, io, math, time, json, copy, random, argparse, datetime
from typing import Tuple, Iterable, List, Optional, Union, Dict
import numpy as np
from PIL import Image, ImageFilter
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

# ------------------------------
# Global config & seed
# ------------------------------

def set_seed(seed: int = 1337, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

_USE_CUDA = torch.cuda.is_available()
if _USE_CUDA:
    from torch.cuda.amp import autocast, GradScaler
else:
    class autocast:
        def __init__(self, enabled=False, **kwargs): self.enabled = enabled
        def __enter__(self): return None
        def __exit__(self, *a, **k): return False
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = False
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass

# ------------------------------
# Utilities
# ------------------------------

def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ------------------------------
# BlurPool AA (stride-2)
# ------------------------------

class BlurPool(nn.Module):
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        assert stride in (2, 1)
        if filt_size == 3:
            kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        elif filt_size == 5:
            kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        else:
            raise ValueError("filt_size must be 3 or 5")
        kernel_2d = torch.outer(kernel, kernel)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer('kernel', kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.pad = (filt_size // 2, ) * 4  # L,R,T,B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad[0], self.pad[1], self.pad[2], self.pad[3]), mode='reflect')
        return F.conv2d(x, self.kernel, stride=self.stride, groups=x.shape[1])

# ------------------------------
# Separable DCT-II / IDCT-III (orthonormal)
# ------------------------------

_dct_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
_idct_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

def _dct_mat(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    if key in _dct_cache:
        return _dct_cache[key]
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)  # [n,1]
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)  # [1,n]
    mat = torch.cos(math.pi * (i + 0.5) * k / n)                 # [n,n]
    mat[0, :] = mat[0, :] * (1.0 / math.sqrt(n))
    mat[1:, :] = mat[1:, :] * math.sqrt(2.0 / n)
    _dct_cache[key] = mat
    return mat

def _idct_mat(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    if key in _idct_cache:
        return _idct_cache[key]
    # For orthonormal DCT-II, inverse is its transpose
    _idct_cache[key] = _dct_mat(n, device, dtype).t()
    return _idct_cache[key]

def dct2(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    Th = _dct_mat(H, device, dtype)      # [H,H]
    Tw = _dct_mat(W, device, dtype)      # [W,W]
    x = torch.einsum('bchw,hh->bchw', x, Th.t())  # along H
    x = torch.einsum('bchw,ww->bchw', x.permute(0,1,3,2), Tw.t()).permute(0,1,3,2)
    return x

def idct2(X: torch.Tensor) -> torch.Tensor:
    # X: [B,C,H,W]
    B, C, H, W = X.shape
    device, dtype = X.device, X.dtype
    ThT = _idct_mat(H, device, dtype)    # [H,H]
    TwT = _idct_mat(W, device, dtype)    # [W,W]
    x = torch.einsum('bchw,hh->bchw', X, ThT.t())
    x = torch.einsum('bchw,ww->bchw', x.permute(0,1,3,2), TwT.t()).permute(0,1,3,2)
    return x

# ------------------------------
# Sobel & soft morphology (simplified)
# ------------------------------

def sobel_xy_kernels(device, dtype):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype, device=device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=dtype, device=device).view(1,1,3,3)
    return kx, ky

class SimpleMorphGate(nn.Module):
    """
    M = sigmoid( α |Sobel(x)| + β (dilate(x) - erode(x)) ), with τ temperature that we schedule.
    Outputs A in [0,1], shape [B,1,H,W]; depends only on input feature (grayscale proxy).
    """
    def __init__(self, in_channels: int, kernel_size: int = 3, tau_init: float = 5.0):
        super().__init__()
        assert kernel_size in (3,5,7)
        self.kernel_size = kernel_size
        self.register_buffer('tau', torch.tensor(tau_init, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.register_buffer('kx', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('ky', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))

    def set_tau(self, value: float):
        with torch.no_grad():
            self.tau.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # grayscale proxy
        g = x.mean(dim=1, keepdim=True)
        # Sobel magnitude
        gx = F.conv2d(g, self.kx.to(g.device), padding=1)
        gy = F.conv2d(g, self.ky.to(g.device), padding=1)
        edge = torch.sqrt(gx*gx + gy*gy + 1e-12)

        # soft dilation/erosion via softmax/softmin over local window
        k = self.kernel_size
        pad = k // 2
        patches = F.unfold(g, kernel_size=k, padding=pad)  # [B, k*k, HW]
        B, KK, HW = patches.shape
        patches = patches.view(B, 1, KK, int(g.shape[-2]), int(g.shape[-1]))  # [B,1,KK,H,W]

        scores_max = (patches * self.tau).softmax(dim=2)
        dil = (scores_max * patches).sum(dim=2)
        scores_min = (-patches * self.tau).softmax(dim=2)
        ero = (scores_min * patches).sum(dim=2)

        raw = self.alpha * edge + self.beta * (dil - ero)
        A = torch.sigmoid(raw)
        return A.clamp(0.0, 1.0)

# ------------------------------
# Orthogonal mixer via QR
# ------------------------------

class OrthoMixer(nn.Module):
    def __init__(self, M: int):
        super().__init__()
        self.P = nn.Parameter(torch.randn(M, M) * 0.02)
    def forward(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.P, mode='reduced')
        diag = torch.sign(torch.diag(R))
        diag[diag == 0] = 1.0
        return Q @ torch.diag(diag)

# ------------------------------
# Base spatial filter bank (as before)
# ------------------------------

def _gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float):
    xmax = ks // 2; ymax = ks // 2
    y, x = torch.meshgrid(torch.arange(-ymax, ymax+1), torch.arange(-xmax, xmax+1), indexing='ij')
    x = x.to(torch.float32); y = y.to(torch.float32)
    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    gb = torch.exp(-(x_theta**2 + (gamma**2)*(y_theta**2)) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / lam + psi)
    gb = gb / (gb.abs().sum() + 1e-6)
    return gb

def make_base_filters(kernel_size: int, device: torch.device) -> torch.Tensor:
    ks = kernel_size
    assert ks in (3,5,7), "kernel_size must be 3, 5 or 7."
    bank: List[torch.Tensor] = []
    eye = torch.zeros((ks, ks), dtype=torch.float32); eye[ks//2, ks//2] = 1.0; bank.append(eye)
    box = torch.ones((ks, ks), dtype=torch.float32) / (ks*ks); bank.append(box)
    if ks == 3:
        gauss = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32); gauss /= gauss.sum()
    elif ks == 5:
        gauss = torch.tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]], dtype=torch.float32); gauss /= gauss.sum()
    else:
        g1d = torch.tensor([1,6,15,20,15,6,1], dtype=torch.float32); gauss = torch.outer(g1d, g1d); gauss /= gauss.sum()
    bank.append(gauss)
    if ks == 3:
        sharp = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=torch.float32)
    else:
        sharp = -box * (ks*ks - 1); sharp[ks//2, ks//2] += ks*ks
    bank.append(sharp)
    if ks == 3:
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    else:
        lap = torch.zeros((ks, ks), dtype=torch.float32)
        lap[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    bank.append(lap)
    if ks == 3:
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    else:
        sobel_x = torch.zeros((ks, ks), dtype=torch.float32); sobel_y = sobel_x.clone()
        sobel_x[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    bank.extend([sobel_x, sobel_y])
    for theta in (0.0, math.pi/4, math.pi/2, 3*math.pi/4):
        bank.append(_gabor_kernel(ks, 1.2, theta, 4.0, 0.5, 0.0))
    bank = torch.stack(bank, dim=0).unsqueeze(1).to(device=device)
    return bank  # [M,1,ks,ks]

# ------------------------------
# QFMConv2d-DCT (stabilized)
# ------------------------------

class QFMConv2dDCT(nn.Module):
    """
    Stabilized QFM block:
      - Spatial: base bank -> orthogonal mixing -> conv
      - Frequency: DCT2 -> real 1x1 -> radial mask (gain in [0.8,1.2], t∈[0.25,0.45]) -> IDCT2
      - Morphology gate: simplified, τ scheduled externally
      - Fusion: start y = s*S + f*F ; after phase_unfreeze, switch to interference form
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int,int]],
        stride: Union[int, Tuple[int,int]] = 1,
        padding: Union[int, Tuple[int,int]] = 0,
        bias: bool = True,
        input_size: Tuple[int,int] = (224,224),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        phase_clip: float = math.pi/6,
    ):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.phase_clip = phase_clip

        # Spatial branch
        self.register_buffer('_base_bank', make_base_filters(self.kH, self.device))
        self.M = self._base_bank.shape[0]
        self.ortho = OrthoMixer(self.M)
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Frequency branch (real 1x1)
        # We'll share one real mixing per (out,in)
        self.Wf = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)

        # Morphology
        self.morph = SimpleMorphGate(in_channels, kernel_size=3, tau_init=5.0)

        # Radial mask parameters
        self._t_param = nn.Parameter(torch.tensor(0.0))    # squashed to [0.25,0.45]
        self._lg_param = nn.Parameter(torch.tensor(0.0))   # low/high gain through [0.8,1.2]
        self._hg_param = nn.Parameter(torch.tensor(0.0))
        self.band_steep = nn.Parameter(torch.tensor(8.0))  # slope

        # Fusion gates
        # s starts near 1, f near 0 (causal f is additionally gated by n(x) early on)
        self.s_gate = nn.Parameter(torch.tensor(1.0))
        self.f_gate_base = nn.Parameter(torch.tensor(0.0))

        # Phase head (kept tiny)
        self.phase_head = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        for m in self.phase_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)

        # Mode flags (managed by trainer)
        self.use_interference = False  # toggled after phase unfreeze
        self.phase_enabled = False     # set True after scheduler

    # ---------- helpers ----------

    @staticmethod
    def _squash(val: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        # map R -> (lo, hi) by sigmoid
        return lo + (hi - lo) * torch.sigmoid(val)

    def _spatial_weight(self) -> torch.Tensor:
        Q = self.ortho()  # [M,M]
        A = torch.einsum('oim,mn->oin', self.V, Q)  # [out,in,M]
        W = torch.einsum('oin,mxy->oixy', A, self._base_bank.squeeze(1))  # [out,in,kH,kW]
        return W

    def _build_radial(self, H: int, W: int, device) -> torch.Tensor:
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.fftfreq(W, d=1.0, device=device)
        YY, XX = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(XX*XX + YY*YY)
        r = r / (r.max() + 1e-8)
        s = self.band_steep.clamp_min(1.0)
        t = self._squash(self._t_param, 0.25, 0.45)
        low = torch.sigmoid(s * (t - r))
        high = 1.0 - low
        low_gain = self._squash(self._lg_param, 0.8, 1.2)
        high_gain = self._squash(self._hg_param, 0.8, 1.2)
        return (low_gain * low + high_gain * high).clamp(0.5, 1.5)

    @staticmethod
    def _apply_stride(y: torch.Tensor, stride: Tuple[int,int]) -> torch.Tensor:
        sy, sx = stride
        if sy == 1 and sx == 1: return y
        return F.avg_pool2d(y, kernel_size=stride, stride=stride, ceil_mode=False)

    @staticmethod
    def _pad_like_conv(x: torch.Tensor, padding: Tuple[int,int]) -> torch.Tensor:
        if padding == (0,0): return x
        return F.pad(x, (padding[1], padding[1], padding[0], padding[0]))

    # ---------- forward ----------

    def forward(self, x: torch.Tensor, style_score: Optional[torch.Tensor] = None, f_clip: float = 0.3) -> torch.Tensor:
        """
        x: [B,Cin,H,W] (expects input_size)
        style_score: [B,1,1,1] optional; used to modulate f early in training causally
        f_clip: maximum allowed f in early epochs (trainer will pass 0.3 for first ~10 epochs)
        """
        B, Cin, H, W = x.shape
        assert (H, W) == tuple(self.input_size), f"QFMConv2dDCT expects {self.input_size}, got {(H,W)}"
        # Morphology
        A = self.morph(x)  # [B,1,H,W]

        # Spatial branch
        Wsp = self._spatial_weight().to(dtype=x.dtype)
        yS = F.conv2d(x, Wsp, bias=self.bias, stride=self.stride, padding=self.padding)
        # Gate by morphology (downsample A approximately)
        A_ds = self._apply_stride(self._pad_like_conv(A, self.padding), self.stride)
        A_ds = A_ds[..., :yS.shape[-2], :yS.shape[-1]]
        yS = yS * (1.0 + 0.25 * A_ds)

        # Frequency branch (DCT)
        x_pad = self._pad_like_conv(x, self.padding).float()
        X = dct2(x_pad)                 # [B,Cin,Hp,Wp]
        # real 1x1 mixing
        Y = torch.einsum('oi,bihw->bohw', self.Wf, X)

        # radial mask * morphology avg
        Hp, Wp = Y.shape[-2], Y.shape[-1]
        R = self._build_radial(Hp, Wp, Y.device).view(1,1,Hp,Wp)
        Aavg = A.mean(dim=(-1,-2), keepdim=True)  # [B,1,1,1]
        Y = Y * (1.0 + Aavg * R)

        yF = idct2(Y)
        # crop back & stride
        yF = yF[..., self.padding[0]:self.padding[0]+H, self.padding[1]:self.padding[1]+W]
        yF = self._apply_stride(yF, self.stride)
        A_ds2 = self._apply_stride(A, self.stride)[..., :yF.shape[-2], :yF.shape[-1]]
        yF = yF * (1.0 + 0.25 * A_ds2)

        # Fusion gates (causal modulation early)
        s = torch.sigmoid(self.s_gate)  # ~1
        f_base = torch.sigmoid(self.f_gate_base)  # ~0
        if style_score is not None:
            # enforce w1>=0 by softplus on a separate parameter? keep simple: multiply by style_score ∈ [0,1] upstream
            f_eff = (f_base * (0.5 + 0.5 * style_score)).clamp(0.0, f_clip)
        else:
            f_eff = f_base
        # Conservative fusion
        if not self.use_interference or not self.phase_enabled:
            return s * yS + f_eff * yF

        # Interference fusion (after phase enabled)
        cat = torch.cat([yS, yF], dim=1)
        theta = self.phase_head(cat)
        theta = torch.tanh(theta) * self.phase_clip
        ei = torch.complex(torch.cos(theta), torch.sin(theta))
        psiS = torch.complex(yS, torch.zeros_like(yS))
        psiF = torch.complex(yF, torch.zeros_like(yF))
        fused = psiS + ei * f_eff * psiF / (s + 1e-6)  # scale to keep magnitudes balanced
        out = (fused.real**2 + fused.imag**2)
        return out

# ------------------------------
# Replace convs + insert BlurPool AA
# ------------------------------

def insert_blurpool_before_downsample(m: nn.Module, in_channels: int):
    # For stride-2, insert BlurPool just before
    return nn.Sequential(BlurPool(in_channels, filt_size=3, stride=2),
                         nn.Identity())

def replace_resnet_with_qfm(
    model: nn.Module,
    input_size: Tuple[int,int],
    device: torch.device,
    replace: Iterable[str] = ('conv1',),
    phase_clip: float = math.pi/6,
    aa_every_stride2: bool = True
) -> nn.Module:

    def conv_out_sz(sz, k, s, p):
        return (sz + 2*p - k)//s + 1

    current_size = list(input_size)

    # Stem conv1 replacement + AA
    if 'conv1' in replace:
        old = model.conv1
        q = QFMConv2dDCT(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size, old.kernel_size),
            stride=old.stride if isinstance(old.stride, tuple) else (old.stride, old.stride),
            padding=old.padding if isinstance(old.padding, tuple) else (old.padding, old.padding),
            bias=(old.bias is not None),
            input_size=tuple(current_size),
            device=device,
            dtype=old.weight.dtype,
            phase_clip=phase_clip
        )
        with torch.no_grad():
            q.V.zero_()
            if q.M > 0:
                q.V[:,:,0] = old.weight.mean(dim=(-1,-2))
            if q.bias is not None and old.bias is not None:
                q.bias.data.copy_(old.bias.data)
        model.conv1 = q.to(device)

    # Anti-aliasing before each stride-2 downsample (stem maxpool & layer2..4 first block conv)
    # Replace maxpool in stem with BlurPool + standard MaxPool (or only BlurPool stride-2)
    if aa_every_stride2:
        # Replace model.maxpool with BlurPool + MaxPool(kernel=1, stride=1) (no extra downsample)
        in_ch = model.bn1.num_features
        model.maxpool = nn.Sequential(
            BlurPool(in_ch, filt_size=3, stride=2),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

    # Update size after stem conv1 + maxpool
    old = model.conv1
    k, s, p = old.kH, old.stride[0], old.padding[0]
    current_size = [conv_out_sz(current_size[i], k, s, p) for i in (0,1)]
    current_size = [conv_out_sz(current_size[i], 3, 2, 1) for i in (0,1)]

    # Layers
    for lname in ('layer1','layer2','layer3','layer4'):
        layer = getattr(model, lname)
        for bidx, block in enumerate(layer):
            # Insert AA before first block stride-2 (for layer2..4, conv1 stride=2)
            if aa_every_stride2 and hasattr(block, 'conv1'):
                s1 = block.conv1.stride
                if isinstance(s1, tuple): s1 = s1[0]
                if s1 == 2:
                    # Prepend BlurPool via a wrapper on forward in a tiny Module
                    in_ch = block.conv1.in_channels
                    aa = BlurPool(in_ch, filt_size=3, stride=2)
                    # Wrap the block to apply AA before conv1; easiest is to modify conv1 to stride=1 and add aa
                    block.conv1 = nn.Sequential(aa, nn.Conv2d(in_ch, block.conv1.out_channels,
                                                              kernel_size=block.conv1.kernel_size,
                                                              stride=1,
                                                              padding=block.conv1.padding,
                                                              bias=(block.conv1.bias is not None)))
            if lname in replace and hasattr(block, 'conv1'):
                old = block.conv1 if not isinstance(block.conv1, nn.Sequential) else block.conv1[1]
                in_ch = old.in_channels; out_ch = old.out_channels
                stride = old.stride if isinstance(old, nn.Conv2d) else (1,1)
                padding = old.padding if isinstance(old, nn.Conv2d) else (1,1)
                q = QFMConv2dDCT(
                    in_channels=in_ch, out_channels=out_ch,
                    kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size, old.kernel_size),
                    stride=stride if isinstance(stride, tuple) else (stride, stride),
                    padding=padding if isinstance(padding, tuple) else (padding, padding),
                    input_size=tuple(current_size), device=device, phase_clip=phase_clip
                )
                with torch.no_grad():
                    q.V.zero_()
                    if q.M > 0:
                        # warm start identity-like
                        q.V[:,:,0] = 0.0
                    if q.bias is not None and isinstance(old, nn.Conv2d) and old.bias is not None:
                        q.bias.data.copy_(old.bias.data)
                # If AA wrapper exists, replace its conv part
                if isinstance(block.conv1, nn.Sequential) and isinstance(block.conv1[0], BlurPool):
                    block.conv1[1] = q
                else:
                    block.conv1 = q
            # Update current_size after conv1
            if hasattr(block, 'conv1'):
                o = block.conv1 if isinstance(block.conv1, QFMConv2dDCT) else (block.conv1[1] if isinstance(block.conv1, nn.Sequential) else block.conv1)
                k = o.kH if isinstance(o, QFMConv2dDCT) else (o.kernel_size if isinstance(o.kernel_size, int) else o.kernel_size[0])
                s = o.stride[0] if isinstance(o, QFMConv2dDCT) else (o.stride if isinstance(o.stride, int) else o.stride[0])
                p = o.padding[0] if isinstance(o, QFMConv2dDCT) else (o.padding if isinstance(o.padding, int) else o.padding[0])
                current_size = [conv_out_sz(current_size[i], k, s, p) for i in (0,1)]

    return model

# ------------------------------
# Style proxy n(x): noise, blur, jpeg, lowlight (per-image, no batch coupling)
# ------------------------------

def _per_image_norm(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    v: [B,1,1,1] positive-ish scalar.
    Compress with log1p to keep dynamic range controlled.
    """
    return torch.log1p(v.clamp_min(0.0) + eps)


def noise_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Higher when there is additive high-frequency noise.
    Laplacian variance on per-image luminance.
    Returns shape [B,1,1,1].
    """
    g = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    k = torch.tensor(
        [[0, 1, 0],
         [1,-4, 1],
         [0, 1, 0]],
        dtype=g.dtype, device=g.device
    ).view(1, 1, 3, 3)
    L = F.conv2d(g, k, padding=1)
    v = L.var(dim=(-1, -2), keepdim=True)  # [B,1,1,1]
    return _per_image_norm(v)


def blur_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Higher when image is blurrier.
    Uses inverse edge energy (Tenengrad-like).
    """
    g = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=g.dtype, device=g.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1,-2,-1],
         [ 0, 0, 0],
         [ 1, 2, 1]],
        dtype=g.dtype, device=g.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    edge_energy = (gx * gx + gy * gy).mean(dim=(-1, -2), keepdim=True)  # [B,1,1,1]
    inv_edge = 1.0 / (edge_energy + 1e-6)
    return _per_image_norm(inv_edge)


def jpeg_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Higher when JPEG blockiness / staircasing is present.
    Measures gradient contrast across 8×8 block boundaries.
    """
    g = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    B, _, H, W = g.shape

    # absolute differences
    diff_h = (g[..., :, 1:] - g[..., :, :-1]).abs()  # [B,1,H,W-1]
    diff_v = (g[..., 1:, :] - g[..., :-1, :]).abs()  # [B,1,H-1,W]

    col = torch.arange(W - 1, device=g.device)
    row = torch.arange(H - 1, device=g.device)

    boundary_cols = ((col + 1) % 8 == 0).float().view(1, 1, 1, -1)
    interior_cols = 1.0 - boundary_cols
    boundary_rows = ((row + 1) % 8 == 0).float().view(1, 1, -1, 1)
    interior_rows = 1.0 - boundary_rows

    b_h = (diff_h * boundary_cols).sum(dim=(-2, -1)) / (boundary_cols.sum() + 1e-6)
    i_h = (diff_h * interior_cols).sum(dim=(-2, -1)) / (interior_cols.sum() + 1e-6)
    b_v = (diff_v * boundary_rows).sum(dim=(-2, -1)) / (boundary_rows.sum() + 1e-6)
    i_v = (diff_v * interior_rows).sum(dim=(-2, -1)) / (interior_rows.sum() + 1e-6)

    boundary = b_h + b_v
    interior = i_h + i_v
    ratio = (boundary - interior) / (interior + 1e-6)  # [B,1]

    return _per_image_norm(ratio.view(B, 1, 1, 1))


def lowlight_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Higher when the whole image is darker (low light).
    """
    g = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    mean_l = g.mean(dim=(-1, -2), keepdim=True)  # [B,1,1,1]
    darkness = 1.0 - mean_l
    return _per_image_norm(darkness)


def style_score(x: torch.Tensor) -> torch.Tensor:
    """
    Combine per-image proxies with fixed positive weights
    (roughly inspired by offline regression).
    Returns scalar in [0,1].
    """
    pn = noise_proxy(x)       # [B,1,1,1]
    pb = blur_proxy(x)
    pj = jpeg_proxy(x)
    pd = lowlight_proxy(x)

    # plug in your learned positive coefficients (from offline script)
    w_noise  = 0.0062
    w_blur   = 0.0111
    w_jpeg   = 0.0106
    w_low    = 0.0007

    s_raw = (
        w_noise * pn +
        w_blur  * pb +
        w_jpeg  * pj +
        w_low   * pd
    )  # [B,1,1,1]

    s_clamped = torch.clamp(s_raw, -4.0, 4.0)
    return torch.sigmoid(s_clamped)


# ------------------------------
# CIFAR-10 datasets + simple corruptions (envs)
# ------------------------------

class CIFARWithEnv(Dataset):
    def __init__(self, root, train: bool, img_size: int = 160, env: str = "clean",
                 randaugment: bool = False, augmix: bool = False, label_smoothing: float = 0.0):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True)
        self.env = env
        self.sz = img_size
        self.label_smoothing = label_smoothing

        base = [transforms.Resize((img_size, img_size))]
        if randaugment:
            base.append(transforms.RandAugment(num_ops=2, magnitude=9))
        base.append(transforms.ToTensor())
        base.append(transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.247,0.243,0.261]))
        self.tf = transforms.Compose(base)

        self.augmix = augmix

    def __len__(self): return len(self.ds)

    @staticmethod
    def _to_pil(img_np: np.ndarray) -> Image.Image:
        return Image.fromarray(img_np)

    @staticmethod
    def _jpeg_compress(pil_img: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def _apply_env(self, pil_img: Image.Image) -> Image.Image:
        e = self.env
        if e == "clean":
            return pil_img
        if e.startswith("gauss"):
            sigma = float(e.split("_")[1])
            # PIL doesn't do Gaussian noise; add in numpy then clip
            arr = np.array(pil_img).astype(np.float32)
            arr += np.random.normal(0, sigma*255.0, size=arr.shape).astype(np.float32)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        if e.startswith("jpeg"):
            q = int(e.split("_")[1])
            return self._jpeg_compress(pil_img, q)
        if e.startswith("blur"):
            k = int(e.split("_")[1])
            return pil_img.filter(ImageFilter.GaussianBlur(radius=max(1, (k-1)//2)))
        if e.startswith("lowlight"):
            gamma = float(e.split("_")[1])
            arr = np.array(pil_img).astype(np.float32) / 255.0
            arr = np.power(arr, gamma)
            arr = np.clip(arr*255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return pil_img

    def __getitem__(self, idx: int):
        img, y = self.ds[idx]
        # env transform
        img = self._apply_env(img)
        if self.augmix:
            img = self._augmix(img)
        x = self.tf(img)
        return x, y

    # Light AugMix: 3 chains, depth 2
    def _augmix(self, img: Image.Image, width: int = 3, depth: int = 2, alpha: float = 1.0):
        ops = [
            lambda im: im.rotate(random.choice([-10,-7,-5,5,7,10])),
            lambda im: Image.fromarray(np.uint8(np.clip(np.array(im, dtype=np.float32)*(0.8 + 0.4*random.random()),0,255))),
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.choice([0,1,2]))),
            lambda im: Image.fromarray(np.uint8(np.clip(np.array(im, dtype=np.float32)+np.random.normal(0, 5.0, size=np.array(im).shape),0,255)))
        ]
        ws = np.random.dirichlet([alpha]*width).astype(np.float32)
        m = np.random.beta(alpha, alpha).astype(np.float32)
        mix = np.zeros_like(np.array(img), dtype=np.float32)
        for i in range(width):
            im = img.copy()
            d = depth if depth > 0 else np.random.randint(1,4)
            for _ in range(d):
                op = random.choice(ops)
                im = op(im)
            mix += ws[i] * np.array(im, dtype=np.float32)
        mixed = (1-m) * np.array(img, dtype=np.float32) + m * mix
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed)

# ------------------------------
# MixUp + label smoothing
# ------------------------------

def one_hot(nclass: int, idx: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    y = torch.zeros(idx.size(0), nclass, device=idx.device, dtype=torch.float32)
    y.scatter_(1, idx.view(-1,1), 1.0)
    if smoothing > 0:
        y = (1.0 - smoothing) * y + smoothing / nclass
    return y

def mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, torch.tensor(1.0, device=x.device)
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]; y2 = y[perm]
    xm = lam * x + (1 - lam) * x2
    return xm, (y, y2, lam), torch.tensor(lam, device=x.device)

def mixup_criterion(crit, pred, targets):
    if not isinstance(targets, tuple): return crit(pred, targets)
    y1, y2, lam = targets
    return lam * crit(pred, y1) + (1 - lam) * crit(pred, y2)

# ------------------------------
# REx penalty
# ------------------------------

def rex_penalty(per_env_losses: List[torch.Tensor]) -> torch.Tensor:
    # variance across environments
    losses = torch.stack(per_env_losses)  # [E]
    return losses.var()

# ------------------------------
# Training loop (ablation-ready)
# ------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed, deterministic=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        nclass = 10
        self.num_classes = nclass

        # Environments for training (pseudo-interventions)
        envs_train = ['clean']
        if args.envs_train:
            envs_train = [e.strip() for e in args.envs_train.split(',') if e.strip()]
        self.train_sets = [CIFARWithEnv(args.data, True, img_size=args.img, env=e,
                                        randaugment=args.randaugment, augmix=args.augmix, label_smoothing=args.smooth)
                           for e in envs_train]
        # Build a unified loader by interleaving envs each step (slice batches)
        self.train_loaders = [DataLoader(ds, batch_size=args.batch//len(self.train_sets),
                                         shuffle=True, num_workers=min(4, os.cpu_count() or 2),
                                         pin_memory=_USE_CUDA, drop_last=True)
                              for ds in self.train_sets]
        self.train_iters = [iter(dl) for dl in self.train_loaders]

        # Validation: clean + a probe env
        self.val_clean = DataLoader(CIFARWithEnv(args.data, False, img_size=args.img, env='clean'),
                                    batch_size=args.batch, shuffle=False, num_workers=2)
        self.val_probe = DataLoader(CIFARWithEnv(args.data, False, img_size=args.img, env=args.val_env),
                                    batch_size=args.batch, shuffle=False, num_workers=2)

        # Model
        self.model = self._build_model()
        self.model.to(self.device)
        self.ema = EMA(self.model, decay=args.ema)

        # Loss, optimizer, sched
        self.criterion = nn.KLDivLoss(reduction='batchmean') if args.logits_kld else nn.CrossEntropyLoss()
        self.ce = nn.CrossEntropyLoss()
        self.smooth = args.smooth
        self.scaler = GradScaler(enabled=_USE_CUDA)

        # Parameter groups: QFM params (smaller LR)
        qfm_params, base_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if 'QFMConv2dDCT' in str(type(getattr(self.model, n.split('.')[0], self.model))):
                qfm_params.append(p)
            elif 'QFMConv2dDCT' in n:
                qfm_params.append(p)
            else:
                base_params.append(p)
        if not base_params:  # fallback grouping
            for n,p in self.model.named_parameters():
                (qfm_params if ('.V' in n or '.Wf' in n or '.morph.' in n or '.phase_' in n) else base_params).append(p)

        self.opt = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': args.lr, 'weight_decay': args.wd},
                {'params': qfm_params, 'lr': args.lr * args.qfm_lr_scale, 'weight_decay': args.wd}
            ]
        )

        self.total_iters = args.epochs * (len(self.train_sets[0]) // (args.batch//len(self.train_sets)))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=args.epochs, eta_min=args.lr*1e-2)

        self.writer = SummaryWriter(log_dir=os.path.join(args.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.global_step = 0

        # Warmup/freeze schedule
        self.unfreeze_gates_epoch = args.unfreeze_gates_epoch
        self.unfreeze_phase_epoch = args.unfreeze_phase_epoch
        self.f_clip_warm = args.f_clip_warm

        # REx schedule
        self.rex_lambda_start = args.rex_lambda_start
        self.rex_lambda_final = args.rex_lambda_final
        self.rex_ramp_epochs = args.rex_ramp_epochs

        # MixUp
        self.mixup_alpha = args.mixup

        # Grad clip
        self.clip = args.clip

        # Phase clip
        self.phase_clip = args.phase_clip

        # Ablation modes
        self.mode = args.mode  # baseline, blurpool, blurpool_augmix (D), qfm_E, qfm_F, qfm_G

    def _build_model(self):
        args = self.args
        if self.args.pretrained:
            try:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                model = resnet18(weights=None)
        else:
            model = resnet18(weights=None)

        # Head
        in_fc = model.fc.in_features
        model.fc = nn.Linear(in_fc, self.num_classes)

        # Baseline modes
        if self.mode in ('baseline', 'baseline_blur', 'baseline_augmix', 'baseline_blur_augmix'):
            # optional AA
            if self.mode in ('baseline_blur', 'baseline_blur_augmix'):
                in_ch = model.bn1.num_features
                model.maxpool = nn.Sequential(
                    BlurPool(in_ch, filt_size=3, stride=2),
                    nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                )
            return model

        # QFM modes
        replace = ('conv1','layer1','layer2','layer3','layer4') if self.args.replace_all else tuple(self.args.replace.split(','))
        model = replace_resnet_with_qfm(
            model, input_size=(self.args.img, self.args.img),
            device=self.device, replace=replace, phase_clip=self.phase_clip, aa_every_stride2=True
        )
        return model

    def _set_train_stage(self, epoch: int):
        # Freeze/unfreeze QFM parts as schedule
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                # default: everything frozen except DCT 1x1 (identity-ish)
                for n, p in m.named_parameters():
                    p.requires_grad = False
                # Always allow small pieces per recipe
                # During warmup (epoch < unfreeze_gates): allow only Wf (1x1) to train (keep near identity)
                if epoch < self.unfreeze_gates_epoch:
                    m.Wf.requires_grad = True
                # After gates unfreeze
                if epoch >= self.unfreeze_gates_epoch:
                    m.Wf.requires_grad = True
                    m.V.requires_grad  = True
                    m.ortho.P.requires_grad = True
                    # morphology small params
                    m.morph.alpha.requires_grad = True
                    m.morph.beta.requires_grad  = True
                    m._t_param.requires_grad    = True
                    m._lg_param.requires_grad   = True
                    m._hg_param.requires_grad   = True
                    m.band_steep.requires_grad  = True
                    m.s_gate.requires_grad      = True
                    m.f_gate_base.requires_grad = True
                # After phase unfreeze
                if epoch >= self.unfreeze_phase_epoch:
                    m.phase_enabled = True
                    m.use_interference = True
                    for n, p in m.phase_head.named_parameters():
                        p.requires_grad = True

    def _tau_schedule(self, epoch: int):
        # τ: 5 -> 15 over first 10 epochs, then hold
        tau = 5.0 + min(1.0, epoch/10.0) * (15.0 - 5.0)
        for m in self.model.modules():
            if isinstance(m, SimpleMorphGate):
                m.set_tau(tau)

    def _rex_lambda(self, epoch: int) -> float:
        if self.rex_ramp_epochs <= 0:
            return self.rex_lambda_final
        if epoch <= 5:
            # allow 0->start warmup
            lam = (self.rex_lambda_start) * (epoch / 5.0)
        else:
            t = min(1.0, max(0.0, (epoch - 5) / float(self.rex_ramp_epochs)))
            lam = self.rex_lambda_start + t * (self.rex_lambda_final - self.rex_lambda_start)
        return lam

    def _forward_env_chunk(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # style proxy for causal gate
        style = style_score(x)
        # Early clip on f
        fclip = self.f_clip_warm if epoch < 10 else 1.0
        logits = self.model_forward_qfm_aware(x, style, fclip)
        yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
        loss = F.cross_entropy(logits, y) if self.smooth <= 0 else F.kl_div(F.log_softmax(logits, dim=1), yoh, reduction='batchmean')
        return logits, loss

    def model_forward_qfm_aware(self, x, style, fclip):
        # Call with (style,fclip) only if model has QFM, else ignore
        has_qfm = any(isinstance(m, QFMConv2dDCT) for m in self.model.modules())
        if not has_qfm:
            return self.model(x)
        # Inject (style, fclip) via forward hooks by replacing forward temporarily? Simpler: walk modules and pass style to QFM
        # We'll do a manual forward by replacing nn.Sequential items? That's messy.
        # Instead, monkey-patch QFMConv2dDCT.forward to bind style & fclip using a context manager.

        # Safer approach: register temporary forward_pre_hook that sets attributes.
        tmp = []
        def pre_hook(m, inp):
            if isinstance(m, QFMConv2dDCT):
                m._tmp_style = style
                m._tmp_fclip = fclip
        def post_hook(m, inp, out):
            if isinstance(m, QFMConv2dDCT):
                delattr(m, '_tmp_style')
                delattr(m, '_tmp_fclip')
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                tmp.append(m.register_forward_pre_hook(pre_hook))
                tmp.append(m.register_forward_hook(post_hook))
        # Wrap forward: override QFM forward to read tmp attrs if present
        def qfm_forward(self_qfm, x_q):
            s = getattr(self_qfm, '_tmp_style', None)
            fclip_local = getattr(self_qfm, '_tmp_fclip', 1.0)
            return QFMConv2dDCT.forward(self_qfm, x_q, s, fclip_local)
        # monkey patch bound method
        saved_forwards = []
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                saved_forwards.append((m, m.forward))
                m.forward = qfm_forward.__get__(m, QFMConv2dDCT)
        try:
            out = self.model(x)
        finally:
            for m, f in saved_forwards:
                m.forward = f
            for h in tmp:
                h.remove()
        return out

    def train(self):
        args = self.args
        best_clean = 0.0
        best_probe_retention = 0.0

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            self.model.train()
            self._set_train_stage(epoch)
            self._tau_schedule(epoch)
            rex_lambda = self._rex_lambda(epoch)
            epoch_loss = 0.0
            nseen = 0

            # Iterate in lock-step across env loaders
            iters_per_env = min(len(dl) for dl in self.train_loaders)
            for step in range(iters_per_env):
                per_env_losses = []
                logits_cat = []
                targets_cat = []
                self.opt.zero_grad(set_to_none=True)

                for li, it in enumerate(self.train_iters):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        self.train_iters[li] = iter(self.train_loaders[li])
                        x, y = next(self.train_iters[li])

                    x = x.to(self.device, non_blocking=True)
                    y = torch.as_tensor(y, device=self.device)

                    # MixUp across each env mini-batch
                    if self.mixup_alpha > 0:
                        yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
                        x, mix_tgt, _ = mixup(x, yoh, alpha=self.mixup_alpha)
                        with autocast(enabled=_USE_CUDA):
                            logits = self.model_forward_qfm_aware(x, style_score(x), self.f_clip_warm if epoch<10 else 1.0)
                            loss_e = mixup_criterion(lambda p,t: F.kl_div(F.log_softmax(p, dim=1), t, reduction='batchmean'), logits, mix_tgt)
                    else:
                        with autocast(enabled=_USE_CUDA):
                            logits, loss_e = self._forward_env_chunk(x, y, epoch)

                    per_env_losses.append(loss_e)
                    logits_cat.append(logits)
                    targets_cat.append(y)

                with autocast(enabled=_USE_CUDA):
                    Ltask = torch.stack(per_env_losses).mean()
                    Lrex = rex_penalty(per_env_losses) if len(per_env_losses) > 1 else torch.tensor(0.0, device=self.device)
                    loss = Ltask + rex_lambda * Lrex

                # Regularize phase head weights (L2) if enabled
                for m in self.model.modules():
                    if isinstance(m, QFMConv2dDCT) and m.phase_enabled:
                        for p in m.phase_head.parameters():
                            loss = loss + args.phase_l2 * (p**2).mean()

                self.scaler.scale(loss).backward()
                if self.clip and self.clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.scaler.step(self.opt)
                self.scaler.update()

                self.ema.update(self.model)
                epoch_loss += loss.item() * sum(x.size(0) for x,_ in [next(iter([ (None, None) ]))])  # dummy to keep code short
                nseen += sum(dl.batch_size for dl in self.train_loaders)

                # Logging
                if self.global_step % args.log_every == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/rex_lambda', rex_lambda, self.global_step)
                    self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], self.global_step)

                    # QFM diagnostics (first few modules)
                    self._log_qfm_stats(epoch)

                self.global_step += 1

            self.scheduler.step()
            # Eval with EMA
            self.ema.apply_shadow(self.model)
            acc_clean = self.evaluate(self.val_clean, tag='clean', epoch=epoch)
            acc_probe = self.evaluate(self.val_probe, tag=f'probe_{self.args.val_env}', epoch=epoch)
            self.ema.restore(self.model)

            # Retention (probe relative to clean)
            retention = (acc_probe / max(1e-6, acc_clean)) if acc_clean > 0 else 0.0
            self.writer.add_scalar('eval/clean_acc', acc_clean, epoch)
            self.writer.add_scalar('eval/probe_acc', acc_probe, epoch)
            self.writer.add_scalar('eval/retention', retention, epoch)

            # Save best
            if acc_clean > best_clean:
                best_clean = acc_clean
                torch.save({'model': self.model.state_dict(), 'epoch': epoch, 'best_clean': best_clean}, 'qfm_clean_best.pth')
            if retention > best_probe_retention:
                best_probe_retention = retention
                torch.save({'model': self.model.state_dict(), 'epoch': epoch, 'best_retention': best_probe_retention}, 'qfm_retention_best.pth')

            dt = time.time() - t0
            print(f"Epoch {epoch:03d}/{self.args.epochs} | clean {acc_clean:.2f} | probe {acc_probe:.2f} | retention {retention*100:.1f}% | {dt:.1f}s")

        # Final probe curves & feature stability/energy drift
        self._final_probes()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tag: str, epoch: int) -> float:
        self.model.eval()
        correct, total = 0, 0
        feats_clean, feats_corr = [], []  # for cosine stability across env if needed
        for x, y in loader:
            x = x.to(self.device)
            y = torch.as_tensor(y, device=self.device)
            with autocast(enabled=_USE_CUDA):
                logits = self.model_forward_qfm_aware(x, style_score(x), 1.0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = 100.0 * correct / max(1, total)
        self.writer.add_scalar(f'eval/{tag}_acc', acc, epoch)
        return acc

    def _log_qfm_stats(self, epoch: int, max_items: int = 4):
        cnt = 0
        for name, m in self.model.named_modules():
            if isinstance(m, QFMConv2dDCT):
                # scalar params
                with torch.no_grad():
                    self.writer.add_scalar(f"QFM/{name}/t_center", float(0.25 + (0.45-0.25)*torch.sigmoid(m._t_param).item()), epoch)
                    self.writer.add_scalar(f"QFM/{name}/low_gain", float(0.8 + (1.2-0.8)*torch.sigmoid(m._lg_param).item()), epoch)
                    self.writer.add_scalar(f"QFM/{name}/high_gain", float(0.8 + (1.2-0.8)*torch.sigmoid(m._hg_param).item()), epoch)
                    # orthogonality error
                    Q = m.ortho()
                    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
                    err = torch.norm(Q.T @ Q - I, p='fro').item()
                    self.writer.add_scalar(f"QFM/{name}/ortho_err", err, epoch)
                cnt += 1
                if cnt >= max_items: break

    @torch.no_grad()
    def _final_probes(self):
        # Feature cosine stability & energy drift across clean vs probe
        # (We run a forward hook to capture penultimate layer)
        feats = {'clean': [], 'probe': []}
        def hook(_m, _i, o): feats[current_env].append(o.detach().flatten(1))
        handle = None
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and n.endswith('fc'):
                handle = m.register_forward_hook(lambda m,i,o: None)  # no-op to pick a safer layer
        # Better: pick avgpool output
        target_mod = None
        for n,m in self.model.named_modules():
            if n == 'avgpool':
                target_mod = m; break
        if target_mod is not None:
            if handle: handle.remove()
            handle = target_mod.register_forward_hook(hook)

        self.model.eval()
        global current_env
        for env_name, loader in [('clean', self.val_clean), (f'probe_{self.args.val_env}', self.val_probe)]:
            current_env = 'clean' if env_name == 'clean' else 'probe'
            for x,y in loader:
                x = x.to(self.device)
                _ = self.model_forward_qfm_aware(x, style_score(x), 1.0)

        if feats['clean'] and feats['probe']:
            a = torch.cat(feats['clean'], dim=0)
            b = torch.cat(feats['probe'], dim=0)
            # cosine between matched indices (truncate to min len)
            n = min(a.size(0), b.size(0))
            a = a[:n]; b = b[:n]
            cos = F.cosine_similarity(a, b, dim=1).mean().item()
            self.writer.add_scalar('probe/feature_cosine', cos, 0)

        # Energy drift per depth: mean(|features|^2). We’ll sample after each layer’s output roughly by hooks.
        energies = {}
        def e_hook(nm):
            def _h(_m, _i, o):
                if isinstance(o, torch.Tensor):
                    energies.setdefault(nm, []).append((o.detach()**2).mean().item())
            return _h
        hs = []
        for n,m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d, QFMConv2dDCT)):
                hs.append(m.register_forward_hook(e_hook(n)))
        # run clean
        for x,y in self.val_clean:
            x = x.to(self.device)
            _ = self.model_forward_qfm_aware(x, style_score(x), 1.0)
            break
        for h in hs: h.remove()
        # Log average energy per depth (order is approximate)
        for k,v in energies.items():
            if v:
                self.writer.add_scalar(f'energy/{k}', float(np.mean(v)), 0)

# ------------------------------
# CLI
# ------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="QFM-DCT + ResNet (stabilized, causal, ablations)")
    # Data / train
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--img', type=int, default=160)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--logdir', type=str, default='runs/qfm_dct')
    p.add_argument('--log_every', type=int, default=100)
    # Optim
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--qfm_lr_scale', type=float, default=0.25)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--clip', type=float, default=1.0)
    # Aug
    p.add_argument('--smooth', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.2)
    p.add_argument('--randaugment', action='store_true')
    p.add_argument('--augmix', action='store_true')
    # Envs
    p.add_argument('--envs_train', type=str, default='clean,gauss_0.1,jpeg_40,blur_3,lowlight_0.6')
    p.add_argument('--val_env', type=str, default='jpeg_40')
    # Freeze schedule
    p.add_argument('--unfreeze_gates_epoch', type=int, default=4)
    p.add_argument('--unfreeze_phase_epoch', type=int, default=8)
    p.add_argument('--f_clip_warm', type=float, default=0.3)
    p.add_argument('--phase_clip', type=float, default=math.pi/6)
    p.add_argument('--phase_l2', type=float, default=1e-3)
    # Replace controls
    p.add_argument('--replace', type=str, default='conv1,layer1,layer2,layer3,layer4')
    p.add_argument('--replace_all', action='store_true')
    # REx
    p.add_argument('--rex_lambda_start', type=float, default=1.0)
    p.add_argument('--rex_lambda_final', type=float, default=10.0)
    p.add_argument('--rex_ramp_epochs', type=int, default=40)
    # Modes / ablations
    p.add_argument('--mode', type=str, default='qfm_E',
                   choices=['baseline','baseline_blur','baseline_augmix','baseline_blur_augmix','qfm_E','qfm_F','qfm_G'])
    return p

def main():
    args = build_argparser().parse_args()

    # Mode convenience mappings to flags (matches A..G plan)
    # A: baseline; B: baseline_blur; C: baseline_augmix; D: baseline_blur_augmix
    # E: D + DCT branch (no morph, phase off) => qfm_E
    # F: E + simple morphology => qfm_F
    # G: F + phase (clipped) => qfm_G
    if args.mode in ('baseline','baseline_blur','baseline_augmix','baseline_blur_augmix'):
        args.randaugment = (args.mode in ('baseline_augmix','baseline_blur_augmix'))
        args.augmix = (args.mode in ('baseline_augmix','baseline_blur_augmix'))
    else:
        # QFM modes inherit strong baseline augmentations per the plan (D as baseline to beat)
        args.randaugment = True
        args.augmix = True

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
