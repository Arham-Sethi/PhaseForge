#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QFM-DCT + ResNet Integration (Stabilized, Causal-Robust, Ablations Included)

Post-stride DCT version:
- QFMConv2dDCT does spatial conv (+stride) first, then DCT/IDCT on the feature map.
- No fixed input_size assumptions; works for any H×W that flows through the block.
- Morphology gate uses interpolate to match feature map sizes.
- Style_score(x) drives fusion gate f in a causal way; phase is gated & clipped.

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
        def __init__(self, enabled=False):
            self.enabled = False
        def scale(self, x):
            return x
        def step(self, opt):
            opt.step()
        def update(self):
            pass
        def unscale_(self, opt):
            # no-op on CPU
            pass

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
    _idct_cache[key] = _dct_mat(n, device, dtype).t()
    return _idct_cache[key]

def dct2(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    Th = _dct_mat(H, device, dtype)      # [H,H]
    Tw = _dct_mat(W, device, dtype)      # [W,W]
    x = torch.einsum('bchw,hh->bchw', x, Th.t())
    x = torch.einsum('bchw,ww->bchw', x.permute(0,1,3,2), Tw.t()).permute(0,1,3,2)
    return x

def idct2(X: torch.Tensor) -> torch.Tensor:
    B, C, H, W = X.shape
    device, dtype = X.device, X.dtype
    ThT = _idct_mat(H, device, dtype)
    TwT = _idct_mat(W, device, dtype)
    x = torch.einsum('bchw,hh->bchw', X, ThT.t())
    x = torch.einsum('bchw,ww->bchw', x.permute(0,1,3,2), TwT.t()).permute(0,1,3,2)
    return x

# ------------------------------
# Sobel & soft morphology
# ------------------------------

class SimpleMorphGate(nn.Module):
    """
    M = sigmoid( α |Sobel(x)| + β (dilate(x) - erode(x)) )
    Outputs A in [0,1], shape [B,1,H,W].
    """
    def __init__(self, in_channels: int, kernel_size: int = 3, tau_init: float = 5.0):
        super().__init__()
        assert kernel_size in (3,5,7)
        self.kernel_size = kernel_size
        self.register_buffer('tau', torch.tensor(tau_init, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.register_buffer('kx', torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32
        ).view(1,1,3,3))
        self.register_buffer('ky', torch.tensor(
            [[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32
        ).view(1,1,3,3))

    def set_tau(self, value: float):
        with torch.no_grad():
            self.tau.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=1, keepdim=True)  # grayscale proxy
        gx = F.conv2d(g, self.kx.to(g.device), padding=1)
        gy = F.conv2d(g, self.ky.to(g.device), padding=1)
        edge = torch.sqrt(gx*gx + gy*gy + 1e-12)

        k = self.kernel_size
        pad = k // 2
        patches = F.unfold(g, kernel_size=k, padding=pad)  # [B, k*k, HW]
        B, KK, HW = patches.shape
        H, W = g.shape[-2:]

        patches = patches.view(B, 1, KK, H, W)  # [B,1,KK,H,W]

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
# Base spatial filter bank
# ------------------------------

def _gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float):
    xmax = ks // 2; ymax = ks // 2
    y, x = torch.meshgrid(torch.arange(-ymax, ymax+1),
                          torch.arange(-xmax, xmax+1),
                          indexing='ij')
    x = x.to(torch.float32); y = y.to(torch.float32)
    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    gb = torch.exp(-(x_theta**2 + (gamma**2)*(y_theta**2)) / (2 * sigma**2)) * \
         torch.cos(2 * math.pi * x_theta / lam + psi)
    gb = gb / (gb.abs().sum() + 1e-6)
    return gb

def make_base_filters(kernel_size: int, device: torch.device) -> torch.Tensor:
    ks = kernel_size
    assert ks in (3,5,7), "kernel_size must be 3, 5 or 7."
    bank: List[torch.Tensor] = []

    eye = torch.zeros((ks, ks), dtype=torch.float32); eye[ks//2, ks//2] = 1.0
    bank.append(eye)

    box = torch.ones((ks, ks), dtype=torch.float32) / (ks*ks)
    bank.append(box)

    if ks == 3:
        gauss = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32); gauss /= gauss.sum()
    elif ks == 5:
        gauss = torch.tensor([[1,4,7,4,1],
                              [4,16,26,16,4],
                              [7,26,41,26,7],
                              [4,16,26,16,4],
                              [1,4,7,4,1]], dtype=torch.float32)
        gauss /= gauss.sum()
    else:
        g1d = torch.tensor([1,6,15,20,15,6,1], dtype=torch.float32)
        gauss = torch.outer(g1d, g1d); gauss /= gauss.sum()
    bank.append(gauss)

    if ks == 3:
        sharp = torch.tensor([[0,-1,0],
                              [-1,5,-1],
                              [0,-1,0]], dtype=torch.float32)
    else:
        sharp = -box * (ks*ks - 1)
        sharp[ks//2, ks//2] += ks*ks
    bank.append(sharp)

    if ks == 3:
        lap = torch.tensor([[0,1,0],
                            [1,-4,1],
                            [0,1,0]], dtype=torch.float32)
    else:
        lap = torch.zeros((ks, ks), dtype=torch.float32)
        lap[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[0,1,0],
                                                              [1,-4,1],
                                                              [0,1,0]], dtype=torch.float32)
    bank.append(lap)

    if ks == 3:
        sobel_x = torch.tensor([[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],
                                [0,0,0],
                                [1,2,1]], dtype=torch.float32)
    else:
        sobel_x = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_y = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_x[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,0,1],
                                                                  [-2,0,2],
                                                                  [-1,0,1]], dtype=torch.float32)
        sobel_y[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,-2,-1],
                                                                  [0,0,0],
                                                                  [1,2,1]], dtype=torch.float32)
    bank.extend([sobel_x, sobel_y])

    for theta in (0.0, math.pi/4, math.pi/2, 3*math.pi/4):
        bank.append(_gabor_kernel(ks, 1.2, theta, 4.0, 0.5, 0.0))

    bank = torch.stack(bank, dim=0).unsqueeze(1).to(device=device)
    return bank  # [M,1,ks,ks]

# ------------------------------
# QFMConv2d-DCT (post-stride)
# ------------------------------

class QFMConv2dDCT(nn.Module):
    """
    Stabilized QFM block (post-stride DCT):
      - Spatial: x -> conv (stride, padding) -> yS
      - Frequency: DCT2(yS) -> channel mix -> radial mask -> IDCT2 -> yF
      - Morphology gate from input x, interpolated to match yS / yF.
      - Fusion: y = s*yS + f*yF; optional interference-with-phase once enabled.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int,int]],
        stride: Union[int, Tuple[int,int]] = 1,
        padding: Union[int, Tuple[int,int]] = 0,
        bias: bool = True,
        phase_clip: float = math.pi/6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
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
        self.phase_clip = phase_clip
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        # Spatial branch
        self.register_buffer('_base_bank', make_base_filters(self.kH, self.device))
        self.M = self._base_bank.shape[0]
        self.ortho = OrthoMixer(self.M)
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Frequency branch: mix across output channels
        self.Wf = nn.Parameter(torch.eye(out_channels, dtype=torch.float32))

        # Morphology gate
        self.morph = SimpleMorphGate(in_channels, kernel_size=3, tau_init=5.0)

        # Radial mask parameters
        self._t_param = nn.Parameter(torch.tensor(0.0))    # squashed to [0.25,0.45]
        self._lg_param = nn.Parameter(torch.tensor(0.0))   # [0.8,1.2]
        self._hg_param = nn.Parameter(torch.tensor(0.0))
        self.band_steep = nn.Parameter(torch.tensor(8.0))  # slope

        # Fusion gates
        self.s_gate = nn.Parameter(torch.tensor(1.0))
        self.f_gate_base = nn.Parameter(torch.tensor(0.0))

        # Phase head
        self.phase_head = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        for m in self.phase_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)

        self.use_interference = False
        self.phase_enabled = False

    @staticmethod
    def _squash(val: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, style_score: Optional[torch.Tensor] = None,
                f_clip: float = 0.3) -> torch.Tensor:
        """
        x: [B,Cin,H,W]
        style_score: [B,1,1,1] in [0,1] (optional)
        """
        B, Cin, H, W = x.shape

        # Morphology map on input
        A = self.morph(x)  # [B,1,H,W]

        # Spatial conv
        Wsp = self._spatial_weight().to(dtype=x.dtype, device=x.device)
        yS = F.conv2d(x, Wsp, bias=self.bias, stride=self.stride, padding=self.padding)

        # Morph gate onto spatial branch
        A_ds = F.interpolate(A, size=yS.shape[-2:], mode='bilinear', align_corners=False)
        yS = yS * (1.0 + 0.25 * A_ds)

        # Frequency branch on yS (post-stride)
        Y = dct2(yS.float())  # [B,Cout,Hs,Ws]
        Y = torch.einsum('oc,bchw->bohw', self.Wf.to(Y.dtype).to(Y.device), Y)

        Hs, Ws = Y.shape[-2], Y.shape[-1]
        R = self._build_radial(Hs, Ws, Y.device).view(1,1,Hs,Ws)
        Aavg = A.mean(dim=(-1,-2), keepdim=True)  # [B,1,1,1]
        Y = Y * (1.0 + Aavg * R)

        yF = idct2(Y)
        yF = yF.to(dtype=x.dtype)
        A_ds2 = F.interpolate(A, size=yF.shape[-2:], mode='bilinear', align_corners=False)
        yF = yF * (1.0 + 0.25 * A_ds2)

        # Fusion gates
        s = torch.sigmoid(self.s_gate)
        f_base = torch.sigmoid(self.f_gate_base)
        if style_score is not None:
            f_eff = (f_base * (0.5 + 0.5 * style_score)).clamp(0.0, f_clip)
        else:
            f_eff = f_base

        if not self.use_interference or not self.phase_enabled:
            return s * yS + f_eff * yF

        # Interference fusion
        cat = torch.cat([yS, yF], dim=1)
        theta = self.phase_head(cat)
        theta = torch.tanh(theta) * self.phase_clip
        ei = torch.complex(torch.cos(theta), torch.sin(theta))
        psiS = torch.complex(yS, torch.zeros_like(yS))
        psiF = torch.complex(yF, torch.zeros_like(yF))
        fused = psiS + ei * f_eff * psiF / (s + 1e-6)
        out = fused.real**2 + fused.imag**2
        return out

# ------------------------------
# Replace convs + insert BlurPool AA
# ------------------------------

def replace_resnet_with_qfm(
    model: nn.Module,
    device: torch.device,
    replace: Iterable[str] = ('conv1',),
    phase_clip: float = math.pi/6,
    aa_every_stride2: bool = True
) -> nn.Module:
    """
    Replace chosen convs with QFMConv2dDCT.
    No shape bookkeeping: QFM handles whatever H×W it gets.
    """

    # Stem conv1
    if 'conv1' in replace:
        old = model.conv1
        q = QFMConv2dDCT(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size, old.kernel_size),
            stride=old.stride if isinstance(old.stride, tuple) else (old.stride, old.stride),
            padding=old.padding if isinstance(old.padding, tuple) else (old.padding, old.padding),
            bias=(old.bias is not None),
            phase_clip=phase_clip,
            device=device,
            dtype=old.weight.dtype
        )
        with torch.no_grad():
            q.V.zero_()
            if q.M > 0:
                q.V[:,:,0] = old.weight.mean(dim=(-1,-2))
            if q.bias is not None and old.bias is not None:
                q.bias.data.copy_(old.bias.data)
        model.conv1 = q.to(device)

    # Stem AA
    if aa_every_stride2:
        in_ch = model.bn1.num_features
        model.maxpool = nn.Sequential(
            BlurPool(in_ch, filt_size=3, stride=2),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

    # Layers
    for lname in ('layer1','layer2','layer3','layer4'):
        layer = getattr(model, lname)
        for bidx, block in enumerate(layer):
            # Insert AA before stride-2 conv1 in block (layers 2..4 usually)
            if aa_every_stride2 and hasattr(block, 'conv1'):
                base_conv = block.conv1
                stride = base_conv.stride if isinstance(base_conv, nn.Conv2d) else getattr(base_conv, 'stride', (1,1))
                if isinstance(stride, tuple): s1 = stride[0]
                else: s1 = stride
                if s1 == 2:
                    in_ch = base_conv.in_channels
                    aa = BlurPool(in_ch, filt_size=3, stride=2)
                    block.conv1 = nn.Sequential(
                        aa,
                        nn.Conv2d(in_ch, base_conv.out_channels,
                                  kernel_size=base_conv.kernel_size,
                                  stride=1,
                                  padding=base_conv.padding,
                                  bias=(base_conv.bias is not None))
                    )

            # Replace conv1 with QFM if requested
            if lname in replace and hasattr(block, 'conv1'):
                old = block.conv1
                if isinstance(old, nn.Sequential) and isinstance(old[0], BlurPool):
                    conv_part = old[1]
                    in_ch = conv_part.in_channels
                    out_ch = conv_part.out_channels
                    k = conv_part.kernel_size if isinstance(conv_part.kernel_size, tuple) else (conv_part.kernel_size,)*2
                    s = conv_part.stride if isinstance(conv_part.stride, tuple) else (conv_part.stride,)*2
                    p = conv_part.padding if isinstance(conv_part.padding, tuple) else (conv_part.padding,)*2
                    q = QFMConv2dDCT(
                        in_channels=in_ch, out_channels=out_ch,
                        kernel_size=k, stride=s, padding=p,
                        bias=(conv_part.bias is not None),
                        phase_clip=phase_clip, device=device
                    )
                    with torch.no_grad():
                        q.V.zero_()
                        if q.M > 0:
                            q.V[:,:,0] = 0.0
                        if q.bias is not None and conv_part.bias is not None:
                            q.bias.data.copy_(conv_part.bias.data)
                    old[1] = q
                else:
                    conv_part = old
                    in_ch = conv_part.in_channels
                    out_ch = conv_part.out_channels
                    k = conv_part.kernel_size if isinstance(conv_part.kernel_size, tuple) else (conv_part.kernel_size,)*2
                    s = conv_part.stride if isinstance(conv_part.stride, tuple) else (conv_part.stride,)*2
                    p = conv_part.padding if isinstance(conv_part.padding, tuple) else (conv_part.padding,)*2
                    q = QFMConv2dDCT(
                        in_channels=in_ch, out_channels=out_ch,
                        kernel_size=k, stride=s, padding=p,
                        bias=(conv_part.bias is not None),
                        phase_clip=phase_clip, device=device
                    )
                    with torch.no_grad():
                        q.V.zero_()
                        if q.M > 0:
                            q.V[:,:,0] = 0.0
                        if q.bias is not None and conv_part.bias is not None:
                            q.bias.data.copy_(conv_part.bias.data)
                    block.conv1 = q

    return model

# ------------------------------
# Style proxy n(x)
# ------------------------------

def _batch_z(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = v.mean(dim=0, keepdim=True)
    s = v.std(dim=0, keepdim=True)
    return (v - m) / (s + eps)

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1,3,1,1)
    g = (x * w).sum(dim=1, keepdim=True)
    return g

def noise_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    k = torch.tensor([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    L = F.conv2d(g, k, padding=1)
    v = L.var(dim=(-1,-2), keepdim=True)
    return _batch_z(v)

def _jpeg_blockiness_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    B, _, H, W = g.shape

    diff_h = (g[..., :, 1:] - g[..., :, :-1]).abs()
    diff_v = (g[..., 1:, :] - g[..., :-1, :]).abs()

    col_idx = torch.arange(W - 1, device=g.device)
    row_idx = torch.arange(H - 1, device=g.device)

    boundary_cols = ((col_idx + 1) % 4 == 0).float().view(1,1,1,-1)
    interior_cols = 1.0 - boundary_cols
    boundary_rows = ((row_idx + 1) % 4 == 0).float().view(1,1,-1,1)
    interior_rows = 1.0 - boundary_rows

    b_h = (diff_h * boundary_cols).sum(dim=(-2,-1)) / (boundary_cols.sum() + 1e-6)
    i_h = (diff_h * interior_cols).sum(dim=(-2,-1)) / (interior_cols.sum() + 1e-6)
    b_v = (diff_v * boundary_rows).sum(dim=(-2,-1)) / (boundary_rows.sum() + 1e-6)
    i_v = (diff_v * interior_rows).sum(dim=(-2,-1)) / (interior_rows.sum() + 1e-6)

    boundary = b_h + b_v
    interior = i_h + i_v
    ratio = (boundary - interior) / (interior + 1e-6)
    return _batch_z(ratio.view(-1,1,1,1))

def _hf_drop_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    box = torch.ones((1,1,3,3), device=g.device, dtype=g.dtype) / 9.0
    g_lp = F.conv2d(g, box, padding=1)
    g_hp = g - g_lp
    hf_energy = (g_hp ** 2).mean(dim=(-1,-2), keepdim=True)
    hf_drop = 1.0 / (hf_energy + 1e-6)
    return _batch_z(hf_drop)

def _fft_blur_proxy(x: torch.Tensor) -> torch.Tensor:
    import torch.fft as tfft
    g = _to_gray(x)
    F_complex = tfft.fft2(g, norm='ortho')
    Fshift = tfft.fftshift(F_complex, dim=(-2,-1))
    mag = (Fshift.real ** 2 + Fshift.imag ** 2) ** 0.5

    B, _, H, W = mag.shape
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=mag.device),
        torch.linspace(-1, 1, W, device=mag.device),
        indexing='ij'
    )
    R = torch.sqrt(X*X + Y*Y)[None,None,:,:]

    lf_mask = (R < 0.15).float()
    hf_mask = (R > 0.5).float()

    lf_energy = (mag * lf_mask).sum(dim=(-2,-1), keepdim=True) + 1e-6
    hf_energy = (mag * hf_mask).sum(dim=(-2,-1), keepdim=True) + 1e-6

    sharpness = hf_energy / lf_energy
    blur_fft = 1.0 / (sharpness + 1e-6)
    return _batch_z(blur_fft)

def lowlight_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    mean_l = g.mean(dim=(-1,-2), keepdim=True)
    inv_l = 1.0 - mean_l
    return _batch_z(inv_l)

def style_score(x: torch.Tensor) -> torch.Tensor:
    pn = noise_proxy(x)
    pj = _jpeg_blockiness_proxy(x)
    pb_fft = _fft_blur_proxy(x)
    pl = lowlight_proxy(x)
    phf = _hf_drop_proxy(x)

    pn_pos  = pn.clamp_min(0.0)
    pj_pos  = pj.clamp_min(0.0)
    pb_pos  = pb_fft.clamp_min(0.0)
    pl_pos  = pl.clamp_min(0.0)
    phf_pos = phf.clamp_min(0.0)

    w_noise   = 0.11
    w_jpeg    = 0.03
    w_blur    = 0.12
    w_low     = 0.05
    w_hf_drop = 0.08

    s = (w_noise * pn_pos +
         w_jpeg * pj_pos +
         w_blur * pb_pos +
         w_low * pl_pos +
         w_hf_drop * phf_pos)

    return torch.sigmoid(1.5 * s)

# ------------------------------
# CIFAR-10 + envs
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
        base.append(transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                                         std=[0.247,0.243,0.261]))
        self.tf = transforms.Compose(base)

        self.augmix = augmix

    def __len__(self): return len(self.ds)

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

    def _augmix(self, img: Image.Image, width: int = 3, depth: int = 2, alpha: float = 1.0):
        ops = [
            lambda im: im.rotate(random.choice([-10,-7,-5,5,7,10])),
            lambda im: Image.fromarray(np.uint8(np.clip(np.array(im, dtype=np.float32) *
                                                        (0.8 + 0.4*random.random()),0,255))),
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.choice([0,1,2]))),
            lambda im: Image.fromarray(np.uint8(np.clip(np.array(im, dtype=np.float32) +
                                                        np.random.normal(0, 5.0, size=np.array(im).shape),
                                                        0,255)))
        ]
        ws = np.random.dirichlet([alpha]*width).astype(np.float32)
        m = float(np.random.beta(alpha, alpha))

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

    def __getitem__(self, idx: int):
        img, y = self.ds[idx]
        img = self._apply_env(img)
        if self.augmix:
            img = self._augmix(img)
        x = self.tf(img)
        return x, y

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
    losses = torch.stack(per_env_losses)
    return losses.var()

# ------------------------------
# Training loop (ablation-ready)
# ------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode

        # Model hyperparameters (placeholders for future use)
        self.phase_clip = getattr(args, "phase_clip", 1.0)
        self.replace = getattr(args, "replace", "conv1,layer1,layer2,layer3,layer4")
        self.style_path = getattr(args, "style_path", None)
        self.qfm_pw = getattr(args, "qfm_pw", 1)
        self.gabor_pw = getattr(args, "gabor_pw", 1)
        self.freqmix_pw = getattr(args, "freqmix_pw", 1)

        set_seed(args.seed, deterministic=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        self.num_classes = 10

        envs_train = ['clean']
        if args.envs_train:
            envs_train = [e.strip() for e in args.envs_train.split(',') if e.strip()]
        self.train_sets = [CIFARWithEnv(args.data, True, img_size=args.img, env=e,
                                        randaugment=args.randaugment, augmix=args.augmix,
                                        label_smoothing=args.smooth)
                           for e in envs_train]
        self.train_loaders = [DataLoader(ds,
                                         batch_size=args.batch//len(self.train_sets),
                                         shuffle=True,
                                         num_workers=min(4, os.cpu_count() or 2),
                                         pin_memory=_USE_CUDA,
                                         drop_last=True)
                              for ds in self.train_sets]
        self.train_iters = [iter(dl) for dl in self.train_loaders]

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

        # Param groups: QFM vs base
        qfm_params, base_params = [], []
        qfm_ids = set()
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                for p in m.parameters():
                    if p.requires_grad:
                        qfm_ids.add(id(p))
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in qfm_ids:
                qfm_params.append(p)
            else:
                base_params.append(p)

        if not base_params:  # fallback
            base_params = [p for p in self.model.parameters() if p.requires_grad]

        self.opt = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': args.lr, 'weight_decay': args.wd},
                {'params': qfm_params, 'lr': args.lr * args.qfm_lr_scale, 'weight_decay': args.wd}
            ]
        )

        steps_per_epoch = min(len(dl) for dl in self.train_loaders)
        self.total_iters = args.epochs * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=args.epochs, eta_min=args.lr*1e-2
        )

        self.writer = SummaryWriter(log_dir=os.path.join(
            args.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ))
        self.global_step = 0

        # Schedules
        self.unfreeze_gates_epoch = args.unfreeze_gates_epoch
        self.unfreeze_phase_epoch = args.unfreeze_phase_epoch
        self.f_clip_warm = args.f_clip_warm

        self.rex_lambda_start = args.rex_lambda_start
        self.rex_lambda_final = args.rex_lambda_final
        self.rex_ramp_epochs = args.rex_ramp_epochs

        self.mixup_alpha = args.mixup
        self.clip = args.clip
        self.phase_clip = args.phase_clip
        self.mode = args.mode

    def _build_model(self):
        args = self.args
        if args.pretrained:
            try:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                model = resnet18(weights=None)
        else:
            model = resnet18(weights=None)

        in_fc = model.fc.in_features
        model.fc = nn.Linear(in_fc, self.num_classes)

        # Baseline ablations
        if self.mode in ('baseline', 'baseline_blur', 'baseline_augmix', 'baseline_blur_augmix'):
            if self.mode in ('baseline_blur', 'baseline_blur_augmix'):
                in_ch = model.bn1.num_features
                model.maxpool = nn.Sequential(
                    BlurPool(in_ch, filt_size=3, stride=2),
                    nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                )
            return model

        # QFM modes
        if self.args.replace_all:
            replace = ('conv1','layer1','layer2','layer3','layer4')
        else:
            replace = tuple(self.args.replace.split(','))

        model = replace_resnet_with_qfm(
            model, device=self.device,
            replace=replace, phase_clip=self.phase_clip,
            aa_every_stride2=True
        )
        return model

    def _set_train_stage(self, epoch: int):
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                for n, p in m.named_parameters():
                    p.requires_grad = False
                if epoch < self.unfreeze_gates_epoch:
                    m.Wf.requires_grad = True
                if epoch >= self.unfreeze_gates_epoch:
                    m.Wf.requires_grad = True
                    m.V.requires_grad  = True
                    m.ortho.P.requires_grad = True
                    m.morph.alpha.requires_grad = True
                    m.morph.beta.requires_grad  = True
                    m._t_param.requires_grad    = True
                    m._lg_param.requires_grad   = True
                    m._hg_param.requires_grad   = True
                    m.band_steep.requires_grad  = True
                    m.s_gate.requires_grad      = True
                    m.f_gate_base.requires_grad = True
                if epoch >= self.unfreeze_phase_epoch:
                    m.phase_enabled = True
                    m.use_interference = True
                    for _, p in m.phase_head.named_parameters():
                        p.requires_grad = True

    def _tau_schedule(self, epoch: int):
        tau = 5.0 + min(1.0, epoch/10.0) * (15.0 - 5.0)
        for m in self.model.modules():
            if isinstance(m, SimpleMorphGate):
                m.set_tau(tau)

    def _rex_lambda(self, epoch: int) -> float:
        if self.rex_ramp_epochs <= 0:
            return self.rex_lambda_final
        if epoch <= 5:
            return (self.rex_lambda_start) * (epoch / 5.0)
        t = min(1.0, max(0.0, (epoch - 5) / float(self.rex_ramp_epochs)))
        return self.rex_lambda_start + t * (self.rex_lambda_final - self.rex_lambda_start)

    def model_forward_qfm_aware(self, x, style, fclip):
        """
        If model has QFMConv2dDCT, temporarily bind their forward
        to use the given (style, fclip). Otherwise, plain forward.
        """
        has_qfm = any(isinstance(m, QFMConv2dDCT) for m in self.model.modules())
        if not has_qfm:
            return self.model(x)

        saved_forwards = []

        def make_forward(m):
            def qfm_forward(x_q, m=m):
                return QFMConv2dDCT.forward(m, x_q, style, fclip)
            return qfm_forward

        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                saved_forwards.append((m, m.forward))
                m.forward = make_forward(m)

        try:
            out = self.model(x)
        finally:
            for m, f in saved_forwards:
                m.forward = f
        return out

    def _forward_env_chunk(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        style = style_score(x)
        fclip = self.f_clip_warm if epoch < 10 else 1.0
        logits = self.model_forward_qfm_aware(x, style, fclip)
        if self.smooth <= 0:
            loss = F.cross_entropy(logits, y)
        else:
            yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
            loss = F.kl_div(F.log_softmax(logits, dim=1), yoh, reduction='batchmean')
        return logits, loss

    def train(self):
        args = self.args
        best_clean = 0.0
        best_probe_retention = 0.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            self.model.train()
            self._set_train_stage(epoch)
            self._tau_schedule(epoch)
            rex_lambda = self._rex_lambda(epoch)
            epoch_loss = 0.0
            nseen = 0

            # lock-step across envs
            iters_per_env = min(len(dl) for dl in self.train_loaders)

            for step in range(iters_per_env):
                per_env_losses = []
                batch_total = 0

                # IMPORTANT: zero grads BEFORE backward
                self.opt.zero_grad(set_to_none=True)

                for li, it in enumerate(self.train_iters):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        self.train_iters[li] = iter(self.train_loaders[li])
                        x, y = next(self.train_iters[li])

                    x = x.to(self.device, non_blocking=True)
                    y = torch.as_tensor(y, device=self.device)
                    bsz = x.size(0)
                    batch_total += bsz

                    if self.mixup_alpha > 0:
                        # MixUp path
                        yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
                        x_mix, mix_tgt, _ = mixup(x, yoh, alpha=self.mixup_alpha)
                        with autocast(enabled=_USE_CUDA):
                            logits = self.model_forward_qfm_aware(
                                x_mix,
                                style_score(x_mix),
                                self.f_clip_warm if epoch < 10 else 1.0
                            )
                            loss_e = mixup_criterion(
                                lambda p, t: F.kl_div(
                                    F.log_softmax(p, dim=1),
                                    t,
                                    reduction='batchmean'
                                ),
                                logits,
                                mix_tgt
                            )
                    else:
                        # No MixUp path
                        with autocast(enabled=_USE_CUDA):
                            logits, loss_e = self._forward_env_chunk(x, y, epoch)

                    per_env_losses.append(loss_e)

                # Combine env losses + REx
                with autocast(enabled=_USE_CUDA):
                    Ltask = torch.stack(per_env_losses).mean()
                    Lrex = rex_penalty(per_env_losses) if len(per_env_losses) > 1 else torch.tensor(
                        0.0, device=self.device
                    )
                    loss = Ltask + rex_lambda * Lrex

                    # Phase regularization
                    for m in self.model.modules():
                        if isinstance(m, QFMConv2dDCT) and m.phase_enabled:
                            for p in m.phase_head.parameters():
                                loss = loss + args.phase_l2 * (p ** 2).mean()

                # -------- BACKWARD WITH SCALER (CRITICAL) --------
                self.scaler.scale(loss).backward()

                # -------- GRAD CLIP (AFTER unscale_) ------------
                if self.clip and self.clip > 0:
                    if _USE_CUDA and hasattr(self.scaler, "unscale_"):
                        # Now safe: scale() was used above, so _scale is set
                        self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)

                # -------- OPTIM STEP + SCALER UPDATE ------------
                self.scaler.step(self.opt)
                self.scaler.update()

                # EMA update
                self.ema.update(self.model)

                epoch_loss += loss.item() * batch_total
                nseen += batch_total

                # Logging
                if self.global_step % args.log_every == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/rex_lambda', rex_lambda, self.global_step)
                    self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], self.global_step)
                    self._log_qfm_stats(epoch)

                self.global_step += 1

            # LR sched per epoch
            self.scheduler.step()

            # EMA evaluation
            self.ema.apply_shadow(self.model)
            acc_clean = self.evaluate(self.val_clean, tag='clean', epoch=epoch)
            acc_probe = self.evaluate(self.val_probe, tag=f'probe_{self.args.val_env}', epoch=epoch)
            self.ema.restore(self.model)

            retention = (acc_probe / max(1e-6, acc_clean)) if acc_clean > 0 else 0.0
            self.writer.add_scalar('eval/clean_acc', acc_clean, epoch)
            self.writer.add_scalar('eval/probe_acc', acc_probe, epoch)
            self.writer.add_scalar('eval/retention', retention, epoch)

            # Save best checkpoints
            if acc_clean > best_clean:
                best_clean = acc_clean
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'best_clean': best_clean
                    },
                    'qfm_clean_best.pth'
                )
            if retention > best_probe_retention:
                best_probe_retention = retention
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'best_retention': best_probe_retention
                    },
                    'qfm_retention_best.pth'
                )

            dt = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{self.args.epochs} | "
                f"clean {acc_clean:.2f} | probe {acc_probe:.2f} | "
                f"retention {retention*100:.1f}% | {dt:.1f}s"
            )

        # Final diagnostics
        self._final_probes()


    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tag: str, epoch: int) -> float:
        self.model.eval()
        correct, total = 0, 0
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
                with torch.no_grad():
                    t_center = float(0.25 + (0.45-0.25)*torch.sigmoid(m._t_param).item())
                    low_gain = float(0.8 + (1.2-0.8)*torch.sigmoid(m._lg_param).item())
                    high_gain = float(0.8 + (1.2-0.8)*torch.sigmoid(m._hg_param).item())
                    Q = m.ortho()
                    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
                    err = torch.norm(Q.T @ Q - I, p='fro').item()
                    self.writer.add_scalar(f"QFM/{name}/t_center", t_center, epoch)
                    self.writer.add_scalar(f"QFM/{name}/low_gain", low_gain, epoch)
                    self.writer.add_scalar(f"QFM/{name}/high_gain", high_gain, epoch)
                    self.writer.add_scalar(f"QFM/{name}/ortho_err", err, epoch)
                cnt += 1
                if cnt >= max_items:
                    break

    @torch.no_grad()
    def _final_probes(self):
        feats = {'clean': [], 'probe': []}
        def hook(_m, _i, o):
            feats[current_env].append(o.detach().flatten(1))
        handle = None
        for n, m in self.model.named_modules():
            if n == 'avgpool':
                handle = m.register_forward_hook(hook)
                break

        self.model.eval()
        global current_env
        for env_name, loader in [('clean', self.val_clean),
                                 (f'probe_{self.args.val_env}', self.val_probe)]:
            current_env = 'clean' if env_name == 'clean' else 'probe'
            for x,y in loader:
                x = x.to(self.device)
                _ = self.model_forward_qfm_aware(x, style_score(x), 1.0)

        if handle is not None:
            handle.remove()

        if feats['clean'] and feats['probe']:
            a = torch.cat(feats['clean'], dim=0)
            b = torch.cat(feats['probe'], dim=0)
            n = min(a.size(0), b.size(0))
            a = a[:n]; b = b[:n]
            cos = F.cosine_similarity(a, b, dim=1).mean().item()
            self.writer.add_scalar('probe/feature_cosine', cos, 0)

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
        for x,y in self.val_clean:
            x = x.to(self.device)
            _ = self.model_forward_qfm_aware(x, style_score(x), 1.0)
            break
        for h in hs: h.remove()
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
    p.add_argument('--envs_train', type=str,
                   default='clean,gauss_0.1,jpeg_40,blur_3,lowlight_0.6')
    p.add_argument('--val_env', type=str, default='jpeg_40')

    # Freeze schedule
    p.add_argument('--unfreeze_gates_epoch', type=int, default=4)
    p.add_argument('--unfreeze_phase_epoch', type=int, default=8)
    p.add_argument('--f_clip_warm', type=float, default=0.3)
    p.add_argument('--phase_clip', type=float, default=math.pi/6)
    p.add_argument('--phase_l2', type=float, default=1e-3)

    # Replace controls
    p.add_argument('--replace', type=str,
                   default='conv1,layer1,layer2,layer3,layer4')
    p.add_argument('--replace_all', action='store_true')

    # REx
    p.add_argument('--rex_lambda_start', type=float, default=1.0)
    p.add_argument('--rex_lambda_final', type=float, default=10.0)
    p.add_argument('--rex_ramp_epochs', type=int, default=40)

    # Extra flags
    p.add_argument('--logits_kld', action='store_true',
                   help='Use KLDivLoss on logits instead of CrossEntropyLoss')
    p.add_argument('--style_path', type=str, default=None,
                   help='(Optional) path to precomputed style stats / proxies')
    p.add_argument('--qfm_pw', type=int, default=1,
                   help='QFM power / strength knob (reserved)')
    p.add_argument('--gabor_pw', type=int, default=1,
                   help='Gabor power / strength knob (reserved)')
    p.add_argument('--freqmix_pw', type=int, default=1,
                   help='Frequency-mix power / strength knob (reserved)')

    # Modes / ablations
    p.add_argument('--mode', type=str, default='qfm_E',
                   choices=[
                       'baseline', 'baseline_blur',
                       'baseline_augmix', 'baseline_blur_augmix',
                       'qfm_E', 'qfm_F', 'qfm_G'
                   ])
    return p

def main():
    args = build_argparser().parse_args()

    if args.mode in ('baseline','baseline_blur','baseline_augmix','baseline_blur_augmix'):
        args.randaugment = (args.mode in ('baseline_augmix','baseline_blur_augmix'))
        args.augmix = (args.mode in ('baseline_augmix','baseline_blur_augmix'))
    else:
        args.randaugment = True
        args.augmix = True

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
