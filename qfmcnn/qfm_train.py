"""
QFM-CNN + ResNet Integration (Refined, Practical, Robust)

Implements the refined Quantum Frequency Morphological CNN as drop-in replacements
for ResNet convs. Includes:
- QFMConv2d: spatial kernel morphing (base-filter superposition with orthogonal mixing),
  frequency branch with complex 1x1 mixing + morpho-frequency modulation,
  morphological attention map, and quantum-like interference fusion with adaptive phase.
- Replace utilities for ResNet18 (conv1 and/or layer blocks).
- Full training loop with AMP, early stopping, TensorBoard, checkpoint/resume.

Author: Arham Sethi
Date: 2025-08-20
"""

import os
import math
import time
import copy
import json
import random
import datetime
from typing import Tuple, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# ------------------------------
# Repro & Backend configuration
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

set_seed(1337, deterministic=True)

# AMP helpers (CPU-safe)
_USE_CUDA = torch.cuda.is_available()
if _USE_CUDA:
    from torch.cuda.amp import autocast, GradScaler
else:
    # Fallback stubs for CPU so "with autocast()" works.
    class autocast:
        def __init__(self, enabled=False, **kwargs): self.enabled = enabled
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = False
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass

# ------------------------------
# Dataset (same CSV contract as your script)
# ------------------------------

class MoistureDataset(Dataset):
    """
    CSV columns:
      - image: path to image file
      - folder: float/number label (moisture level)
    """
    def __init__(self, csv_path: str, augment: bool = False, resize: Tuple[int,int]=(224,224)):
        import pandas as pd
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        self.resize = resize

        base_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            self.tf = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.ToTensor(),
                base_norm
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                base_norm
            ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = str(row['image'])
        # label is numeric (float)
        label_val = float(row['folder'])
        label = torch.tensor(label_val, dtype=torch.float32)

        try:
            img = Image.open(img_path).convert('RGB')
            img = self.tf(img)
        except Exception as e:
            # Robust fallback (will still backprop)
            print(f"[WARN] Failed to load image: {img_path} ({e}). Using zeros.")
            img = torch.zeros((3, self.resize[0], self.resize[1]), dtype=torch.float32)

        return img, label


# ------------------------------
# Base Filter Bank (spatial)
# ------------------------------

def _gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float):
    """
    Create a 2D Gabor kernel (float32 tensor).
    """
    xmax = ks // 2
    ymax = ks // 2
    y, x = torch.meshgrid(torch.arange(-ymax, ymax+1), torch.arange(-xmax, xmax+1), indexing='ij')
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    # Rotation
    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    gb = torch.exp(-(x_theta**2 + (gamma**2)*(y_theta**2)) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / lam + psi)
    gb = gb / (gb.abs().sum() + 1e-6)
    return gb

def make_base_filters(kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns base filters tensor of shape [M, 1, kH, kW].
    Includes: identity-ish, blur, sharpen, Laplacian, Sobel X/Y, and several Gabors.
    """
    ks = kernel_size
    assert ks in (3,5,7), "QFMConv2d: kernel_size should be 3, 5, or 7 for provided base bank."
    M_bank: List[torch.Tensor] = []

    # Identity-like (delta)
    eye = torch.zeros((ks, ks), dtype=torch.float32)
    eye[ks//2, ks//2] = 1.0
    M_bank.append(eye)

    # Box blur
    box = torch.ones((ks, ks), dtype=torch.float32) / (ks * ks)
    M_bank.append(box)

    # Gaussian blur (approx)
    if ks == 3:
        gauss = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32); gauss /= gauss.sum()
    elif ks == 5:
        gauss = torch.tensor([
            [1,  4,  7,  4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,  4,  7,  4, 1],
        ], dtype=torch.float32); gauss /= gauss.sum()
    else:
        # 7x7 gaussian via separable approx
        g1d = torch.tensor([1,6,15,20,15,6,1], dtype=torch.float32)
        gauss = torch.outer(g1d, g1d); gauss /= gauss.sum()
    M_bank.append(gauss)

    # Sharpen
    if ks == 3:
        sharp = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=torch.float32)
    elif ks == 5:
        sharp = -box * (ks*ks - 1); sharp[ks//2, ks//2] += ks*ks
    else:
        sharp = -box * (ks*ks - 1); sharp[ks//2, ks//2] += ks*ks
    M_bank.append(sharp)

    # Laplacian
    if ks == 3:
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    else:
        # scale 3x3 into center of larger kernel
        lap = torch.zeros((ks, ks), dtype=torch.float32)
        lap[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    M_bank.append(lap)

    # Sobel X / Y (centered)
    if ks == 3:
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    else:
        sobel_x = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_y = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_x[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    M_bank.append(sobel_x)
    M_bank.append(sobel_y)

    # Several Gabor orientations
    g_params = [
        (ks, 1.2, 0.0, 4.0, 0.5, 0.0),
        (ks, 1.2, math.pi/4, 4.0, 0.5, 0.0),
        (ks, 1.2, math.pi/2, 4.0, 0.5, 0.0),
        (ks, 1.2, 3*math.pi/4, 4.0, 0.5, 0.0),
    ]
    for (ks_, sigma, theta, lam, gamma, psi) in g_params:
        M_bank.append(_gabor_kernel(ks_, sigma, theta, lam, gamma, psi))

    bank = torch.stack(M_bank, dim=0).unsqueeze(1)  # [M, 1, ks, ks]
    bank = bank.to(device=device)
    return bank


# ------------------------------
# Morphology (soft, differentiable)
# ------------------------------

class SoftMorphology(nn.Module):
    """
    Produces a morphological attention map A in [0,1] of shape [B,1,H,W].
    Combines:
      - Soft dilation/erosion (via max/min pooling approximations),
      - Sobel edges,
    Then fuses with learnable scalars and a 1x1 conv gate.
    """
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: float = 10.0):
        super().__init__()
        assert kernel_size in (3,5,7)
        self.kernel_size = kernel_size
        self.tau = tau

        # Learnable fusers
        self.w_dil = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.w_ero = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.w_edge = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Simple 1x1 conv gate to allow data-dependent scaling
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] normalized image/feature map
        returns A: [B,1,H,W] in [0,1]
        """
        B, C, H, W = x.shape
        # grayscale proxy (mean over channels)
        g = x.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # Soft dilation/erosion using softmax/softmin over local window
        # unfold then softmax pooling
        k = self.kernel_size
        pad = k // 2
        patches = F.unfold(g, kernel_size=k, padding=pad)        # [B, k*k, HW]
        patches = patches.view(B, 1, k*k, H, W)                   # [B,1,KK,H,W]

        # Softmax along neighborhood dims
        scores = (patches * self.tau).softmax(dim=2)
        dil = (scores * patches).sum(dim=2)                       # ~soft dilation  [B,1,H,W]

        # Softmin via softmax on negative
        scores_min = (-patches * self.tau).softmax(dim=2)
        ero = (scores_min * patches).sum(dim=2)                   # ~soft erosion  [B,1,H,W]

        # Edge magnitude
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8)

        # Fuse
        raw = self.w_dil * dil + self.w_edge * edge - self.w_ero * ero + self.bias
        gate = torch.sigmoid(self.gate(x))                        # [B,1,H,W]
        attn = torch.sigmoid(raw) * gate + torch.sigmoid(raw) * (1 - gate)

        return attn.clamp(0.0, 1.0)


# ------------------------------
# Unitary/Orthogonal mixing (QR)
# ------------------------------

class OrthoMixer(nn.Module):
    """
    Parametrizes an orthogonal matrix Q via QR of a free parameter P.
    Returns Q with shape [M, M].
    """
    def __init__(self, M: int):
        super().__init__()
        self.P = nn.Parameter(torch.randn(M, M) * 0.02)

    def forward(self) -> torch.Tensor:
        # QR decomposition: P = QR. Ensure deterministic sign (R diag positive)
        Q, R = torch.linalg.qr(self.P, mode='reduced')
        # Make diag(R) positive to remove sign ambiguity
        diag = torch.sign(torch.diag(R))
        diag[diag == 0] = 1.0
        D = torch.diag(diag)
        Q = Q @ D
        return Q


# ------------------------------
# QFMConv2d
# ------------------------------

class QFMConv2d(nn.Module):
    """
    Drop-in replacement for Conv2d:
      - Spatial branch: base filter superposition with orthogonal mixing.
      - Frequency branch: FFT2 -> complex 1x1 mixing -> morpho-frequency mask -> IFFT2.
      - Morphological attention gates both branches.
      - Quantum-like interference fusion with adaptive phase θ(x).

    Output shape matches a Conv2d with same (out_channels, stride, padding).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int,int]],
        stride: Union[int, Tuple[int,int]] = 1,
        padding: Union[int, Tuple[int,int]] = 0,
        bias: bool = True,
        input_size: Tuple[int,int] = (224, 224),
        base_filter_count: int = 10,   # will be set by bank size automatically
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
        self.input_size = input_size
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        # Base bank: [M, 1, kH, kW]
        self.register_buffer('_base_bank', make_base_filters(self.kH, self.device))
        self.M = self._base_bank.shape[0]  # base_filter_count derived

        # Orthogonal mixer Q (M x M)
        self.ortho = OrthoMixer(self.M)

        # Coefficients V for each (out,in,m)
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.01)

        # Optional bias for spatial conv
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Frequency branch: complex 1x1 mixing weights [out, in], init small
        self.Wc_real = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)
        self.Wc_imag = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)

        # Morphology
        self.morph = SoftMorphology(in_channels, kernel_size=3, tau=10.0)

        # Morpho-frequency modulation parameters
        self.low_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.high_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.band_steep = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))   # radial transfer slope

        # Adaptive phase network θ(x): small 1x1 conv predicting phase per (B, out, H', W')
        self.phase_net = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        )
        # Initial small weights -> small phases at start
        for m in self.phase_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)

        # Fusion scaling
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    # ---------- Helpers ----------

    def _spatial_weight(self) -> torch.Tensor:
        """
        Build spatial kernel weights W of shape [out, in, kH, kW] from base bank.
        W(out,in,:,:)= sum_m A[out,in,m] * base_bank[m]
        where A = V @ Q (Q orthogonal).
        """
        Q = self.ortho()                               # [M,M]
        A = torch.einsum('oim,mn->oin', self.V, Q)     # [out,in,M]
        # Combine with base filters
        # _base_bank: [M,1,kH,kW]
        W = torch.einsum('oin,mxy->oixy', A, self._base_bank.squeeze(1))  # [out,in,kH,kW]
        return W

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(x, norm='ortho')

    def _ifft2(self, X: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifft2(X, norm='ortho')
        # Return real part (imag should cancel if signals are real-symmetric after processing)
        return x.real

    def _complex_1x1(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, Cin, H, W] complex
        Returns: [B, Cout, H, W] complex using complex matmul per spatial location.
        """
        Wr = self.Wc_real
        Wi = self.Wc_imag
        # Z_out = (Wr + i Wi) @ (Xr + i Xi) = (Wr@Xr - Wi@Xi) + i(Wr@Xi + Wi@Xr)
        Xr = X.real
        Xi = X.imag
        Yr = torch.einsum('oi,bihw->bohw', Wr, Xr) - torch.einsum('oi,bihw->bohw', Wi, Xi)
        Yi = torch.einsum('oi,bihw->bohw', Wr, Xi) + torch.einsum('oi,bihw->bohw', Wi, Xr)
        Y = torch.complex(Yr, Yi)
        return Y

    def _freq_grid_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Builds a radial frequency grid r \in [0, 1] and produces two masks:
        low_mask = sigmoid(s * (t - r))
        high_mask = 1 - low_mask
        Then combines via learnable low/high gains.
        """
        fy = torch.fft.fftfreq(H, d=1.0, device=device)  # [-0.5, 0.5) scaled
        fx = torch.fft.fftfreq(W, d=1.0, device=device)
        YY, XX = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(XX*XX + YY*YY)  # [H,W], max ~ sqrt(0.5^2+0.5^2)
        r = r / (r.max() + 1e-8)       # normalize to [0,1]
        # fixed threshold at ~0.35 radial (can be made learnable)
        t = 0.35
        s = self.band_steep.clamp_min(1.0)
        low = torch.sigmoid(s * (t - r))     # emphasize low frequencies near center
        high = 1.0 - low
        # stack to [2,H,W] then apply gains
        mask = torch.stack([self.low_gain * low, self.high_gain * high], dim=0)  # [2,H,W]
        # sum
        combined = mask.sum(dim=0).clamp(0.0, 10.0)                               # [H,W]
        return combined

    def _apply_stride(self, y: torch.Tensor) -> torch.Tensor:
        sy, sx = self.stride
        if sy == 1 and sx == 1:
            return y
        # Use average pooling to mimic downsampling for frequency branch
        return F.avg_pool2d(y, kernel_size=self.stride, stride=self.stride, ceil_mode=False)

    # ---------- Forward ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Cin, H, W] (expects H,W == input_size that was passed during module construction)
        """
        B, Cin, H, W = x.shape
        assert Cin == self.in_channels, f"QFMConv2d expected {self.in_channels} channels, got {Cin}"
        assert (H, W) == tuple(self.input_size), f"QFMConv2d expected input {self.input_size}, got {(H,W)}"

        # Morphology map from input
        A = self.morph(x)                        # [B,1,H,W], in [0,1]

        # ---- Spatial branch: conv with morphed kernels ----
        W_spatial = self._spatial_weight().to(dtype=x.dtype)  # [out,in,kH,kW]
        y_spatial = F.conv2d(x, W_spatial, bias=self.bias, stride=self.stride, padding=self.padding)
        # Gate by morphology (downsample A to output res if stride>1)
        if self.stride != (1,1) or self.padding != (0,0):
            # compute output size for conv to match gating map
            # Instead of exact conv receptive gating, we approximate by downsampling A
            A_ds = self._apply_stride(F.pad(A, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])))
        else:
            A_ds = A
        A_ds = A_ds[..., :y_spatial.shape[-2], :y_spatial.shape[-1]].clamp(0,1)  # crop if any mismatch
        y_spatial = y_spatial * (1.0 + 0.5 * A_ds)   # modest morph gating

        # ---- Frequency branch: FFT -> complex 1x1 -> freq mask -> IFFT ----
        # Pad before FFT to mimic conv padding influence (same as spatial branch padding)
        if self.padding != (0,0):
            x_pad = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        else:
            x_pad = x
        x_pad = x_pad.float() 
        Xc = self._fft2(x_pad)                   # [B,Cin,Hp,Wp] complex
        Yc = self._complex_1x1(Xc)               # [B,Cout,Hp,Wp] complex

        # Morpho-frequency modulation: build radial mask and gate by morphology avg
        Hp, Wp = Yc.shape[-2], Yc.shape[-1]
        freq_mask = self._freq_grid_mask(Hp, Wp, Yc.device)    # [Hp,Wp]
        # influence of morphology onto frequency: global avg of A per sample -> [B,1,1,1], then scale
        A_avg = A.mean(dim=(-1,-2), keepdim=True)              # [B,1,1,1]
        # Combine: Yc *= (1 + A_avg * freq_mask) with broadcast
        freq_mask_bc = freq_mask.view(1,1,Hp,Wp)
        Yc = Yc * (1.0 + A_avg * freq_mask_bc)

        yf = self._ifft2(Yc)                    # [B,Cout,Hp,Wp] real
        # Crop back to match padded conv output spatial dims prior to stride
        yf = yf[..., :x_pad.shape[-2], :x_pad.shape[-1]]       # ensure already that Hp=H+2p etc.
        # Remove pad region to original H,W then downsample for stride
        if self.padding != (0,0):
            yf = yf[..., self.padding[0]:self.padding[0]+H, self.padding[1]:self.padding[1]+W]
        yf = self._apply_stride(yf)                                # [B,Cout,H',W']

        # Morphology gate also on frequency branch (downsampled A)
        A_ds2 = self._apply_stride(A)
        A_ds2 = A_ds2[..., :yf.shape[-2], :yf.shape[-1]]
        y_freq = yf * (1.0 + 0.5 * A_ds2)

        # ---- Quantum-like interference fusion ----
        # Create complex reps (psi_spatial, psi_freq) as real-only complex
        psi_spatial = torch.complex(y_spatial, torch.zeros_like(y_spatial))
        psi_freq    = torch.complex(y_freq,    torch.zeros_like(y_freq))

        # Compute phase θ(x) from concatenated real maps
        # Stack along channel dim: [B, 2*Cout, H', W']
        cat = torch.cat([y_spatial, y_freq], dim=1)
        theta = self.phase_net(cat)                         # [B, Cout, H', W']
        # limit magnitude to reasonable range initially
        theta = torch.tanh(theta) * math.pi                # phase in (-π, π)

        ei_theta = torch.complex(torch.cos(theta), torch.sin(theta))
        fused_complex = psi_spatial + ei_theta * psi_freq
        fused = (fused_complex.real**2 + fused_complex.imag**2)    # | . |^2

        # Final affine (per-layer scaling)
        out = self.gamma * fused + self.beta

        return out


# ------------------------------
# Replace ResNet convs with QFMConv2d
# ------------------------------

def replace_resnet_convs_with_qfm(
    model: nn.Module,
    input_size: Tuple[int,int] = (224,224),
    layers_to_replace: Iterable[str] = ('conv1',),
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Replaces specified convs in ResNet18 with QFMConv2d:
      - 'conv1'
      - 'layer1', 'layer2', 'layer3', 'layer4' (replaces block.conv1 within each)
    Computes and threads the expected input spatial sizes for shape-safe FFT masks.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_size = list(input_size)

    def conv_out_sz(sz, k, s, p):
        # (sz + 2p - k) // s + 1
        return (sz + 2*p - k) // s + 1

    # Replace stem conv1
    if 'conv1' in layers_to_replace:
        old = model.conv1
        q = QFMConv2d(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size, old.kernel_size),
            stride=old.stride if isinstance(old.stride, tuple) else (old.stride, old.stride),
            padding=old.padding if isinstance(old.padding, tuple) else (old.padding, old.padding),
            bias=(old.bias is not None),
            input_size=tuple(current_size),
            device=device,
            dtype=old.weight.dtype
        )
        # warm start: copy old weight into V by projecting delta kernel onto base bank
        with torch.no_grad():
            # naive init: set V such that identity-like base dominates
            # find index of identity-like base (0)
            q.V.zero_()
            if q.M > 0:
                # set center coefficient to mimic identity conv distribution per out/in
                q.V[:,:,0] = old.weight.mean(dim=(-1,-2))  # rough seeding
            if q.bias is not None and old.bias is not None:
                q.bias.data.copy_(old.bias.data)
        model.conv1 = q.to(device)

        # Update current_size after conv1 and maxpool
        k, s, p = old.kernel_size, old.stride, old.padding
        if isinstance(k, tuple): k = k[0]
        if isinstance(s, tuple): s = s[0]
        if isinstance(p, tuple): p = p[0]
        current_size = [conv_out_sz(current_size[i], k, s, p) for i in (0,1)]
        # maxpool(3,2,1)
        current_size = [conv_out_sz(current_size[i], 3, 2, 1) for i in (0,1)]

    # Replace block conv1 in residual layers
    for layer_name in ('layer1','layer2','layer3','layer4'):
        layer = getattr(model, layer_name)
        # compute downsample at start of layer (layer1 has stride=1, layer2->4 have stride=2 on first block)
        for block_idx, block in enumerate(layer):
            if layer_name in layers_to_replace:
                if hasattr(block, 'conv1'):
                    old = block.conv1
                    q = QFMConv2d(
                        in_channels=old.in_channels,
                        out_channels=old.out_channels,
                        kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size, old.kernel_size),
                        stride=old.stride if isinstance(old.stride, tuple) else (old.stride, old.stride),
                        padding=old.padding if isinstance(old.padding, tuple) else (old.padding, old.padding),
                        bias=(old.bias is not None),
                        input_size=tuple(current_size),
                        device=device,
                        dtype=old.weight.dtype
                    )
                    with torch.no_grad():
                        q.V.zero_()
                        if q.M > 0:
                            q.V[:,:,0] = old.weight.mean(dim=(-1,-2))
                        if q.bias is not None and old.bias is not None:
                            q.bias.data.copy_(old.bias.data)
                    block.conv1 = q.to(device)
                # Update current_size after this conv1
                k, s, p = old.kernel_size, old.stride, old.padding
                if isinstance(k, tuple): k = k[0]
                if isinstance(s, tuple): s = s[0]
                if isinstance(p, tuple): p = p[0]
                current_size = [conv_out_sz(current_size[i], k, s, p) for i in (0,1)]
            else:
                # Still need to update sizes to keep computed input_size correct even if we didn't replace
                old = block.conv1
                k, s, p = old.kernel_size, old.stride, old.padding
                if isinstance(k, tuple): k = k[0]
                if isinstance(s, tuple): s = s[0]
                if isinstance(p, tuple): p = p[0]
                current_size = [conv_out_sz(current_size[i], k, s, p) for i in (0,1)]

    return model


# ------------------------------
# TensorBoard Logging for Attentions
# ------------------------------

def log_qfm_maps(model: nn.Module, writer: SummaryWriter, epoch: int, max_items: int = 4, tag_prefix: str = "QFM"):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, QFMConv2d):
            # Log morphology weights as scalars
            writer.add_scalar(f"{tag_prefix}/{name}/low_gain", module.low_gain.item(), epoch)
            writer.add_scalar(f"{tag_prefix}/{name}/high_gain", module.high_gain.item(), epoch)
            writer.add_scalar(f"{tag_prefix}/{name}/band_steep", module.band_steep.item(), epoch)
            # Log orthogonality dev: ||Q^T Q - I||
            with torch.no_grad():
                Q = module.ortho()
                I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
                ortho_err = torch.norm(Q.T @ Q - I, p='fro').item()
                writer.add_scalar(f"{tag_prefix}/{name}/ortho_err", ortho_err, epoch)
            cnt += 1
            if cnt >= max_items:
                break


# ------------------------------
# Training / Evaluation
# ------------------------------

def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def try_load_resnet18_pretrained():
    try:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[WARN] Could not load pretrained weights ({e}). Using random init.")
        return resnet18(weights=None)

def train_qfm(
    train_csv: str = 'train_clean_fixed.csv',
    val_csv: str = 'val_clean_fixed.csv',
    test_csv: str = 'test_clean_fixed.csv',
    replace_layers: Tuple[str,...] = ('conv1','layer1'),
    input_size: Tuple[int,int] = (224,224),
    batch_size: int = 8,
    num_epochs: int = 40,
    warmup_epochs: int = 3,
    lr: float = 1e-4,
    lr_unitary: float = 5e-4,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    logdir: str = 'runs/qfm_experiment',
    resume_path: Optional[str] = None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_ds = MoistureDataset(train_csv, augment=True,  resize=input_size)
    val_ds   = MoistureDataset(val_csv,   augment=False, resize=input_size)
    test_ds  = MoistureDataset(test_csv,  augment=False, resize=input_size)

    nw = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=_USE_CUDA)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=max(1,nw//2))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=max(1,nw//2))

    # Model
    model = try_load_resnet18_pretrained()
    model = replace_resnet_convs_with_qfm(model, input_size=input_size, layers_to_replace=replace_layers, device=device)

    # Replace head for regression
    in_fc = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_fc, 1)
    )
    model = model.to(device)

    print(f"[INFO] Trainable params: {count_parameters(model):,}")

    # Loss / Opt
    criterion = nn.SmoothL1Loss(beta=0.5)

    # Parameter groups: split unitary mixer P (in OrthoMixer) and others
    unitary_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if '.ortho.P' in n:
            unitary_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': unitary_params, 'lr': lr_unitary, 'weight_decay': 0.0},  # encourage exact orthogonality
    ])

    scaler = GradScaler(enabled=_USE_CUDA)
    writer = SummaryWriter(log_dir=os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Resume
    start_epoch = 1
    best_val = float('inf')
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        best_val = ckpt.get('best_val', best_val)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[INFO] Resumed from {resume_path} (epoch {start_epoch-1}, best_val={best_val:.4f})")

    # Warm-up: train only Ortho + head initially (and QFM special small parts) to stabilize
    def set_requires_grad(stage: str):
        for n, p in model.named_parameters():
            req = True
            if stage == 'warmup':
                # train only qfm mixers + last fc + small scalars
                req = ('.ortho.P' in n) or (n.startswith('fc')) or ('.low_gain' in n) or ('.high_gain' in n) or ('.band_steep' in n) or ('.phase_net' in n) or ('.V' in n) or ('.Wc_' in n) or ('.morph.' in n)
            p.requires_grad = req

    if start_epoch == 1:
        set_requires_grad('warmup')
        print(f"[INFO] Warm-up for {warmup_epochs} epochs...")
    else:
        print(f"[INFO] Skipping warm-up due to resume.")
        set_requires_grad('full')

    patience = 10
    no_improve = 0
    global_step = 0

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()
        if epoch == warmup_epochs + 1 and start_epoch == 1:
            set_requires_grad('full')
            print("[INFO] Unfroze all parameters after warm-up.")

        # ---- Train ----
        running_loss, running_count = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=_USE_CUDA):
                preds = model(imgs)
                loss  = criterion(preds, labels)

                # Optional: small regularizer on V magnitude to keep kernels stable
                reg = 0.0
                for m in model.modules():
                    if isinstance(m, QFMConv2d):
                        reg = reg + 1e-5 * (m.V**2).mean()
                loss = loss + reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            running_count += imgs.size(0)
            global_step += 1

        train_loss = running_loss / max(1, running_count)
        writer.add_scalar('loss/train', train_loss, epoch)

        # Log QFM stats
        log_qfm_maps(model, writer, epoch, max_items=6, tag_prefix="QFM")

        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            val_sum, val_n = 0.0, 0
            val_rmse_sum = 0.0
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device).unsqueeze(1)
                with autocast(enabled=_USE_CUDA):
                    preds = model(imgs)
                    l = criterion(preds, labels)
                    r = rmse(preds, labels)
                val_sum += l.item() * imgs.size(0)
                val_rmse_sum += r.item() * imgs.size(0)
                val_n += imgs.size(0)
            val_loss = val_sum / max(1, val_n)
            val_rm = val_rmse_sum / max(1, val_n)

        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('metric/val_rmse', val_rm, epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} (rmse {val_rm:.4f}) | {dt:.1f}s")

        # Save best
        if val_rm < best_val:
            best_val = val_rm
            no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'cfg': {
                    'replace_layers': replace_layers,
                    'input_size': input_size,
                }
            }, 'qfm_best.pth')
            print(f"[INFO] Saved new best model (val RMSE={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Early Stop] Patience {patience} exceeded (best val RMSE={best_val:.4f}).")
                break

        # Save resume checkpoint every epoch
        torch.save({
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': epoch,
            'best_val': best_val,
        }, 'qfm_last.pth')

    # ---- Test ----
    # Load best
    if os.path.isfile('qfm_best.pth'):
        ckpt = torch.load('qfm_best.pth', map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"[INFO] Loaded best checkpoint (val RMSE={ckpt.get('best_val', -1):.4f})")

    model.eval()
    with torch.no_grad():
        test_sum_rmse, n = 0.0, 0
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device).unsqueeze(1)
            with autocast(enabled=_USE_CUDA):
                preds = model(imgs)
                r = rmse(preds, labels)
            test_sum_rmse += r.item() * imgs.size(0)
            n += imgs.size(0)
        test_rm = test_sum_rmse / max(1, n)

    writer.add_scalar('metric/test_rmse', test_rm, 0)
    writer.close()
    print(f"[TEST] RMSE: {test_rm:.4f}")

    return model, test_rm


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QFM-CNN + ResNet Training")
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--val_csv', type=str, default='val.csv')
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_unitary', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--replace', type=str, default='conv1,layer1', help="comma-separated among conv1,layer1,layer2,layer3,layer4")
    parser.add_argument('--logdir', type=str, default='runs/qfm_experiment')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = tuple([s.strip() for s in args.replace.split(',') if s.strip()])

    train_qfm(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        replace_layers=layers,
        input_size=(args.height, args.width),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup,
        lr=args.lr,
        lr_unitary=args.lr_unitary,
        weight_decay=args.wd,
        device=device,
        logdir=args.logdir,
        resume_path=args.resume if args.resume else None,
    )
