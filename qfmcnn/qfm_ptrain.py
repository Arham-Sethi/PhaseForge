"""
QFM-CNN Parali Multi-Task Post-Training Script

Tasks:
- Moisture regression (required)
- Crop-type classification (optional; trains only if labels present)
- Particle-size regression (optional; trains only if labels present)

Features:
- Reuses QFM-ResNet18 backbone (conv1, layer1 replaced by QFMConv2d by default)
- Uncertainty-weighted multi-task loss (learnable log-sigmas) with per-sample label masking
- Warm-up (train heads + QFM mixers), then unfreeze all
- Mixed precision (AMP), early stopping on composite validation score
- TensorBoard logging (scalars), best-checkpointing, resume
- TorchScript export for on-device inference (recommended)
- Robust CSV handling; optional class mapping file

CSV schema (minimum):
  image, moisture [, crop_type] [, psize]

Author: Arham Sethi
Date: 2025-08-20
"""

from __future__ import annotations
import os
import math
import time
import copy
import json
import random
import argparse
import datetime
from typing import Tuple, Iterable, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# =============================
# Utils: seeds, device, AMP
# =============================

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
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = False
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass

# =============================
# Dataset
# =============================

class ParaliDataset(Dataset):
    """CSV columns:
       - image: path to image file (required)
       - moisture: float (required)
       - crop_type: str (optional)
       - psize: float (optional, mean particle length)

       Normalization of targets is done outside the dataset (in trainer),
       dataset returns raw labels if present, or None if missing.
    """
    def __init__(self, csv_path: str, augment: bool = False, resize: Tuple[int,int]=(224,224)):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        if 'image' not in self.df.columns or 'moisture' not in self.df.columns:
            raise ValueError("CSV must contain at least 'image' and 'moisture' columns")
        self.augment = augment
        self.resize = resize

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.ToTensor(),
                self.norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                self.norm,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row['image'])
        try:
            img = Image.open(img_path).convert('RGB')
            x = self.tf(img)
        except Exception as e:
            print(f"[WARN] Failed to load image: {img_path} ({e}). Using zeros.")
            x = torch.zeros((3, self.resize[0], self.resize[1]), dtype=torch.float32)

        # moisture (required)
        m = None
        try:
            m = float(row['moisture'])
        except Exception:
            raise ValueError("'moisture' must be a numeric column")
        m = torch.tensor(m, dtype=torch.float32)

        # crop_type (optional)
        crop = row['crop_type'] if 'crop_type' in self.df.columns else None
        crop = None if (isinstance(crop, float) and math.isnan(crop)) else crop

        # psize (optional)
        psize = None
        if 'psize' in self.df.columns:
            try:
                psize = float(row['psize'])
            except Exception:
                psize = None
        p = None if psize is None else torch.tensor(psize, dtype=torch.float32)

        return x, m, crop, p

# =============================
# Base Filter Bank and QFMConv2d
# (same core as earlier, trimmed for clarity)
# =============================

def _gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float):
    xmax = ks // 2
    ymax = ks // 2
    y, x = torch.meshgrid(torch.arange(-ymax, ymax+1), torch.arange(-xmax, xmax+1), indexing='ij')
    x = x.to(torch.float32); y = y.to(torch.float32)
    c = torch.tensor(theta)
    x_t = x*torch.cos(c) + y*torch.sin(c)
    y_t = -x*torch.sin(c) + y*torch.cos(c)
    gb = torch.exp(-(x_t**2 + (gamma**2)*(y_t**2))/(2*sigma**2)) * torch.cos(2*math.pi*x_t/lam + psi)
    gb = gb / (gb.abs().sum() + 1e-6)
    return gb

def make_base_filters(kernel_size: int, device: torch.device) -> torch.Tensor:
    ks = kernel_size
    if ks not in (3,5,7):
        raise ValueError("kernel_size must be 3,5,7 for built-in base bank")
    bank: List[torch.Tensor] = []
    eye = torch.zeros((ks,ks), dtype=torch.float32); eye[ks//2, ks//2] = 1.0
    bank.append(eye)
    box = torch.ones((ks,ks), dtype=torch.float32)/(ks*ks)
    bank.append(box)
    if ks == 3:
        gauss = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32); gauss/=gauss.sum()
    elif ks == 5:
        gauss = torch.tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]], dtype=torch.float32); gauss/=gauss.sum()
    else:
        g1d = torch.tensor([1,6,15,20,15,6,1], dtype=torch.float32)
        gauss = torch.outer(g1d,g1d); gauss/=gauss.sum()
    bank.append(gauss)
    if ks == 3:
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    else:
        lap = torch.zeros((ks,ks), dtype=torch.float32)
        lap[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    bank.append(lap)
    if ks == 3:
        sobx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        soby = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    else:
        sobx = torch.zeros((ks,ks), dtype=torch.float32)
        soby = torch.zeros((ks,ks), dtype=torch.float32)
        sobx[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        soby[ks//2-1:ks//2+2, ks//2-1:ks//2+2] = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
    bank.append(sobx); bank.append(soby)
    g_params = [
        (ks, 1.2, 0.0, 4.0, 0.5, 0.0),
        (ks, 1.2, math.pi/4, 4.0, 0.5, 0.0),
        (ks, 1.2, math.pi/2, 4.0, 0.5, 0.0),
        (ks, 1.2, 3*math.pi/4, 4.0, 0.5, 0.0),
    ]
    for (_, sigma, theta, lam, gamma, psi) in g_params:
        bank.append(_gabor_kernel(ks, sigma, theta, lam, gamma, psi))
    return torch.stack(bank, dim=0).unsqueeze(1).to(device)

class SoftMorphology(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 3, tau: float = 10.0):
        super().__init__()
        self.k = kernel_size
        self.tau = tau
        self.w_dil = nn.Parameter(torch.tensor(0.7))
        self.w_ero = nn.Parameter(torch.tensor(0.3))
        self.w_edge = nn.Parameter(torch.tensor(0.6))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.register_buffer('sobx', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('soby', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        g = x.mean(dim=1, keepdim=True)
        k = self.k; pad = k//2
        patches = F.unfold(g, kernel_size=k, padding=pad).view(B,1,k*k,H,W)
        scores = (patches * self.tau).softmax(dim=2)
        dil = (scores * patches).sum(dim=2)
        scores_min = (-patches * self.tau).softmax(dim=2)
        ero = (scores_min * patches).sum(dim=2)
        gx = F.conv2d(g, self.sobx, padding=1)
        gy = F.conv2d(g, self.soby, padding=1)
        edge = torch.sqrt(gx*gx + gy*gy + 1e-8)
        raw = self.w_dil*dil + self.w_edge*edge - self.w_ero*ero + self.bias
        gate = torch.sigmoid(self.gate(x))
        attn = torch.sigmoid(raw) * gate + torch.sigmoid(raw) * (1 - gate)
        return attn.clamp(0.0, 1.0)

class OrthoMixer(nn.Module):
    def __init__(self, M: int):
        super().__init__()
        self.P = nn.Parameter(torch.randn(M, M) * 0.02)
    def forward(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.P, mode='reduced')
        diag = torch.sign(torch.diag(R)); diag[diag==0]=1.0
        D = torch.diag(diag)
        return Q @ D

class QFMConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int,int]|int, stride=1, padding=0, bias=True,
                 input_size: Tuple[int,int]=(224,224), device: Optional[torch.device]=None,
                 dtype: torch.dtype=torch.float32):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size=(kernel_size,kernel_size)
        if isinstance(stride, int): stride=(stride,stride)
        if isinstance(padding, int): padding=(padding,padding)
        self.in_ch = in_channels; self.out_ch = out_channels
        self.kH, self.kW = kernel_size
        self.stride = stride; self.padding = padding
        self.input_size = input_size
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.register_buffer('base_bank', make_base_filters(self.kH, self.device))
        self.M = self.base_bank.shape[0]
        self.ortho = OrthoMixer(self.M)
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.Wc_real = nn.Parameter(torch.randn(out_channels, in_channels)*0.01)
        self.Wc_imag = nn.Parameter(torch.randn(out_channels, in_channels)*0.01)
        self.morph = SoftMorphology(in_channels, kernel_size=3, tau=10.0)
        self.low_gain = nn.Parameter(torch.tensor(1.0))
        self.high_gain = nn.Parameter(torch.tensor(1.0))
        self.band_steep = nn.Parameter(torch.tensor(8.0))
        self.phase_net = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        )
        for m in self.phase_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def _spatial_weight(self) -> torch.Tensor:
        Q = self.ortho()                                        # [M,M]
        A = torch.einsum('oim,mn->oin', self.V, Q)              # [out,in,M]
        W = torch.einsum('oin,mxy->oixy', A, self.base_bank.squeeze(1))
        return W
    def _fft2(self, x): return torch.fft.fft2(x, norm='ortho')
    def _ifft2(self, X): return torch.fft.ifft2(X, norm='ortho').real
    def _complex_1x1(self, X: torch.Tensor) -> torch.Tensor:
        Wr, Wi = self.Wc_real, self.Wc_imag
        Xr, Xi = X.real, X.imag
        Yr = torch.einsum('oi,bihw->bohw', Wr, Xr) - torch.einsum('oi,bihw->bohw', Wi, Xi)
        Yi = torch.einsum('oi,bihw->bohw', Wr, Xi) + torch.einsum('oi,bihw->bohw', Wi, Xr)
        return torch.complex(Yr, Yi)
    def _freq_grid_mask(self, H:int, W:int, device: torch.device):
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.fftfreq(W, d=1.0, device=device)
        YY, XX = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(XX*XX + YY*YY); r = r / (r.max() + 1e-8)
        t = 0.35; s = self.band_steep.clamp_min(1.0)
        low = torch.sigmoid(s * (t - r)); high = 1.0 - low
        mask = torch.stack([self.low_gain*low, self.high_gain*high], dim=0)
        return mask.sum(dim=0).clamp(0.0, 10.0)
    def _apply_stride(self, y: torch.Tensor) -> torch.Tensor:
        sy, sx = self.stride
        return y if (sy==1 and sx==1) else F.avg_pool2d(y, kernel_size=self.stride, stride=self.stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        assert C==self.in_ch, f"QFMConv2d expected {self.in_ch} channels, got {C}"
        assert (H,W)==tuple(self.input_size), f"QFMConv2d expected input {self.input_size}, got {(H,W)}"
        A = self.morph(x)                                      # [B,1,H,W]
        W_spatial = self._spatial_weight().to(dtype=x.dtype)   # [out,in,kH,kW]
        y_sp = F.conv2d(x, W_spatial, bias=self.bias, stride=self.stride, padding=self.padding)
        if self.stride!=(1,1) or self.padding!=(0,0):
            A_ds = self._apply_stride(F.pad(A, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])))
        else:
            A_ds = A
        A_ds = A_ds[..., :y_sp.shape[-2], :y_sp.shape[-1]].clamp(0,1)
        y_sp = y_sp * (1.0 + 0.5*A_ds)
        x_pad = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])) if self.padding!=(0,0) else x
        Xc = self._fft2(x_pad)
        Yc = self._complex_1x1(Xc)
        Hp, Wp = Yc.shape[-2], Yc.shape[-1]
        freq_mask = self._freq_grid_mask(Hp, Wp, Yc.device).view(1,1,Hp,Wp)
        A_avg = A.mean(dim=(-1,-2), keepdim=True)
        Yc = Yc * (1.0 + A_avg * freq_mask)
        yf = self._ifft2(Yc)
        yf = yf[..., :x_pad.shape[-2], :x_pad.shape[-1]]
        if self.padding!=(0,0):
            yf = yf[..., self.padding[0]:self.padding[0]+H, self.padding[1]:self.padding[1]+W]
        yf = self._apply_stride(yf)
        A_ds2 = self._apply_stride(A)[..., :yf.shape[-2], :yf.shape[-1]]
        y_fr = yf * (1.0 + 0.5*A_ds2)
        psi_sp = torch.complex(y_sp, torch.zeros_like(y_sp))
        psi_fr = torch.complex(y_fr, torch.zeros_like(y_fr))
        cat = torch.cat([y_sp, y_fr], dim=1)
        theta = self.phase_net(cat); theta = torch.tanh(theta) * math.pi
        ei = torch.complex(torch.cos(theta), torch.sin(theta))
        fused = psi_sp + ei * psi_fr
        out = (fused.real**2 + fused.imag**2)
        return self.gamma * out + self.beta

# =============================
# ResNet replace utility
# =============================

def replace_resnet_convs_with_qfm(model: nn.Module, input_size: Tuple[int,int]=(224,224),
                                  layers_to_replace: Iterable[str]=( 'conv1','layer1' ),
                                  device: Optional[torch.device]=None) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cur = list(input_size)
    def out_sz(sz,k,s,p): return (sz + 2*p - k)//s + 1
    # stem
    if 'conv1' in layers_to_replace:
        old = model.conv1
        q = QFMConv2d(old.in_channels, old.out_channels,
                      kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size,old.kernel_size),
                      stride=old.stride if isinstance(old.stride, tuple) else (old.stride,old.stride),
                      padding=old.padding if isinstance(old.padding, tuple) else (old.padding,old.padding),
                      bias=(old.bias is not None), input_size=tuple(cur), device=device, dtype=old.weight.dtype)
        with torch.no_grad():
            q.V.zero_()
            if q.M>0:
                q.V[:,:,0] = old.weight.mean(dim=(-1,-2))
            if q.bias is not None and old.bias is not None:
                q.bias.data.copy_(old.bias.data)
        model.conv1 = q.to(device)
        k = old.kernel_size[0] if isinstance(old.kernel_size, tuple) else old.kernel_size
        s = old.stride[0] if isinstance(old.stride, tuple) else old.stride
        p = old.padding[0] if isinstance(old.padding, tuple) else old.padding
        cur = [out_sz(cur[i], k, s, p) for i in (0,1)]
        cur = [out_sz(cur[i], 3, 2, 1) for i in (0,1)]  # maxpool
    # layers
    for lname in ('layer1','layer2','layer3','layer4'):
        layer = getattr(model, lname)
        for block in layer:
            old = block.conv1
            if lname in layers_to_replace:
                q = QFMConv2d(old.in_channels, old.out_channels,
                              kernel_size=old.kernel_size if isinstance(old.kernel_size, tuple) else (old.kernel_size,old.kernel_size),
                              stride=old.stride if isinstance(old.stride, tuple) else (old.stride,old.stride),
                              padding=old.padding if isinstance(old.padding, tuple) else (old.padding,old.padding),
                              bias=(old.bias is not None), input_size=tuple(cur), device=device, dtype=old.weight.dtype)
                with torch.no_grad():
                    q.V.zero_();
                    if q.M>0: q.V[:,:,0] = old.weight.mean(dim=(-1,-2))
                    if q.bias is not None and old.bias is not None:
                        q.bias.data.copy_(old.bias.data)
                block.conv1 = q.to(device)
            # update spatial size through this conv1
            k = old.kernel_size[0] if isinstance(old.kernel_size, tuple) else old.kernel_size
            s = old.stride[0] if isinstance(old.stride, tuple) else old.stride
            p = old.padding[0] if isinstance(old.padding, tuple) else old.padding
            cur = [out_sz(cur[i], k, s, p) for i in (0,1)]
    return model

# =============================
# Heads and Multi-task Loss
# =============================

class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int|None):
        super().__init__()
        self.cls = None
        if num_classes and num_classes > 1:
            self.cls = nn.Linear(in_features, num_classes)
        self.moist = nn.Linear(in_features, 1)
        self.psize = nn.Linear(in_features, 1)  # optional; only used if labels exist
        self.dropout = nn.Dropout(0.25)
    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.dropout(feat)
        out = {'moist': self.moist(z)}
        out['psize'] = self.psize(z)
        if self.cls is not None:
            out['cls'] = self.cls(z)
        return out

class UncertaintyWeightedLoss(nn.Module):
    """Kendall & Gal style uncertainty weighting with per-task log_sigma.
       Supports missing labels via masks.
    """
    def __init__(self, use_cls: bool, use_psize: bool):
        super().__init__()
        self.use_cls = use_cls
        self.use_psize = use_psize
        # learnable log sigmas
        self.log_sigma_m = nn.Parameter(torch.zeros(1))
        self.log_sigma_p = nn.Parameter(torch.zeros(1)) if use_psize else None
        self.log_sigma_c = nn.Parameter(torch.zeros(1)) if use_cls else None
        self.ce = nn.CrossEntropyLoss(reduction='none') if use_cls else None
        self.l1 = nn.SmoothL1Loss(reduction='none', beta=0.5)
    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
                mask: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total = 0.0
        # moisture
        if mask['moist'].any():
            l = self.l1(pred['moist'].flatten(), target['moist'].flatten())
            l = l[mask['moist']]
            sigma2 = torch.exp(2*self.log_sigma_m)
            losses['moist'] = (l / (2*sigma2) + self.log_sigma_m).mean()
            total = total + losses['moist']
        else:
            losses['moist'] = torch.tensor(0.0, device=pred['moist'].device)
        # psize
        if self.use_psize and mask['psize'].any():
            l = self.l1(pred['psize'].flatten(), target['psize'].flatten())
            l = l[mask['psize']]
            sigma2 = torch.exp(2*self.log_sigma_p)
            losses['psize'] = (l / (2*sigma2) + self.log_sigma_p).mean()
            total = total + losses['psize']
        else:
            losses['psize'] = torch.tensor(0.0, device=pred['moist'].device)
        # classification
        if self.use_cls and mask['cls'].any():
            l = self.ce(pred['cls'][mask['cls']], target['cls'][mask['cls']])
            sigma2 = torch.exp(2*self.log_sigma_c)
            losses['cls'] = (l / (2*sigma2) + self.log_sigma_c).mean()
            total = total + losses['cls']
        else:
            if self.use_cls:
                losses['cls'] = torch.tensor(0.0, device=pred['moist'].device)
        return total, {k: float(v.detach().cpu()) for k,v in losses.items()}

# =============================
# Metrics
# =============================

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(a, b)).item())

def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    # compute per-class precision/recall/f1 without sklearn
    preds = torch.argmax(logits, dim=1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds==c) & (labels==c)).sum().item()
        fp = ((preds==c) & (labels!=c)).sum().item()
        fn = ((preds!=c) & (labels==c)).sum().item()
        if tp==0 and fp==0 and fn==0:
            f1s.append(1.0)  # no samples for this class in batch; neutral
        else:
            prec = tp / (tp+fp+1e-8)
            rec = tp / (tp+fn+1e-8)
            f1 = 2*prec*rec/(prec+rec+1e-8)
            f1s.append(f1)
    return float(np.mean(f1s))

# =============================
# Trainer
# =============================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def try_load_backbone() -> nn.Module:
    try:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[WARN] pretrained weights unavailable: {e}\nUsing random init.")
        return resnet18(weights=None)

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        set_seed(args.seed, deterministic=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Data
        self.train_ds = ParaliDataset(args.train_csv, augment=True,  resize=(args.height, args.width))
        self.val_ds   = ParaliDataset(args.val_csv,   augment=False, resize=(args.height, args.width))
        self.test_ds  = ParaliDataset(args.test_csv,  augment=False, resize=(args.height, args.width))
        nw = min(4, os.cpu_count() or 2)
        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=nw, pin_memory=_USE_CUDA)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=max(1,nw//2))
        self.test_loader  = DataLoader(self.test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=max(1,nw//2))
        # Class map
        self.class_map = None
        if args.class_map and os.path.isfile(args.class_map):
            with open(args.class_map, 'r') as f:
                self.class_map = json.load(f)  # {name: idx}
        else:
            # build from train if present
            names = []
            if 'crop_type' in self.train_ds.df.columns:
                names = [c for c in self.train_ds.df['crop_type'].dropna().unique().tolist()]
            if len(names)==0:
                self.num_classes = None
                self.class_map = {}
            else:
                # If user wants a fixed set but only has wheat, we'll just map what's present
                names = sorted(list(set(names)))
                self.class_map = {n:i for i,n in enumerate(names)}
        self.num_classes = None if len(self.class_map)==0 else len(self.class_map)
        print(f"[INFO] Classes: {self.class_map if self.class_map else 'None (classification disabled)'}")
        # Standardize targets using train split
        m_vals = self.train_ds.df['moisture'].astype(float).values
        self.m_mean = float(np.mean(m_vals)); self.m_std = float(np.std(m_vals) + 1e-8)
        if 'psize' in self.train_ds.df.columns:
            p_vals = self.train_ds.df['psize'].astype(float).dropna().values
            self.p_mean = float(np.mean(p_vals)) if len(p_vals)>0 else 0.0
            self.p_std  = float(np.std(p_vals) + 1e-8) if len(p_vals)>0 else 1.0
        else:
            self.p_mean, self.p_std = 0.0, 1.0
        # Model
        self.model = try_load_backbone()
        layers = tuple([s.strip() for s in args.replace.split(',') if s.strip()])
        self.model = replace_resnet_convs_with_qfm(self.model, input_size=(args.height, args.width), layers_to_replace=layers, device=self.device)
        in_features = self.model.fc.in_features
        self.head = MultiTaskHead(in_features, self.num_classes)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.head = self.head.to(self.device)
        print(f"[INFO] Params (backbone+head): {count_params(self.model)+count_params(self.head):,}")
        # Loss & Optim
        self.crit = UncertaintyWeightedLoss(use_cls=(self.num_classes is not None and self.num_classes>1),
                                            use_psize=('psize' in self.train_ds.df.columns))
        # Param groups
        unitary_params=[]; other_params=[]; head_params=list(self.head.parameters())
        for n,p in self.model.named_parameters():
            if not p.requires_grad: continue
            if '.ortho.P' in n: unitary_params.append(p)
            else: other_params.append(p)
        self.opt = torch.optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd},
            {'params': unitary_params, 'lr': args.lr_unitary, 'weight_decay': 0.0},
            {'params': head_params, 'lr': args.lr_head, 'weight_decay': args.wd}
        ])
        self.scaler = GradScaler(enabled=_USE_CUDA)
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(args.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.best_score = float('inf'); self.start_epoch=1
        # Resume
        if args.resume and os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.head.load_state_dict(ckpt['head'])
            self.opt.load_state_dict(ckpt['opt'])
            self.best_score = ckpt.get('best_score', self.best_score)
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.m_mean, self.m_std = ckpt.get('m_mean', self.m_mean), ckpt.get('m_std', self.m_std)
            self.p_mean, self.p_std = ckpt.get('p_mean', self.p_mean), ckpt.get('p_std', self.p_std)
            print(f"[INFO] Resumed from {args.resume} @ epoch {self.start_epoch-1}")

    # ---- helpers ----
    def encode_crop(self, names: List[Optional[str]]) -> torch.Tensor:
        if self.num_classes is None: return torch.empty((0,), dtype=torch.long)
        idxs=[]
        for n in names:
            if n is None:
                idxs.append(-1)
            else:
                idxs.append(self.class_map.get(n, -1))
        return torch.tensor(idxs, dtype=torch.long)

    def standardize(self, m: torch.Tensor, p: Optional[torch.Tensor]):
        m_z = (m - self.m_mean)/self.m_std
        p_z = None if p is None else (p - self.p_mean)/self.p_std
        return m_z, p_z

    def unstandardize(self, m_z: torch.Tensor, p_z: Optional[torch.Tensor]):
        m = m_z*self.m_std + self.m_mean
        p = None if p_z is None else (p_z*self.p_std + self.p_mean)
        return m, p

    # ---- training ----
    def set_stage(self, stage: str):
        if stage=='warmup':
            for n,p in self.model.named_parameters():
                req = ('.ortho.P' in n) or ('.V' in n) or ('.Wc_' in n) or ('.phase_net' in n) or ('.low_gain' in n) or ('.high_gain' in n) or ('.band_steep' in n)
                p.requires_grad = req
            for p in self.head.parameters(): p.requires_grad=True
        else:
            for p in self.model.parameters(): p.requires_grad=True
            for p in self.head.parameters(): p.requires_grad=True

    def composite_val_score(self, logs: Dict[str, float]) -> float:
        # Lower is better. Combine normalized RMSEs and classification error if present.
        # moisture_rmse is dominant; psize_rmse optional; cls_error=1-macro_f1
        score = logs['moist_rmse']
        if 'psize_rmse' in logs and not math.isnan(logs['psize_rmse']):
            score += 0.2 * logs['psize_rmse']
        if 'macro_f1' in logs and logs['macro_f1']>=0:
            score += 0.5 * (1.0 - logs['macro_f1'])
        return score

    def train(self):
        args = self.args
        self.set_stage('warmup')
        print(f"[INFO] Warm-up for {args.warmup} epochs...")
        global_step=0; patience=args.patience; no_improve=0
        for epoch in range(self.start_epoch, args.epochs+1):
            t0=time.time(); self.model.train(); self.head.train()
            if epoch==args.warmup+1 and self.start_epoch==1:
                self.set_stage('full')
                print("[INFO] Unfroze all parameters after warm-up.")
            # train epoch
            tr_sum=0.0; tr_n=0
            for x, m, crop, p in self.train_loader:
                x = x.to(self.device)
                m = m.to(self.device)
                cz = self.encode_crop(crop).to(self.device)
                p = None if (p is None or (isinstance(p, torch.Tensor) and p.numel()==0)) else p.to(self.device)
                m_z, p_z = self.standardize(m, p)
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=_USE_CUDA):
                    feat = self.model(x)
                    out = self.head(feat)
                    # build targets & masks
                    tgt = {'moist': m_z}
                    msk = {'moist': torch.ones_like(m, dtype=torch.bool)}
                    if 'psize' in self.train_ds.df.columns:
                        if p_z is None:
                            p_mask = torch.zeros_like(m, dtype=torch.bool)
                            tgt['psize'] = torch.zeros_like(m)
                        else:
                            p_mask = torch.ones_like(m, dtype=torch.bool)
                            tgt['psize'] = p_z
                        msk['psize'] = p_mask
                    if self.num_classes is not None and self.num_classes>1:
                        cls_mask = (cz>=0)
                        tgt['cls'] = cz[cls_mask]
                        msk['cls'] = cls_mask
                    loss, parts = self.crit(out, tgt, msk)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                tr_sum += loss.item() * x.size(0); tr_n += x.size(0); global_step+=1
            tr_loss = tr_sum/max(1,tr_n)
            self.writer.add_scalar('loss/train', tr_loss, epoch)
            # validate
            logs = self.validate(epoch)
            score = self.composite_val_score(logs)
            self.writer.add_scalar('score/val_composite', score, epoch)
            dt=time.time()-t0
            print(f"Epoch {epoch:03d}/{args.epochs} | train {tr_loss:.4f} | val_moist_rmse {logs['moist_rmse']:.4f} | score {score:.4f} | {dt:.1f}s")
            # checkpoint
            if score < self.best_score:
                self.best_score = score; no_improve=0
                torch.save({
                    'model': self.model.state_dict(),
                    'head': self.head.state_dict(),
                    'opt': self.opt.state_dict(),
                    'epoch': epoch,
                    'best_score': self.best_score,
                    'm_mean': self.m_mean, 'm_std': self.m_std,
                    'p_mean': self.p_mean, 'p_std': self.p_std,
                    'class_map': self.class_map,
                }, 'qfm_parali_best.pth')
                print(f"[INFO] Saved best checkpoint (score={score:.4f})")
            else:
                no_improve+=1
                if no_improve>=patience:
                    print(f"[Early Stop] patience {patience} reached (best score={self.best_score:.4f}).")
                    break
            # save last
            torch.save({
                'model': self.model.state_dict(),
                'head': self.head.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': epoch,
                'best_score': self.best_score,
                'm_mean': self.m_mean, 'm_std': self.m_std,
                'p_mean': self.p_mean, 'p_std': self.p_std,
                'class_map': self.class_map,
            }, 'qfm_parali_last.pth')
        # Final test
        self.test()
        self.writer.close()

    @torch.no_grad()
    def validate(self, epoch:int) -> Dict[str,float]:
        self.model.eval(); self.head.eval()
        m_preds=[]; m_trues=[]
        p_preds=[]; p_trues=[]; have_p=False
        macro_f1_vals=[]; have_cls=False
        for x, m, crop, p in self.val_loader:
            x=x.to(self.device); m=m.to(self.device)
            cz=self.encode_crop(crop).to(self.device)
            p = None if (p is None or (isinstance(p, torch.Tensor) and p.numel()==0)) else p.to(self.device)
            with autocast(enabled=_USE_CUDA):
                feat = self.model(x); out = self.head(feat)
            # moisture
            m_hat_z = out['moist'].squeeze(1)
            m_hat, _ = self.unstandardize(m_hat_z, None)
            m_preds.append(m_hat.detach().cpu()); m_trues.append(m.detach().cpu())
            # psize
            if 'psize' in self.val_ds.df.columns and p is not None:
                p_hat_z = out['psize'].squeeze(1)
                _, p_hat = self.unstandardize(None, p_hat_z)
                p_preds.append(p_hat.detach().cpu()); p_trues.append(p.detach().cpu()); have_p=True
            # cls
            if self.num_classes is not None and self.num_classes>1 and (cz>=0).any():
                logits = out['cls'][cz>=0]
                labels = cz[cz>=0]
                macro_f1_vals.append(macro_f1_from_logits(logits, labels, self.num_classes))
                have_cls=True
        m_pred = torch.cat(m_preds) if len(m_preds)>0 else torch.tensor([])
        m_true = torch.cat(m_trues) if len(m_trues)>0 else torch.tensor([])
        moist_rmse = rmse(m_pred, m_true) if m_pred.numel()>0 else float('nan')
        logs={'moist_rmse': moist_rmse}
        if have_p:
            p_pred = torch.cat(p_preds); p_true = torch.cat(p_trues)
            logs['psize_rmse'] = rmse(p_pred, p_true)
        if have_cls:
            logs['macro_f1'] = float(np.mean(macro_f1_vals))
        return logs

    @torch.no_grad()
    def test(self):
        # Load best ckpt if exists
        if os.path.isfile('qfm_parali_best.pth'):
            ckpt = torch.load('qfm_parali_best.pth', map_location=self.device)
            self.model.load_state_dict(ckpt['model']); self.head.load_state_dict(ckpt['head'])
        self.model.eval(); self.head.eval()
        m_preds=[]; m_trues=[]; have_p=False; p_preds=[]; p_trues=[]
        for x,m,_,p in self.test_loader:
            x=x.to(self.device); m=m.to(self.device)
            p = None if (p is None or (isinstance(p, torch.Tensor) and p.numel()==0)) else p.to(self.device)
            with autocast(enabled=_USE_CUDA):
                feat=self.model(x); out=self.head(feat)
            m_hat, _ = self.unstandardize(out['moist'].squeeze(1), None)
            m_preds.append(m_hat.detach().cpu()); m_trues.append(m.detach().cpu())
            if 'psize' in self.test_ds.df.columns and p is not None:
                _, p_hat = self.unstandardize(None, out['psize'].squeeze(1))
                p_preds.append(p_hat.detach().cpu()); p_trues.append(p.detach().cpu()); have_p=True
        if len(m_preds)>0:
            m_pred = torch.cat(m_preds); m_true = torch.cat(m_trues)
            print(f"[TEST] Moisture RMSE: {rmse(m_pred, m_true):.4f}")
        if have_p:
            p_pred = torch.cat(p_preds); p_true = torch.cat(p_trues)
            print(f"[TEST] PSize   RMSE: {rmse(p_pred, p_true):.4f}")

    # ---- export ----
    def export_torchscript(self, out_path: str='qfm_parali_ts.pt'):
        self.model.eval(); self.head.eval()
        # Bundle into a single module for export
        class Wrapped(nn.Module):
            def __init__(self, backbone, head, m_mean, m_std, p_mean, p_std, num_classes, class_map):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.m_mean = torch.tensor(m_mean)
                self.m_std  = torch.tensor(m_std)
                self.p_mean = torch.tensor(p_mean)
                self.p_std  = torch.tensor(p_std)
                self.num_classes = -1 if (num_classes is None) else int(num_classes)
                # store names in same order as indices for app-side mapping
                if class_map and len(class_map)>0:
                    idx_to_name = [None]*len(class_map)
                    for name, idx in class_map.items():
                        idx_to_name[idx] = name
                    self.class_names = idx_to_name
                else:
                    self.class_names = []
            def forward(self, x: torch.Tensor):
                # x: [B,3,H,W], normalized already
                feat = self.backbone(x)
                out = self.head(feat)
                m = out['moist'].squeeze(1) * self.m_std + self.m_mean
                p = out['psize'].squeeze(1) * self.p_std + self.p_mean
                if self.num_classes and self.num_classes>1 and 'cls' in out:
                    probs = F.softmax(out['cls'], dim=1)
                else:
                    probs = torch.empty((x.size(0),0), dtype=x.dtype, device=x.device)
                return probs, m, p
        wrapped = Wrapped(self.model, self.head, self.m_mean, self.m_std, self.p_mean, self.p_std, self.num_classes, self.class_map)
        scripted = torch.jit.script(wrapped)
        scripted.save(out_path)
        print(f"[EXPORT] TorchScript saved to {out_path}")

# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(description='QFM-CNN Parali Post-Training')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_unitary', type=float, default=5e-4)
    parser.add_argument('--lr_head', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--replace', type=str, default='conv1,layer1', help='comma list among conv1,layer1,layer2,layer3,layer4')
    parser.add_argument('--logdir', type=str, default='runs/qfm_parali')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--class_map', type=str, default='')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--export_ts', action='store_true', help='export TorchScript at the end')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
    if args.export_ts:
        trainer.export_torchscript('qfm_parali_ts.pt')

if __name__ == '__main__':
    main()
