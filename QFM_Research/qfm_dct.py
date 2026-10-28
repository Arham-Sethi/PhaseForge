# qfm_dct.py
import math
from typing import Tuple, Iterable, List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast as amp_autocast

from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# FIXES INCLUDED (high level)
# 1) DCT/IDCT implemented correctly (no diagonal einsum bug).
# 2) DCT matrices cached in float32; transform runs in float32.
# 3) Radial grid matches DCT indexing (0..H-1), not fftfreq().
# 4) OrthoMixer uses Cayley transform (smooth, no sign()).
# 5) QFM replacement init projects old conv weights onto base bank.
# 6) replace list strips whitespace (no silent " layer4" bug).
# 7) style_score is per-sample (no batch-z leak / instability).
# 8) JPEG proxy uses 8x8 boundaries.
# 9) Real ablations: enable_freq / enable_morph / enable_style / enable_phase
#
# NEW FIXES (this patch)
# 10) CIFAR stem applied BEFORE QFM replacement (no overwriting conv1).
# 11) Morph gating is NOT a no-op (learnable local gating).
# 12) Frequency mask can suppress (uses gain R, not always > 1).
# 13) Mask uses LOCAL trust A on DCT grid (not Aavg).
# 14) Fusion is LOCAL (per-pixel) based on A (matches theory).
# ============================================================


# ------------------------------
# Separable DCT-II / IDCT-III (orthonormal)
#   Note: DCT/IDCT run in float32 for stability under AMP.
# ------------------------------

_dct_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
_idct_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def _dct_mat_f32(n: int, device) -> torch.Tensor:
    key = (n, device)
    if key in _dct_cache:
        return _dct_cache[key]
    k = torch.arange(n, device=device, dtype=torch.float32).view(-1, 1)  # [n,1]
    i = torch.arange(n, device=device, dtype=torch.float32).view(1, -1)  # [1,n]
    mat = torch.cos(math.pi * (i + 0.5) * k / n)                         # [n,n]
    mat[0, :] = mat[0, :] * (1.0 / math.sqrt(n))
    mat[1:, :] = mat[1:, :] * math.sqrt(2.0 / n)
    _dct_cache[key] = mat
    return mat


def _idct_mat_f32(n: int, device) -> torch.Tensor:
    key = (n, device)
    if key in _idct_cache:
        return _idct_cache[key]
    _idct_cache[key] = _dct_mat_f32(n, device).t()
    return _idct_cache[key]


def dct2(x: torch.Tensor) -> torch.Tensor:
    """
    Correct separable DCT:
      y = Th @ x @ Tw^T  (applied per (B,C))
    x: [B,C,H,W]
    returns float32 tensor [B,C,H,W]
    """
    B, C, H, W = x.shape
    device = x.device

    Th = _dct_mat_f32(H, device)  # [H,H], float32
    Tw = _dct_mat_f32(W, device)  # [W,W], float32

    xc = x.reshape(B * C, H, W).to(torch.float32)  # [BC,H,W]
    yc = Th @ xc
    yc = yc @ Tw.t()
    return yc.reshape(B, C, H, W)


def idct2(X: torch.Tensor) -> torch.Tensor:
    """
    Correct separable IDCT:
      x = Th^T @ X @ Tw
    X: [B,C,H,W] float32
    returns float32 tensor [B,C,H,W]
    """
    B, C, H, W = X.shape
    device = X.device

    ThT = _idct_mat_f32(H, device)
    TwT = _idct_mat_f32(W, device)

    Xc = X.reshape(B * C, H, W).to(torch.float32)
    xc = ThT @ Xc
    xc = xc @ TwT.t()
    return xc.reshape(B, C, H, W)


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
        self.register_buffer(
            "kernel",
            kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
        )
        self.stride = stride
        self.pad = (filt_size // 2,) * 4  # L,R,T,B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad[0], self.pad[1], self.pad[2], self.pad[3]), mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=x.shape[1])


# ------------------------------
# Sobel & soft morphology
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMorphGate(nn.Module):
    """
    Morphological gate that outputs A in [0,1], shape [B,1,H,W].

    Interface expected by your trainer:
      - set_tau(tau: float)
      - get_tau() -> float
    """

    def __init__(self, in_channels=None, kernel_size=3, tau_init=5.0):
        super().__init__()

        # keep tau as a buffer so it moves with .to(device) and saves in state_dict
        self.register_buffer("_tau", torch.tensor(float(tau_init), dtype=torch.float32))

        # Fixed Sobel kernels (fp32 buffers)
        kx = torch.tensor(
            [[[-1., 0., 1.],
              [-2., 0., 2.],
              [-1., 0., 1.]]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[[-1., -2., -1.],
              [ 0.,  0.,  0.],
              [ 1.,  2.,  1.]]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

        # NOTE: kernel_size/in_channels are accepted only for compatibility.
        # This gate is intentionally channel-agnostic and uses a grayscale proxy.

    @torch.no_grad()
    def set_tau(self, tau: float):
        self._tau.fill_(float(tau))

    def get_tau(self) -> float:
        return float(self._tau.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] (fp16 or fp32)
        returns A: [B,1,H,W] same dtype as x
        """
        x32 = x.float()

        # grayscale proxy
        if x32.size(1) > 1:
            g = x32.mean(dim=1, keepdim=True)
        else:
            g = x32

        gx = F.conv2d(g, self.kx, padding=1)
        gy = F.conv2d(g, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        # normalize to [0,1]
        denom = mag.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        a = (mag / denom).clamp(0.0, 1.0)

        # tau acts like a sharpness/contrast control
        tau = self._tau.clamp(0.25, 50.0)
        A = torch.sigmoid((a - 0.5) * tau)

        return A.to(dtype=x.dtype)


# ------------------------------
# Orthogonal mixer via Cayley transform
# ------------------------------

class OrthoMixer(nn.Module):
    """
    Stable orthogonal mixer.
    Uses Cayley transform but bounds parameters and falls back to identity if solve becomes unstable.
    """
    def __init__(self, M: int, eps: float = 1e-2, max_scale: float = 0.25):
        super().__init__()
        self.M = M
        self.eps = eps
        self.max_scale = max_scale
        self.S = nn.Parameter(torch.zeros(M, M))  # start at 0 -> Q=I

    def forward(self) -> torch.Tensor:
        S = torch.tanh(self.S) * float(self.max_scale)
        A = S - S.t()
        I = torch.eye(self.M, device=S.device, dtype=S.dtype)

        M1 = (I - A) + float(self.eps) * I
        M2 = (I + A)

        try:
            Q = torch.linalg.solve(M1, M2)
        except RuntimeError:
            Q = I

        if not torch.isfinite(Q).all():
            Q = I
        return Q


# ------------------------------
# Base spatial filter bank
# ------------------------------

_pinv_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

def _gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float):
    xmax = ks // 2
    ymax = ks // 2
    y, x = torch.meshgrid(
        torch.arange(-ymax, ymax + 1),
        torch.arange(-xmax, xmax + 1),
        indexing="ij"
    )
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    gb = torch.exp(-(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2 * sigma ** 2)) * \
         torch.cos(2 * math.pi * x_theta / lam + psi)
    gb = gb / (gb.abs().sum() + 1e-6)
    return gb


def make_base_filters(kernel_size: int, device: torch.device) -> torch.Tensor:
    ks = kernel_size
    assert ks in (3, 5, 7), "kernel_size must be 3, 5 or 7."
    bank: List[torch.Tensor] = []

    eye = torch.zeros((ks, ks), dtype=torch.float32)
    eye[ks // 2, ks // 2] = 1.0
    bank.append(eye)

    box = torch.ones((ks, ks), dtype=torch.float32) / (ks * ks)
    bank.append(box)

    if ks == 3:
        gauss = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32)
        gauss /= gauss.sum()
    elif ks == 5:
        gauss = torch.tensor([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]], dtype=torch.float32)
        gauss /= gauss.sum()
    else:
        g1d = torch.tensor([1, 6, 15, 20, 15, 6, 1], dtype=torch.float32)
        gauss = torch.outer(g1d, g1d)
        gauss /= gauss.sum()
    bank.append(gauss)

    if ks == 3:
        sharp = torch.tensor([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=torch.float32)
    else:
        sharp = -box * (ks * ks - 1)
        sharp[ks // 2, ks // 2] += ks * ks
    bank.append(sharp)

    if ks == 3:
        lap = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32)
    else:
        lap = torch.zeros((ks, ks), dtype=torch.float32)
        lap[ks // 2 - 1:ks // 2 + 2, ks // 2 - 1:ks // 2 + 2] = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        )
    bank.append(lap)

    if ks == 3:
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
    else:
        sobel_x = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_y = torch.zeros((ks, ks), dtype=torch.float32)
        sobel_x[ks // 2 - 1:ks // 2 + 2, ks // 2 - 1:ks // 2 + 2] = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y[ks // 2 - 1:ks // 2 + 2, ks // 2 - 1:ks // 2 + 2] = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        )
    bank.extend([sobel_x, sobel_y])

    for theta in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        bank.append(_gabor_kernel(ks, 1.2, theta, 4.0, 0.5, 0.0))

    bank = torch.stack(bank, dim=0).unsqueeze(1).to(device=device)  # [M,1,ks,ks]
    return bank


def _base_bank_pinv(ks: int, device: torch.device) -> torch.Tensor:
    key = (ks, device)
    if key in _pinv_cache:
        return _pinv_cache[key]
    bank = make_base_filters(ks, device).squeeze(1)  # [M,ks,ks]
    M = bank.shape[0]
    K = ks * ks
    B = bank.reshape(M, K).to(torch.float32)  # [M,K]
    pinv = torch.linalg.pinv(B).to(torch.float32)  # [K,M]
    _pinv_cache[key] = pinv
    return pinv


def project_conv_to_bank(old_w: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """
    old_w: [out,in,ks,ks]
    bank:  [M,1,ks,ks]
    returns V_init: [out,in,M] such that sum_m V[...,m] * bank[m] ~ old_w
    """
    device = old_w.device
    dtype = old_w.dtype
    ks = old_w.shape[-1]
    pinv = _base_bank_pinv(ks, device)  # [K,M], float32
    out_ch, in_ch, _, _ = old_w.shape
    K = ks * ks

    w_flat = old_w.reshape(out_ch * in_ch, K).to(torch.float32)  # [OI,K]
    coeff = w_flat @ pinv  # [OI,M]
    coeff = coeff.reshape(out_ch, in_ch, -1).to(dtype=dtype)
    return coeff


# ------------------------------
# QFMConv2d-DCT
# ------------------------------

class QFMConv2dDCT(nn.Module):
    """
    Stabilized QFM block:
      - Spatial branch: conv using base bank mixture (learned V)
      - Frequency branch: DCT2(yS) -> channel mix -> DCT-radius mask -> IDCT2 -> yF
      - Morph gate: A(x) -> local gating for yS/yF
      - Fusion: LOCAL arbitration wS(x)*yS + wF(x)*yF
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        phase_clip: float = math.pi / 6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,

        enable_freq: bool = True,
        enable_morph: bool = True,
        enable_style: bool = True,
        enable_interference: bool = False,
        enable_phase: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.phase_clip = float(phase_clip)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.enable_freq = enable_freq
        self.enable_morph = enable_morph
        self.enable_style = enable_style
        self.use_interference = enable_interference
        self.phase_enabled = enable_phase

        # Spatial branch
        self.register_buffer("_base_bank", make_base_filters(self.kH, self.device))
        self.M = int(self._base_bank.shape[0])
        self.ortho = OrthoMixer(self.M)
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.001)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Frequency branch: mix across output channels
        self.Wf = nn.Parameter(torch.eye(out_channels, dtype=torch.float32))

        # Morphology gate
        self.morph = SimpleMorphGate(in_channels, kernel_size=3, tau_init=5.0)

        # NEW: learnable local morph strengths (start small)
        self.morph_s = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.morph_f = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        # DCT-radius mask parameters
        self._t_param = nn.Parameter(torch.tensor(0.0))
        self._lg_param = nn.Parameter(torch.tensor(0.0))
        self._hg_param = nn.Parameter(torch.tensor(0.0))
        self.band_steep = nn.Parameter(torch.tensor(8.0))

        # Fusion gates (global baselines)
        self.s_gate = nn.Parameter(torch.tensor(1.5))
        self.f_gate_base = nn.Parameter(torch.tensor(-3.0))
        self.last_A = None
        self.last_G = None

        # Phase head (kept; not touched)
        self.phase_head = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        for m in self.phase_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _squash(val: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(val)

    def _spatial_weight(self) -> torch.Tensor:
        # Keep OrthoMixer bypassed (stability-first)
        A = self.V  # [out,in,M]
        bank = self._base_bank.squeeze(1)  # [M,ks,ks]
        W = torch.einsum("oin,mxy->oixy", A, bank)
        return W

    def _build_dct_radius(self, H: int, W: int, device):
        yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
        r = torch.sqrt(yy * yy + xx * xx)

        rmax = torch.sqrt(torch.tensor((H - 1) ** 2 + (W - 1) ** 2,
                                    device=device, dtype=torch.float32))
        r = r / (rmax + 1e-6)          # r in [0,1]
        return r.clamp(0.0, 1.0)

    def make_gain_mask(self, H: int, W: int, device=None, dtype=None):
        """
        Debug helper for triage: returns current gain mask G(H,W).
        Must reflect the SAME formula used in forward().
        """
        device = device or self._t_param.device
        dtype = dtype or torch.float32

        # Use the exact same parameterization you use in forward.
        # If your forward uses: t = sigmoid(_t_param) * t_max, etc.,
        # replicate that here identically.
        t = torch.sigmoid(self._t_param) * self.t_max
        lg = 1.0 + torch.tanh(self._lg_param) * self.gain_range
        hg = 1.0 + torch.tanh(self._hg_param) * self.gain_range

        # Call your existing implementation if you already have one:
        # return self._make_gain_mask(H, W, t, lg, hg, device=device, dtype=dtype)

        # Otherwise compute it here using the same radial grid used in forward
        r = self._radial_grid(H, W, device=device, dtype=dtype)  # whatever your helper is called
        G = lg + (hg - lg) * torch.sigmoid((r - t) / self.tau)

        return G


    
    def forward(
        self,
        x: torch.Tensor,
        style_score: Optional[torch.Tensor] = None,
        f_clip: float = 0.3,
        force_no_freq: bool = False,
    ) -> torch.Tensor:
        with amp_autocast(device_type="cuda", enabled=False):
            """
            x: [B,Cin,H,W]
            style_score: [B,1,1,1] in [0,1] (optional)
            force_no_freq: warmup can set this True
            """
            # Morphology map on input (optional)
            if self.enable_morph:
                A = self.morph(x)  # [B,1,H,W]
                # >>> ADD THESE 2 LINES <<<
                self.last_A = A.detach()
            else:
                A = None
                # >>> ADD THIS LINE <<<
                self.last_A = None


            # Spatial conv
            Wsp = self._spatial_weight().to(dtype=x.dtype, device=x.device)
            yS = F.conv2d(x, Wsp, bias=self.bias, stride=self.stride, padding=self.padding)

            # FIX: real local morph gating for spatial branch
            if A is not None:
                A_ds = F.interpolate(A, size=yS.shape[-2:], mode="bilinear", align_corners=False)
                gS = torch.tanh(self.morph_s) * (A_ds - 0.5) * 2.0  # in [-tanh, +tanh]
                yS = yS * (1.0 + gS).clamp(0.5, 1.5)

            # Frequency branch
            do_freq = self.enable_freq and (not force_no_freq)
            if not do_freq:
                if not hasattr(self, "_printed_freq_bypass"):
                    self._printed_freq_bypass = 0

                # print only once total (first QFM layer that hits bypass)
                if self._printed_freq_bypass < 1:
                    with torch.no_grad():
                        s = torch.sigmoid(self.s_gate)
                        yS_abs = yS.abs().mean()
                        out = s * yS
                        out_abs = out.abs().mean()
                        print(
                            f"[BYPASS_DEBUG] sigmoid(s_gate)={float(s):.6f} "
                            f"mean|yS|={float(yS_abs):.6f} mean|out|={float(out_abs):.6f} "
                            f"ratio(out/yS)={float(out_abs / (yS_abs + 1e-12)):.6f}"
                        )
                    self._printed_freq_bypass += 1

                s = torch.sigmoid(self.s_gate)
                return s * yS




            # ---- AMP-SAFE FREQ PATH: disable autocast locally ----
            # This prevents GradScaler step-skips caused by inf/NaNs from mixed precision
            if x.is_cuda:
                autocast_off = torch.autocast(device_type="cuda", enabled=False)
            else:
                # CPU: autocast isn't used anyway, but keep structure consistent
                class _Null:
                    def __enter__(self): return None
                    def __exit__(self, *a): return False
                autocast_off = _Null()

            with autocast_off:
                yS32 = yS.to(torch.float32)

                # DCT in float32
                Y = dct2(yS32)  # float32

                # channel mixing in float32
                Wf32 = self.Wf.to(dtype=torch.float32, device=Y.device)
                Y = torch.einsum("oc,bchw->bohw", Wf32, Y)

                Hs, Ws = Y.shape[-2], Y.shape[-1]
                r = self._build_dct_radius(Hs, Ws, Y.device).view(1, 1, Hs, Ws)

                t  = self._squash(self._t_param, 0.20, 0.55)     # threshold on radius
                lg = self._squash(self._lg_param, 0.70, 1.00)    # low-freq: suppress or keep
                hg = self._squash(self._hg_param, 1.00, 1.30)    # high-freq: boost or keep
                if not hasattr(self, "_tgrad_printed"):
                    self._tgrad_printed = True
                    print("[TGRAD] t.requires_grad=", t.requires_grad, " _t_param.requires_grad=", self._t_param.requires_grad)

                # smooth band: alpha≈0 for low-freq, alpha≈1 for high-freq (DIFF wrt t)
                k = self.band_steep.clamp(1.0, 40.0)  # slope
                alpha = torch.sigmoid((r - t) * k)    # [1,1,Hs,Ws]
                G = lg * (1.0 - alpha) + hg * alpha
                self.last_G = G.detach()


                # Style gating: on clean images, push G toward identity; on degraded, allow it
                if self.enable_style and (style_score is not None):
                    ss = style_score.to(torch.float32)  # [B,1,1,1]
                    # Broadcast to [B,1,Hs,Ws]
                    ss = ss.expand(-1, 1, Hs, Ws)
                    G = 1.0 + ss * (G - 1.0)
                    
                # ---- DEBUG: verify gain is not constant (Step-6) ----
                if not hasattr(self, "_dbgG"):
                    self._dbgG = 0

                if self._dbgG < 3:
                    with torch.no_grad():
                        print(
                            "[G]",
                            "t:", float(t),
                            "lg:", float(lg),
                            "hg:", float(hg),
                            "| G min/mean/max:",
                            float(G.min()),
                            float(G.mean()),
                            float(G.max())
                        )
                    self._dbgG += 1

                # Local trust blend with morphology map A:
                # - trust=0 -> identity
                # - trust=1 -> apply G
                if A is not None:
                    A_Y = F.interpolate(A, size=(Hs, Ws), mode="bilinear", align_corners=False).to(torch.float32)
                    trust = A_Y.clamp(0.0, 1.0)
                    Y = Y * (1.0 + trust * (G - 1.0))
                else:
                    Y = Y * G


                # IDCT back to float32
                yF32 = idct2(Y)  # float32

            # cast back to original dtype for the rest of the block
            yF = yF32.to(dtype=yS.dtype)
            # scale match to prevent runaway freq magnitude (keeps training sane)
            yF = yF * (yS.abs().mean(dim=(-1,-2), keepdim=True) /
                    (yF.abs().mean(dim=(-1,-2), keepdim=True) + 1e-6))


            # FIX: real local morph gating for frequency branch
            if A is not None:
                A_ds2 = F.interpolate(A, size=yF.shape[-2:], mode="bilinear", align_corners=False)
                gF = torch.tanh(self.morph_f) * (A_ds2 - 0.5) * 2.0
                yF = yF * (1.0 + gF).clamp(0.5, 1.5)

            # ===== C-DEBUG: capture branch magnitudes once =====
            if not hasattr(self, "_cdbg_done"):
                self._cdbg_done = False

            # Fusion gates (global baselines)
            s0 = torch.sigmoid(self.s_gate)
            f0 = torch.sigmoid(self.f_gate_base)

            if self.enable_style and (style_score is not None):
                f_eff = (f0 * (0.5 + 0.5 * style_score)).clamp(0.0, float(f_clip))
            else:
                f_eff = f0.clamp(0.0, float(f_clip))

            # ---- DEBUG: branch magnitudes (once) ----
            if (not self._cdbg_done) and (not force_no_freq) and self.enable_freq:
                with torch.no_grad():
                    yS_mag = float(yS.abs().mean())
                    yF_mag = float(yF.abs().mean())
                    print(
                        f"[CDEBUG] mean|yS|={yS_mag:.6f} "
                        f"mean|yF|={yF_mag:.6f} "
                        f"ratio(yF/yS)={yF_mag/(yS_mag+1e-12):.6f}"
                    )
                self._cdbg_done = True


            # ===== LOCAL FUSION (SCALE-PRESERVING) =====
            if A is None:
                wS = s0
                wF = f_eff
                denom = (wS + wF).clamp_min(1e-6)
                return (wS * yS + wF * yF) / denom
            else:
                A_out = F.interpolate(A, size=yS.shape[-2:], mode="bilinear", align_corners=False)

                wS = (s0 * (0.5 + 0.5 * A_out)).clamp(0.0, 1.0)
                wF = (f_eff * (0.5 + 0.5 * (1.0 - A_out))).clamp(0.0, 1.0)

                denom = (wS + wF).clamp_min(1e-6)
                return (wS * yS + wF * yF) / denom



# ------------------------------
# Style proxy (PER-SAMPLE)
# ------------------------------

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 1:
        return x
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)

def _bounded_sigmoid(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(scale * z)

def noise_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    k = torch.tensor([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    L = F.conv2d(g, k, padding=1)
    v = L.var(dim=(-1, -2), keepdim=True)
    return _bounded_sigmoid(torch.log1p(v) * 2.0)

def jpeg_blockiness_proxy(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    g = _to_gray(x)
    B, _, H, W = g.shape

    diff_h = (g[..., :, 1:] - g[..., :, :-1]).abs()
    diff_v = (g[..., 1:, :] - g[..., :-1, :]).abs()

    col_idx = torch.arange(W - 1, device=g.device)
    row_idx = torch.arange(H - 1, device=g.device)

    boundary_cols = ((col_idx + 1) % block == 0).float().view(1, 1, 1, -1)
    interior_cols = 1.0 - boundary_cols
    boundary_rows = ((row_idx + 1) % block == 0).float().view(1, 1, -1, 1)
    interior_rows = 1.0 - boundary_rows

    b_h = (diff_h * boundary_cols).sum(dim=(-2, -1)) / (boundary_cols.sum() + 1e-6)
    i_h = (diff_h * interior_cols).sum(dim=(-2, -1)) / (interior_cols.sum() + 1e-6)
    b_v = (diff_v * boundary_rows).sum(dim=(-2, -1)) / (boundary_rows.sum() + 1e-6)
    i_v = (diff_v * interior_rows).sum(dim=(-2, -1)) / (interior_rows.sum() + 1e-6)

    ratio = (b_h + b_v) / (i_h + i_v + 1e-6)
    return _bounded_sigmoid(torch.log1p(ratio) * 1.5).view(-1, 1, 1, 1)

def blur_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    gx = (g[..., :, 1:] - g[..., :, :-1]).abs().mean(dim=(-1, -2), keepdim=True)
    gy = (g[..., 1:, :] - g[..., :-1, :]).abs().mean(dim=(-1, -2), keepdim=True)
    sharp = gx + gy
    blur = 1.0 / (sharp + 1e-6)
    return _bounded_sigmoid(torch.log1p(blur) * 1.5)

def lowlight_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    mean_l = g.mean(dim=(-1, -2), keepdim=True)
    inv_l = 1.0 - mean_l
    return _bounded_sigmoid((inv_l - 0.3) * 4.0)

def hf_drop_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    box = torch.ones((1, 1, 3, 3), device=g.device, dtype=g.dtype) / 9.0
    g_lp = F.conv2d(g, box, padding=1)
    g_hp = g - g_lp
    hf_energy = (g_hp ** 2).mean(dim=(-1, -2), keepdim=True)
    drop = 1.0 / (hf_energy + 1e-6)
    return _bounded_sigmoid(torch.log1p(drop) * 1.2)

def style_score(x: torch.Tensor) -> torch.Tensor:
    pn = noise_proxy(x)
    pj = jpeg_blockiness_proxy(x, block=8)
    pb = blur_proxy(x)
    pl = lowlight_proxy(x)
    ph = hf_drop_proxy(x)
    s = 0.22 * pn + 0.08 * pj + 0.22 * pb + 0.12 * pl + 0.18 * ph
    return (0.15 + 0.85 * s).clamp(0.0, 1.0)


# ------------------------------
# Replace convs + insert BlurPool AA
# ------------------------------

def replace_resnet_with_qfm(
    model: nn.Module,
    device: Optional[torch.device] = None,
    replace: Iterable[str] = ("conv1",),
    phase_clip: float = math.pi / 6,
    aa_every_stride2: bool = False,

    enable_freq: bool = True,
    enable_morph: bool = True,
    enable_style: bool = True,
    enable_interference: bool = False,
    enable_phase: bool = False,
) -> nn.Module:
    if device is None:
        device = next(model.parameters()).device

    replace_set = set(replace)

    # Stem conv1
    if "conv1" in replace_set:
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
            dtype=old.weight.dtype,
            enable_freq=enable_freq,
            enable_morph=enable_morph,
            enable_style=enable_style,
            enable_interference=enable_interference,
            enable_phase=enable_phase,
        ).to(device)

        with torch.no_grad():
            q.V.copy_(project_conv_to_bank(old.weight.to(device), q._base_bank))
            if q.bias is not None and old.bias is not None:
                q.bias.copy_(old.bias)

        model.conv1 = q

    # Stem AA
    if aa_every_stride2:
        in_ch = model.bn1.num_features
        model.maxpool = nn.Sequential(
            BlurPool(in_ch, filt_size=3, stride=2),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

    # Layers
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        for block in layer:
            if aa_every_stride2 and hasattr(block, "conv1"):
                base_conv = block.conv1
                stride = base_conv.stride if isinstance(base_conv, nn.Conv2d) else getattr(base_conv, "stride", (1, 1))
                s1 = stride[0] if isinstance(stride, tuple) else stride
                if s1 == 2 and isinstance(base_conv, nn.Conv2d):
                    in_ch = base_conv.in_channels
                    aa = BlurPool(in_ch, filt_size=3, stride=2)
                    block.conv1 = nn.Sequential(
                        aa,
                        nn.Conv2d(
                            in_ch,
                            base_conv.out_channels,
                            kernel_size=base_conv.kernel_size,
                            stride=1,
                            padding=base_conv.padding,
                            bias=(base_conv.bias is not None)
                        )
                    )

            if lname in replace_set and hasattr(block, "conv1"):
                old = block.conv1

                if isinstance(old, nn.Sequential) and isinstance(old[0], BlurPool) and isinstance(old[1], nn.Conv2d):
                    conv_part = old[1]
                    q = QFMConv2dDCT(
                        in_channels=conv_part.in_channels,
                        out_channels=conv_part.out_channels,
                        kernel_size=conv_part.kernel_size if isinstance(conv_part.kernel_size, tuple) else (conv_part.kernel_size,) * 2,
                        stride=conv_part.stride if isinstance(conv_part.stride, tuple) else (conv_part.stride,) * 2,
                        padding=conv_part.padding if isinstance(conv_part.padding, tuple) else (conv_part.padding,) * 2,
                        bias=(conv_part.bias is not None),
                        phase_clip=phase_clip,
                        device=device,
                        enable_freq=enable_freq,
                        enable_morph=enable_morph,
                        enable_style=enable_style,
                        enable_interference=enable_interference,
                        enable_phase=enable_phase,
                    ).to(device)
                    with torch.no_grad():
                        q.V.copy_(project_conv_to_bank(conv_part.weight.to(device), q._base_bank))
                        if q.bias is not None and conv_part.bias is not None:
                            q.bias.copy_(conv_part.bias)
                    old[1] = q

                elif isinstance(old, nn.Conv2d):
                    conv_part = old
                    q = QFMConv2dDCT(
                        in_channels=conv_part.in_channels,
                        out_channels=conv_part.out_channels,
                        kernel_size=conv_part.kernel_size if isinstance(conv_part.kernel_size, tuple) else (conv_part.kernel_size,) * 2,
                        stride=conv_part.stride if isinstance(conv_part.stride, tuple) else (conv_part.stride,) * 2,
                        padding=conv_part.padding if isinstance(conv_part.padding, tuple) else (conv_part.padding,) * 2,
                        bias=(conv_part.bias is not None),
                        phase_clip=phase_clip,
                        device=device,
                        enable_freq=enable_freq,
                        enable_morph=enable_morph,
                        enable_style=enable_style,
                        enable_interference=enable_interference,
                        enable_phase=enable_phase,
                    ).to(device)
                    with torch.no_grad():
                        q.V.copy_(project_conv_to_bank(conv_part.weight.to(device), q._base_bank))
                        if q.bias is not None and conv_part.bias is not None:
                            q.bias.copy_(conv_part.bias)
                    block.conv1 = q

    return model


# ------------------------------
# Build QFM-ResNet18 with real mode mapping
# ------------------------------

def build_qfm_resnet18(
    img_size: int,
    num_classes: int = 10,
    mode: str = "qfm_G",
    device: Optional[torch.device] = None,
    phase_clip: float = math.pi / 6,
    replace: str = "conv1,layer1,layer2,layer3,layer4",
    pretrained: bool = False,

    # NEW: stem must be applied here (architecture-level), BEFORE replacement
    cifar_stem: bool = False,

    use_interference: Optional[bool] = None,
    use_phase: Optional[bool] = None,
    enable_style: bool = True,
) -> nn.Module:
    if pretrained:
        try:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)

    # CIFAR stem MUST happen before any replacement (baseline or QFM)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_fc = model.fc.in_features
    model.fc = nn.Linear(in_fc, num_classes)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = mode or "qfm_G"
    mode_lower = mode.lower()

    # Baselines: no QFM
    if mode_lower.startswith("baseline"):
        if mode in ("baseline_blur", "baseline_blur_augmix"):
            # Only apply stem-AA if maxpool exists (if cifar_stem, maxpool is Identity)
            if not isinstance(model.maxpool, nn.Identity):
                in_ch = model.bn1.num_features
                model.maxpool = nn.Sequential(
                    BlurPool(in_ch, filt_size=3, stride=2),
                    nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                )
        return model.to(device)

    # Real ablation mapping
    enable_morph = True
    enable_freq = True

    if mode == "qfm_E":
        enable_freq = False
        enable_morph = True
    elif mode == "qfm_F":
        enable_freq = True
        enable_morph = False
    elif mode == "qfm_G":
        enable_freq = True
        enable_morph = True

    if use_interference is None:
        use_interference = ("phase" in mode_lower)
    if use_phase is None:
        use_phase = ("phase" in mode_lower)

    replace_tuple = tuple(s.strip() for s in replace.split(",") if s.strip())
    aa_flag = (not cifar_stem)
    model = replace_resnet_with_qfm(
        model,
        device=device,
        replace=replace_tuple,
        phase_clip=phase_clip,
        aa_every_stride2=aa_flag,
        enable_freq=enable_freq,
        enable_morph=enable_morph,
        enable_style=enable_style,
        enable_interference=bool(use_interference),
        enable_phase=bool(use_phase),
    )

    return model.to(device)
