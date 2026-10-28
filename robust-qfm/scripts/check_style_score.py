#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_style_score.py — fixed + stabilized version

Fully rewritten with:
  - Correct per-image JPEG proxy (4x4 grid)
  - FFT blur proxy (HF/LF ratio)
  - HF-drop proxy
  - Clean-relative baseline normalization
  - Winsorized z-scores to remove insane outliers
  - Stable manual n(x)
  - Positive regression learned n(x)
"""

import csv
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
EQUALIZE_SCALES = True   # gentle equalization after clean-relative baseline

# --------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


# --------------------------------------------------------
# FFT blur proxy
# --------------------------------------------------------
@torch.no_grad()
def fft_blur_proxy_batch(gray_batch: torch.Tensor, eps: float = 1e-6):
    """
    gray_batch: [B,1,H,W]

    Returns:
        blur_fft: numpy array [B], larger = blurrier
    """
    F = fft.fft2(gray_batch, norm='ortho')
    Fshift = fft.fftshift(F, dim=(-2, -1))

    mag = (Fshift.real**2 + Fshift.imag**2).sqrt()

    B, _, H, W = mag.shape
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=mag.device),
        torch.linspace(-1, 1, W, device=mag.device),
        indexing='ij'
    )
    R = (X*X + Y*Y).sqrt()[None, None, :, :]

    lf_mask = (R < 0.15).float()
    hf_mask = (R > 0.50).float()

    lf_energy = (mag * lf_mask).sum(dim=(-2,-1)) + eps
    hf_energy = (mag * hf_mask).sum(dim=(-2,-1)) + eps

    sharpness = hf_energy / lf_energy
    blur_fft = 1.0 / (sharpness + eps)

    return blur_fft.squeeze(1).detach().cpu().numpy()


# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------
def load_dataset(csv_path: str):
    tfm = transforms.ToTensor()

    filepaths, envs, params, severities, base_ids, tensors = [], [], [], [], [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        required = {'filepath', 'env', 'param', 'severity', 'base_id'}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            fp = row['filepath']
            ev = row['env']
            pr = float(row['param'])
            sv = float(row['severity'])
            bid = int(row['base_id'])

            img = Image.open(fp).convert('RGB')
            tensors.append(tfm(img))
            filepaths.append(fp)
            envs.append(ev)
            params.append(pr)
            severities.append(sv)
            base_ids.append(bid)

    return (
        filepaths,
        np.array(envs, dtype=object),
        np.array(params, dtype=np.float32),
        np.array(severities, dtype=np.float32),
        np.array(base_ids, dtype=np.int64),
        tensors
    )


# --------------------------------------------------------
# PROXIES
# --------------------------------------------------------
@torch.no_grad()
def compute_batch_proxies(batch: torch.Tensor, eps: float = 1e-6):
    device = batch.device
    w = LUMA_WEIGHTS.to(device).view(1,3,1,1)
    y = (batch * w).sum(dim=1, keepdim=True)  # [B,1,H,W]

    B,_,H,W = y.shape

    # FFT blur proxy
    blur_fft = fft_blur_proxy_batch(y)

    # Lowpass + highpass
    box = torch.ones((1,1,3,3), device=device) / 9.0
    y_lp = F.conv2d(y, box, padding=1)
    y_hp = y - y_lp

    # Noise proxy = MAD of high-pass
    hp_flat = y_hp.view(B, -1)
    med = hp_flat.median(dim=1, keepdim=True).values
    mad = (hp_flat - med).abs().median(dim=1).values + eps
    noise_mad = mad

    # JPEG blockiness proxy — **correct per-image version**
    diff_h = (y_lp[:,:,:,1:] - y_lp[:,:,:,:-1]).abs()
    diff_v = (y_lp[:,:,1:,:] - y_lp[:,:,:-1,:]).abs()

    col_idx = torch.arange(W-1, device=device)
    row_idx = torch.arange(H-1, device=device)

    boundary_cols = ((col_idx+1) % 4 == 0).float().view(1,1,1,-1)
    interior_cols = 1.0 - boundary_cols

    boundary_rows = ((row_idx+1) % 4 == 0).float().view(1,1,-1,1)
    interior_rows = 1.0 - boundary_rows

    b_h = (diff_h * boundary_cols).sum(dim=(-2,-1)) / (boundary_cols.sum() + eps)
    i_h = (diff_h * interior_cols).sum(dim=(-2,-1)) / (interior_cols.sum() + eps)

    b_v = (diff_v * boundary_rows).sum(dim=(-2,-1)) / (boundary_rows.sum() + eps)
    i_v = (diff_v * interior_rows).sum(dim=(-2,-1)) / (interior_rows.sum() + eps)

    boundary = b_h + b_v
    interior = i_h + i_v
    jpeg_ratio = (boundary - interior) / (interior + eps)  # shape [B,1]

    # Low-light proxy
    lowlight = 1.0 - y.mean(dim=(-2,-1)).squeeze(1)

    # HF-drop
    hf_energy = y_hp.pow(2).mean(dim=(-2,-1)).squeeze(1)
    hf_drop = 1.0 / (hf_energy + eps)

    return {
        'noise': noise_mad.cpu().numpy(),
        'jpeg': jpeg_ratio.squeeze(1).cpu().numpy(),
        'lowlight': lowlight.cpu().numpy(),
        'hf_drop': hf_drop.cpu().numpy(),
        'blur_fft': blur_fft,
    }


# --------------------------------------------------------
# CLEAN-RELATIVE NORMALIZATION
# --------------------------------------------------------
def apply_clean_baseline(proxies: dict, base_ids: np.ndarray, envs: np.ndarray, eps=1e-6):

    adjusted = {k: v.copy() for k,v in proxies.items()}

    clean_values = {k:{} for k in proxies}
    for i,(bid,ev) in enumerate(zip(base_ids, envs)):
        if ev == 'clean':
            for k in proxies:
                clean_values[k][bid] = proxies[k][i]

    for i,bid in enumerate(base_ids):
        for k in proxies:
            if bid in clean_values[k]:
                c = clean_values[k][bid]
                adjusted[k][i] = (proxies[k][i] - c) / (abs(c) + eps)

    return adjusted


# --------------------------------------------------------
# ROBUST Z + CLAMPING
# --------------------------------------------------------
def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad), med, mad

def clamp_z(z, cap=5.0):
    return np.clip(z, -cap, cap)


# --------------------------------------------------------
# STYLE SCORE
# --------------------------------------------------------
def compute_style_score(proxies_rel):

    # Optional global equalization
    if EQUALIZE_SCALES:
        eq = {}
        for k,v in proxies_rel.items():
            eq[k] = (v - v.mean()) / (v.std() + 1e-6)
        proxies_rel = eq

    # Robust z
    z_noise, n_med, n_mad = robust_z(proxies_rel['noise'])
    z_jpeg,  j_med, j_mad = robust_z(proxies_rel['jpeg'])
    z_blur,  b_med, b_mad = robust_z(proxies_rel['blur_fft'])
    z_low,   l_med, l_mad = robust_z(proxies_rel['lowlight'])
    z_hf,    h_med, h_mad = robust_z(proxies_rel['hf_drop'])

    # Clamp extreme z-scores
    z_noise = clamp_z(z_noise)
    z_jpeg  = clamp_z(z_jpeg)
    z_blur  = clamp_z(z_blur)
    z_low   = clamp_z(z_low)
    z_hf    = clamp_z(z_hf)

    # Positive part
    z_noise_pos = np.maximum(z_noise, 0)
    z_jpeg_pos  = np.maximum(z_jpeg, 0)
    z_blur_pos  = np.maximum(z_blur, 0)
    z_low_pos   = np.maximum(z_low, 0)

    # Manual n(x)
    n_manual = (
        0.15 * z_noise_pos +
        0.25 * z_jpeg_pos  +
        0.45 * z_blur_pos  +
        0.15 * z_low_pos
    )

    # Scale to [0,1]
    p10, p90 = np.percentile(n_manual, [10,90])
    n_scaled = np.clip((n_manual - p10) / (p90 - p10 + 1e-6), 0, 1)

    stats = {
        'noise': {'median':n_med, 'mad':n_mad},
        'jpeg':  {'median':j_med, 'mad':j_mad},
        'blur_fft': {'median':b_med, 'mad':b_mad},
        'lowlight': {'median':l_med, 'mad':l_mad},
        'hf_drop': {'median':h_med, 'mad':h_mad},
        'p10': p10,
        'p90': p90
    }

    z_dict = {
        'noise': z_noise, 'jpeg': z_jpeg,
        'blur_fft': z_blur, 'lowlight': z_low,
        'hf_drop': z_hf
    }
    z_pos_dict = {
        'noise': z_noise_pos, 'jpeg': z_jpeg_pos,
        'blur_fft': z_blur_pos, 'lowlight': z_low_pos,
        'hf_drop': np.maximum(z_hf,0)
    }

    return n_manual, n_scaled, stats, z_dict, z_pos_dict


# --------------------------------------------------------
# PER-FAMILY CORRELATIONS
# --------------------------------------------------------
def per_family_correlations(envs, severities, n_vals):
    fam_idxs = defaultdict(list)
    for i,ev in enumerate(envs):
        fam = ev.split('_')[0]
        fam_idxs[fam].append(i)

    out = {}
    for fam,idxs in sorted(fam_idxs.items()):
        sev = severities[idxs]
        if sev.std() < 1e-8:
            continue
        n_f = n_vals[idxs]
        r,_ = pearsonr(sev, n_f)
        out[fam] = {'pearson':r, 'count':len(idxs)}
    return out


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    csv_path = "style.csv"
    print(f"Reading {csv_path}...")
    (filepaths, envs, params, severities, base_ids, tensors) = load_dataset(csv_path)
    N = len(tensors)
    print(f"Loaded {N} images. Computing proxies...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 512

    proxy_buffers = defaultdict(list)

    for start in range(0, N, bs):
        batch = torch.stack(tensors[start:start+bs]).to(device)
        proxy_batch = compute_batch_proxies(batch)
        for k,v in proxy_batch.items():
            proxy_buffers[k].append(v)

        print(f"  Processed {min(start+bs, N)}/{N}...")

    proxies = {k: np.concatenate(v, axis=0) for k,v in proxy_buffers.items()}

    # Normalize via clean baseline
    proxies_rel = apply_clean_baseline(proxies, base_ids, envs)

    # Compute scores
    n_manual, n_scaled, stats, z_dict, z_pos_dict = compute_style_score(proxies_rel)

    perfam = per_family_correlations(envs, severities, n_manual)
    overall_r,_ = pearsonr(severities, n_manual)
    overall_s,_ = spearmanr(severities, n_manual)

    print("\nProxy statistics (median/MAD):")
    for k,v in stats.items():
        if isinstance(v,dict):
            print(f"  {k:12s} median={v['median']:.6f} mad={v['mad']:.6f}")
        else:
            print(f"  {k:12s}: {v}")

    print("\nPer-family Pearson r (manual n):")
    for fam,info in perfam.items():
        print(f"  {fam:12s} -> r={info['pearson']:+.4f} (n={info['count']})")

    print(f"\nOverall Pearson r (manual n):   {overall_r:+.4f}")
    print(f"Overall Spearman rho (manual n): {overall_s:+.4f}")

    print(f"\nScaled n(x) range: [{n_scaled.min():.4f}, {n_scaled.max():.4f}]")
    print(f"Manual n(x) range: [{n_manual.min():.4f}, {n_manual.max():.4f}]")

    # Learned regression (positive weights)
    Z = np.stack([
        z_pos_dict['noise'],
        z_pos_dict['jpeg'],
        z_pos_dict['blur_fft'],
        z_pos_dict['lowlight'],
        z_pos_dict['hf_drop']
    ], axis=1)

    reg = LinearRegression(positive=True)
    reg.fit(Z, severities)
    n_learned = reg.predict(Z)
    learned_weights = np.concatenate(([reg.intercept_], reg.coef_))

    perfam_learned = per_family_correlations(envs, severities, n_learned)
    overall_rL,_ = pearsonr(severities, n_learned)
    overall_sL,_ = spearmanr(severities, n_learned)

    print("\nLearned positive linear weights (intercept + 5 features):")
    print(" ", learned_weights)

    print("\nPer-family Pearson r (learned n):")
    for fam,info in perfam_learned.items():
        print(f"  {fam:12s} -> r={info['pearson']:+.4f} (n={info['count']})")

    print(f"\nOverall Pearson r (learned n):   {overall_rL:+.4f}")
    print(f"Overall Spearman rho (learned n): {overall_sL:+.4f}")

    # Save all
    np.savez(
        "style_score_stats.npz",
        filepaths=np.array(filepaths, dtype=object),
        envs=envs,
        params=params,
        severity=severities,
        base_id=base_ids,
        n_manual=n_manual,
        n_scaled=n_scaled,
        n_learned=n_learned,
        proxies_raw=proxies,
        proxies_rel=proxies_rel,
        learned_weights=learned_weights,
        p10=stats['p10'],
        p90=stats['p90'],
        z_noise=z_dict['noise'],
        z_jpeg=z_dict['jpeg'],
        z_blur=z_dict['blur_fft'],
        z_low=z_dict['lowlight'],
        z_hf=z_dict['hf_drop']
    )

    print("\nSaved stats to 'style_score_stats.npz'.")


if __name__ == "__main__":
    main()
