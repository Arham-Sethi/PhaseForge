#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a style severity dataset from CIFAR-10 images.

- Saves multiple corruptions per image with *monotonic* severities in [0,1].
- CSV columns: filepath, env, param, severity
  env   ∈ {clean, gauss, jpeg, blur, lowlight}
  param = numeric corruption parameter (e.g., 0.1, 40, 5, 0.6)
"""

import os
import io
import csv
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import datasets

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)

def jpeg_compress(pil_img, quality):
    """Apply JPEG compression, preserving artifacts even if saved as PNG."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def apply_gaussian_noise(pil_img, sigma):
    """Add zero-mean Gaussian noise with std = sigma*255 (per channel)."""
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0.0, float(sigma) * 255.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_blur(pil_img, radius):
    """Apply Gaussian blur with given PIL radius."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=float(radius)))

def apply_lowlight_scale(pil_img, scale):
    """Linear dimming by a scale factor in [0,1]."""
    scale = float(scale)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.clip(arr * scale, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    set_seed(1337)

    # Output directories
    output_dir = 'style_images'
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10 train split
    print("Loading CIFAR-10...")
    cifar = datasets.CIFAR10(root='./data', train=True, download=True)

    # Sample 1000 random images
    num_samples = 1000
    indices = random.sample(range(len(cifar)), num_samples)

    # Corruption specs (monotonic severities)
    corruptions = []

    # Clean
    corruptions.append(('clean', lambda img, _: img, None, 0.0))  # env, func, param, severity

    # Gaussian noise: σ ∈ {0.0, 0.1, 0.3} → severity {0, 0.5, 1}
    gauss_map = {0.0: 0.0, 0.1: 0.5, 0.3: 1.0}
    for sigma, sev in gauss_map.items():
        if sigma == 0.0:
            continue  # already have clean
        name = f'gauss_{sigma:.1f}'
        corruptions.append((name, apply_gaussian_noise, sigma, sev))

    # JPEG: q ∈ {100, 80, 40, 20} → severity = (100 - q) / 80
    for q in (100, 80, 40, 20):
        sev = (100 - q) / 80.0
        name = f'jpeg_{q}'
        corruptions.append((name, jpeg_compress, q, sev))

    # Blur: remove radius 0 (clean), keep {3:0.5, 5:1.0}
    blur_map = {3: 0.5, 5: 1.0}
    for radius, sev in blur_map.items():
        name = f'blur_{radius}'
        corruptions.append((name, apply_blur, radius, sev))

    # Low-light (linear dimming): scale ∈ {1.0, 0.8, 0.6, 0.4}
    # severity = (1 - scale) / (1 - 0.4)  → {0, 0.33, 0.67, 1}
    for scale in (1.0, 0.8, 0.6, 0.4):
        sev = (1.0 - scale) / (1.0 - 0.4)
        name = f'lowlight_{scale:.1f}'
        corruptions.append((name, apply_lowlight_scale, scale, sev))

    # Prepare CSV rows
    csv_rows = []
    img_idx = 0
    processed = 0
    total = len(indices) * len(corruptions)

    print(f"Processing {len(indices)} images with {len(corruptions)} corruptions each...")
    for orig_idx in indices:
        img, _ = cifar[orig_idx]
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        for corr_name, corr_func, corr_param, severity in corruptions:
            # Apply corruption
            corrupted = corr_func(img, corr_param) if corr_param is not None else corr_func(img, None)

            # Save as PNG (JPEG artifacts persist from the in-memory JPEG step)
            filename = f'img_{img_idx:05d}_{corr_name}.png'
            filepath = os.path.join(output_dir, filename)
            corrupted.save(filepath)

            # Parse env and param for CSV
            env = corr_name.split('_')[0]  # clean, gauss, jpeg, blur, lowlight
            param = 0.0 if corr_param is None else float(corr_param)

            csv_rows.append({
                'filepath': filepath,
                'env': env,
                'param': param,
                'severity': float(severity)
            })

            img_idx += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{total} images...")

    # Write CSV
    csv_path = 'style.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'env', 'param', 'severity'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone! Created {len(csv_rows)} images in '{output_dir}/'")
    print(f"CSV saved to '{csv_path}'")

if __name__ == '__main__':
    main()
