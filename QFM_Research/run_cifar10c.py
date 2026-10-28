#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10-C robustness evaluation for QFM / baseline ResNet18.

- Loads a model with build_qfm_resnet18() from qfm_dct.py
- Loads a checkpoint (.pt / .pth)
- Evaluates accuracy on all CIFAR-10-C corruptions (15 types × 5 severities)
- Prints a small table + saves JSON with per-corruption accuracies.

Usage example (baseline):

  python run_cifar10c.py \
      --ckpt ckpts/baseline_clean25.pt \
      --mode baseline_blur_augmix \
      --name baseline \
      --data_root ./CIFAR-10-C \
      --batch 256 \
      --gpu 0

Make sure CIFAR-10-C has been downloaded and extracted so that
data_root contains files like "gaussian_noise.npy" and "labels.npy".
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from qfm_dct import build_qfm_resnet18


# ------------------------------
# CIFAR-10-C dataset
# ------------------------------

CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


class CIFAR10C(Dataset):
    """
    CIFAR-10-C corruption dataset.

    Each corruption file is shape [50000, 32, 32, 3].
    5 severities, each 10k images. We pick a single severity block.
    """

    def __init__(self, root, corruption, severity, transform=None):
        assert corruption in CIFAR10C_CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, "severity must be in [1..5]"

        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        images_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Could not find CIFAR-10-C files at {root}. "
                f"Expected {images_path} and {labels_path}."
            )

        data = np.load(images_path)  # [50000, 32, 32, 3], uint8
        labels = np.load(labels_path)  # [50000], int64

        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = data[start:end]
        self.labels = labels[start:end]

        assert self.data.shape[0] == 10000, "Unexpected CIFAR-10-C shape"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]  # HWC, uint8
        target = int(self.labels[idx])

        # convert to tensor CHW in [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ------------------------------
# Helper: build model + load checkpoint
# ------------------------------

def build_model(args, device):
    """
    Build model using the same mode you used in training.
    'mode' must match the config's mode (e.g. "baseline", "baseline_blur_augmix",
    "qfm_G", "qfm_full", etc.).
    """
    model = build_qfm_resnet18(
        img_size=32,
        num_classes=10,
        mode=args.mode,
        device=device,
        phase_clip=args.phase_clip,
        replace=args.replace,
        pretrained=False,
    )
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # assume this is already a state_dict
            state_dict = ckpt
    else:
        # very unlikely, but be defensive
        state_dict = ckpt

    # strip "module." if coming from DataParallel / DDP
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] Missing keys in state_dict:", missing)
    if unexpected:
        print("[WARN] Unexpected keys in state_dict:", unexpected)

    return model


# ------------------------------
# Evaluation
# ------------------------------

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    acc = 100.0 * correct / max(total, 1)
    return acc


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10-C robustness eval")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt / .pth)")
    parser.add_argument("--mode", type=str, required=True,
                        help="Model mode (must match training config 'mode'). "
                             "E.g. 'baseline', 'baseline_blur_augmix', 'qfm_G', 'qfm_full'.")
    parser.add_argument("--name", type=str, default="model",
                        help="Name label for logs / output JSON")
    parser.add_argument("--data_root", type=str, default="./CIFAR-10-C",
                        help="Path to CIFAR-10-C folder")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id; use -1 for CPU")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--phase_clip", type=float, default=3.14159 / 6.0,
                        help="phase_clip passed to build_qfm_resnet18")
    parser.add_argument("--replace", type=str,
                        default="conv1,layer1,layer2,layer3,layer4",
                        help="Which layers QFM replaced in training (same as config).")
    parser.add_argument("--output_dir", type=str, default="cifar10c_results",
                        help="Where to write JSON summary")

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("==> Device:", device)
    print("==> Loading model in mode:", args.mode)
    model = build_model(args, device)
    model = load_checkpoint(model, args.ckpt, device)
    model.to(device)

    # CIFAR-10 normalization (standard)
    normalize = T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    transform = T.Compose([
        # input already in [0,1] tensor, so just normalize
        normalize,
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    t0 = time.time()
    print("==> Evaluating on CIFAR-10-C from:", args.data_root)
    print("Corruption        Sev1   Sev2   Sev3   Sev4   Sev5   Mean")

    overall_accs = []

    for corr in CIFAR10C_CORRUPTIONS:
        sev_accs = []
        for sev in range(1, 6):
            ds = CIFAR10C(args.data_root, corr, sev, transform=transform)
            loader = DataLoader(
                ds,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            acc = eval_loader(model, loader, device)
            sev_accs.append(acc)

        mean_acc = float(np.mean(sev_accs))
        overall_accs.append(mean_acc)
        results[corr] = {
            f"severity_{i+1}": float(sev_accs[i]) for i in range(5)
        }
        results[corr]["mean_acc"] = mean_acc

        print(
            f"{corr:<16} "
            + " ".join(f"{a:5.1f}" for a in sev_accs)
            + f"  {mean_acc:6.2f}"
        )

    overall_mean = float(np.mean(overall_accs))
    results_summary = {
        "name": args.name,
        "ckpt": args.ckpt,
        "mode": args.mode,
        "overall_mean_acc": overall_mean,
        "per_corruption": results,
    }

    out_path = os.path.join(args.output_dir, f"{args.name}_cifar10c.json")
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    dt = time.time() - t0
    print("=" * 60)
    print(f"Overall CIFAR-10-C mean accuracy: {overall_mean:.2f}%")
    print(f"Saved JSON results to: {out_path}")
    print(f"Total eval time: {dt/60.0:.1f} min")


if __name__ == "__main__":
    main()
