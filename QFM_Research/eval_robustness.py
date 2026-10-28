import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import train_cifar  # must be importable from your code/ folder


# -------------------------
# Eval helpers
# -------------------------
@torch.no_grad()
def eval_one_env(model: torch.nn.Module, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        # Common patterns: (x,y) or dict
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise RuntimeError(f"Unknown batch format: {type(batch)}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / max(1, total)


def load_model_from_run(run_dir: str, device: str) -> torch.nn.Module:
    """
    Expects:
      run_dir/config.json (saved by your trainer)
      run_dir/best.pth (or last.pth)
    """
    cfg_path = os.path.join(run_dir, "config.json")
    ckpt_best = os.path.join(run_dir, "best.pth")
    ckpt_last = os.path.join(run_dir, "last.pth")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    # Build model using your existing builder
    model = train_cifar.build_qfm_resnet18(
        mode=cfg["mode"],
        replace=cfg["replace"],
        img_size=int(cfg.get("img", 32)),
        cifar_stem=bool(cfg.get("cifar_stem", True)),
    ).to(device)

    ckpt_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_last
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint best.pth/last.pth in: {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("[LOAD]", os.path.basename(run_dir), "mode=", cfg.get("mode"), "replace=", cfg.get("replace"), "ckpt=", os.path.basename(ckpt_path))
    print("[LOAD] ckpt keys:", list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt))

    # Handle common checkpoint formats
    state = None
    for k in ["model", "state_dict", "net", "model_state", "model_state_dict"]:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    if state is None and isinstance(ckpt, dict):
        # maybe the dict is already a state_dict
        state = ckpt
    if state is None:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt)} keys={list(ckpt.keys()) if isinstance(ckpt, dict) else None}")

    # Strip "module." if needed
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): e.g. {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): e.g. {unexpected[:5]}")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)  # /content/drive/MyDrive/QFM_PROJECT
    ap.add_argument("--runs_root", type=str, required=True)     # /content/drive/MyDrive/QFM_PROJECT/RUNS
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--envs", type=str, default="clean,jpeg_40")
    ap.add_argument("--run_names", type=str, default="baseline,qfm_E,qfm_F,qfm_G")
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--qfm_E", type=str, required=True)
    ap.add_argument("--qfm_F", type=str, required=True)
    ap.add_argument("--qfm_G", type=str, required=True)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print("device:", device)

    # Map names -> run folder
    run_map = {
        "baseline": os.path.join(args.runs_root, args.baseline),
        "qfm_E": os.path.join(args.runs_root, args.qfm_E),
        "qfm_F": os.path.join(args.runs_root, args.qfm_F),
        "qfm_G": os.path.join(args.runs_root, args.qfm_G),
    }

    env_list = [e.strip() for e in args.envs.split(",") if e.strip()]
    order = [n.strip() for n in args.run_names.split(",") if n.strip()]

    results: Dict[str, Dict[str, float]] = {name: {} for name in order}

    for env in env_list:
        print(f"\n=== ENV: {env} ===")
        loader, _ = train_cifar.build_eval_loader(
            project_root=args.project_root,
            env=env,
            batch=args.batch,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        for name in order:
            run_dir = run_map[name]
            model = load_model_from_run(run_dir, device=device)
            acc = eval_one_env(model, loader, device=device)
            results[name][env] = acc
            print(f"{name:8s} | {env:10s} | acc={acc:.2f}")

    # Print a compact table
    print("\n=== SUMMARY TABLE (acc %) ===")
    header = "model".ljust(10) + "".join([env.rjust(12) for env in env_list])
    print(header)
    for name in order:
        row = name.ljust(10)
        for env in env_list:
            row += f"{results[name].get(env, float('nan')):12.2f}"
        print(row)


if __name__ == "__main__":
    main()
