# run_cifar1.py
import os, json, argparse
import torch
from torch.utils.data import DataLoader

from train_cifar import CIFARWithEnv
from qfm_dct import build_qfm_resnet18


def strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return ckpt, missing, unexpected

@torch.no_grad()
def eval_env(model, data_root, img, batch, env, device, num_workers=0):
    ds = CIFARWithEnv(data_root, train=False, img_size=img, env=env)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--img", type=int, default=32)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--replace", type=str, default="conv1,layer1,layer2")
    p.add_argument("--phase_clip", type=float, default=3.14159/6.0)
    p.add_argument("--use_interference", type=lambda s: str(s).lower() in ("1","true","yes"), default=None)
    p.add_argument("--use_phase", type=lambda s: str(s).lower() in ("1","true","yes"), default=None)
    p.add_argument("--disable_style", action="store_true")
    p.add_argument("--cifar_stem", action="store_true")
    p.add_argument("--out", type=str, default="eval_results.json")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_qfm_resnet18(
        img_size=args.img,
        num_classes=10,
        mode=args.mode,
        device=device,
        phase_clip=args.phase_clip,
        replace=args.replace,
        pretrained=False,
        use_interference=args.use_interference,
        use_phase=args.use_phase,
        enable_style=not args.disable_style,
    ).to(device)

    if args.cifar_stem:
        # mirror train_cifar behavior
        from train_cifar import apply_cifar_stem
        apply_cifar_stem(model)

    ckpt, missing, unexpected = load_ckpt(model, args.ckpt, device)
    if missing:
        print("[WARN] missing keys:", len(missing))
    if unexpected:
        print("[WARN] unexpected keys:", len(unexpected))

    envs = {
        "clean": ["clean"],
        "gauss": ["gauss_0.05", "gauss_0.1", "gauss_0.2"],
        "jpeg":  ["jpeg_80", "jpeg_60", "jpeg_40", "jpeg_20"],
        "blur":  ["blur_1", "blur_3", "blur_5"],
        "lowlight": ["lowlight_0.8", "lowlight_0.6", "lowlight_0.4"],
    }

    metrics = {}
    for group, lst in envs.items():
        metrics[group] = {}
        for e in lst:
            acc = eval_env(model, args.data, args.img, args.batch, e, device, num_workers=args.num_workers)
            metrics[group][e] = float(acc)
            print(f"{group:8s} {e:12s} -> {acc:.2f}")

    out = {
        "ckpt": args.ckpt,
        "epoch": ckpt.get("epoch", None) if isinstance(ckpt, dict) else None,
        "acc_clean_logged": ckpt.get("acc_clean", None) if isinstance(ckpt, dict) else None,
        "mode": args.mode,
        "img": args.img,
        "metrics": metrics
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", args.out)

if __name__ == "__main__":
    main()
