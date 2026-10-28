# train_cifar.py
import os, io, math, time, json, random, argparse, datetime, shutil
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from qfm_dct import build_qfm_resnet18, QFMConv2dDCT, SimpleMorphGate, style_score


# ------------------------------
# Seed / determinism
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
_DEVICE = torch.device("cuda" if _USE_CUDA else "cpu")

# Modern AMP
if _USE_CUDA:
    from torch.amp import autocast, GradScaler
else:
    class autocast:
        def __init__(self, device_type: str = "cpu", enabled: bool = False, **kwargs):
            self.enabled = False
        def __enter__(self): return None
        def __exit__(self, *a, **k): return False

# ------------------------------
# CIFAR-10 + envs
# ------------------------------
class CIFARWithEnv(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        img_size: int = 32,
        env: str = "clean",
        randaugment: bool = False,
        augmix: bool = False,
        label_smoothing: float = 0.0
    ):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True)
        self.env = env
        self.sz = img_size
        self.label_smoothing = label_smoothing

        base = [transforms.Resize((img_size, img_size))]
        if randaugment:
            base.append(transforms.RandAugment(num_ops=2, magnitude=9))
        base.append(transforms.ToTensor())
        base.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.247, 0.243, 0.261]))
        self.tf = transforms.Compose(base)
        self.augmix = augmix

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _jpeg_compress(pil_img: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _apply_env(self, pil_img: Image.Image) -> Image.Image:
        e = self.env
        if e == "clean":
            return pil_img
        if e.startswith("gauss"):
            sigma = float(e.split("_")[1])
            arr = np.array(pil_img).astype(np.float32)
            arr += np.random.normal(0, sigma * 255.0, size=arr.shape).astype(np.float32)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        if e.startswith("jpeg"):
            q = int(e.split("_")[1])
            return self._jpeg_compress(pil_img, q)
        if e.startswith("blur"):
            k = int(e.split("_")[1])
            return pil_img.filter(ImageFilter.GaussianBlur(radius=max(1, (k - 1) // 2)))
        if e.startswith("lowlight"):
            gamma = float(e.split("_")[1])
            arr = np.array(pil_img).astype(np.float32) / 255.0
            arr = np.power(arr, gamma)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return pil_img

    def _augmix(self, img: Image.Image, width: int = 3, depth: int = 2, alpha: float = 1.0):
        ops = [
            lambda im: im.rotate(random.choice([-10, -7, -5, 5, 7, 10])),
            lambda im: Image.fromarray(
                np.uint8(np.clip(np.array(im, dtype=np.float32) * (0.8 + 0.4 * random.random()), 0, 255))
            ),
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.choice([0, 1, 2]))),
            lambda im: Image.fromarray(
                np.uint8(np.clip(np.array(im, dtype=np.float32) + np.random.normal(0, 5.0, size=np.array(im).shape), 0, 255))
            ),
        ]
        ws = np.random.dirichlet([alpha] * width).astype(np.float32)
        m = float(np.random.beta(alpha, alpha))

        mix = np.zeros_like(np.array(img), dtype=np.float32)
        for i in range(width):
            im = img.copy()
            d = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = random.choice(ops)
                im = op(im)
            mix += ws[i] * np.array(im, dtype=np.float32)

        mixed = (1 - m) * np.array(img, dtype=np.float32) + m * mix
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
# EMA
# ------------------------------
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
# MixUp + label smoothing helpers
# ------------------------------
def one_hot(nclass: int, idx: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    y = torch.zeros(idx.size(0), nclass, device=idx.device, dtype=torch.float32)
    y.scatter_(1, idx.view(-1, 1), 1.0)
    if smoothing > 0:
        y = (1.0 - smoothing) * y + smoothing / nclass
    return y

def mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, torch.tensor(1.0, device=x.device)
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]
    y2 = y[perm]
    xm = lam * x + (1 - lam) * x2
    return xm, (y, y2, lam), torch.tensor(lam, device=x.device)

def mixup_criterion(crit, pred, targets):
    if not isinstance(targets, tuple):
        return crit(pred, targets)
    y1, y2, lam = targets
    return lam * crit(pred, y1) + (1 - lam) * crit(pred, y2)

def rex_penalty(per_env_losses: List[torch.Tensor]) -> torch.Tensor:
    losses = torch.stack(per_env_losses)
    return losses.var()


# ------------------------------
# Trainer
# ------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed, deterministic=True)
        self.device = _DEVICE
        self.num_classes = 10

        envs_train = ["clean"] if not args.envs_train else [e.strip() for e in args.envs_train.split(",") if e.strip()]
        self.train_sets = [
            CIFARWithEnv(args.data, True, img_size=args.img, env=e,
                         randaugment=args.randaugment, augmix=args.augmix,
                         label_smoothing=args.smooth)
            for e in envs_train
        ]

        n_envs = max(1, len(self.train_sets))
        per_env_batch = max(1, args.batch // n_envs)

        nw = int(args.num_workers)
        pin = bool(_USE_CUDA) and bool(args.pin_memory)

        self.train_loaders = [
            DataLoader(ds, batch_size=per_env_batch, shuffle=True,
                       num_workers=nw, pin_memory=pin,
                       persistent_workers=False, drop_last=True)
            for ds in self.train_sets
        ]

        self.val_clean = DataLoader(
            CIFARWithEnv(args.data, False, img_size=args.img, env="clean"),
            batch_size=args.batch, shuffle=False, num_workers=min(2, nw),
            pin_memory=pin, persistent_workers=False
        )
        self.val_probe = DataLoader(
            CIFARWithEnv(args.data, False, img_size=args.img, env=args.val_env),
            batch_size=args.batch, shuffle=False, num_workers=min(2, nw),
            pin_memory=pin, persistent_workers=False
        )

        # Model
        self.model = self._build_model().to(self.device)
        self.ema = EMA(self.model, decay=args.ema)

        self.smooth = float(args.smooth)
        self.scaler = GradScaler(
            enabled=bool(_USE_CUDA and args.amp),
            init_scale=2.0**10,
            growth_interval=2000
        )

        # Optim groups
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
            (qfm_params if id(p) in qfm_ids else base_params).append(p)
        if not base_params:
            base_params = [p for p in self.model.parameters() if p.requires_grad]

        self.opt = torch.optim.AdamW(
            [{"params": base_params, "lr": args.lr, "weight_decay": args.wd},
             {"params": qfm_params, "lr": args.lr * args.qfm_lr_scale, "weight_decay": args.wd}]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=args.epochs, eta_min=args.lr * 1e-2
        )

        self.run_dir = args.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.global_step = 0

        # schedules
        self.unfreeze_gates_epoch = int(args.unfreeze_gates_epoch)
        self.unfreeze_phase_epoch = int(args.unfreeze_phase_epoch)
        self.f_clip_warm = float(args.f_clip_warm)
        self.freq_warmup_epochs = int(args.freq_warmup_epochs)
        print("[CFG] f_clip_warm=", self.f_clip_warm, "freq_warmup_epochs=", self.freq_warmup_epochs)

        # REx
        self.rex_lambda_start = float(args.rex_lambda_start)
        self.rex_lambda_final = float(args.rex_lambda_final)
        self.rex_ramp_epochs = int(args.rex_ramp_epochs)

        self.mixup_alpha = float(args.mixup)
        self.clip = float(args.clip)
        self.phase_l2 = float(args.phase_l2)

        self.export_dir = args.export_dir if args.export_dir else None
        if self.export_dir:
            os.makedirs(self.export_dir, exist_ok=True)

        if args.resume:
            self._load_resume(args.resume)

    def _build_model(self):
        args = self.args
        # FIX: cifar_stem is applied INSIDE build_qfm_resnet18 BEFORE replacement.
        model = build_qfm_resnet18(
            img_size=args.img,
            num_classes=self.num_classes,
            mode=args.mode,
            device=self.device,
            phase_clip=args.phase_clip,
            replace=args.replace,
            pretrained=args.pretrained,
            use_interference=args.use_interference,
            use_phase=args.use_phase,
            enable_style=not args.disable_style,
            cifar_stem=args.cifar_stem,
        )
        print("[STEM] conv1 type:", type(model.conv1))
        print("[STEM] maxpool:", model.maxpool)
        try:
            print("[STEM] conv1 kH,kW,stride:", model.conv1.kH, model.conv1.kW, model.conv1.stride)
        except Exception:
            print("[STEM] conv1 kernel_size,stride:", getattr(model.conv1, "kernel_size", None), getattr(model.conv1, "stride", None))

        return model

    def _tau_schedule(self, epoch: int):
        tau = 5.0 + min(1.0, epoch / 10.0) * (15.0 - 5.0)
        for m in self.model.modules():
            if isinstance(m, SimpleMorphGate):
                m.set_tau(tau)

    def _rex_lambda(self, epoch: int) -> float:
        if self.rex_ramp_epochs <= 0:
            return self.rex_lambda_final
        if epoch <= 5:
            return self.rex_lambda_start * (epoch / 5.0)
        t = min(1.0, max(0.0, (epoch - 5) / float(self.rex_ramp_epochs)))
        return self.rex_lambda_start + t * (self.rex_lambda_final - self.rex_lambda_start)

    def _set_train_stage(self, epoch: int):
        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                for _, p in m.named_parameters():
                    p.requires_grad = True

                if epoch < self.unfreeze_gates_epoch:
                    m._t_param.requires_grad = False
                    m._lg_param.requires_grad = False
                    m._hg_param.requires_grad = False
                    m.band_steep.requires_grad = False

                if epoch < self.unfreeze_phase_epoch:
                    m.phase_enabled = False
                    m.use_interference = False
                    for _, p in m.phase_head.named_parameters():
                        p.requires_grad = False
                else:
                    m.phase_enabled = True
                    m.use_interference = True
                    for _, p in m.phase_head.named_parameters():
                        p.requires_grad = True

    def model_forward_qfm_aware(self, x, style, fclip, force_no_freq: bool):
        if not hasattr(self, "_printed_train_force_flag"):
            self._printed_train_force_flag = True
            print(f"[TRAIN_FORWARD] force_no_freq={force_no_freq}")

        has_qfm = any(isinstance(m, QFMConv2dDCT) for m in self.model.modules())
        if not has_qfm:
            return self.model(x)

        saved = []

        def make_forward(mm):
            def _f(xx):
                return QFMConv2dDCT.forward(mm, xx, style_score=style, f_clip=fclip, force_no_freq=force_no_freq)
            return _f

        for m in self.model.modules():
            if isinstance(m, QFMConv2dDCT):
                saved.append((m, m.forward))
                m.forward = make_forward(m)

        try:
            out = self.model(x)
        finally:
            for m, f in saved:
                m.forward = f

        return out

    def _forward_env_chunk(self, x: torch.Tensor, y: torch.Tensor, epoch: int):
        s = style_score(x)
        fclip = self.f_clip_warm if epoch < 10 else 1.0
        force_no_freq = (epoch <= self.freq_warmup_epochs)
        logits = self.model_forward_qfm_aware(x, s, fclip, force_no_freq)

        if self.smooth <= 0:
            loss = F.cross_entropy(logits, y)
        else:
            yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
            loss = F.kl_div(F.log_softmax(logits, dim=1), yoh, reduction="batchmean")
        return logits, loss

    def _save_ckpt(self, path: str, epoch: int, extra: dict):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if (_USE_CUDA and self.args.amp) else None,
            "epoch": epoch,
            "global_step": self.global_step,
        }
        ckpt.update(extra)
        torch.save(ckpt, path)

    def _export_copy(self, src_path: str, name: str):
        if not self.export_dir:
            return
        dst = os.path.join(self.export_dir, name)
        try:
            shutil.copy2(src_path, dst)
        except Exception:
            pass

    def _load_resume(self, resume_path: str):
        ckpt = torch.load(resume_path, map_location=self.device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(sd, strict=False)

        if isinstance(ckpt, dict) and "opt" in ckpt and ckpt["opt"] is not None:
            try: self.opt.load_state_dict(ckpt["opt"])
            except Exception: pass
        if isinstance(ckpt, dict) and "sched" in ckpt and ckpt["sched"] is not None:
            try: self.scheduler.load_state_dict(ckpt["sched"])
            except Exception: pass
        if isinstance(ckpt, dict) and "scaler" in ckpt and ckpt["scaler"] is not None and (_USE_CUDA and self.args.amp):
            try: self.scaler.load_state_dict(ckpt["scaler"])
            except Exception: pass

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        print(f"[RESUME] Loaded {resume_path} -> start_epoch={self.start_epoch} global_step={self.global_step}")

    def train(self):
        args = self.args
        best_clean = -1.0
        best_ret = -1.0

        start_epoch = getattr(self, "start_epoch", 1)

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            self.model.train()

            # ===== DEBUG PROBE (RUNS ONCE, NO-AUTOGRAD) =====
            if epoch == 1 and self.global_step == 0:
                x0, y0 = next(iter(self.train_loaders[0]))
                x0 = x0.to(self.device)
                y0 = torch.as_tensor(y0, device=self.device)

                qfm_mod = None
                for m in self.model.modules():
                    if isinstance(m, QFMConv2dDCT):
                        qfm_mod = m
                        break
                print("[DEBUG] enable_freq:", getattr(qfm_mod, "enable_freq", None))
                print("[DEBUG] enable_morph:", getattr(qfm_mod, "enable_morph", None))
                print("[DEBUG] enable_style:", getattr(qfm_mod, "enable_style", None))

                self.opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    logits0 = self.model_forward_qfm_aware(
                        x0[:16],
                        style_score(x0[:16]),
                        1.0,
                        force_no_freq=False
                    )
                    loss0 = F.cross_entropy(logits0, y0[:16])

                    pred = logits0.argmax(1)
                    print("[DEBUG] loss:", float(loss0.item()))
                    print("[DEBUG] logits mean/std:", float(logits0.mean()), float(logits0.std()))
                    print("[DEBUG] unique preds in batch:", int(pred.unique().numel()))

                    if qfm_mod is None:
                        print("[DEBUG] Baseline model detected (no QFMConv2dDCT).")
                    else:
                        # --- A) Morph changes output? ---
                        if not getattr(qfm_mod, "enable_morph", True):
                            print("[DEBUG] morph disabled → skipping morph-effect test")
                        else:
                            try:
                                A = qfm_mod.morph(x0[:4])
                                A_mid = torch.full_like(A, 0.5)

                                Wsp = qfm_mod._spatial_weight().to(dtype=x0.dtype, device=x0.device)
                                yS = F.conv2d(
                                    x0[:4], Wsp, bias=qfm_mod.bias,
                                    stride=qfm_mod.stride, padding=qfm_mod.padding
                                )

                                A_ds = F.interpolate(A, size=yS.shape[-2:], mode="bilinear", align_corners=False)
                                A_ds_mid = F.interpolate(A_mid, size=yS.shape[-2:], mode="bilinear", align_corners=False)

                                gS = torch.tanh(qfm_mod.morph_s) * (A_ds - 0.5) * 2.0
                                gS_mid = torch.tanh(qfm_mod.morph_s) * (A_ds_mid - 0.5) * 2.0

                                yA = yS * (1.0 + gS).clamp(0.5, 1.5)
                                yMid = yS * (1.0 + gS_mid).clamp(0.5, 1.5)
                                diff = (yA - yMid).abs().mean()
                                print("[DEBUG] morph local effect |mean abs diff|:", float(diff))
                            except Exception as e:
                                print("[DEBUG] morph local effect check failed:", str(e))

                        # --- B) Mask can suppress? ---
                        try:
                            # use yS_dbg which is always defined in E-block OR define it here safely
                            Wsp = qfm_mod._spatial_weight().to(dtype=x0.dtype, device=x0.device)
                            yS_dbg = F.conv2d(x0[:16], Wsp, bias=qfm_mod.bias,
                                            stride=qfm_mod.stride, padding=qfm_mod.padding)
                            YR = qfm_mod._build_dct_radius(yS_dbg.shape[-2], yS_dbg.shape[-1], yS_dbg.device)
                            print("[DEBUG] radius r min/mean/max:", float(YR.min()), float(YR.mean()), float(YR.max()))
                        except Exception as e:
                            print("[DEBUG] R stats check failed:", str(e))


                        # --- C) A variance ---
                        if not getattr(qfm_mod, "enable_morph", True):
                            print("[DEBUG] morph disabled → skipping A-variance test")
                        else:
                            try:
                                A = qfm_mod.morph(x0[:4])
                                A_ds = F.interpolate(A, size=(16, 16), mode="bilinear", align_corners=False)
                                var_sp = A_ds.var(dim=(-1, -2)).mean()
                                print("[DEBUG] A spatial variance (should be >0):", float(var_sp))
                            except Exception as e:
                                print("[DEBUG] A variance check failed:", str(e))


                        # --- D) Why is R ~ 1? show params + implied gains ---
                        try:
                            print("[DEBUG] _t_param/_lg_param/_hg_param:",
                                float(qfm_mod._t_param), float(qfm_mod._lg_param), float(qfm_mod._hg_param))
                            t = qfm_mod._squash(qfm_mod._t_param, 0.20, 0.55)
                            lg = qfm_mod._squash(qfm_mod._lg_param, 0.8, 1.2)
                            hg = qfm_mod._squash(qfm_mod._hg_param, 0.8, 1.2)
                            print("[DEBUG] t/low_gain/high_gain:", float(t), float(lg), float(hg))
                        except Exception as e:
                            print("[DEBUG] gain param check failed:", str(e))

                        # --- E) spatial scale sanity ---
                        try:
                            Wsp = qfm_mod._spatial_weight().to(dtype=x0.dtype, device=x0.device)
                            print("[DEBUG] Wsp |mean| / |max|:", float(Wsp.abs().mean()), float(Wsp.abs().max()))
                            yS_dbg = F.conv2d(
                                x0[:16], Wsp, bias=qfm_mod.bias,
                                stride=qfm_mod.stride, padding=qfm_mod.padding
                            )
                            print("[DEBUG] yS mean/std:", float(yS_dbg.mean()), float(yS_dbg.std()))
                        except Exception as e:
                            print("[DEBUG] yS scale check failed:", str(e))
            # ===== END DEBUG PROBE =====


            self._set_train_stage(epoch)
            self._tau_schedule(epoch)
            rex_lambda = self._rex_lambda(epoch)

            train_iters = [iter(dl) for dl in self.train_loaders]
            iters_per_env = min(len(dl) for dl in self.train_loaders)

            for _step in range(iters_per_env):
                per_env_losses = []
                self.opt.zero_grad(set_to_none=True)

                # ---- fetch per-env batches, compute per-env losses ----
                for li, it in enumerate(train_iters):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        train_iters[li] = iter(self.train_loaders[li])
                        x, y = next(train_iters[li])

                    x = x.to(self.device, non_blocking=True)
                    y = torch.as_tensor(y, device=self.device)

                    if self.mixup_alpha > 0:
                        yoh = one_hot(self.num_classes, y, smoothing=self.smooth)
                        x_mix, mix_tgt, _ = mixup(x, yoh, alpha=self.mixup_alpha)

                        with autocast(device_type="cuda", enabled=bool(_USE_CUDA and args.amp)):
                            logits = self.model_forward_qfm_aware(
                                x_mix,
                                style_score(x_mix),
                                self.f_clip_warm if epoch < 10 else 1.0,
                                force_no_freq=(epoch <= self.freq_warmup_epochs)
                            )
                            loss_e = mixup_criterion(
                                lambda p, t: F.kl_div(F.log_softmax(p, dim=1), t, reduction="batchmean"),
                                logits,
                                mix_tgt
                            )
                    else:
                        with autocast(device_type="cuda", enabled=bool(_USE_CUDA and args.amp)):
                            _logits, loss_e = self._forward_env_chunk(x, y, epoch)

                    per_env_losses.append(loss_e)

                # ---- combine losses (task + rex + optional phase l2) ----
                with autocast(device_type="cuda", enabled=bool(_USE_CUDA and args.amp)):
                    Ltask = torch.stack(per_env_losses).mean()
                    Lrex = rex_penalty(per_env_losses) if len(per_env_losses) > 1 else torch.tensor(0.0, device=self.device)
                    loss = Ltask + rex_lambda * Lrex

                    for mm in self.model.modules():
                        if isinstance(mm, QFMConv2dDCT) and getattr(mm, "phase_enabled", False):
                            for _, p in mm.phase_head.named_parameters():
                                loss = loss + self.phase_l2 * (p ** 2).mean()

                # ---- STEP2 delta check: capture weights BEFORE step (only once) ----
                do_param_delta_check = (epoch == 1 and _step == 0 and self.global_step == 0)
                fc_w_before = None
                qfmV_before = None
                first_qfm = None

                if do_param_delta_check:
                    with torch.no_grad():
                        fc_w_before = self.model.fc.weight.detach().float().clone()
                        for mm in self.model.modules():
                            if isinstance(mm, QFMConv2dDCT):
                                first_qfm = mm
                                qfmV_before = mm.V.detach().float().clone()
                                break

                # ===============================
                # BACKWARD + STEP (single path)
                # ===============================
                use_amp = bool(_USE_CUDA and args.amp)

                if use_amp:
                    self.scaler.scale(loss).backward()

                    # Always unscale before stepping manually
                    self.scaler.unscale_(self.opt)
                    if do_param_delta_check:
                        bad_grad = None
                        for name, p in self.model.named_parameters():
                            if p.grad is None:
                                continue
                            if not torch.isfinite(p.grad).all():
                                bad_grad = name
                                break
                        print("[AMP] first_nonfinite_grad:", bad_grad)

                        bad_param = None
                        for name, p in self.model.named_parameters():
                            if not torch.isfinite(p.data).all():
                                bad_param = name
                                break
                        print("[AMP] first_nonfinite_param_pre_step:", bad_param)


                    if self.clip and self.clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)

                    # Manual step: bypass GradScaler.step() skip logic
                    self.opt.step()
                    self.scaler.update()

                else:
                    loss.backward()
                    if self.clip and self.clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    self.opt.step()

                # ---- STEP2 delta print AFTER step (only once) ----
                if do_param_delta_check:
                    with torch.no_grad():
                        fc_delta = (self.model.fc.weight.detach().float() - fc_w_before).abs().mean().item() if fc_w_before is not None else float("nan")
                        if (first_qfm is not None) and (qfmV_before is not None):
                            qfmV_delta = (first_qfm.V.detach().float() - qfmV_before).abs().mean().item()
                        else:
                            qfmV_delta = float("nan")

                        # try to detect scaler skip (found_inf) if available
                        found_inf = None
                        if use_amp:
                            try:
                                st = self.scaler._per_optimizer_states[id(self.opt)]
                                found_inf = float(next(iter(st["found_inf_per_device"].values())).item())
                            except Exception:
                                found_inf = None

                        try:
                            scale = float(self.scaler.get_scale()) if use_amp else -1.0
                        except Exception:
                            scale = -1.0

                        print(
                            "[STEP2] loss:", float(loss.item()),
                            "use_amp:", use_amp,
                            "scaler_scale:", scale,
                            "found_inf:", found_inf,
                            "Δfc(mean|abs|):", fc_delta,
                            "ΔV(mean|abs|):", qfmV_delta
                        )

                self.ema.update(self.model)

                if self.global_step % args.log_every == 0:
                    self.writer.add_scalar("train/loss", float(loss.item()), self.global_step)
                    self.writer.add_scalar("train/rex_lambda", float(rex_lambda), self.global_step)
                    self.writer.add_scalar("train/lr", float(self.opt.param_groups[0]["lr"]), self.global_step)

                self.global_step += 1


            self.scheduler.step()

            self.ema.apply_shadow(self.model)
            acc_clean = self.evaluate(self.val_clean, tag="clean", epoch=epoch)
            acc_probe = self.evaluate(self.val_probe, tag=f"probe_{args.val_env}", epoch=epoch)
            self.ema.restore(self.model)

            retention = (acc_probe / max(1e-6, acc_clean)) if acc_clean > 0 else 0.0
            self.writer.add_scalar("eval/clean_acc", acc_clean, epoch)
            self.writer.add_scalar("eval/probe_acc", acc_probe, epoch)
            self.writer.add_scalar("eval/retention", retention, epoch)

            last_path = os.path.join(self.run_dir, "last.pth")
            self._save_ckpt(last_path, epoch, {"acc_clean": acc_clean, "acc_probe": acc_probe, "retention": retention})
            self._export_copy(last_path, "last.pth")

            if acc_clean > best_clean:
                best_clean = acc_clean
                best_path = os.path.join(self.run_dir, "best.pth")
                self._save_ckpt(best_path, epoch, {"acc_clean": acc_clean})
                self._export_copy(best_path, "best.pth")

            if retention > best_ret:
                best_ret = retention
                ret_path = os.path.join(self.run_dir, f"{args.mode}_seed{args.seed}_retention_best.pth")
                self._save_ckpt(ret_path, epoch, {"retention": retention})

            dt = time.time() - t0
            print(f"Epoch {epoch:03d}/{args.epochs} | clean {acc_clean:.2f} | probe {acc_probe:.2f} | retention {retention*100:.1f}% | {dt:.1f}s")

        self.writer.flush()
        self.writer.close()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tag: str, epoch: int) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = torch.as_tensor(y, device=self.device)
            with autocast(device_type="cuda", enabled=bool(_USE_CUDA and self.args.amp)):
                logits = self.model_forward_qfm_aware(x, style_score(x), 1.0, force_no_freq=False)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = 100.0 * correct / max(1, total)
        self.writer.add_scalar(f"eval/{tag}_acc", acc, epoch)
        return acc


# ------------------------------
# CLI
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="QFM-DCT ResNet18 (resume-safe)")

    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--img", type=int, default=32)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--pretrained", action="store_true")

    p.add_argument("--logdir", type=str, default="runs/exp")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--export_dir", type=str, default=None)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--qfm_lr_scale", type=float, default=0.25)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0)

    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--randaugment", action="store_true")
    p.add_argument("--augmix", action="store_true")

    p.add_argument("--envs_train", type=str, default=None)
    p.add_argument("--val_env", type=str, default="jpeg_40")

    # IMPORTANT: now this is safe; it no longer overwrites QFM conv1
    p.add_argument("--cifar_stem", action="store_true")

    p.add_argument("--unfreeze_gates_epoch", type=int, default=4)
    p.add_argument("--unfreeze_phase_epoch", type=int, default=9999)
    p.add_argument("--freq_warmup_epochs", type=int, default=5)
    p.add_argument("--f_clip_warm", type=float, default=0.3)
    p.add_argument("--phase_clip", type=float, default=math.pi / 6)
    p.add_argument("--phase_l2", type=float, default=1e-3)

    p.add_argument("--replace", type=str, default="conv1,layer1,layer2")

    p.add_argument("--rex_lambda_start", type=float, default=0.0)
    p.add_argument("--rex_lambda_final", type=float, default=0.0)
    p.add_argument("--rex_ramp_epochs", type=int, default=0)

    p.add_argument("--use_interference", type=lambda s: str(s).lower() in ("1","true","yes"), default=None)
    p.add_argument("--use_phase", type=lambda s: str(s).lower() in ("1","true","yes"), default=None)
    p.add_argument("--disable_style", action="store_true")

    p.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "baseline_blur", "baseline_augmix", "baseline_blur_augmix",
                 "qfm_E", "qfm_F", "qfm_G"]
    )
    return p


def _apply_config(args, cfg: dict):
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


def main():
    args = build_argparser().parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        _apply_config(args, cfg)

    if args.envs_train is None:
        args.envs_train = "clean"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.run_dir = os.path.join(args.logdir, ts)
    os.makedirs(args.run_dir, exist_ok=True)

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    trainer = Trainer(args)
    trainer.train()

def build_eval_loader(project_root: str, env: str, batch: int = 256, num_workers: int = 2, pin_memory: bool = True):
    """
    Public entrypoint for eval_robustness.py.

    Returns: (loader, num_classes)
    IMPORTANT: We do NOT instantiate Trainer here (it expects different signature).
    We directly build the CIFAR eval dataset the same way Trainer does.
    """
    # Prefer your project data folder, fallback to local ./data
    data_root = os.path.join(project_root, "data")
    if not os.path.exists(data_root):
        data_root = "./data"

    ds = CIFARWithEnv(
        root=data_root,
        train=False,
        img_size=32,
        env=env,
        randaugment=False,
        augmix=False,
        label_smoothing=0.0,
    )

    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(use_cuda and pin_memory),
        persistent_workers=False,
    )

    return loader, 10


if __name__ == "__main__":
    main()
