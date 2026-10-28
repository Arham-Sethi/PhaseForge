# phaseforge_bench.py
# Robustness & Data-Efficiency Benchmark for QFM-ResNet18 vs Baselines
# Author: Arham Sethi (with helper scaffolding)
# Requires: qfm_train.py in the same folder exposing replace_resnet_convs_with_qfm, set_seed

import os, time, argparse, math, random, warnings
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models, utils as vutils
from torch.utils.tensorboard import SummaryWriter

# ---- QFM imports (from your file) ----
from qfm_train import replace_resnet_convs_with_qfm, set_seed

# ---------------------------
# Utilities / Environment
# ---------------------------
def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def amp_autocast():
    # modern API (works on CPU too with disabled flag)
    return torch.amp.autocast('cuda', enabled=torch.cuda.is_available())

def make_writer(logdir, tag):
    os.makedirs(logdir, exist_ok=True)
    return SummaryWriter(os.path.join(logdir, tag))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_cuda_allocator():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

# ---------------------------
# Data: loaders + fraction
# ---------------------------
def get_dataloaders(name: str, img_size=160, batch=32, workers=4, fraction=1.0):
    """Return train/test loaders, num_classes. Auto-downloads datasets."""
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    name = name.lower()
    if name == "cifar10":
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), norm])
        train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
        test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
        C = 10

    elif name == "dtd":
        train_tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), norm])
        tr  = datasets.DTD(root="./data", split="train", download=True, transform=train_tf)
        val = datasets.DTD(root="./data", split="val",   download=True, transform=train_tf)
        test= datasets.DTD(root="./data", split="test",  download=True, transform=test_tf)
        train = ConcatDataset([tr, val])  # more data for train
        C = 47

    elif name == "eurosat":
        train_tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), norm])
        full = datasets.EuroSAT(root="./data", download=True, transform=train_tf)
        n = len(full); ntr = int(0.8*n)
        train, test = torch.utils.data.random_split(full, [ntr, n-ntr], generator=torch.Generator().manual_seed(1337))
        train.dataset.transform = train_tf
        test.dataset.transform  = test_tf
        C = 10
    else:
        raise ValueError("dataset must be one of: cifar10, dtd, eurosat")

    # Low-data fraction
    if fraction < 1.0:
        tr_n = len(train)
        sub_n = max(1, int(fraction * tr_n))
        idx = torch.randperm(tr_n)[:sub_n].tolist()
        if isinstance(train, ConcatDataset):
            # flatten indices across parts
            # simplest: sample indices over range(len(train))
            idx = torch.randperm(len(train))[:sub_n].tolist()
            train = Subset(train, idx)
        else:
            train = Subset(train, idx)

    train_loader = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch, shuffle=False, num_workers=workers)
    return train_loader, test_loader, C

# ---------------------------
# Corruptions / robustness
# ---------------------------
def corrupt(x, kind="gauss", severity=0.3):
    if kind == "gauss":
        return (x + severity*torch.randn_like(x)).clamp(0,1)
    if kind == "blur":
        k = 3 if severity < 0.35 else 5
        return F.avg_pool2d(x, k, stride=1, padding=k//2)
    if kind == "jpeg":
        # poor man's quantization
        q = 1.0 - min(0.8, severity)  # lower q -> more loss
        y = (x * 255.0).round()
        y = (y * q).round()
        y = (y / max(1.0, q)).clamp(0,255) / 255.0
        return y
    if kind == "frost":
        # simple frost overlay (noise pattern)
        noise = torch.randn_like(x) * severity * 0.5 + 0.5
        return (0.7*x + 0.3*noise).clamp(0,1)
    return x

@torch.no_grad()
def corruption_curve(model, loader, device, severities=(0.1,0.2,0.3,0.4,0.5), kinds=("gauss","blur","jpeg","frost")):
    model.eval()
    curves = {k: [] for k in kinds}
    for k in kinds:
        for s in severities:
            ok, tot = 0, 0
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                x = corrupt(x, k, s)
                p = model(x).argmax(1)
                ok += (p==y).sum().item(); tot += y.numel()
            curves[k].append(100.0*ok/tot)
    return curves

# ---------------------------
# Feature stability & energy
# ---------------------------
class FeatureTap(nn.Module):
    """Register-forward tap to capture features & per-layer energy."""
    def __init__(self, model: nn.Module, take='avgpool'):
        super().__init__()
        self.model = model
        self.take = take
        self.clean = None
        self.corr  = None
        self.energies = []  # collect per-layer mean square
        self._hooks = []
        # tap common blocks
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
                self._hooks.append(m.register_forward_hook(self._energy_hook(name)))
        # penultimate feature tap (before FC)
        target = getattr(model, take) if hasattr(model, take) else None
        if target is not None:
            self._hooks.append(target.register_forward_hook(self._penult_hook))

    def _energy_hook(self, name):
        def hook(_, inp, out):
            with torch.no_grad():
                t = out.detach()
                e = (t.float()**2).mean().item()
                self.energies.append((name, e))
        return hook

    def _penult_hook(self, module, inp, out):
        # out shape [B, 512, 1, 1] for resnet18
        with torch.no_grad():
            f = torch.flatten(out.detach(), 1)
            if self.clean is None:
                self.clean = f
            else:
                self.corr = f

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

def cosine_feature_stability(model, x_clean, x_corr, device):
    tap = FeatureTap(model, take='avgpool')
    model.eval()
    with torch.no_grad():
        _ = model(x_clean.to(device))
        _ = model(x_corr.to(device))
        if tap.clean is None or tap.corr is None:
            tap.close()
            return None, []
        # cosine similarity batch-wise then mean
        c = F.cosine_similarity(tap.clean, tap.corr, dim=1).mean().item()
        energies = tap.energies[:]  # [(layer, energy), ...]
    tap.close()
    return c, energies

# ---------------------------
# Baseline variants
# ---------------------------
class Cutout(object):
    def __init__(self, size=32):
        self.size = size
    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError("Cutout expects tensor after ToTensor")
        _, H, W = img.shape
        s = self.size
        cy = random.randint(0, H-1); cx = random.randint(0, W-1)
        y1 = max(0, cy - s//2); y2 = min(H, cy + s//2)
        x1 = max(0, cx - s//2); x2 = min(W, cx + s//2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

class DropoutHead(nn.Module):
    def __init__(self, in_features, num_classes, p=0.3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.seq(x)

class TinyDAE(nn.Module):
    """Very small denoising autoencoder to prefilter inputs."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1),
        )
    def forward(self, x): return self.dec(self.enc(x))

class PrefilterModel(nn.Module):
    """Wrap a prefilter (e.g., DAE) in front of a classifier."""
    def __init__(self, pre: nn.Module, backbone: nn.Module):
        super().__init__()
        self.pre = pre
        self.backbone = backbone
    def forward(self, x):
        x = self.pre(x)
        return self.backbone(x)

def try_wavelet_stem_available():
    try:
        import pywt  # noqa: F401
        return True
    except Exception:
        return False

class DWTStem(nn.Module):
    """Simple wavelet stem: replace first conv by LL band of DWT + 1x1 conv to match channels."""
    def __init__(self, orig_stem: nn.Conv2d):
        super().__init__()
        import pywt
        self.wave = "haar"
        self.post = nn.Conv2d(orig_stem.in_channels, orig_stem.out_channels, kernel_size=1, bias=False)
        # init post to roughly identity scaling
        nn.init.kaiming_normal_(self.post.weight, nonlinearity='relu')
        self.bn   = nn.BatchNorm2d(orig_stem.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # DWT per channel -> LL band only
        B,C,H,W = x.shape
        LLs = []
        for b in range(B):
            # channel-wise DWT
            chans = []
            for c in range(C):
                arr = x[b, c].detach().cpu().numpy()
                LL, (LH, HL, HH) = __import__('pywt').dwt2(arr, self.wave)
                chans.append(torch.from_numpy(LL).to(x.device, x.dtype))
            LLs.append(torch.stack(chans, dim=0))
        LL = torch.stack(LLs, dim=0)  # [B,C,H/2,W/2]
        y = self.post(LL)
        y = self.bn(y); y = self.relu(y)
        y = self.pool(y)
        return y

def build_model(model_type: str, num_classes: int, img_size: int, device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if model_type == "baseline":
        pass
    elif model_type == "baseline_dropout_cutout":
        pass  # augment will add Cutout; head has dropout below
    elif model_type == "baseline_dwt":
        if try_wavelet_stem_available():
            m.conv1 = DWTStem(m.conv1)
        else:
            warnings.warn("pywt not available; falling back to baseline.")
    elif model_type == "baseline_dae":
        # will wrap later after pretrain
        pass
    elif model_type == "qfm":
        # QFM on conv1 only for stability/memory
        m = replace_resnet_convs_with_qfm(
            m, input_size=(img_size, img_size),
            layers_to_replace=("conv1",),
            device=device
        )
    else:
        raise ValueError("Unknown model_type")

    # replace FC
    in_fc = m.fc.in_features
    if model_type == "baseline_dropout_cutout":
        m.fc = DropoutHead(in_fc, num_classes, p=0.3)
    else:
        m.fc = nn.Linear(in_fc, num_classes)

    return m.to(device)

# ---------------------------
# Training / Evaluation
# ---------------------------
@torch.no_grad()
def top1(model, loader, device):
    model.eval()
    ok, tot = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        ok += (p==y).sum().item(); tot += y.numel()
    return 100.0*ok/tot

def pretrain_dae(dae: nn.Module, loader: DataLoader, steps=300, lr=1e-3, device=None):
    """Quick DAE pretrain on-the-fly noise; keeps it short to save time."""
    device = device or device_auto()
    dae.to(device).train()
    opt = torch.optim.Adam(dae.parameters(), lr=lr)
    it = 0
    for (x, _) in loader:
        x = x.to(device)
        noisy = corrupt(x, "gauss", 0.2)
        opt.zero_grad(set_to_none=True)
        with amp_autocast():
            out = dae(noisy)
            loss = F.mse_loss(out, x)
        loss.backward(); opt.step()
        it += 1
        if it >= steps: break

def attach_prefilter(model, dataset_name, num_classes, img_size, device):
    """Wrap baseline with a tiny DAE prefilter (pretrained for a few steps)."""
    dae = TinyDAE(in_ch=3)
    # Build a tiny loader for quick pretrain (same dataset)
    tr, _, _ = get_dataloaders(dataset_name, img_size=img_size, batch=64, workers=2, fraction=0.1)
    pretrain_dae(dae, tr, steps=300, lr=1e-3, device=device)
    return PrefilterModel(dae, model).to(device)

def train_one(
    dataset="cifar10", model_type="baseline",
    epochs=8, img_size=160, batch=32, lr=3e-4, fraction=1.0,
    logdir="runs/phaseforge", workers=4
):
    set_seed(1337, deterministic=True)
    set_cuda_allocator()
    device = device_auto()

    # Data (cutout aug only if baseline_dropout_cutout)
    base_tf_extra = []
    if model_type == "baseline_dropout_cutout":
        # We add Cutout AFTER ToTensor & Normalize using a wrapper below.
        base_tf_extra = [Cutout(size=max(8, img_size//6))]

    train_loader, test_loader, C = get_dataloaders(dataset, img_size=img_size, batch=batch, workers=workers, fraction=fraction)

    # If we need to insert Cutout, wrap the loader collate
    if base_tf_extra:
        # we implement a very small custom dataset wrapper rather than re-download transforms
        class CutoutWrap(torch.utils.data.Dataset):
            def __init__(self, base_ds, extra_ops):
                self.ds = base_ds; self.ops = extra_ops
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                x,y = self.ds[i]
                # ensure tensor then apply ops
                for op in self.ops:
                    x = op(x)
                return x,y
        train_loader = DataLoader(CutoutWrap(train_loader.dataset, base_tf_extra),
                                  batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)

    # Model
    model = build_model(model_type, C, img_size, device)

    if model_type == "baseline_dae":
        model = attach_prefilter(model, dataset, C, img_size, device)

    # Opt & sched
    # Use smaller LR for QFM parts if present
    qfm_params, base_params = [], []
    for n,p in model.named_parameters():
        if "ortho.P" in n or ".V" in n or ".Wc_" in n or ".phase_net" in n or ".morph." in n:
            qfm_params.append(p)
        else:
            base_params.append(p)

    if qfm_params:
        opt = torch.optim.AdamW([
            {"params": base_params, "lr": lr,       "weight_decay": 1e-4},
            {"params": qfm_params, "lr": lr*0.25,   "weight_decay": 1e-5},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    tag = f"{dataset}_{model_type}_img{img_size}_b{batch}_frac{fraction}_{int(time.time())}"
    writer = make_writer(logdir, tag)

    # Log a grid of few training images
    try:
        bx, by = next(iter(train_loader))
        writer.add_image("sample/train_batch", vutils.make_grid(bx[:16], nrow=8), 0)
    except Exception:
        pass

    # Warm-up (QFM only): freeze all but small parts for 2 epochs (if epochs >= 5)
    def set_requires_grad(stage):
        if not qfm_params: return
        for n,p in model.named_parameters():
            req = True
            if stage == "warmup":
                req = ("ortho.P" in n) or (".V" in n) or (".Wc_" in n) or (".phase_net" in n) or (".morph." in n) or ("fc" in n)
            p.requires_grad = req

    if qfm_params and epochs >= 5:
        set_requires_grad("warmup")
        warm_epochs = min(2, epochs//4)
    else:
        warm_epochs = 0

    best = 0.0
    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        loss_sum, nsum = 0.0, 0

        if qfm_params and warm_epochs and ep == warm_epochs+1:
            set_requires_grad("full")

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with amp_autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * y.size(0); nsum += y.size(0)

        sched.step()
        train_loss = loss_sum / max(1, nsum)
        acc_clean = top1(model, test_loader, device)

        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("acc/clean", acc_clean, ep)

        # Evaluate robustness curve (lightweight: every 2 epochs)
        if ep % 2 == 0 or ep == epochs:
            curves = corruption_curve(model, test_loader, device)
            for kind, vals in curves.items():
                for i, sv in enumerate(vals):
                    writer.add_scalar(f"robust/{kind}_s{i+1}", sv, ep)

        # Feature stability + energy (use small batch)
        try:
            bx, by = next(iter(test_loader))
            bx = bx[:16]
            bx_corr = corrupt(bx.clone(), kind="gauss", severity=0.3)
            cs, energies = cosine_feature_stability(model, bx, bx_corr, device)
            if cs is not None:
                writer.add_scalar("stability/cosine_clean_vs_gauss0.3", cs, ep)
            # log few energy taps
            for li, (lname, e) in enumerate(energies[:12]):
                writer.add_scalar(f"energy/{lname}", e, ep)
        except Exception:
            pass

        print(f"[{dataset}][{model_type}] ep {ep:02d}/{epochs} | loss {train_loss:.3f} | acc {acc_clean:.2f}% | {time.time()-t0:.1f}s")
        if acc_clean > best:
            best = acc_clean
            torch.save(model.state_dict(), f"best_{dataset}_{model_type}.pth")

    writer.close()
    print(f"[DONE] Best clean acc: {best:.2f}%")
    return best

# ---------------------------
# Orchestrator
# ---------------------------
def run_phaseforge(
    datasets: List[str],
    models: List[str],
    epochs=8, img=160, batch=32, lr=3e-4, fraction=1.0,
    logdir="runs/phaseforge", workers=4
):
    results = {}
    for ds in datasets:
        results[ds] = {}
        for mt in models:
            print(f"\n=== Running {ds} | {mt} | epochs={epochs} img={img} batch={batch} frac={fraction} ===")
            try:
                acc = train_one(
                    dataset=ds, model_type=mt,
                    epochs=epochs, img_size=img, batch=batch, lr=lr,
                    fraction=fraction, logdir=logdir, workers=workers
                )
                results[ds][mt] = acc
            except Exception as e:
                warnings.warn(f"Run failed for {ds}-{mt}: {e}")
                results[ds][mt] = None
    print("\nRESULTS:")
    for ds, row in results.items():
        print(ds, row)
    return results

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="cifar10,dtd,eurosat",
                    help="comma-separated: cifar10,dtd,eurosat")
    ap.add_argument("--models", type=str, default="baseline,qfm,baseline_dropout_cutout,baseline_dae,baseline_dwt",
                    help="comma-separated models")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--logdir", type=str, default="runs/phaseforge")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    md_list = [s.strip() for s in args.models.split(",") if s.strip()]

    run_phaseforge(
        datasets=ds_list, models=md_list,
        epochs=args.epochs, img=args.img, batch=args.batch, lr=args.lr,
        fraction=args.fraction, logdir=args.logdir, workers=args.workers
    )
