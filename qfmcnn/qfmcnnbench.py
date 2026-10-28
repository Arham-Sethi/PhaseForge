# qfm_quickbench.py
import os, time, argparse, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# ---- import your QFM bits ----
from qfm_train import replace_resnet_convs_with_qfm   # you already have this
from qfm_train import set_seed                         # same helper from your script

def get_dataloaders(name, img_size=224, batch=128, num_workers=4):
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    if name.lower() == "cifar10":
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), norm])
        train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
        test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
        num_classes = 10

    elif name.lower() == "dtd":  # textures
        train_tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), norm])
        train = datasets.DTD(root="./data", split="train", download=True, transform=train_tf)
        val   = datasets.DTD(root="./data", split="val",   download=True, transform=test_tf)
        test  = datasets.DTD(root="./data", split="test",  download=True, transform=test_tf)
        train = torch.utils.data.ConcatDataset([train, val])  # train+val, hold out test
        num_classes = 47

    elif name.lower() == "eurosat":
        train_tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm
        ])
        test_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), norm])
        full = datasets.EuroSAT(root="./data", download=True, transform=train_tf)
        # quick split 80/20
        n = len(full); ntr = int(0.8*n)
        train, test = torch.utils.data.random_split(full, [ntr, n-ntr])
        train.dataset.transform = train_tf
        test.dataset.transform  = test_tf
        num_classes = 10
    else:
        raise ValueError("dataset must be one of: cifar10, dtd, eurosat")

    train_loader = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_classes

def build_model(model_type, num_classes, img_size, device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if model_type == "qfm":
        m = replace_resnet_convs_with_qfm(m, input_size=(img_size,img_size), layers_to_replace=("conv1"), device=device)
    # replace head
    in_fc = m.fc.in_features
    m.fc = nn.Linear(in_fc, num_classes)
    return m.to(device)

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    ok, tot = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        ok += (p==y).sum().item()
        tot += y.numel()
    return 100.0*ok/tot

# simple synthetic corruptions for a robustness sanity check
def corrupt_batch(x, kind="gauss", severity=0.2):
    if kind=="gauss":   return (x + severity*torch.randn_like(x)).clamp(0,1)
    if kind=="blur":    return torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
    if kind=="jpeg":    return x.add(-severity).clamp(0,1) + severity*(x**2)
    return x

@torch.no_grad()
def cifar10_corruption_eval(model, loader, device):
    kinds = ["gauss","blur","jpeg"]
    model.eval()
    outs = {}
    for k in kinds:
        ok, tot = 0, 0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            x = corrupt_batch(x, k, 0.3)
            p = model(x).argmax(1)
            ok += (p==y).sum().item(); tot += y.numel()
        outs[k] = 100.0*ok/tot
    return outs

def train_quick(dataset="cifar10", model_type="baseline", epochs=8, img_size=224, batch=128, lr=3e-4, logdir="runs/qfm_quickbench"):
    set_seed(1337, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te, C = get_dataloaders(dataset, img_size=img_size, batch=batch, num_workers=min(4, os.cpu_count() or 2))
    model = build_model("qfm" if model_type=="qfm" else "baseline", C, img_size, device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    writer = SummaryWriter(os.path.join(logdir, f"{dataset}_{model_type}_{int(time.time())}"))

    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        running=0; n=0
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item()*y.size(0); n += y.size(0)
        sched.step()
        train_loss = running/max(1,n)
        acc = accuracy(model, te, device)
        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("acc/val", acc, ep)
        print(f"[{dataset}][{model_type}] epoch {ep:02d}/{epochs} | loss {train_loss:.3f} | val acc {acc:.2f}% | {time.time()-t0:.1f}s")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), f"best_{dataset}_{model_type}.pth")

    # quick robustness if cifar10
    if dataset.lower()=="cifar10":
        rb = cifar10_corruption_eval(model, te, device)
        for k,v in rb.items():
            writer.add_scalar(f"robust/{k}", v, 0)
        print("Robustness (CIFAR10 synthetic):", rb)

    writer.close()
    print(f"[DONE] Best {dataset} {model_type}: {best:.2f}%")
    return best

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","dtd","eurosat"], default="cifar10")
    ap.add_argument("--model", choices=["baseline","qfm"], default="qfm")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    train_quick(args.dataset, args.model, args.epochs, args.img, args.batch, args.lr)
