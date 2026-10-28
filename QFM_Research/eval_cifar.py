# eval_cifar10c.py
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math

from qfm_dct import build_qfm_resnet18, style_score

# --- CIFAR-10-C loader ---

class CIFAR10C(Dataset):
    def __init__(self, root, corruption, level, img_size=32):
        assert 1 <= level <= 5
        self.root = root
        self.corruption = corruption
        self.level = level
        data_path = os.path.join(root, corruption + ".npy")
        labels_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path)    # [50000,32,32,3]
        self.labels = np.load(labels_path)

        # pick severity slice
        start = (level-1) * 10000
        end   = level * 10000
        self.data = self.data[start:end]
        self.labels = self.labels[start:end]

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                                 std=[0.247,0.243,0.261]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        x = self.tf(img)
        y = int(self.labels[idx])
        return x, y

def evaluate_cifar10c(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_qfm_resnet18(
        img_size=args.img,
        num_classes=10,
        mode=args.mode,
        device=device,
        phase_clip=args.phase_clip,
        replace=args.replace,
        pretrained=False
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform',
        'pixelate', 'jpeg_compression'
    ]

    all_acc = []

    for corr in corruptions:
        acc_levels = []
        for level in range(1, 6):
            ds = CIFAR10C(args.cifar10c_dir, corr, level, img_size=args.img)
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
            correct, total = 0, 0
            with torch.no_grad():
                for x,y in dl:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)  # style_score used only during training via Trainer
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = 100.0 * correct / max(1,total)
            acc_levels.append(acc)
        mean_corr = float(np.mean(acc_levels))
        print(f"{corr}: {mean_corr:.2f}")
        all_acc.append(mean_corr)

    overall = float(np.mean(all_acc))
    print(f"Mean CIFAR-10-C accuracy: {overall:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifar10c_dir', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--img', type=int, default=160)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--mode', type=str, default='qfm_G')
    ap.add_argument('--replace', type=str, default='conv1,layer1,layer2,layer3,layer4')
    ap.add_argument('--phase_clip', type=float, default=float(math.pi/6))
    args = ap.parse_args()
    evaluate_cifar10c(args)
