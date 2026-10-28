# tests/test_qfm_shapes.py
import os
import sys

# Add project root (QFM_Research) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from qfm_dct import build_qfm_resnet18

def run_one(mode="qfm_G", img=160):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_qfm_resnet18(
        img_size=img,
        num_classes=10,
        mode=mode,
        device=device,
        phase_clip=3.14159/6,
        replace="conv1,layer1,layer2,layer3,layer4",
        pretrained=False
    )
    model.eval()
    x = torch.randn(4, 3, img, img, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 10), f"{mode}: expected (4,10), got {y.shape}"
    assert torch.isfinite(y).all(), f"{mode}: found NaNs/Infs"

if __name__ == "__main__":
    for m in ["baseline_blur_augmix", "qfm_E", "qfm_F", "qfm_G"]:
        if m.startswith("baseline"):
            mode_use = m
        else:
            mode_use = m
        print("Testing", mode_use)
        run_one(mode=mode_use, img=160)
    print("All good.")
