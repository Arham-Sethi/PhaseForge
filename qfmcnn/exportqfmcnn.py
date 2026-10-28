# qfmcnn/export_qfmcnn.py
# Export the trained QFM-ResNet (moisture regressor) to TorchScript / ONNX.

import os
import argparse
import torch

# Reuse your training code pieces
from qfm_train import (
    try_load_resnet18_pretrained,
    replace_resnet_convs_with_qfm,
)

def build_qfm_resnet(input_size=(224, 224), replace_layers=("conv1", "layer1")):
    """
    Rebuild the same architecture your training script uses:
      - ResNet18 backbone (pretrained if available)
      - Replace specified convs with QFMConv2d
      - Replace classifier head with 1-dim regressor
    """
    model = try_load_resnet18_pretrained()
    model = replace_resnet_convs_with_qfm(
        model, input_size=input_size, layers_to_replace=replace_layers, device=None
    )
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.25),
        torch.nn.Linear(in_features, 1)
    )
    return model

def load_ckpt_flex(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # support raw state_dict as well
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint (qfm_best.pth or best_afsc_dct_model.pth)")
    ap.add_argument("--ts",   required=True, help="Output TorchScript .ts path")
    ap.add_argument("--onnx", default=None,  help="(Optional) Output ONNX .onnx path")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width",  type=int, default=224)
    ap.add_argument("--replace", type=str, default="conv1,layer1",
                    help="Comma separated among conv1,layer1,layer2,layer3,layer4")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    layers = tuple([s.strip() for s in args.replace.split(",") if s.strip()])
    input_size = (args.height, args.width)

    print(f"[build] input_size={input_size}, replace_layers={layers}, device={device}")
    model = build_qfm_resnet(input_size=input_size, replace_layers=layers).to(device).eval()

    print(f"[load] checkpoint: {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, device)

    # Dummy input (1,3,H,W) — your model outputs shape (1,1)
    example = torch.randn(1, 3, args.height, args.width, device=device)

    # TorchScript export
    os.makedirs(os.path.dirname(os.path.abspath(args.ts)), exist_ok=True)
    with torch.no_grad():
        try:
            scripted = torch.jit.trace(model, example, strict=False)
        except Exception as e:
            print("[info] trace failed, trying script:", e)
            scripted = torch.jit.script(model)
    scripted.save(args.ts)
    print(f"✅ Saved TorchScript: {args.ts}")

    # Optional: ONNX export of the moisture output only
    if args.onnx:
        class MoistureOnly(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x):
                y = self.core(x)  # expect (B,1)
                return y

        wrapper = MoistureOnly(model).eval().to(device)
        torch.onnx.export(
            wrapper, example, args.onnx,
            input_names=["input"], output_names=["moisture"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=17
        )
        print(f"✅ Saved ONNX: {args.onnx}")

if __name__ == "__main__":
    main()
