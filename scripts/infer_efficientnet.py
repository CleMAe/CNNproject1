#!/usr/bin/env python3
"""加载本地模型进行单张图片推理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.efficientnet import create_efficientnet  # noqa: E402
from cnnproject1.trainer import choose_device  # noqa: E402
from cnnproject1.transforms import build_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地 EfficientNet 模型推理图片")
    parser.add_argument("--checkpoint", required=True, help="本地模型路径")
    parser.add_argument("--image", required=True, help="待推理图片路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint["config"]
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}

    device = choose_device()
    model = create_efficientnet(config_dict["model_name"], num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transforms(image_size=config_dict["image_size"], is_train=False)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    print("推理结果：")
    for idx, prob in enumerate(probs.detach().cpu().tolist()):
        print(f"  {idx_to_class[idx]}: {prob:.6f}")
    print(f"最终预测类别: {idx_to_class[pred_idx]}")


if __name__ == "__main__":
    main()

