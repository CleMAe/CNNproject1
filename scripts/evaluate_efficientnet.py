#!/usr/bin/env python3
"""使用本地模型进行评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.datasets import create_dataloaders  # noqa: E402
from cnnproject1.efficientnet import create_efficientnet  # noqa: E402
from cnnproject1.trainer import choose_device, evaluate_predictions, run_epoch, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估本地 EfficientNet 模型")
    parser.add_argument("--checkpoint", required=True, help="本地模型路径")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint["config"]
    set_seed(config_dict["seed"])

    bundle = create_dataloaders(
        image_size=config_dict["image_size"],
        batch_size=config_dict["batch_size"],
        num_workers=args.num_workers,
        val_ratio=config_dict["val_ratio"],
        seed=config_dict["seed"],
        per_class_limit=config_dict.get("per_class_limit"),
        drop_cross_split_duplicates=config_dict.get("drop_cross_split_duplicates", True),
    )
    loader = bundle.val_loader if args.split == "val" else bundle.test_loader

    model = create_efficientnet(config_dict["model_name"], num_classes=len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(choose_device())

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        device=choose_device(),
        max_batches=args.max_batches,
    )
    output_dir = Path(args.checkpoint).resolve().parent / "evaluation"
    summary = evaluate_predictions(
        targets=metrics["targets"],
        preds=metrics["preds"],
        probs=metrics["probs"],
        output_dir=output_dir,
        idx_to_class=checkpoint["idx_to_class"],
        split_name="验证集" if args.split == "val" else "测试集",
    )
    print("评估完成：", summary)


if __name__ == "__main__":
    main()

