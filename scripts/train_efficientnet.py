#!/usr/bin/env python3
"""EfficientNet 训练脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.datasets import create_dataloaders  # noqa: E402
from cnnproject1.efficientnet import create_efficientnet  # noqa: E402
from cnnproject1.trainer import (  # noqa: E402
    FocalLoss,
    TrainConfig,
    choose_device,
    plot_training_history,
    run_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 EfficientNet 铸件缺陷分类模型")
    parser.add_argument("--model-name", default="b2", choices=["b0", "b1", "b2", "b3"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/efficientnet_b2")
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--keep-cross-split-duplicates", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """执行模型训练。"""
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        per_class_limit=args.per_class_limit,
        drop_cross_split_duplicates=not args.keep_cross_split_duplicates,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )

    set_seed(config.seed)
    device = choose_device()
    output_dir = ROOT_DIR / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_dataloaders(
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
        per_class_limit=config.per_class_limit,
        drop_cross_split_duplicates=config.drop_cross_split_duplicates,
    )

    model = create_efficientnet(config.model_name, num_classes=len(bundle.class_to_idx)).to(device)
    criterion = FocalLoss(alpha=bundle.class_weights.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history = []
    best_val_f1 = -1.0
    best_checkpoint_path = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            max_batches=config.max_val_batches,
        )
        scheduler.step(val_metrics["f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"第 {epoch} 轮 | "
            f"训练损失 {train_metrics['loss']:.4f} | "
            f"训练F1 {train_metrics['f1']:.4f} | "
            f"验证损失 {val_metrics['loss']:.4f} | "
            f"验证F1 {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_checkpoint_path = save_checkpoint(
                model=model,
                config=config,
                class_to_idx=bundle.class_to_idx,
                output_dir=output_dir,
                history=history,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("触发早停，结束训练。")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "训练历史.csv", index=False, encoding="utf-8-sig")
    plot_training_history(history_df, output_dir)
    (output_dir / "训练配置.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"训练完成，最佳模型保存在: {best_checkpoint_path}")


if __name__ == "__main__":
    main()

