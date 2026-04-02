#!/usr/bin/env python3
"""GAN 训练入口脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.gan_trainer import GANTrainConfig, train_gan  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练基于铸件图像数据集的 DCGAN 模型")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--generator-feature-maps", type=int, default=64)
    parser.add_argument("--discriminator-feature-maps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--no-balance-labels", action="store_true")
    parser.add_argument("--keep-cross-split-duplicates", action="store_true")
    parser.add_argument("--output-dir", default="gan——results")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--discriminator-steps", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--defect-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """执行 GAN 训练。"""
    args = parse_args()
    config = GANTrainConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        generator_feature_maps=args.generator_feature_maps,
        discriminator_feature_maps=args.discriminator_feature_maps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_interval=args.sample_interval,
        per_class_limit=args.per_class_limit,
        balance_labels=not args.no_balance_labels,
        drop_cross_split_duplicates=not args.keep_cross_split_duplicates,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        discriminator_steps=args.discriminator_steps,
        grad_clip_norm=args.grad_clip_norm,
        use_mixed_precision=args.use_mixed_precision,
        defect_multiplier=args.defect_multiplier,
    )
    train_gan(config)


if __name__ == "__main__":
    main()
