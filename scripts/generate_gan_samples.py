#!/usr/bin/env python3
"""加载本地 GAN 检查点并重新生成图像。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.gan_advanced_models import ConditionalResGenerator  # noqa: E402
from cnnproject1.gan_utils import configure_matplotlib_chinese, ensure_output_dir, save_fake_image_grid, set_seed  # noqa: E402


LABEL_NAME_TO_INDEX = {
    "合格件": 0,
    "缺陷件": 1,
}

LABEL_MODE_TO_TEXT = {
    "defect": "缺陷件",
    "ok": "合格件",
    "alternate": "交替条件",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 GAN 生成器检查点重新生成图像")
    parser.add_argument("--checkpoint", required=True, help="生成器检查点路径")
    parser.add_argument("--output-dir", required=True, help="生成图片的输出目录")
    parser.add_argument("--num-samples", type=int, default=16, help="生成样本数量")
    parser.add_argument(
        "--label-mode",
        choices=["defect", "ok", "alternate"],
        default="defect",
        help="控制生成样本的类别标签",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def build_labels(num_samples: int, label_mode: str, device: torch.device) -> torch.Tensor:
    """根据生成模式构造条件标签。"""
    if label_mode == "defect":
        return torch.full((num_samples,), LABEL_NAME_TO_INDEX["缺陷件"], dtype=torch.long, device=device)
    if label_mode == "ok":
        return torch.full((num_samples,), LABEL_NAME_TO_INDEX["合格件"], dtype=torch.long, device=device)

    labels = [LABEL_NAME_TO_INDEX["合格件"], LABEL_NAME_TO_INDEX["缺陷件"]]
    repeated = [labels[idx % 2] for idx in range(num_samples)]
    return torch.tensor(repeated, dtype=torch.long, device=device)


def main() -> None:
    """执行 GAN 采样。"""
    args = parse_args()
    set_seed(args.seed)
    configure_matplotlib_chinese()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = ensure_output_dir(Path(args.output_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    generator = ConditionalResGenerator(
        latent_dim=config["latent_dim"],
        num_classes=2,
        base_channels=config["generator_feature_maps"],
        image_channels=3,
        image_size=config["image_size"],
    ).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()

    num_samples = args.num_samples
    noise = torch.randn(num_samples, config["latent_dim"], 1, 1, device=device)
    labels = build_labels(num_samples=num_samples, label_mode=args.label_mode, device=device)

    with torch.no_grad():
        fake_images = generator(noise, labels)

    epoch = int(checkpoint.get("epoch", 0))
    grid_size = int(math.sqrt(num_samples))
    if grid_size * grid_size != num_samples:
        grid_size = 4

    output_path = output_dir / f"regenerated_epoch_{epoch:04d}_{args.label_mode}.png"
    save_fake_image_grid(
        fake_images=fake_images[: max(grid_size * grid_size, 1)],
        output_path=output_path,
        title=f"第 {epoch} 轮模型重生成样本（{LABEL_MODE_TO_TEXT[args.label_mode]}）",
    )

    print(f"使用设备：{device}")
    print(f"检查点路径：{checkpoint_path}")
    print(f"输出图片：{output_path}")


if __name__ == "__main__":
    main()
