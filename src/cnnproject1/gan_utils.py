"""GAN 通用工具函数。"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自动检测运行设备。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def configure_matplotlib_chinese() -> None:
    """全局配置中文字体显示。"""
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    matplotlib.use("Agg")
    font_candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Songti SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    selected_font = "DejaVu Sans"
    for candidate in font_candidates:
        try:
            font_path = font_manager.findfont(candidate, fallback_to_default=False)
            font_manager.fontManager.addfont(font_path)
            selected_font = font_manager.FontProperties(fname=font_path).get_name()
            break
        except Exception:  # noqa: BLE001
            continue
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir(output_dir: Path) -> Path:
    """确保输出目录存在。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """将 [-1,1] 图像还原到 [0,1]。"""
    return images.detach().cpu().clamp(-1, 1).add(1).div(2)


def make_image_grid(images: torch.Tensor, nrow: int = 4) -> np.ndarray:
    """将一批图片拼成网格。"""
    images = denormalize_images(images)
    batch, channels, height, width = images.shape
    nrow = min(nrow, batch)
    ncol = int(np.ceil(batch / nrow))
    canvas = np.ones((ncol * height, nrow * width, channels), dtype=np.float32)
    for idx in range(batch):
        row = idx // nrow
        col = idx % nrow
        image = images[idx].permute(1, 2, 0).numpy()
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = image
    return np.clip(canvas, 0.0, 1.0)


def save_fake_image_grid(fake_images: torch.Tensor, output_path: Path, title: str) -> None:
    """保存生成图片拼图。"""
    configure_matplotlib_chinese()
    grid = make_image_grid(fake_images, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_real_fake_comparison(real_images: torch.Tensor, fake_images: torch.Tensor, output_path: Path, title: str) -> None:
    """保存真实样本与生成样本对比图。"""
    configure_matplotlib_chinese()
    real_grid = make_image_grid(real_images[:8], nrow=4)
    fake_grid = make_image_grid(fake_images[:8], nrow=4)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].imshow(real_grid)
    axes[0].set_title("真实样本拼图")
    axes[0].axis("off")
    axes[1].imshow(fake_grid)
    axes[1].set_title("生成样本拼图")
    axes[1].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    """保存 GAN 损失曲线。"""
    configure_matplotlib_chinese()
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["generator_loss"], label="生成器损失")
    plt.plot(history_df["epoch"], history_df["discriminator_loss"], label="判别器损失")
    plt.xlabel("轮数")
    plt.ylabel("损失值")
    plt.title("GAN 训练损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_history_csv(history: list[dict], output_path: Path) -> pd.DataFrame:
    """保存训练历史记录。"""
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return history_df


def save_json(data: dict, output_path: Path) -> None:
    """保存 JSON 文件。"""
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_model_checkpoint(model: torch.nn.Module, output_path: Path, extra_state: dict) -> None:
    """保存模型检查点。"""
    checkpoint = {"model_state_dict": model.state_dict(), **extra_state}
    torch.save(checkpoint, output_path)

