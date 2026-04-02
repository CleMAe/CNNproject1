"""训练、评估、可视化与推理工具。"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


@dataclass
class TrainConfig:
    """训练配置。"""

    model_name: str = "b2"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_ratio: float = 0.15
    seed: int = 42
    output_dir: str = "outputs/efficientnet_b2"
    per_class_limit: int | None = None
    drop_cross_split_duplicates: bool = True
    early_stopping_patience: int = 4
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None


def set_seed(seed: int) -> None:
    """固定随机种子，保证复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    """优先选择可用设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_plot_style() -> None:
    """设置中文可视化风格。"""
    font_candidates = [
        "Songti SC",
        "Hiragino Sans GB",
        "Heiti TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
    ]
    selected_font_name = "DejaVu Sans"
    for candidate in font_candidates:
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
            font_manager.fontManager.addfont(path)
            selected_font_name = font_manager.FontProperties(fname=path).get_name()
            break
        except Exception:  # noqa: BLE001
            continue
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["font.family"] = selected_font_name
    plt.rcParams["font.sans-serif"] = [selected_font_name]
    plt.rcParams["axes.unicode_minus"] = False


class FocalLoss(nn.Module):
    """带类别权重的 Focal Loss。"""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = (1.0 - target_probs) ** self.gamma
        return (focal_weight * ce_loss).mean()


def run_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """执行单轮训练或验证。"""
    is_train = optimizer is not None
    model.train(is_train)

    all_targets = []
    all_preds = []
    all_probs = []
    losses = []

    for batch_idx, (images, targets, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        losses.append(float(loss.item()))
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(all_targets, all_preds) if all_targets else 0.0,
        "f1": f1_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
    }
    return metrics


def save_checkpoint(model, config: TrainConfig, class_to_idx: dict[str, int], output_dir: Path, history: list[dict]):
    """保存本地模型权重与配置信息。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "class_to_idx": class_to_idx,
            "idx_to_class": {idx: label for label, idx in class_to_idx.items()},
            "history": history,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        checkpoint_path,
    )
    return checkpoint_path


def plot_training_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制训练过程曲线。"""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="训练损失")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="验证损失")
    axes[0].set_title("训练与验证损失曲线")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel("损失")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_f1"], label="训练F1")
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="验证F1")
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="训练准确率")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="验证准确率")
    axes[1].set_title("训练与验证指标曲线")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("指标值")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "训练过程曲线.png", dpi=180)
    plt.close()


def evaluate_predictions(targets, preds, probs, output_dir: Path, idx_to_class: dict[int, str], split_name: str) -> dict:
    """生成评估报告与相关曲线。"""
    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(
        targets,
        preds,
        labels=[0, 1],
        target_names=[idx_to_class[0], idx_to_class[1]],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / f"{split_name}_分类报告.csv", encoding="utf-8-sig")

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values())
    plt.title(f"{split_name}混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_混淆矩阵.png", dpi=180)
    plt.close()

    if len(set(targets)) >= 2:
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"{split_name} ROC 曲线")
        plt.xlabel("假阳性率")
        plt.ylabel("真正率")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_ROC曲线.png", dpi=180)
        plt.close()

        precision, recall, _ = precision_recall_curve(targets, probs)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.title(f"{split_name} PR 曲线")
        plt.xlabel("召回率")
        plt.ylabel("精确率")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_PR曲线.png", dpi=180)
        plt.close()
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    summary = {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "roc_auc": roc_auc if len(set(targets)) >= 2 else None,
        "pr_auc": pr_auc,
    }
    (output_dir / f"{split_name}_评估摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
