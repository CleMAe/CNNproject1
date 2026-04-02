#!/usr/bin/env python3
"""铸造件图像数据集 EDA 脚本。

功能概览：
1. 扫描解压后的训练集、测试集与原始 512x512 图像目录
2. 统计样本数量、类别分布、图像尺寸、文件大小、像素分布等信息
3. 检查损坏文件、重复文件、跨划分重复文件
4. 生成中文可视化图表与中文 Markdown 报告
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from PIL import Image, ImageFile

# 允许读取轻微损坏但仍可恢复的图片，避免遍历时中断
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 为 matplotlib 指定可写缓存目录，避免权限问题
ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


DATA_ROOT = ROOT_DIR / "data"
REPORT_ROOT = ROOT_DIR / "reports"
FIGURE_ROOT = REPORT_ROOT / "figures"
SUMMARY_CSV = REPORT_ROOT / "image_eda_summary.csv"
SUMMARY_JSON = REPORT_ROOT / "image_eda_summary.json"
REPORT_MD = REPORT_ROOT / "EDA报告.md"


def configure_plot_style() -> None:
    """设置中文绘图主题。"""
    font_candidates = ["Songti SC", "Hiragino Sans GB", "Heiti TC"]
    selected_font_name = None
    for candidate in font_candidates:
        try:
            font_path = font_manager.findfont(candidate, fallback_to_default=False)
            font_manager.fontManager.addfont(font_path)
            selected_font_name = font_manager.FontProperties(fname=font_path).get_name()
            break
        except Exception:  # noqa: BLE001
            continue

    if selected_font_name is None:
        selected_font_name = "DejaVu Sans"

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["font.family"] = selected_font_name
    plt.rcParams["font.sans-serif"] = [selected_font_name]
    plt.rcParams["axes.unicode_minus"] = False


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """将 DataFrame 转为简单 Markdown 表格，避免额外依赖。"""
    table_df = df.head(max_rows).copy() if max_rows is not None else df.copy()
    if isinstance(table_df.columns, pd.MultiIndex):
        table_df.columns = ["_".join(map(str, col)).strip("_") for col in table_df.columns]

    headers = [str(col) for col in table_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_df.itertuples(index=False, name=None):
        row_values = []
        for value in row:
            if pd.isna(value):
                row_values.append("")
            else:
                row_values.append(str(value))
        lines.append("| " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def discover_dataset_dirs() -> dict[str, Path]:
    """定位需要分析的三个目录。"""
    candidates = {
        "train": DATA_ROOT / "casting_data" / "casting_data" / "train",
        "test": DATA_ROOT / "casting_data" / "casting_data" / "test",
        "raw_512": DATA_ROOT / "casting_512x512" / "casting_512x512",
    }
    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少数据目录: {missing}")
    return candidates


def iter_image_files(folder: Path) -> Iterable[Path]:
    """递归遍历图像文件。"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def infer_label(path: Path) -> str:
    """从目录名推断标签。"""
    parts = {part.lower() for part in path.parts}
    if "def_front" in parts:
        return "缺陷件"
    if "ok_front" in parts:
        return "合格件"
    return "未知"


def infer_split(path: Path, dataset_name: str) -> str:
    """从路径或数据源名称推断划分。"""
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "训练集"
    if "test" in parts:
        return "测试集"
    if dataset_name == "raw_512":
        return "原始512图"
    return dataset_name


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件哈希，用于精确重复检测。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def gradient_sharpness(gray_array: np.ndarray) -> float:
    """用梯度能量粗略衡量图像清晰度。"""
    gy, gx = np.gradient(gray_array.astype(np.float32))
    return float(np.mean(gx**2 + gy**2))


def analyze_single_image(path: Path, dataset_name: str) -> dict[str, object]:
    """提取单张图片的统计特征。"""
    result: dict[str, object] = {
        "file_path": str(path.relative_to(ROOT_DIR)),
        "dataset_name": dataset_name,
        "split": infer_split(path, dataset_name),
        "label": infer_label(path),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "file_size_kb": round(path.stat().st_size / 1024, 3),
        "sha256": "",
        "is_valid": 0,
        "error_message": "",
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "pixel_count": np.nan,
        "mode": "",
        "mean_r": np.nan,
        "mean_g": np.nan,
        "mean_b": np.nan,
        "mean_gray": np.nan,
        "std_gray": np.nan,
        "sharpness_score": np.nan,
        "near_black_ratio": np.nan,
        "near_white_ratio": np.nan,
    }
    try:
        result["sha256"] = sha256_of_file(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            arr = np.asarray(image)
            gray = arr.mean(axis=2)
            result.update(
                {
                    "is_valid": 1,
                    "width": int(image.width),
                    "height": int(image.height),
                    "aspect_ratio": round(image.width / image.height, 4),
                    "pixel_count": int(image.width * image.height),
                    "mode": "RGB",
                    "mean_r": round(float(arr[:, :, 0].mean()), 4),
                    "mean_g": round(float(arr[:, :, 1].mean()), 4),
                    "mean_b": round(float(arr[:, :, 2].mean()), 4),
                    "mean_gray": round(float(gray.mean()), 4),
                    "std_gray": round(float(gray.std()), 4),
                    "sharpness_score": round(gradient_sharpness(gray), 4),
                    "near_black_ratio": round(float((gray <= 15).mean()), 6),
                    "near_white_ratio": round(float((gray >= 240).mean()), 6),
                }
            )
    except Exception as exc:  # noqa: BLE001
        result["error_message"] = str(exc)
    return result


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """构建后续报告需要的统计表。"""
    valid_df = df[df["is_valid"] == 1].copy()

    split_label_table = (
        valid_df.groupby(["split", "label"])
        .size()
        .reset_index(name="样本数")
        .sort_values(["split", "label"])
    )
    dataset_label_table = (
        valid_df.groupby(["dataset_name", "label"])
        .size()
        .reset_index(name="样本数")
        .sort_values(["dataset_name", "label"])
    )
    image_size_table = (
        valid_df.groupby(["width", "height"])
        .size()
        .reset_index(name="样本数")
        .sort_values("样本数", ascending=False)
    )
    quality_table = (
        valid_df.groupby(["split", "label"])[
            [
                "file_size_kb",
                "mean_gray",
                "std_gray",
                "sharpness_score",
                "near_black_ratio",
                "near_white_ratio",
            ]
        ]
        .agg(["mean", "median", "min", "max"])
        .round(4)
    )

    return {
        "split_label_table": split_label_table,
        "dataset_label_table": dataset_label_table,
        "image_size_table": image_size_table,
        "quality_table": quality_table,
    }


def plot_split_class_distribution(df: pd.DataFrame) -> None:
    """绘制不同划分下的类别分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=valid_df, x="split", hue="label", order=["训练集", "测试集", "原始512图"])
    ax.set_title("不同数据划分下的类别样本分布", fontsize=15)
    ax.set_xlabel("数据划分")
    ax.set_ylabel("样本数量")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "01_不同数据划分类别分布.png", dpi=180)
    plt.close()


def plot_image_size_distribution(df: pd.DataFrame) -> None:
    """绘制图像宽高分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=valid_df,
        x="width",
        y="height",
        hue="split",
        style="label",
        alpha=0.55,
        s=28,
    )
    plt.title("图像宽高分布散点图", fontsize=15)
    plt.xlabel("图像宽度")
    plt.ylabel("图像高度")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "02_图像宽高分布.png", dpi=180)
    plt.close()


def plot_file_size_distribution(df: pd.DataFrame) -> None:
    """绘制文件大小分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=valid_df,
        x="file_size_kb",
        hue="label",
        bins=40,
        kde=True,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.title("按类别划分的图像文件大小分布", fontsize=15)
    plt.xlabel("文件大小（KB）")
    plt.ylabel("样本数量")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "03_文件大小分布.png", dpi=180)
    plt.close()


def plot_brightness_and_contrast(df: pd.DataFrame) -> None:
    """绘制亮度与对比度分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=valid_df, x="label", y="mean_gray", ax=axes[0])
    axes[0].set_title("不同类别的平均亮度分布", fontsize=14)
    axes[0].set_xlabel("类别")
    axes[0].set_ylabel("平均灰度值")

    sns.boxplot(data=valid_df, x="label", y="std_gray", ax=axes[1])
    axes[1].set_title("不同类别的灰度标准差分布", fontsize=14)
    axes[1].set_xlabel("类别")
    axes[1].set_ylabel("灰度标准差")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "04_亮度与对比度分布.png", dpi=180)
    plt.close()


def plot_sharpness_distribution(df: pd.DataFrame) -> None:
    """绘制清晰度分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=valid_df, x="label", y="sharpness_score", hue="split", cut=0)
    plt.title("不同类别与划分下的清晰度分布", fontsize=15)
    plt.xlabel("类别")
    plt.ylabel("清晰度分数")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "05_清晰度分布.png", dpi=180)
    plt.close()


def plot_rgb_channel_distribution(df: pd.DataFrame) -> None:
    """绘制 RGB 三通道均值分布。"""
    valid_df = df[df["is_valid"] == 1].copy()
    channel_df = valid_df.melt(
        id_vars=["split", "label"],
        value_vars=["mean_r", "mean_g", "mean_b"],
        var_name="通道",
        value_name="像素均值",
    )
    channel_df["通道"] = channel_df["通道"].map(
        {"mean_r": "红色通道", "mean_g": "绿色通道", "mean_b": "蓝色通道"}
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=channel_df, x="通道", y="像素均值", hue="label")
    plt.title("不同类别的 RGB 通道像素均值分布", fontsize=15)
    plt.xlabel("颜色通道")
    plt.ylabel("像素均值")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "06_RGB通道分布.png", dpi=180)
    plt.close()


def plot_extreme_pixel_ratio(df: pd.DataFrame) -> None:
    """绘制近黑/近白像素占比。"""
    valid_df = df[df["is_valid"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=valid_df, x="label", y="near_black_ratio", ax=axes[0])
    axes[0].set_title("近黑像素占比分布", fontsize=14)
    axes[0].set_xlabel("类别")
    axes[0].set_ylabel("占比")

    sns.boxplot(data=valid_df, x="label", y="near_white_ratio", ax=axes[1])
    axes[1].set_title("近白像素占比分布", fontsize=14)
    axes[1].set_xlabel("类别")
    axes[1].set_ylabel("占比")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "07_极端像素占比分布.png", dpi=180)
    plt.close()


def plot_sample_mosaic(df: pd.DataFrame) -> None:
    """绘制样本拼图，帮助人工直观看图。"""
    valid_df = df[df["is_valid"] == 1].copy()
    candidates = (
        valid_df[valid_df["split"].isin(["训练集", "测试集"])]
        .groupby(["split", "label"], group_keys=False)
        .head(6)
    )
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for idx, (_, row) in enumerate(candidates.iterrows()):
        if idx >= len(axes):
            break
        image_path = ROOT_DIR / row["file_path"]
        with Image.open(image_path) as image:
            axes[idx].imshow(image.convert("RGB"))
        axes[idx].set_title(f"{row['split']}\n{row['label']}", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("训练集与测试集样本示例拼图", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGURE_ROOT / "08_样本拼图.png", dpi=180)
    plt.close()


def collect_duplicate_info(df: pd.DataFrame) -> dict[str, object]:
    """收集重复文件与跨划分重复情况。"""
    valid_df = df[df["is_valid"] == 1].copy()
    duplicate_groups = valid_df.groupby("sha256").filter(lambda group: len(group) > 1)
    duplicate_count = int(duplicate_groups.shape[0])
    exact_duplicate_groups = (
        duplicate_groups.groupby("sha256")["file_path"].apply(list).to_dict()
        if duplicate_count
        else {}
    )

    split_duplicate_groups = {}
    for sha256, group in valid_df.groupby("sha256"):
        splits = sorted(set(group["split"]))
        if len(splits) > 1:
            split_duplicate_groups[sha256] = {
                "splits": splits,
                "files": group["file_path"].tolist(),
            }

    return {
        "duplicate_file_records": duplicate_count,
        "duplicate_hash_groups": len(exact_duplicate_groups),
        "cross_split_duplicate_groups": len(split_duplicate_groups),
        "exact_duplicate_examples": dict(list(exact_duplicate_groups.items())[:10]),
        "cross_split_duplicate_examples": dict(list(split_duplicate_groups.items())[:10]),
    }


def write_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    duplicate_info: dict[str, object],
) -> None:
    """输出中文 Markdown 报告。"""
    valid_df = df[df["is_valid"] == 1].copy()
    invalid_df = df[df["is_valid"] == 0].copy()

    split_counts = (
        valid_df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(
            ["训练集", "测试集", "原始512图"], fill_value=0
        )
    )
    imbalance_desc = []
    for split_name, row in split_counts.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        major_ratio = float(row.max() / total)
        imbalance_desc.append(
            f"- {split_name}：总样本 {total} 张，最大类别占比 {major_ratio:.2%}"
        )

    width_mode = valid_df["width"].mode().iloc[0]
    height_mode = valid_df["height"].mode().iloc[0]
    mean_brightness = valid_df.groupby("label")["mean_gray"].mean().round(2).to_dict()
    mean_sharpness = valid_df.groupby("label")["sharpness_score"].mean().round(2).to_dict()

    report = f"""# 铸造件图像数据集 EDA 报告

## 1. 分析目标

本报告针对项目中的铸造件图像数据集开展完整的数据探索分析，重点服务于后续神经网络建模训练任务。分析重点包括：

- 数据结构与标签划分是否清晰
- 类别分布是否存在不均衡
- 图像尺寸、亮度、对比度、清晰度等分布特征
- 是否存在损坏图片、重复图片、跨训练/测试重复等数据质量问题
- 对后续 CNN 建模、数据清洗、数据增强和评估策略的建议

## 2. 数据概览

- 数据源目录：`data/casting_data/casting_data/train`、`data/casting_data/casting_data/test`、`data/casting_512x512/casting_512x512`
- 标签定义：`ok_front` 映射为“合格件”，`def_front` 映射为“缺陷件”
- 有效图片总数：{len(valid_df)} 张
- 无效/损坏图片数：{len(invalid_df)} 张
- 最常见图像尺寸：{int(width_mode)} x {int(height_mode)}

### 2.1 各划分样本统计

{dataframe_to_markdown(tables["split_label_table"])}

### 2.2 各数据源样本统计

{dataframe_to_markdown(tables["dataset_label_table"])}

### 2.3 图像尺寸统计（前 10 种）

{dataframe_to_markdown(tables["image_size_table"], max_rows=10)}

## 3. 数据质量检查

- 损坏图片数量：{len(invalid_df)} 张
- 精确重复文件记录数：{duplicate_info["duplicate_file_records"]}
- 精确重复哈希组数：{duplicate_info["duplicate_hash_groups"]}
- 跨数据划分重复哈希组数：{duplicate_info["cross_split_duplicate_groups"]}

### 3.1 数据质量结论

- 该数据集采用目录作为标签来源，标签结构较清晰，便于直接接入图像分类训练流程。
- 若跨训练集与测试集存在重复图片，需要格外关注评估结果是否被高估。
- 若发现损坏样本，应在训练前剔除，防止 DataLoader 中断或训练噪声增加。

## 4. 数据分布观察

### 4.1 类别分布

{chr(10).join(imbalance_desc)}

### 4.2 亮度与纹理特征

- 合格件平均灰度均值：{mean_brightness.get("合格件", "N/A")}
- 缺陷件平均灰度均值：{mean_brightness.get("缺陷件", "N/A")}
- 合格件平均清晰度分数：{mean_sharpness.get("合格件", "N/A")}
- 缺陷件平均清晰度分数：{mean_sharpness.get("缺陷件", "N/A")}

### 4.3 质量统计汇总

{dataframe_to_markdown(tables["quality_table"].reset_index())}

## 5. 对后续神经网络训练的建议

- 若训练集类别明显不均衡，建议使用 `class_weight`、`WeightedRandomSampler` 或面向少数类的增强策略。
- 若图像尺寸高度统一，可直接采用固定输入尺寸；若存在多个尺寸，建议在训练前统一 resize，并记录是否拉伸变形。
- 若缺陷件与合格件在亮度、对比度、清晰度上分布不同，建议在增强中加入亮度扰动、对比度扰动和轻微模糊增强，提升模型鲁棒性。
- 若检测到跨划分重复图片，建议重新划分训练/验证/测试集，避免数据泄漏。
- 对工业表面缺陷任务，推荐后续重点尝试：`ResNet18/34`、`EfficientNet-B0/B1`、`MobileNetV3` 作为 baseline；同时关注召回率、F1 和混淆矩阵，而不仅是准确率。

## 6. 产出文件说明

- 明细统计表：`reports/image_eda_summary.csv`
- JSON 汇总：`reports/image_eda_summary.json`
- 图表目录：`reports/figures/`
- 当前报告：`reports/EDA报告.md`
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    """执行完整 EDA 流程。"""
    configure_plot_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_dirs = discover_dataset_dirs()
    records: list[dict[str, object]] = []

    for dataset_name, folder in dataset_dirs.items():
        for image_path in iter_image_files(folder):
            records.append(analyze_single_image(image_path, dataset_name))

    df = pd.DataFrame(records)
    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    duplicate_info = collect_duplicate_info(df)
    tables = build_summary_tables(df)

    json_summary = {
        "total_records": int(len(df)),
        "valid_records": int((df["is_valid"] == 1).sum()),
        "invalid_records": int((df["is_valid"] == 0).sum()),
        "dataset_counts": df.groupby("dataset_name").size().to_dict(),
        "split_counts": df.groupby("split").size().to_dict(),
        "label_counts": df.groupby("label").size().to_dict(),
        "duplicate_info": duplicate_info,
    }
    SUMMARY_JSON.write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_split_class_distribution(df)
    plot_image_size_distribution(df)
    plot_file_size_distribution(df)
    plot_brightness_and_contrast(df)
    plot_sharpness_distribution(df)
    plot_rgb_channel_distribution(df)
    plot_extreme_pixel_ratio(df)
    plot_sample_mosaic(df)
    write_report(df, tables, duplicate_info)

    print("EDA 分析完成")
    print(f"明细文件: {SUMMARY_CSV}")
    print(f"汇总文件: {SUMMARY_JSON}")
    print(f"报告文件: {REPORT_MD}")
    print(f"图表目录: {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
