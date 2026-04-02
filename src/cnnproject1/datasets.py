"""数据加载与划分逻辑。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .transforms import build_transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT_DIR / "reports" / "image_eda_summary.csv"

LABEL_TO_INDEX = {"合格件": 0, "缺陷件": 1}
INDEX_TO_LABEL = {0: "合格件", 1: "缺陷件"}


@dataclass
class DatasetBundle:
    """打包返回训练相关对象。"""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weights: torch.Tensor
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


class CastingImageDataset(Dataset):
    """铸件图像分类数据集。"""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = ROOT_DIR / row["file_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_index"])
        return image, label, row["file_path"]


def load_metadata(summary_csv: Path = SUMMARY_CSV, drop_cross_split_duplicates: bool = True) -> pd.DataFrame:
    """读取 EDA 产出的明细表，并执行必要清洗。"""
    df = pd.read_csv(summary_csv)
    df = df[df["is_valid"] == 1].copy()
    df = df[df["dataset_name"].isin(["train", "test"])].copy()
    df = df[df["label"].isin(LABEL_TO_INDEX.keys())].copy()
    df["label_index"] = df["label"].map(LABEL_TO_INDEX)

    if drop_cross_split_duplicates:
        train_hashes = set(df.loc[df["split"] == "训练集", "sha256"])
        duplicate_test_mask = (df["split"] == "测试集") & (df["sha256"].isin(train_hashes))
        df = df.loc[~duplicate_test_mask].copy()

    return df.reset_index(drop=True)


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按标签分层拆分训练集与验证集。"""
    train_full_df = df[df["split"] == "训练集"].copy()
    test_df = df[df["split"] == "测试集"].copy()

    train_idx, val_idx = train_test_split(
        train_full_df.index.tolist(),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_full_df["label_index"],
    )
    train_df = train_full_df.loc[train_idx].copy().reset_index(drop=True)
    val_df = train_full_df.loc[val_idx].copy().reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """根据训练集类别频次计算类别权重。"""
    class_counts = train_df["label_index"].value_counts().sort_index()
    total = float(class_counts.sum())
    weights = [total / (len(class_counts) * class_counts[idx]) for idx in class_counts.index]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(train_df: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    """构建加权采样器，提升少数类被采样概率。"""
    class_counts = train_df["label_index"].value_counts().sort_index()
    sample_weights = train_df["label_index"].map(lambda x: 1.0 / class_counts[x]).tolist()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def limit_dataframe_per_class(df: pd.DataFrame, per_class_limit: int | None, seed: int) -> pd.DataFrame:
    """按类别裁剪样本数，方便做小规模冒烟测试。"""
    if per_class_limit is None:
        return df.reset_index(drop=True)

    random.seed(seed)
    parts = []
    for _, group in df.groupby("label_index"):
        parts.append(group.sample(n=min(per_class_limit, len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def create_dataloaders(
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    per_class_limit: int | None = None,
    drop_cross_split_duplicates: bool = True,
) -> DatasetBundle:
    """创建训练、验证、测试数据加载器。"""
    df = load_metadata(drop_cross_split_duplicates=drop_cross_split_duplicates)
    train_df, val_df, test_df = split_train_val(df=df, val_ratio=val_ratio, seed=seed)
    train_df = limit_dataframe_per_class(train_df, per_class_limit, seed)
    val_df = limit_dataframe_per_class(val_df, per_class_limit, seed)
    test_df = limit_dataframe_per_class(test_df, per_class_limit, seed)

    train_dataset = CastingImageDataset(train_df, transform=build_transforms(image_size=image_size, is_train=True))
    val_dataset = CastingImageDataset(val_df, transform=build_transforms(image_size=image_size, is_train=False))
    test_dataset = CastingImageDataset(test_df, transform=build_transforms(image_size=image_size, is_train=False))

    sampler = build_weighted_sampler(train_df, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_weights=compute_class_weights(train_df),
        class_to_idx=LABEL_TO_INDEX,
        idx_to_class=INDEX_TO_LABEL,
    )

