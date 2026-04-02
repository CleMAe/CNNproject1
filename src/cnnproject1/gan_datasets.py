"""GAN 数据加载模块。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .datasets import ROOT_DIR, load_metadata
from .transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor


@dataclass
class GANDatasetBundle:
    """GAN 训练数据打包对象。"""

    train_loader: DataLoader
    train_df: pd.DataFrame


class UnlabeledCastingDataset(Dataset):
    """用于 GAN 的无标签图像数据集。"""

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


def build_gan_transforms(image_size: int, is_train: bool = True) -> Compose:
    """构造 GAN 使用的图像变换，将像素缩放到 [-1, 1]。"""
    transforms = [Resize((image_size, image_size))]
    if is_train:
        transforms.append(RandomHorizontalFlip(p=0.5))
    transforms.extend(
        [
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return Compose(transforms)


def balance_dataframe_by_label(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """为避免 GAN 过度偏向多数类，默认按最小类别数量做平衡采样。"""
    min_count = int(df["label"].value_counts().min())
    parts = []
    for _, group in df.groupby("label"):
        parts.append(group.sample(n=min_count, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def limit_dataframe(df: pd.DataFrame, per_class_limit: int | None, seed: int) -> pd.DataFrame:
    """用于本地/云端小规模测试时截断样本。"""
    if per_class_limit is None:
        return df.reset_index(drop=True)

    random.seed(seed)
    parts = []
    for _, group in df.groupby("label"):
        parts.append(group.sample(n=min(per_class_limit, len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def oversample_defect_class(df: pd.DataFrame, defect_multiplier: float, seed: int) -> pd.DataFrame:
    """通过复制缺陷件样本提升其在训练集中的占比。"""
    if defect_multiplier <= 1.0:
        return df.reset_index(drop=True)

    defect_df = df[df["label"] == "缺陷件"].copy()
    ok_df = df[df["label"] == "合格件"].copy()
    if defect_df.empty:
        return df.reset_index(drop=True)

    repeat_times = int(defect_multiplier)
    fractional = defect_multiplier - repeat_times

    parts = [ok_df, defect_df]
    for _ in range(max(repeat_times - 1, 0)):
        parts.append(defect_df.copy())
    if fractional > 0:
        extra_size = max(1, int(len(defect_df) * fractional))
        parts.append(defect_df.sample(n=min(extra_size, len(defect_df)), replace=True, random_state=seed))

    merged = pd.concat(parts, ignore_index=True)
    return merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def create_gan_dataloader(
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    per_class_limit: int | None = None,
    drop_cross_split_duplicates: bool = True,
    balance_labels: bool = True,
    defect_multiplier: float = 1.0,
) -> GANDatasetBundle:
    """创建 GAN 训练数据加载器。"""
    df = load_metadata(drop_cross_split_duplicates=drop_cross_split_duplicates)
    train_df = df[df["split"] == "训练集"].copy().reset_index(drop=True)
    if balance_labels:
        train_df = balance_dataframe_by_label(train_df, seed=seed)
    train_df = oversample_defect_class(train_df, defect_multiplier=defect_multiplier, seed=seed)
    train_df = limit_dataframe(train_df, per_class_limit=per_class_limit, seed=seed)

    dataset = UnlabeledCastingDataset(
        train_df,
        transform=build_gan_transforms(image_size=image_size, is_train=True),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )
    return GANDatasetBundle(train_loader=loader, train_df=train_df)
