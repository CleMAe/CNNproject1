"""不依赖 torchvision 的基础图像变换。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance


class Compose:
    """将多个变换按顺序组合。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


@dataclass
class Resize:
    """将图片缩放到固定大小。"""

    size: tuple[int, int]

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, Image.Resampling.BILINEAR)


@dataclass
class RandomHorizontalFlip:
    """按概率进行水平翻转。"""

    p: float = 0.5

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


@dataclass
class RandomBrightnessContrast:
    """进行轻量亮度与对比度扰动。"""

    brightness: float = 0.15
    contrast: float = 0.15

    def __call__(self, image: Image.Image) -> Image.Image:
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        return image


class ToTensor:
    """将 PIL 图片转为 PyTorch 张量。"""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


@dataclass
class Normalize:
    """执行通道归一化。"""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean_tensor) / self.std_tensor


def build_transforms(image_size: int, is_train: bool) -> Compose:
    """根据训练/验证阶段构造变换流水线。"""
    transforms = [Resize((image_size, image_size))]
    if is_train:
        transforms.extend(
            [
                RandomHorizontalFlip(p=0.5),
                RandomBrightnessContrast(brightness=0.18, contrast=0.18),
            ]
        )
    transforms.extend(
        [
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return Compose(transforms)

