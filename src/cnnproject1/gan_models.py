"""DCGAN 模型定义。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def weights_init(module: nn.Module) -> None:
    """按 DCGAN 经验初始化权重。"""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    """支持 64/128/256 输出的生成器。"""

    def __init__(self, latent_dim: int = 128, image_channels: int = 3, feature_maps: int = 64, image_size: int = 128):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size 必须是大于等于 32 的 2 的幂")

        num_upsamples = int(math.log2(image_size)) - 2
        current_channels = feature_maps * (2 ** (num_upsamples - 1))

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, current_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(True),
        ]

        for _ in range(num_upsamples - 1):
            next_channels = max(feature_maps, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True),
                ]
            )
            current_channels = next_channels

        layers.extend(
            [
                nn.ConvTranspose2d(current_channels, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """支持 64/128/256 输入的判别器。"""

    def __init__(self, image_channels: int = 3, feature_maps: int = 64, image_size: int = 128):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size 必须是大于等于 32 的 2 的幂")

        num_downsamples = int(math.log2(image_size)) - 2
        current_channels = feature_maps
        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, current_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(num_downsamples - 1):
            next_channels = min(current_channels * 2, feature_maps * 8)
            layers.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_channels = next_channels

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.network = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).view(-1)

