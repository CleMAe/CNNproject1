"""简化版 EfficientNet 实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BlockConfig:
    """单个 MBConv 阶段的配置。"""

    expand_ratio: int
    kernel_size: int
    stride: int
    in_channels: int
    out_channels: int
    num_layers: int
    se_ratio: float = 0.25


def round_filters(filters: int, width_mult: float, divisor: int = 8) -> int:
    """按照 EfficientNet 规则调整通道数。"""
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_mult: float) -> int:
    """按照 EfficientNet 规则调整层重复次数。"""
    return int(math.ceil(depth_mult * repeats))


class ConvBNAct(nn.Sequential):
    """卷积 + BN + SiLU。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    """SE 注意力模块。"""

    def __init__(self, in_channels: int, se_ratio: float):
        super().__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class StochasticDepth(nn.Module):
    """随机深度，提升大模型训练稳定性。"""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class MBConv(nn.Module):
    """EfficientNet 的核心 MBConv 模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        se_ratio: float,
        drop_path: float,
    ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, stride=1))
        layers.extend(
            [
                ConvBNAct(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim),
                SqueezeExcite(hidden_dim, se_ratio=se_ratio),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self.stochastic_depth(out)
            out = out + x
        return out


class EfficientNet(nn.Module):
    """可配置的 EfficientNet 二分类模型。"""

    def __init__(
        self,
        width_mult: float,
        depth_mult: float,
        dropout: float,
        num_classes: int = 2,
        stochastic_depth_prob: float = 0.2,
    ):
        super().__init__()
        base_blocks = [
            BlockConfig(1, 3, 1, 32, 16, 1),
            BlockConfig(6, 3, 2, 16, 24, 2),
            BlockConfig(6, 5, 2, 24, 40, 2),
            BlockConfig(6, 3, 2, 40, 80, 3),
            BlockConfig(6, 5, 1, 80, 112, 3),
            BlockConfig(6, 5, 2, 112, 192, 4),
            BlockConfig(6, 3, 1, 192, 320, 1),
        ]

        stem_channels = round_filters(32, width_mult)
        self.stem = ConvBNAct(3, stem_channels, kernel_size=3, stride=2)

        total_blocks = sum(round_repeats(config.num_layers, depth_mult) for config in base_blocks)
        block_index = 0
        in_channels = stem_channels
        blocks: list[nn.Module] = []
        for config in base_blocks:
            out_channels = round_filters(config.out_channels, width_mult)
            repeats = round_repeats(config.num_layers, depth_mult)
            for repeat_idx in range(repeats):
                stride = config.stride if repeat_idx == 0 else 1
                current_in = in_channels if repeat_idx == 0 else out_channels
                drop_path = stochastic_depth_prob * block_index / max(total_blocks - 1, 1)
                blocks.append(
                    MBConv(
                        in_channels=current_in,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=config.expand_ratio,
                        kernel_size=config.kernel_size,
                        se_ratio=config.se_ratio,
                        drop_path=drop_path,
                    )
                )
                block_index += 1
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        head_channels = round_filters(1280, width_mult)
        self.head = ConvBNAct(in_channels, head_channels, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(head_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


MODEL_SPECS = {
    "b0": (1.0, 1.0, 0.2),
    "b1": (1.0, 1.1, 0.2),
    "b2": (1.1, 1.2, 0.3),
    "b3": (1.2, 1.4, 0.3),
}


def create_efficientnet(model_name: str, num_classes: int = 2) -> EfficientNet:
    """根据模型名创建 EfficientNet。"""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"不支持的模型名称: {model_name}")
    width_mult, depth_mult, dropout = MODEL_SPECS[model_name]
    return EfficientNet(
        width_mult=width_mult,
        depth_mult=depth_mult,
        dropout=dropout,
        num_classes=num_classes,
    )

