"""更高级的条件式残差 GAN 模型。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_conv2d(*args, **kwargs) -> nn.Module:
    """带谱归一化的卷积层。"""
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


def spectral_linear(*args, **kwargs) -> nn.Module:
    """带谱归一化的线性层。"""
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


class SelfAttention(nn.Module):
    """自注意力模块，增强全局结构建模。"""

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 8, 8)
        self.query = spectral_conv2d(channels, reduced, kernel_size=1)
        self.key = spectral_conv2d(channels, reduced, kernel_size=1)
        self.value = spectral_conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = self.query(x).view(batch, -1, height * width).transpose(1, 2)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, channels, height * width)
        out = torch.bmm(value, attention.transpose(1, 2)).view(batch, channels, height, width)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    """条件批归一化。"""

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(embed_dim, num_features)
        self.beta = nn.Linear(embed_dim, num_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(y).unsqueeze(-1).unsqueeze(-1)
        return out * (1 + gamma) + beta


class GenResBlock(nn.Module):
    """生成器残差块。"""

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, embed_dim)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, embed_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.shortcut(residual)

        out = self.cbn1(x, y)
        out = self.activation(out)
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.cbn2(out, y)
        out = self.activation(out)
        out = self.conv2(out)
        return out + residual


class DiscResBlock(nn.Module):
    """判别器残差块。"""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.conv1 = spectral_conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = spectral_conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = spectral_conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        if self.downsample:
            residual = F.avg_pool2d(residual, kernel_size=2)

        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, kernel_size=2)
        return out + residual


class ConditionalResGenerator(nn.Module):
    """条件式残差生成器。"""

    def __init__(
        self,
        latent_dim: int = 256,
        num_classes: int = 2,
        base_channels: int = 96,
        image_channels: int = 3,
        image_size: int = 128,
        class_embed_dim: int = 128,
    ):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size 必须是大于等于 32 的 2 的幂")
        self.image_size = image_size
        self.init_size = 4
        num_upsamples = int(math.log2(image_size // self.init_size))
        channel_schedule = []
        current = base_channels * (2 ** max(num_upsamples - 1, 0))
        for _ in range(num_upsamples):
            channel_schedule.append(current)
            current = max(base_channels, current // 2)

        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.project = nn.Linear(latent_dim + class_embed_dim, channel_schedule[0] * self.init_size * self.init_size)

        blocks = []
        in_out = list(zip(channel_schedule[:-1], channel_schedule[1:])) + [(channel_schedule[-1], base_channels)]
        for idx, (in_channels, out_channels) in enumerate(in_out):
            blocks.append(GenResBlock(in_channels, out_channels, class_embed_dim))
            if idx == len(in_out) // 2:
                blocks.append(SelfAttention(out_channels))
        self.blocks = nn.ModuleList(blocks)
        self.bn = ConditionalBatchNorm2d(base_channels, class_embed_dim)
        self.activation = nn.ReLU(inplace=False)
        self.to_rgb = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_vec = self.class_embedding(labels)
        latent = torch.cat([noise.view(noise.size(0), -1), class_vec], dim=1)
        out = self.project(latent)
        out = out.view(noise.size(0), -1, self.init_size, self.init_size)
        for block in self.blocks:
            if isinstance(block, GenResBlock):
                out = block(out, class_vec)
            else:
                out = block(out)
        out = self.bn(out, class_vec)
        out = self.activation(out)
        out = torch.tanh(self.to_rgb(out))
        return out


class ProjectionDiscriminator(nn.Module):
    """带投影头的条件式判别器。"""

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 96,
        image_channels: int = 3,
        image_size: int = 128,
    ):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size 必须是大于等于 32 的 2 的幂")
        num_downsamples = int(math.log2(image_size)) - 2
        channels = [base_channels]
        for _ in range(num_downsamples):
            channels.append(min(channels[-1] * 2, base_channels * 8))

        self.from_rgb = spectral_conv2d(image_channels, channels[0], kernel_size=3, padding=1)
        blocks = []
        for idx in range(len(channels) - 1):
            blocks.append(DiscResBlock(channels[idx], channels[idx + 1], downsample=True))
            if idx == len(channels) // 2:
                blocks.append(SelfAttention(channels[idx + 1]))
        blocks.append(DiscResBlock(channels[-1], channels[-1], downsample=False))
        self.blocks = nn.Sequential(*blocks)
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.fc = spectral_linear(channels[-1], 1)
        self.class_embedding = nn.Embedding(num_classes, channels[-1])
        nn.init.normal_(self.class_embedding.weight, 0.0, 0.02)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.from_rgb(images)
        out = self.blocks(out)
        out = self.activation(out)
        features = torch.sum(out, dim=(2, 3))
        logits = self.fc(features).view(-1)
        projection = torch.sum(self.class_embedding(labels) * features, dim=1)
        return logits + projection
