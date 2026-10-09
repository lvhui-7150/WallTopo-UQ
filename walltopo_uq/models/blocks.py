"""Shared convolutional blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalization_groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class LayerNorm2d(nn.Module):
    """Layer normalization over channels for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_normalization_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvNormAct(channels, hidden, kernel_size=1),
            ConvNormAct(hidden, hidden, kernel_size=3, groups=hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class GaussianHighPass(nn.Module):
    """Fixed Gaussian high-pass operator from the local morphology branch."""

    def __init__(self, channels: int = 3, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        coordinates = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coordinates.pow(2)) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel[None, None].repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blurred = F.conv2d(x, self.kernel, padding=self.padding, groups=x.shape[1])
        return x - blurred


class LocalEdgeBranch(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.high_pass = GaussianHighPass(channels=3)
        self.network = nn.Sequential(
            ConvNormAct(6, out_channels, kernel_size=3),
            ResidualBlock(out_channels),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_frequency = self.high_pass(x)
        return self.network(torch.cat([x, high_frequency], dim=1))
