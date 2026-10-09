"""Topology-cross decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, LayerNorm2d


def _pool_tokens(feature: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if feature.shape[-2] * feature.shape[-1] <= max_tokens:
        return feature
    aspect = feature.shape[-2] / max(1, feature.shape[-1])
    height = max(1, int(math.sqrt(max_tokens * aspect)))
    width = max(1, max_tokens // height)
    return F.adaptive_avg_pool2d(feature, (height, width))


class CrossAttention2D(nn.Module):
    """Memory-bounded multi-head cross-attention between two feature maps."""

    def __init__(
        self,
        query_channels: int,
        key_value_channels: int,
        out_channels: int,
        num_heads: int = 4,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads")
        self.query_channels = query_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.max_tokens = max_tokens

        self.query = nn.Conv2d(query_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(key_value_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(key_value_channels, out_channels, kernel_size=1)
        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm = LayerNorm2d(out_channels)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        tensor = tensor.reshape(batch, self.num_heads, self.head_dim, height * width)
        return tensor.transpose(-2, -1)

    def forward(self, query_feature: torch.Tensor, key_value_feature: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(query_feature))
        pooled = _pool_tokens(key_value_feature, self.max_tokens)
        key = self._split_heads(self.key(pooled))
        value = self._split_heads(self.value(pooled))
        attended = F.scaled_dot_product_attention(query, key, value)
        batch, heads, tokens, head_dim = attended.shape
        attended = attended.transpose(1, 2).reshape(batch, heads * head_dim, tokens)
        height, width = query_feature.shape[-2:]
        attended = attended.reshape(batch, self.out_channels, height, width)
        return self.norm(self.output(attended))


class DualCrossAttention(nn.Module):
    """Semantic and detail exchange in both directions."""

    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int,
        out_channels: int,
        num_heads: int = 4,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.semantic_to_detail = CrossAttention2D(
            semantic_channels,
            detail_channels,
            out_channels,
            num_heads=num_heads,
            max_tokens=max_tokens,
        )
        self.detail_to_semantic = CrossAttention2D(
            detail_channels,
            out_channels,
            out_channels,
            num_heads=num_heads,
            max_tokens=max_tokens,
        )
        self.semantic_residual = (
            nn.Identity()
            if semantic_channels == out_channels
            else nn.Conv2d(semantic_channels, out_channels, kernel_size=1)
        )
        self.detail_residual = (
            nn.Identity()
            if detail_channels == out_channels
            else nn.Conv2d(detail_channels, out_channels, kernel_size=1)
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvNormAct(out_channels * 2, out_channels, kernel_size=1)

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
    ) -> torch.Tensor:
        semantic_base = self.semantic_residual(semantic)
        detail_base = self.detail_residual(detail)
        semantic_updated = semantic_base + self.semantic_to_detail(semantic, detail_base)
        detail_updated = detail_base + self.detail_to_semantic(detail_base, semantic_updated)
        if detail_updated.shape[-2:] != semantic_updated.shape[-2:]:
            detail_updated = F.interpolate(
                detail_updated,
                size=semantic_updated.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = self.mix(torch.cat([semantic_updated, detail_updated], dim=1))
        return self.fuse(
            torch.cat(
                [
                    semantic_updated * gate,
                    detail_updated * (1.0 - gate),
                ],
                dim=1,
            )
        )


class TopologyTokenBlock(nn.Module):
    """Pixel-topology bidirectional interaction using four token groups."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        if self.head_dim * num_heads != channels:
            raise ValueError("channels must be divisible by num_heads")
        self.max_tokens = max_tokens

        self.anchor_head = nn.Conv2d(channels, 4, kernel_size=1)
        self.type_embedding = nn.Parameter(torch.zeros(1, 4, channels))
        self.pixel_to_token_query = nn.Linear(channels, channels)
        self.pixel_to_token_key = nn.Conv2d(channels, channels, kernel_size=1)
        self.pixel_to_token_value = nn.Conv2d(channels, channels, kernel_size=1)
        self.pixel_to_token_output = nn.Linear(channels, channels)

        self.token_to_pixel_query = nn.Conv2d(channels, channels, kernel_size=1)
        self.token_to_pixel_key = nn.Linear(channels, channels)
        self.token_to_pixel_value = nn.Linear(channels, channels)
        self.token_to_pixel_output = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = LayerNorm2d(channels)

    @staticmethod
    def _heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        return tensor.reshape(batch, tokens, num_heads, channels // num_heads).transpose(1, 2)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feature.shape
        anchor_logits = self.anchor_head(feature)
        anchor_attention = anchor_logits.flatten(start_dim=2).softmax(dim=-1)
        flat_feature = feature.flatten(start_dim=2)
        tokens = torch.bmm(anchor_attention, flat_feature.transpose(1, 2))
        tokens = tokens + self.type_embedding

        pooled = _pool_tokens(feature, self.max_tokens)
        pooled_height, pooled_width = pooled.shape[-2:]
        pooled_flat = pooled.flatten(start_dim=2).transpose(1, 2)
        query_tokens = self._heads(self.pixel_to_token_query(tokens), self.num_heads)
        key_pixels = self._heads(self.pixel_to_token_key(pooled).flatten(2).transpose(1, 2), self.num_heads)
        value_pixels = self._heads(
            self.pixel_to_token_value(pooled).flatten(2).transpose(1, 2),
            self.num_heads,
        )
        token_update = F.scaled_dot_product_attention(query_tokens, key_pixels, value_pixels)
        token_update = token_update.transpose(1, 2).reshape(batch, 4, channels)
        tokens = tokens + self.pixel_to_token_output(token_update)

        pixel_query = self._heads(
            self.token_to_pixel_query(feature).flatten(2).transpose(1, 2),
            self.num_heads,
        )
        token_key = self._heads(self.token_to_pixel_key(tokens), self.num_heads)
        token_value = self._heads(self.token_to_pixel_value(tokens), self.num_heads)
        pixel_update = F.scaled_dot_product_attention(pixel_query, token_key, token_value)
        pixel_update = pixel_update.transpose(1, 2).reshape(batch, height * width, channels)
        pixel_update = self.token_to_pixel_output(
            pixel_update.transpose(1, 2).reshape(batch, channels, height, width)
        )
        output = self.norm(feature + pixel_update)
        return output, tokens, anchor_logits


class TCDDecoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        num_heads: int = 4,
        max_tokens: int = 512,
        use_dual_cross: bool = True,
        use_topology_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.use_dual_cross = bool(use_dual_cross)
        self.use_topology_tokens = bool(use_topology_tokens)
        self.stage4_to_3 = (
            DualCrossAttention(
                channels[3],
                channels[2],
                channels[2],
                num_heads=num_heads,
                max_tokens=max_tokens,
            )
            if use_dual_cross
            else nn.Identity()
        )
        self.stage3_to_2 = (
            DualCrossAttention(
                channels[2],
                channels[1],
                channels[1],
                num_heads=num_heads,
                max_tokens=max_tokens,
            )
            if use_dual_cross
            else nn.Identity()
        )
        self.stage2_to_1 = (
            DualCrossAttention(
                channels[1],
                channels[0],
                channels[0],
                num_heads=num_heads,
                max_tokens=max_tokens,
            )
            if use_dual_cross
            else nn.Identity()
        )
        self.topology = (
            TopologyTokenBlock(
                channels[0],
                num_heads=num_heads,
                max_tokens=max_tokens,
            )
            if use_topology_tokens
            else nn.Identity()
        )
        self.bypass4_to_3 = (
            ConvNormAct(
                channels[3] + channels[2],
                channels[2],
                kernel_size=1,
            )
            if not use_dual_cross
            else nn.Identity()
        )
        self.bypass3_to_2 = (
            ConvNormAct(
                channels[2] + channels[1],
                channels[1],
                kernel_size=1,
            )
            if not use_dual_cross
            else nn.Identity()
        )
        self.bypass2_to_1 = (
            ConvNormAct(
                channels[1] + channels[0],
                channels[0],
                kernel_size=1,
            )
            if not use_dual_cross
            else nn.Identity()
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def align(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            if other.shape[-2:] == reference.shape[-2:]:
                return other
            return F.interpolate(
                other,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        feature = features[3]
        detail = F.interpolate(
            features[2],
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = (
            self.stage4_to_3(feature, detail)
            if self.use_dual_cross
            else self.bypass4_to_3(
                torch.cat([feature, align(feature, detail)], dim=1)
            )
        )
        feature = F.interpolate(
            feature,
            size=features[2].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = (
            self.stage3_to_2(feature, features[1])
            if self.use_dual_cross
            else self.bypass3_to_2(
                torch.cat([feature, align(feature, features[1])], dim=1)
            )
        )
        feature = F.interpolate(
            feature,
            size=features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = (
            self.stage2_to_1(feature, features[0])
            if self.use_dual_cross
            else self.bypass2_to_1(
                torch.cat([feature, align(feature, features[0])], dim=1)
            )
        )
        if self.use_topology_tokens:
            return self.topology(feature)
        anchor_logits = feature.new_zeros(
            feature.shape[0],
            4,
            feature.shape[-2],
            feature.shape[-1],
        )
        tokens = feature.new_zeros(feature.shape[0], 4, feature.shape[1])
        return feature, tokens, anchor_logits
