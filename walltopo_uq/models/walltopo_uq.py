"""End-to-end WallTopo-UQ model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DepthwiseSeparableConv, LocalEdgeBranch, ResidualBlock
from .mdsm import MDSMBlock
from .tcd import TCDDecoder


class UNetShortcutDecoder(nn.Module):
    """A lightweight U-Net decoder used as a segmentation-stability shortcut."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        channels = list(channels)
        self.up3 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            kernel_size=2,
            stride=2,
        )
        self.decode3 = nn.Sequential(
            ConvNormAct(channels[2] * 2, channels[2], kernel_size=1),
            ResidualBlock(channels[2]),
        )
        self.up2 = nn.ConvTranspose2d(
            channels[2],
            channels[1],
            kernel_size=2,
            stride=2,
        )
        self.decode2 = nn.Sequential(
            ConvNormAct(channels[1] * 2, channels[1], kernel_size=1),
            ResidualBlock(channels[1]),
        )
        self.up1 = nn.ConvTranspose2d(
            channels[1],
            channels[0],
            kernel_size=2,
            stride=2,
        )
        self.decode1 = nn.Sequential(
            ConvNormAct(channels[0] * 2, channels[0], kernel_size=1),
            ResidualBlock(channels[0]),
        )
        self.refine = ConvNormAct(channels[0], channels[0], kernel_size=3)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        value = self.decode3(
            torch.cat([self.up3(features[3]), features[2]], dim=1)
        )
        value = self.decode2(
            torch.cat([self.up2(value), features[1]], dim=1)
        )
        value = self.decode1(
            torch.cat([self.up1(value), features[0]], dim=1)
        )
        return self.refine(value)


class PredictionHeads(nn.Module):
    def __init__(
        self,
        channels: int,
        num_severity_classes: int,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            ResidualBlock(channels),
        )
        self.mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.skeleton = nn.Conv2d(channels, 1, kernel_size=1)
        self.endpoint = nn.Conv2d(channels, 1, kernel_size=1)
        self.junction = nn.Conv2d(channels, 1, kernel_size=1)
        self.orientation = nn.Conv2d(channels, 2, kernel_size=1)
        self.width = nn.Conv2d(channels, 1, kernel_size=1)
        self.evidence = nn.Conv2d(channels, 2, kernel_size=1)
        self.severity = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, num_severity_classes - 1),
        )
        nn.init.constant_(self.mask.bias, 0.0)
        nn.init.constant_(self.skeleton.bias, -1.0)
        nn.init.constant_(self.endpoint.bias, -2.0)
        nn.init.constant_(self.junction.bias, -2.0)
        nn.init.constant_(self.width.bias, -2.5)

    def forward(
        self,
        feature: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        feature = self.shared(feature)
        mask_logits = self.mask(feature)
        probability = torch.sigmoid(mask_logits)
        denominator = probability.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        pooled = (probability * feature).sum(dim=(2, 3), keepdim=True) / denominator
        pooled = pooled.flatten(1)

        outputs = {
            "mask_logits": mask_logits,
            "mask_prob": probability,
            "skeleton_logits": self.skeleton(feature),
            "endpoint_logits": self.endpoint(feature),
            "junction_logits": self.junction(feature),
            "orientation": F.normalize(self.orientation(feature), dim=1, eps=1.0e-6),
            "width": F.softplus(self.width(feature)),
            "evidence": torch.nn.functional.softplus(self.evidence(feature)) + 1.0,
            "severity_logits": self.severity(pooled),
        }
        for key in (
            "mask_logits",
            "skeleton_logits",
            "endpoint_logits",
            "junction_logits",
            "orientation",
            "width",
            "evidence",
        ):
            if outputs[key].shape[-2:] != output_size:
                outputs[key] = F.interpolate(
                    outputs[key],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
        outputs["mask_prob"] = torch.sigmoid(outputs["mask_logits"])
        return outputs


class WallTopoUQ(nn.Module):
    """MDSM encoder, TCD decoder, MGE heads, and EUS evidence head."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 80, 128),
        mdsm_stages: Sequence[int] = (2, 3),
        delays: Sequence[int] = (1, 2, 4, 8),
        max_attention_tokens: int = 512,
        num_severity_classes: int = 5,
        num_heads: int = 4,
        mdsm_use_morphology: bool = True,
        use_dual_cross: bool = True,
        use_topology_tokens: bool = True,
        use_segmentation_shortcut: bool = False,
        shortcut_gate_bias: float = -1.5,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if len(channels) != 4:
            raise ValueError("channels must contain four encoder widths")
        self.channels = channels
        self.mdsm_stages = set(int(stage) for stage in mdsm_stages)

        self.local_stem = LocalEdgeBranch(channels[0])
        self.local_fuse = ConvNormAct(channels[0] * 2, channels[0], kernel_size=1)
        self.stem = ConvNormAct(3, channels[0], kernel_size=3, stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(channels[0], channels[1], stride=2),
            ResidualBlock(channels[1]),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(channels[1], channels[2], stride=2),
            ResidualBlock(channels[2]),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(channels[2], channels[3], stride=2),
            ResidualBlock(channels[3]),
        )

        self.local_down2 = DepthwiseSeparableConv(channels[0], channels[1], stride=2)
        self.local_down3 = DepthwiseSeparableConv(channels[1], channels[2], stride=2)
        self.local_down4 = DepthwiseSeparableConv(channels[2], channels[3], stride=2)
        self.mdsm = nn.ModuleDict(
            {
                str(stage): MDSMBlock(
                    channels[stage],
                    delays=delays,
                    use_morphology=mdsm_use_morphology,
                )
                for stage in self.mdsm_stages
            }
        )
        self.decoder = TCDDecoder(
            channels,
            num_heads=num_heads,
            max_tokens=max_attention_tokens,
            use_dual_cross=use_dual_cross,
            use_topology_tokens=use_topology_tokens,
        )
        self.use_segmentation_shortcut = bool(use_segmentation_shortcut)
        self.segmentation_shortcut = (
            UNetShortcutDecoder(channels)
            if self.use_segmentation_shortcut
            else None
        )
        self.shortcut_gate = (
            nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1)
            if self.use_segmentation_shortcut
            else None
        )
        if self.shortcut_gate is not None:
            nn.init.zeros_(self.shortcut_gate.weight)
            nn.init.constant_(self.shortcut_gate.bias, float(shortcut_gate_bias))
        self.heads = PredictionHeads(channels[0], num_severity_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        local = self.local_stem(image)
        feature1 = self.stem(image)
        local_decoder = F.adaptive_avg_pool2d(local, feature1.shape[-2:])
        feature1 = self.local_fuse(torch.cat([feature1, local_decoder], dim=1))
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)

        local2 = self.local_down2(local)
        local3 = self.local_down3(local2)
        local4 = self.local_down4(local3)
        features = [feature1, feature2, feature3, feature4]
        local_features = [local, local2, local3, local4]
        for stage in sorted(self.mdsm_stages):
            features[stage] = self.mdsm[str(stage)](
                features[stage],
                local_features[stage],
            )

        decoder_feature, tokens, anchors = self.decoder(features)
        if self.segmentation_shortcut is not None:
            shortcut_feature = self.segmentation_shortcut(features)
            if shortcut_feature.shape[-2:] != decoder_feature.shape[-2:]:
                shortcut_feature = F.interpolate(
                    shortcut_feature,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate = torch.sigmoid(
                self.shortcut_gate(
                    torch.cat([decoder_feature, shortcut_feature], dim=1)
                )
            )
            decoder_feature = shortcut_feature + gate * (
                decoder_feature - shortcut_feature
            )
        outputs = self.heads(decoder_feature, input_size)
        outputs["topology_tokens"] = tokens
        outputs["topology_anchor_logits"] = anchors
        return outputs
