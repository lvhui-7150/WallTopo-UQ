"""Standard crack-segmentation baselines with a shared task head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DepthwiseSeparableConv, ResidualBlock
from .walltopo_uq import PredictionHeads


class UNetBaseline(nn.Module):
    """Four-level U-Net with the shared WallTopo-UQ multi-task head."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 80, 128),
        num_severity_classes: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if len(channels) != 4:
            raise ValueError("channels must contain four encoder widths")
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    DepthwiseSeparableConv(3, channels[0], stride=2),
                    ResidualBlock(channels[0]),
                ),
                nn.Sequential(
                    DepthwiseSeparableConv(channels[0], channels[1], stride=2),
                    ResidualBlock(channels[1]),
                ),
                nn.Sequential(
                    DepthwiseSeparableConv(channels[1], channels[2], stride=2),
                    ResidualBlock(channels[2]),
                ),
                nn.Sequential(
                    DepthwiseSeparableConv(channels[2], channels[3], stride=2),
                    ResidualBlock(channels[3]),
                ),
            ]
        )
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
        self.heads = PredictionHeads(channels[0], num_severity_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        features = []
        value = image
        for encoder in self.encoders:
            value = encoder(value)
            features.append(value)

        value = self.decode3(
            torch.cat(
                [
                    self.up3(features[3]),
                    features[2],
                ],
                dim=1,
            )
        )
        value = self.decode2(
            torch.cat(
                [
                    self.up2(value),
                    features[1],
                ],
                dim=1,
            )
        )
        value = self.decode1(
            torch.cat(
                [
                    self.up1(value),
                    features[0],
                ],
                dim=1,
            )
        )
        value = self.refine(value)
        outputs = self.heads(value, input_size)
        value = F.interpolate(
            value,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        token = F.adaptive_avg_pool2d(value, 1).flatten(1)
        outputs["topology_tokens"] = token[:, None, :].repeat(1, 4, 1)
        outputs["topology_anchor_logits"] = value.new_zeros(
            value.shape[0],
            4,
            *input_size,
        )
        return outputs


class _DenseNode(nn.Module):
    """One nested skip node used by the dependency-free U-Net++ fallback."""

    def __init__(
        self,
        skip_channels: Sequence[int],
        up_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.fuse = ConvNormAct(
            int(sum(skip_channels)) + int(up_channels),
            int(out_channels),
            kernel_size=1,
        )
        self.refine = ResidualBlock(int(out_channels))

    def forward(
        self,
        skips: Sequence[torch.Tensor],
        upsampled: torch.Tensor,
    ) -> torch.Tensor:
        return self.refine(self.fuse(torch.cat([upsampled, *skips], dim=1)))


class UNetPlusPlusBaseline(nn.Module):
    """Dependency-free nested U-Net++ baseline with the shared task heads."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 80, 128),
        num_severity_classes: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        channels = [int(value) for value in channels]
        if len(channels) != 4:
            raise ValueError("channels must contain four encoder widths")
        c0, c1, c2, c3 = channels

        self.stem = nn.Sequential(
            ConvNormAct(3, c0, kernel_size=3),
            ResidualBlock(c0),
        )
        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(c0, c1, stride=2),
            ResidualBlock(c1),
        )
        self.down2 = nn.Sequential(
            DepthwiseSeparableConv(c1, c2, stride=2),
            ResidualBlock(c2),
        )
        self.down3 = nn.Sequential(
            DepthwiseSeparableConv(c2, c3, stride=2),
            ResidualBlock(c3),
        )

        self.node_01 = _DenseNode([c0], c1, c1)
        self.node_11 = _DenseNode([c1], c2, c2)
        self.node_21 = _DenseNode([c2], c3, c3)
        self.node_02 = _DenseNode([c0, c1], c2, c1)
        self.node_12 = _DenseNode([c1, c2], c3, c2)
        self.node_03 = _DenseNode([c0, c1, c1], c2, c3)
        self.refine = ConvNormAct(c3, c0, kernel_size=3)
        self.heads = PredictionHeads(c0, num_severity_classes)

    @staticmethod
    def _upsample_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        x00 = self.stem(image)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)

        x01 = self.node_01([x00], self._upsample_like(x10, x00))
        x11 = self.node_11([x10], self._upsample_like(x20, x10))
        x21 = self.node_21([x20], self._upsample_like(x30, x20))
        x02 = self.node_02(
            [x00, x01],
            self._upsample_like(x11, x00),
        )
        x12 = self.node_12(
            [x10, x11],
            self._upsample_like(x21, x10),
        )
        x03 = self.node_03(
            [x00, x01, x02],
            self._upsample_like(x12, x00),
        )

        value = self.refine(x03)
        outputs = self.heads(value, input_size)
        token = F.adaptive_avg_pool2d(value, 1).flatten(1)
        outputs["topology_tokens"] = token[:, None, :].repeat(1, 4, 1)
        outputs["topology_anchor_logits"] = value.new_zeros(
            value.shape[0],
            4,
            *input_size,
        )
        return outputs


class DeepLabV3PlusBaseline(nn.Module):
    """Torchvision DeepLabv3+ with optional ImageNet initialization."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 80, 128),
        num_severity_classes: int = 5,
        pretrained_backbone: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50

        weights_backbone = (
            ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        )
        self.network = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=1,
            aux_loss=True,
        )
        head_channels = int(channels[0])
        self.detail = nn.Sequential(
            ConvNormAct(3, head_channels // 2, kernel_size=3, stride=2),
            ConvNormAct(head_channels // 2, head_channels, kernel_size=3, stride=2),
        )
        self.task_feature = ConvNormAct(head_channels + 1, head_channels, kernel_size=3)
        self.heads = PredictionHeads(head_channels, num_severity_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        network_output = self.network(image)
        mask_logits = network_output["out"]
        if mask_logits.shape[-2:] != input_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        detail = self.detail(image)
        probability = torch.sigmoid(mask_logits)
        task_probability = F.interpolate(
            probability,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        task_feature = self.task_feature(
            torch.cat([detail, task_probability], dim=1)
        )
        outputs = self.heads(task_feature, input_size)
        outputs["mask_logits"] = mask_logits
        outputs["mask_prob"] = probability
        token = F.adaptive_avg_pool2d(task_feature, 1).flatten(1)
        outputs["topology_tokens"] = token[:, None, :].repeat(1, 4, 1)
        outputs["topology_anchor_logits"] = task_feature.new_zeros(
            task_feature.shape[0],
            4,
            *input_size,
        )
        return outputs


class SMPBaseline(nn.Module):
    """Optional segmentation-models-pytorch baseline with the shared task heads."""

    _ARCHITECTURES = {
        "unet": "Unet",
        "unetplusplus": "UnetPlusPlus",
        "deeplabv3plus": "DeepLabV3Plus",
        "fpn": "FPN",
        "pan": "PAN",
        "manet": "MAnet",
        "linknet": "Linknet",
    }

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 80, 128),
        num_severity_classes: int = 5,
        smp_arch: str = "unetplusplus",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        **_: object,
    ) -> None:
        super().__init__()
        self._fallback: nn.Module | None = None
        architecture = self._ARCHITECTURES.get(str(smp_arch).lower())
        if architecture is None:
            raise ValueError(
                f"Unsupported smp_arch={smp_arch!r}. "
                f"Expected one of {sorted(self._ARCHITECTURES)}."
            )
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            if architecture == "UnetPlusPlus":
                self._fallback = UNetPlusPlusBaseline(
                    channels=channels,
                    num_severity_classes=num_severity_classes,
                )
                return
            raise ImportError(
                "SMP baselines require segmentation-models-pytorch. "
                "Install it with: python -m pip install "
                "segmentation-models-pytorch timm"
            ) from error
        model_class = getattr(smp, architecture)
        self.network = model_class(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        head_channels = int(channels[0])
        self.detail = nn.Sequential(
            ConvNormAct(3, head_channels // 2, kernel_size=3, stride=2),
            ConvNormAct(head_channels // 2, head_channels, kernel_size=3, stride=2),
        )
        self.task_feature = ConvNormAct(
            head_channels + 1,
            head_channels,
            kernel_size=3,
        )
        self.heads = PredictionHeads(head_channels, num_severity_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._fallback is not None:
            return self._fallback(image)
        input_size = image.shape[-2:]
        mask_logits = self.network(image)
        if mask_logits.shape[-2:] != input_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        detail = self.detail(image)
        probability = torch.sigmoid(mask_logits)
        task_probability = F.interpolate(
            probability,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        task_feature = self.task_feature(
            torch.cat([detail, task_probability], dim=1)
        )
        outputs = self.heads(task_feature, input_size)
        outputs["mask_logits"] = mask_logits
        outputs["mask_prob"] = probability
        token = F.adaptive_avg_pool2d(task_feature, 1).flatten(1)
        outputs["topology_tokens"] = token[:, None, :].repeat(1, 4, 1)
        outputs["topology_anchor_logits"] = task_feature.new_zeros(
            task_feature.shape[0],
            4,
            *input_size,
        )
        return outputs
