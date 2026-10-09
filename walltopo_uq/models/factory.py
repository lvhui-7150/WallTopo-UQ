"""Model registry for the proposed method and baselines."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baselines import DeepLabV3PlusBaseline, SMPBaseline, UNetBaseline
from .walltopo_uq import WallTopoUQ


def build_model(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("model_name", "walltopo_uq")).lower()
    common = {
        "channels": config["channels"],
        "num_severity_classes": int(config["num_severity_classes"]),
    }
    if name in {"walltopo_uq", "walltopo"}:
        return WallTopoUQ(
            **common,
            mdsm_stages=config["mdsm_stages"],
            delays=config["delays"],
            max_attention_tokens=int(config["max_attention_tokens"]),
            mdsm_use_morphology=bool(config.get("mdsm_use_morphology", True)),
            use_dual_cross=bool(config.get("use_dual_cross", True)),
            use_topology_tokens=bool(config.get("use_topology_tokens", True)),
            use_segmentation_shortcut=bool(
                config.get("use_segmentation_shortcut", False)
            ),
            shortcut_gate_bias=float(config.get("shortcut_gate_bias", -1.5)),
        )
    if name in {"unet", "u_net"}:
        return UNetBaseline(**common)
    if name in {"deeplabv3plus", "deeplabv3+"}:
        return DeepLabV3PlusBaseline(
            **common,
            pretrained_backbone=bool(config.get("pretrained_backbone", False)),
        )
    if name.startswith("smp_"):
        return SMPBaseline(
            **common,
            smp_arch=str(config.get("smp_arch", name[4:])),
            encoder_name=str(config.get("encoder_name", "resnet34")),
            encoder_weights=config.get("encoder_weights", "imagenet"),
        )
    raise ValueError(
        f"Unknown model_name={name!r}. Expected walltopo_uq, unet, "
        "deeplabv3plus, or an SMP name such as smp_unetplusplus."
    )
