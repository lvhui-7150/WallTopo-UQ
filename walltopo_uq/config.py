"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "output_dir": "runs/default",
    "num_workers": 0,
    "amp": True,
    "epochs": 100,
    "batch_size": 4,
    "grad_accum_steps": 1,
    "model_name": "walltopo_uq",
    "pretrained_backbone": False,
    "image_size": 256,
    "channels": [24, 48, 80, 128],
    "mdsm_stages": [3],
    "delays": [1, 2, 4, 8],
    "topology_tokens": 4,
    "max_attention_tokens": 512,
    "num_severity_classes": 5,
    "severity_aux_weight": 0.0,
    "mdsm_use_morphology": True,
    "use_dual_cross": True,
    "use_topology_tokens": True,
    "use_segmentation_shortcut": False,
    "shortcut_gate_bias": -1.5,
    "ema": {
        "enabled": False,
        "decay": 0.999,
        "start_epoch": 10,
    },
    "evaluation": {
        "threshold": 0.5,
        "threshold_metric": "composite",
    },
    "uncertainty": {
        "calibrate": True,
        "tta_views": 4,
        "max_calibration_samples": 128,
        "target_coverage": 0.8,
        "target_error": None,
    },
    "optimizer": {
        "lr": 2.0e-4,
        "weight_decay": 1.0e-4,
    },
    "scheduler": {
        "name": "cosine",
        "min_lr": 1.0e-6,
    },
    "loss": {
        "seg": 1.0,
        "topo": 0.7,
        "geo": 0.7,
        "ordinal": 0.2,
        "consistency": 0.1,
        "evidence": 0.25,
        "cl_dice": 1.0,
        "node": 1.0,
        "width": 1.0,
        "width_consistency": 0.25,
        "orientation": 0.25,
        "evidence_kl": 0.05,
    },
    "data": {
        "train_manifest": "data/synthetic/train/manifest.csv",
        "val_manifest": "data/synthetic/val/manifest.csv",
        "test_manifest": "data/synthetic/test/manifest.csv",
        "image_size": 256,
        "cache_derived": True,
        "use_skeleton": True,
        "use_endpoints": True,
        "use_junctions": True,
        "use_orientation": True,
        "use_width": True,
        "use_severity": False,
        "foreground_ratio": 0.65,
        "max_rotation": 10.0,
    },
    "validation": {
        "interval": 1,
        "metric": "composite",
        "patience": 15,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and merge it with package defaults."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_update(DEFAULTS, loaded)
    config["config_path"] = str(config_path)
    config["output_dir"] = str((config_path.parent.parent / config["output_dir"]).resolve())
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if len(config["channels"]) != 4:
        raise ValueError("channels must contain exactly four encoder widths")
    if any(stage not in (0, 1, 2, 3) for stage in config["mdsm_stages"]):
        raise ValueError("mdsm_stages must be indices in [0, 3]")
    if config["image_size"] % 16 != 0:
        raise ValueError("image_size must be divisible by 16")
    if config["topology_tokens"] != 4:
        raise ValueError("The manuscript defines four topology token groups")
    if config["delays"] != sorted(config["delays"]):
        raise ValueError("delays must be sorted")
