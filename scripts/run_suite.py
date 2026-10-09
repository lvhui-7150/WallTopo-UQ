"""Generate and execute a reproducible smoke-level baseline/ablation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config


VARIANTS: dict[str, dict[str, Any]] = {
    "proposed": {
        "base": "configs/smoke.yaml",
        "overrides": {},
    },
    "unet": {
        "base": "configs/unet_smoke.yaml",
        "overrides": {},
    },
    "deeplabv3plus": {
        "base": "configs/deeplabv3plus_smoke.yaml",
        "overrides": {},
    },
    "ablation_no_mdsm": {
        "base": "configs/ablation_no_mdsm_smoke.yaml",
        "overrides": {},
    },
    "ablation_single_delay": {
        "base": "configs/ablation_single_delay_smoke.yaml",
        "overrides": {},
    },
    "ablation_no_topology_tokens": {
        "base": "configs/ablation_no_topology_tokens_smoke.yaml",
        "overrides": {},
    },
    "ablation_no_morphology": {
        "base": "configs/smoke.yaml",
        "overrides": {"mdsm_use_morphology": False},
    },
    "ablation_no_dual_cross": {
        "base": "configs/smoke.yaml",
        "overrides": {"use_dual_cross": False},
    },
    "ablation_no_topology_loss": {
        "base": "configs/smoke.yaml",
        "overrides": {"loss": {"topo": 0.0}},
    },
    "ablation_no_geometry_consistency": {
        "base": "configs/smoke.yaml",
        "overrides": {
            "loss": {
                "width_consistency": 0.0,
                "orientation": 0.0,
            }
        },
    },
    "ablation_no_uncertainty": {
        "base": "configs/smoke.yaml",
        "overrides": {
            "loss": {"evidence": 0.0, "evidence_kl": 0.0},
            "uncertainty": {"calibrate": False, "tta_views": 1},
        },
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _selected_variants(suite: str) -> list[str]:
    if suite == "core":
        return ["proposed", "unet", "deeplabv3plus"]
    if suite == "baselines":
        return ["unet", "deeplabv3plus"]
    if suite == "ablations":
        return [name for name in VARIANTS if name.startswith("ablation_")]
    return list(VARIANTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["core", "baselines", "ablations", "all"],
        default="core",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--name",
        nargs="+",
        choices=list(VARIANTS),
        help="Run only the named variants instead of a predefined suite.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generated_dir = ROOT / "configs" / "generated" / "smoke_matrix"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
    selected = args.name or _selected_variants(args.suite)
    for name in selected:
        specification = VARIANTS[name]
        source = ROOT / str(specification["base"])
        config = load_config(source)
        _deep_update(config, specification["overrides"])
        config["output_dir"] = str(
            (ROOT / "runs" / "smoke_matrix" / name).resolve()
        )
        config["device"] = args.device
        if args.epochs is not None:
            config["epochs"] = args.epochs
        config.pop("config_path", None)
        target = generated_dir / f"{name}.yaml"
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        config_paths.append(target)
        print(f"prepared: {name} -> {target}")

    if args.dry_run:
        return
    for config_path in config_paths:
        checkpoint = (
            ROOT
            / "runs"
            / "smoke_matrix"
            / config_path.stem
            / "checkpoints"
            / "best.pt"
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(config_path),
                "--device",
                args.device,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--device",
                args.device,
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_results.py"),
            "--root",
            str(ROOT / "runs" / "smoke_matrix"),
            "--pattern",
            "test_metrics.json",
            "--output",
            str(ROOT / "runs" / "smoke_matrix" / "aggregated_metrics.csv"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
