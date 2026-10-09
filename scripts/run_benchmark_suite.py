"""Generate and execute a reproducible full dataset/model/seed benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config


DATASET_CONFIGS = {
    "dais": "configs/dais.yaml",
    "deepcrack": "configs/deepcrack.yaml",
    "crack500": "configs/crack500.yaml",
    "crackforest": "configs/crackforest.yaml",
}

MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "walltopo_uq": {},
    "walltopo_uq_v2": {
        "model_name": "walltopo_uq",
        "use_segmentation_shortcut": True,
        "shortcut_gate_bias": -2.0,
        "loss": {
            "topo": 0.5,
            "geo": 0.5,
            "ordinal": 0.1,
            "evidence": 0.15,
            "consistency": 0.05,
        },
        "validation": {
            "metric": "dice",
            "patience": 20,
        },
    },
    "unet": {"model_name": "unet"},
    "unet_cldice": {
        "model_name": "unet",
        "loss": {
            "seg": 1.0,
            "topo": 0.5,
            "geo": 0.0,
            "ordinal": 0.0,
            "consistency": 0.0,
            "evidence": 0.0,
            "cl_dice": 1.0,
            "node": 0.0,
            "width": 0.0,
            "width_consistency": 0.0,
            "orientation": 0.0,
            "evidence_kl": 0.0,
        },
        "uncertainty": {"calibrate": False, "tta_views": 1},
    },
    "unet_connectivity": {
        "model_name": "unet",
        "loss": {
            "seg": 1.0,
            "topo": 0.7,
            "geo": 0.0,
            "ordinal": 0.0,
            "consistency": 0.0,
            "evidence": 0.0,
            "cl_dice": 1.0,
            "node": 1.0,
            "width": 0.0,
            "width_consistency": 0.0,
            "orientation": 0.0,
            "evidence_kl": 0.0,
        },
        "uncertainty": {"calibrate": False, "tta_views": 1},
    },
    "deeplabv3plus": {
        "model_name": "deeplabv3plus",
        "pretrained_backbone": True,
    },
    "smp_unetplusplus": {
        "model_name": "smp_unetplusplus",
        "smp_arch": "unetplusplus",
        "encoder_name": "resnet34",
        "encoder_weights": None,
    },
    "ablation_no_mdsm": {"mdsm_stages": []},
    "ablation_single_delay": {"delays": [1]},
    "ablation_no_morphology": {"mdsm_use_morphology": False},
    "ablation_no_topology_tokens": {"use_topology_tokens": False},
    "ablation_no_dual_cross": {"use_dual_cross": False},
    "ablation_no_topology_loss": {"loss": {"topo": 0.0}},
    "ablation_no_geometry_consistency": {
        "loss": {
            "width_consistency": 0.0,
            "orientation": 0.0,
        }
    },
    "ablation_no_uncertainty": {
        "loss": {"evidence": 0.0, "evidence_kl": 0.0},
        "uncertainty": {"calibrate": False, "tta_views": 1},
    },
}

CORE_MODELS = [
    "walltopo_uq",
    "unet",
    "deeplabv3plus",
]
ABLATION_MODELS = [
    name for name in MODEL_VARIANTS if name.startswith("ablation_")
]

PROFILES = {
    "smoke": {
        "epochs": 2,
        "patience": 2,
        "validation_interval": 1,
        "num_workers": 0,
        "ema": False,
    },
    "pilot": {
        "epochs": 30,
        "patience": 8,
        "validation_interval": 1,
        "num_workers": 4,
        "ema": True,
    },
    "full": {
        "epochs": 120,
        "patience": 25,
        "validation_interval": 1,
        "num_workers": 4,
        "ema": True,
    },
    "paper": {
        "epochs": 150,
        "patience": 30,
        "validation_interval": 1,
        "num_workers": 4,
        "ema": True,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _absolute_manifest_paths(config: dict[str, Any]) -> None:
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        value = Path(str(config["data"][key]))
        if not value.is_absolute():
            config["data"][key] = str((ROOT / value).resolve())


def _selected_variants(args: argparse.Namespace) -> list[str]:
    if args.models:
        selected = list(dict.fromkeys(args.models))
    elif args.suite == "core":
        selected = list(CORE_MODELS)
    elif args.suite == "ablations":
        selected = list(ABLATION_MODELS)
    else:
        selected = list(CORE_MODELS) + list(ABLATION_MODELS)
    unknown = sorted(set(selected) - set(MODEL_VARIANTS))
    if unknown:
        raise SystemExit(
            f"Unknown model variants: {unknown}. "
            f"Available: {sorted(MODEL_VARIANTS)}"
        )
    return selected


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["core", "ablations", "all"],
        default="core",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="pilot",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=["dais"],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-prefix", default="benchmark")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--collect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect all runs after this invocation finishes.",
    )
    parser.add_argument(
        "--pretrained-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet initialization for the DeepLabV3+ baseline.",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    variants = _selected_variants(args)
    generated_root = ROOT / "configs" / "generated" / args.run_prefix
    runs_root = ROOT / "runs" / args.run_prefix
    jobs: list[dict[str, Any]] = []

    for dataset in args.datasets:
        base_path = ROOT / DATASET_CONFIGS[dataset]
        for variant in variants:
            for seed in args.seeds:
                config = load_config(base_path)
                config.update(
                    {
                        "seed": int(seed),
                        "device": args.device,
                        "epochs": int(args.epochs or profile["epochs"]),
                        "num_workers": int(
                            args.num_workers
                            if args.num_workers is not None
                            else profile["num_workers"]
                        ),
                    }
                )
                config["validation"]["interval"] = int(
                    profile["validation_interval"]
                )
                config["validation"]["patience"] = int(profile["patience"])
                config["evaluation"]["threshold_metric"] = "composite"
                config["ema"] = {
                    "enabled": bool(profile["ema"]),
                    "decay": 0.999,
                    "start_epoch": max(3, int(config["epochs"]) // 10),
                }
                if args.batch_size is not None:
                    config["batch_size"] = int(args.batch_size)
                if args.image_size is not None:
                    config["image_size"] = int(args.image_size)
                    config["data"]["image_size"] = int(args.image_size)
                _deep_update(config, MODEL_VARIANTS[variant])
                if variant == "deeplabv3plus" and not args.pretrained_baseline:
                    config["pretrained_backbone"] = False
                _absolute_manifest_paths(config)

                run_dir = (
                    runs_root / dataset / variant / f"seed_{seed}"
                ).resolve()
                config["output_dir"] = str(run_dir)
                config.pop("config_path", None)
                config_path = (
                    generated_root
                    / dataset
                    / variant
                    / f"seed_{seed}.yaml"
                ).resolve()
                _write_yaml(config_path, config)
                jobs.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": int(seed),
                        "config_path": str(config_path),
                        "output_dir": str(run_dir),
                        "model_name": str(config.get("model_name", "walltopo_uq")),
                        "pretrained_backbone": bool(
                            config.get("pretrained_backbone", False)
                        ),
                    }
                )

    print(f"Prepared {len(jobs)} benchmark jobs.")
    if args.dry_run:
        for job in jobs:
            print(
                f"{job['dataset']:<10} {job['variant']:<30} "
                f"seed={job['seed']} -> {job['output_dir']}"
            )
        return

    completed = 0
    for job in jobs:
        run_dir = Path(job["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "run_metadata.json"
        metadata = {
            **job,
            "profile": args.profile,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        metrics_path = run_dir / "test_metrics.json"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if args.resume and metrics_path.exists():
            print(f"skip completed: {run_dir}")
            completed += 1
            continue

        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(job["config_path"]),
                "--device",
                args.device,
            ]
        )
        if args.tune_threshold:
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "tune_threshold.py"),
                    "--config",
                    str(job["config_path"]),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "val",
                    "--metric",
                    "composite",
                    "--device",
                    args.device,
                ]
            )
        if args.evaluate:
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate.py"),
                    "--config",
                    str(job["config_path"]),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "test",
                    "--device",
                    args.device,
                ]
            )
        completed += 1
        print(f"completed {completed}/{len(jobs)}: {run_dir}")

    if args.collect:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "collect_benchmark_results.py"),
                "--root",
                str(runs_root),
                "--output",
                str(ROOT / "results" / args.run_prefix),
            ]
        )


if __name__ == "__main__":
    main()
