"""Train source-domain models and evaluate them on shifted target domains."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config


DATASETS = {
    "crack500": ROOT / "configs" / "crack500.yaml",
    "crackforest": ROOT / "configs" / "crackforest.yaml",
    "dais": ROOT / "configs" / "dais.yaml",
    "deepcrack": ROOT / "configs" / "deepcrack.yaml",
}

SCENARIOS = {
    "pavement_to_masonry": {
        "train": ["crack500", "crackforest", "deepcrack"],
        "val": "dais",
        "tests": ["dais"],
    },
    "masonry_to_pavement": {
        "train": ["dais"],
        "val": "dais",
        "tests": ["crack500", "crackforest", "deepcrack"],
    },
    "leave_crack500": {
        "train": ["crackforest", "dais", "deepcrack"],
        "val": "crackforest",
        "tests": ["crack500"],
    },
    "leave_crackforest": {
        "train": ["crack500", "dais", "deepcrack"],
        "val": "crack500",
        "tests": ["crackforest"],
    },
    "leave_dais": {
        "train": ["crack500", "crackforest", "deepcrack"],
        "val": "crack500",
        "tests": ["dais"],
    },
    "leave_deepcrack": {
        "train": ["crack500", "crackforest", "dais"],
        "val": "crack500",
        "tests": ["deepcrack"],
    },
}

MODEL_OVERRIDES = {
    "unet": {"model_name": "unet"},
    "walltopo_uq": {"model_name": "walltopo_uq"},
    "walltopo_uq_v2": {
        "model_name": "walltopo_uq",
        "use_segmentation_shortcut": True,
        "shortcut_gate_bias": -2.0,
    },
    "smp_unetplusplus": {
        "model_name": "smp_unetplusplus",
        "smp_arch": "unetplusplus",
        "encoder_name": "resnet34",
        "encoder_weights": None,
    },
}


def _manifest_path(dataset: str, split: str) -> Path:
    return ROOT / "data" / "raw" / dataset / f"{split}.csv"


def _load_rows(path: Path, domain: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in ("image", "mask"):
                value = converted.get(key, "")
                if not value:
                    continue
                source = Path(value)
                if not source.is_absolute():
                    source = (path.parent / source).resolve()
                converted[key] = str(source)
            if not converted.get("domain"):
                converted["domain"] = domain
            rows.append(converted)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _combine(sources: list[tuple[str, str]], output: Path) -> Path:
    rows: list[dict[str, str]] = []
    for dataset, split in sources:
        rows.extend(_load_rows(_manifest_path(dataset, split), dataset))
    _write_manifest(rows, output)
    return output.resolve()


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_OVERRIDES), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-prefix", default="reviewer_crossdomain")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    data_root = ROOT / "data" / "processed" / args.run_prefix / args.scenario
    train_manifest = _combine(
        [(dataset, "train") for dataset in scenario["train"]],
        data_root / "train.csv",
    )
    val_manifest = _combine(
        [(scenario["val"], "val")],
        data_root / "val.csv",
    )
    test_manifests = {
        dataset: _combine(
            [(dataset, "test")],
            data_root / f"test_{dataset}.csv",
        )
        for dataset in scenario["tests"]
    }

    config = load_config(DATASETS["crack500"])
    config.update(
        {
            "seed": int(args.seed),
            "device": args.device,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "image_size": int(args.image_size),
        }
    )
    config["data"].update(
        {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_manifest": str(next(iter(test_manifests.values()))),
            "image_size": int(args.image_size),
        }
    )
    config["validation"]["patience"] = int(args.patience)
    config["evaluation"]["threshold_metric"] = "composite"
    config["ema"] = {
        "enabled": True,
        "decay": 0.999,
        "start_epoch": max(3, int(args.epochs) // 10),
    }
    _deep_update(config, MODEL_OVERRIDES[args.model])
    run_dir = (
        ROOT
        / "runs"
        / args.run_prefix
        / args.scenario
        / args.model
        / f"seed_{args.seed}"
    ).resolve()
    config["output_dir"] = str(run_dir)
    config.pop("config_path", None)
    config_path = (
        ROOT
        / "configs"
        / "generated"
        / args.run_prefix
        / args.scenario
        / args.model
        / f"seed_{args.seed}.yaml"
    ).resolve()
    _write_yaml(config_path, config)

    expected_outputs = [
        run_dir / f"test_{dataset}_metrics.json"
        for dataset in scenario["tests"]
    ]
    if args.resume and all(path.exists() for path in expected_outputs):
        print(f"skip completed cross-domain run: {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--device",
            args.device,
        ]
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tune_threshold.py"),
            "--config",
            str(config_path),
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

    target_metrics: dict[str, dict] = {}
    for dataset, manifest in test_manifests.items():
        output = run_dir / f"test_{dataset}_metrics.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--device",
                args.device,
            ]
        )
        with output.open("r", encoding="utf-8") as handle:
            target_metrics[dataset] = json.load(handle)

    macro = {}
    if target_metrics:
        common_keys = set.intersection(
            *(set(metrics) for metrics in target_metrics.values())
        )
        for key in common_keys:
            values = [
                float(metrics[key])
                for metrics in target_metrics.values()
                if isinstance(metrics.get(key), (int, float))
            ]
            if values:
                macro[key] = sum(values) / len(values)
    with (run_dir / "cross_domain_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "scenario": args.scenario,
                "model": args.model,
                "seed": args.seed,
                "targets": target_metrics,
                "macro_mean": macro,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved cross-domain summary to {run_dir}")


if __name__ == "__main__":
    main()
