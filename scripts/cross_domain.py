"""Evaluate one checkpoint on every domain in a manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.trainer import Trainer, build_dataloader
from walltopo_uq.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "full.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    trainer = Trainer(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])

    manifest_value = args.manifest or config["data"].get(f"{args.split}_manifest")
    if not manifest_value:
        raise SystemExit(f"No {args.split}_manifest configured.")
    manifest = Path(manifest_value).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        domains = sorted(
            {
                row.get("domain", "").strip()
                for row in csv.DictReader(handle)
                if row.get("domain", "").strip()
            }
        )
    if len(domains) < 2:
        raise SystemExit(
            "Cross-domain evaluation requires at least two non-empty domain values "
            "in the manifest."
        )

    results: dict[str, dict[str, float]] = {}
    for domain in domains:
        loader = build_dataloader(
            manifest,
            config,
            training=False,
            domain=domain,
        )
        print(f"Evaluating domain={domain} ({len(loader.dataset)} images)")
        results[domain] = trainer.validate(loader, use_tta=args.tta)

    numeric_keys = sorted(
        {
            key
            for metrics in results.values()
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    )
    mean_metrics = {
        key: sum(float(metrics[key]) for metrics in results.values()) / len(results)
        for key in numeric_keys
    }
    output = {
        "manifest": str(manifest),
        "split": args.split,
        "domains": results,
        "macro_mean": mean_metrics,
    }
    output_path = Path(config["output_dir"]) / "cross_domain_metrics.json"
    write_json(output_path, output)
    print(f"Saved cross-domain metrics to {output_path}")


if __name__ == "__main__":
    main()
