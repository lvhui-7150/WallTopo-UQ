"""Evaluate a WallTopo-UQ checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.trainer import Trainer, build_dataloader
from walltopo_uq.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the manifest configured for the selected split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the metrics output path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-selected mask threshold.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate with the calibrated geometric-TTA report uncertainty.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    trainer = Trainer(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])
    if args.threshold is not None:
        trainer.mask_threshold = float(args.threshold)
    manifest_key = f"{args.split}_manifest"
    manifest = args.manifest or config["data"].get(manifest_key)
    if not manifest:
        raise SystemExit(f"No {manifest_key} configured.")
    loader = build_dataloader(manifest, config, training=False)
    metrics = trainer.validate(
        loader,
        use_tta=args.tta,
        threshold=trainer.mask_threshold,
    )
    output_path = args.output or (
        Path(config["output_dir"]) / f"{args.split}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, metrics)
    print(metrics)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
