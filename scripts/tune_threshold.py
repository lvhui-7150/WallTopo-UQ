"""Select a mask threshold on a validation split and save it for evaluation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.metrics import composite_score, validate_batch
from walltopo_uq.trainer import Trainer, build_dataloader
from walltopo_uq.utils import resolve_device, write_json


def _mean_metrics(per_image: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in per_image for key in item})
    metrics = {
        key: float(np.mean([item.get(key, 0.0) for item in per_image]))
        for key in keys
    }
    metrics["composite"] = composite_score(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--metric", choices=["dice", "composite"], default="composite")
    parser.add_argument("--min", dest="minimum", type=float, default=0.25)
    parser.add_argument("--max", dest="maximum", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    trainer = Trainer(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.model.eval()

    manifest = config["data"].get(f"{args.split}_manifest")
    if not manifest:
        raise SystemExit(f"No {args.split}_manifest configured.")
    loader = build_dataloader(manifest, config, training=False)
    thresholds = np.linspace(
        float(args.minimum),
        float(args.maximum),
        max(2, int(args.steps)),
    )
    results: dict[float, list[dict[str, float]]] = defaultdict(list)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="threshold-search", leave=False):
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            if args.tta:
                outputs, _ = trainer._forward_tta(batch["image"])
            else:
                outputs = trainer.model(batch["image"])
            for threshold in thresholds:
                _, image_metrics = validate_batch(
                    outputs,
                    batch,
                    threshold=float(threshold),
                )
                results[float(threshold)].extend(image_metrics)

    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = _mean_metrics(results[float(threshold)])
        candidates.append(
            {
                "threshold": float(threshold),
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "cl_dice": float(metrics["cl_dice"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "composite": float(metrics["composite"]),
            }
        )
    best = max(candidates, key=lambda item: float(item[args.metric]))

    output_dir = Path(config["output_dir"])
    write_json(
        output_dir / "threshold.json",
        {
            "mask_threshold": float(best["threshold"]),
            "selection_metric": args.metric,
            "split": args.split,
            "tta": bool(args.tta),
            "minimum": float(args.minimum),
            "maximum": float(args.maximum),
            "steps": int(args.steps),
            "best": best,
            "candidates": candidates,
        },
    )
    with (output_dir / "threshold_search.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    print(
        f"Selected threshold={best['threshold']:.3f} "
        f"using {args.metric}={best[args.metric]:.4f}"
    )


if __name__ == "__main__":
    main()
