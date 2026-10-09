"""Evaluate a probability ensemble from multiple trained checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.metrics import composite_score, validate_batch
from walltopo_uq.models import build_model
from walltopo_uq.trainer import build_dataloader
from walltopo_uq.utils import resolve_device, write_json


def _restore(tensor: torch.Tensor, view: str) -> torch.Tensor:
    if view == "hflip":
        return tensor.flip(-1)
    if view == "vflip":
        return tensor.flip(-2)
    if view == "rot180":
        return tensor.flip(-1).flip(-2)
    return tensor


def _orient_view(tensor: torch.Tensor, view: str) -> torch.Tensor:
    restored = _restore(tensor, view).clone()
    if view in {"hflip", "rot180"}:
        restored[:, 0] *= -1.0
    if view in {"vflip", "rot180"}:
        restored[:, 1] *= -1.0
    return restored


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    tta_views: int,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    views = {
        "identity": image,
        "hflip": image.flip(-1),
        "vflip": image.flip(-2),
        "rot180": image.flip(-1).flip(-2),
    }
    names = ["identity", "hflip", "vflip", "rot180"][: max(1, min(4, tta_views))]
    accumulated: dict[str, torch.Tensor] = {}
    for name in names:
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(views[name])
        probability = _restore(torch.sigmoid(outputs["mask_logits"]), name)
        skeleton = _restore(torch.sigmoid(outputs["skeleton_logits"]), name)
        width = _restore(outputs["width"], name)
        orientation = _orient_view(outputs["orientation"], name)
        evidence = _restore(outputs["evidence"], name)
        for key, value in {
            "mask_prob": probability,
            "skeleton_prob": skeleton,
            "width": width,
            "orientation": orientation,
            "evidence": evidence,
            "severity_logits": outputs["severity_logits"],
        }.items():
            accumulated[key] = (
                accumulated[key] + value if key in accumulated else value
            )
    for key in accumulated:
        accumulated[key] = accumulated[key] / len(names)
    return accumulated


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
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta-views", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    manifest = config["data"].get(f"{args.split}_manifest")
    if not manifest:
        raise SystemExit(f"No {args.split}_manifest configured.")
    loader = build_dataloader(manifest, config, training=False)

    models: list[torch.nn.Module] = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)

    per_image: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="ensemble", leave=False):
        batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        member_outputs = [
            _predict(
                model,
                batch["image"],
                tta_views=int(args.tta_views),
                use_amp=device.type == "cuda",
            )
            for model in models
        ]
        averaged = {
            key: torch.stack([item[key] for item in member_outputs], dim=0).mean(dim=0)
            for key in member_outputs[0]
        }
        outputs = {
            "mask_logits": torch.logit(
                averaged["mask_prob"].clamp(1.0e-6, 1.0 - 1.0e-6)
            ),
            "skeleton_logits": torch.logit(
                averaged["skeleton_prob"].clamp(1.0e-6, 1.0 - 1.0e-6)
            ),
            "width": averaged["width"],
            "orientation": averaged["orientation"],
            "evidence": averaged["evidence"],
            "severity_logits": averaged["severity_logits"],
        }
        _, image_metrics = validate_batch(
            outputs,
            batch,
            threshold=float(args.threshold),
        )
        per_image.extend(image_metrics)

    metrics = _mean_metrics(per_image)
    metrics["members"] = float(len(models))
    metrics["mask_threshold"] = float(args.threshold)
    output = args.output or (
        Path(args.checkpoints[0]).parent.parent / "ensemble_test_metrics.json"
    )
    write_json(output, metrics)
    print(metrics)
    print(f"Saved ensemble metrics to {output.resolve()}")


if __name__ == "__main__":
    main()
