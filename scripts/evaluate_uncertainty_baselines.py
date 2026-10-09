"""Evaluate softmax, MC-dropout, and deep-ensemble uncertainty baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.metrics import composite_score, validate_batch
from walltopo_uq.models import build_model
from walltopo_uq.trainer import build_dataloader
from walltopo_uq.uncertainty import selective_metrics
from walltopo_uq.utils import resolve_device, write_json


OUTPUT_KEYS = (
    "mask_logits",
    "skeleton_logits",
    "endpoint_logits",
    "junction_logits",
    "orientation",
    "width",
    "evidence",
    "severity_logits",
)


class _DropoutHook:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def __call__(self, module, inputs, output):
        del module, inputs
        if not torch.is_tensor(output):
            return output
        if output.ndim == 4:
            return F.dropout2d(output, self.probability, training=True)
        if output.ndim == 2:
            return F.dropout(output, self.probability, training=True)
        return output


def _register_dropout_hooks(
    model: torch.nn.Module,
    probability: float,
    targets: Iterable[str],
) -> list[torch.utils.hooks.RemovableHandle]:
    targets = set(targets)
    hook = _DropoutHook(probability)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for module in model.modules():
        if type(module).__name__ in targets:
            handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError(
            "No dropout hooks were registered. Check --dropout-targets "
            "against the selected model architecture."
        )
    return handles


@torch.no_grad()
def _forward_once(
    model: torch.nn.Module,
    image: torch.Tensor,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    with torch.autocast(
        device_type=image.device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        outputs = model(image)
    return {
        key: outputs[key].detach().float()
        for key in OUTPUT_KEYS
        if key in outputs
    }


def _mean_outputs(
    outputs: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = sorted(set.intersection(*(set(item) for item in outputs)))
    return {
        key: torch.stack([item[key] for item in outputs], dim=0).mean(dim=0)
        for key in keys
    }


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    return -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )


def _mean_metrics(per_image: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in per_image for key in item})
    metrics = {
        key: float(np.mean([item.get(key, 0.0) for item in per_image]))
        for key in keys
    }
    metrics["composite"] = composite_score(metrics)
    return metrics


def _load_threshold(checkpoint: Path, fallback: float) -> float:
    threshold_path = checkpoint.parent.parent / "threshold.json"
    if not threshold_path.exists():
        return float(fallback)
    with threshold_path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle).get("mask_threshold", fallback))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["softmax", "mc_dropout", "ensemble"],
        default=["softmax", "mc_dropout", "ensemble"],
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument(
        "--dropout-targets",
        nargs="+",
        default=["ConvNormAct", "ResidualBlock"],
    )
    parser.add_argument("--coverage", type=float, default=0.80)
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
        models.append(model)

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else _load_threshold(args.checkpoints[0], 0.5)
    )
    use_amp = device.type == "cuda"
    results: dict[str, dict] = {}

    for method in args.methods:
        per_image: list[dict[str, float]] = []
        records: list[dict[str, float]] = []
        start = time.time()
        handles: list[torch.utils.hooks.RemovableHandle] = []
        if method == "mc_dropout":
            handles = _register_dropout_hooks(
                models[0],
                probability=float(args.dropout_p),
                targets=args.dropout_targets,
            )

        for batch in tqdm(loader, desc=method, leave=False):
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            if method == "softmax":
                models[0].eval()
                outputs = [_forward_once(models[0], batch["image"], use_amp)]
            elif method == "mc_dropout":
                models[0].train()
                outputs = [
                    _forward_once(models[0], batch["image"], use_amp)
                    for _ in range(max(2, int(args.passes)))
                ]
            else:
                outputs = []
                for model in models:
                    model.eval()
                    outputs.append(_forward_once(model, batch["image"], use_amp))

            averaged = _mean_outputs(outputs)
            probability = torch.sigmoid(averaged["mask_logits"])
            uncertainty = _binary_entropy(probability).mean(dim=(1, 2, 3))
            _, image_metrics = validate_batch(
                averaged,
                batch,
                threshold=threshold,
            )
            for index, metrics in enumerate(image_metrics):
                metrics["report_uncertainty"] = float(
                    uncertainty[index].detach().cpu()
                )
            per_image.extend(image_metrics)
            records.extend(
                {
                    "image_path": batch["meta"]["image_path"][index],
                    "dice": image_metrics[index]["dice"],
                    "image_error": image_metrics[index]["image_error"],
                    "width_mae_px": image_metrics[index].get("width_mae_px", 0.0),
                    "uncertainty": float(uncertainty[index].detach().cpu()),
                }
                for index in range(len(image_metrics))
            )

        for handle in handles:
            handle.remove()

        metrics = _mean_metrics(per_image)
        errors = np.asarray([item["image_error"] for item in per_image])
        width_errors = np.asarray(
            [item.get("width_mae_px", 0.0) for item in per_image]
        )
        uncertainty_values = np.asarray(
            [item["report_uncertainty"] for item in per_image]
        )
        metrics.update(
            selective_metrics(
                errors,
                uncertainty_values,
                secondary_error=width_errors,
                coverage=float(args.coverage),
            )
        )
        elapsed = time.time() - start
        metrics.update(
            {
                "mask_threshold": threshold,
                "elapsed_seconds": elapsed,
                "seconds_per_image": elapsed / max(len(per_image), 1),
                "members": float(
                    1 if method in {"softmax", "mc_dropout"} else len(models)
                ),
                "stochastic_passes": float(
                    max(2, int(args.passes)) if method == "mc_dropout" else 1
                ),
            }
        )
        results[method] = {"metrics": metrics, "records": records}
        print(f"{method}: {metrics}")

    output = args.output or (
        args.checkpoints[0].parent.parent
        / f"uncertainty_baselines_{args.split}.json"
    )
    write_json(output, results)
    print(f"Saved uncertainty baselines to {output.resolve()}")


if __name__ == "__main__":
    main()
