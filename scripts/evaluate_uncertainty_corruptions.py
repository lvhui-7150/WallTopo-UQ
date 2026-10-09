"""Evaluate EUS and uncertainty baselines under deterministic corruptions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from evaluate_uncertainty_baselines import (
    _binary_entropy,
    _forward_once,
    _load_threshold,
    _mean_metrics,
    _mean_outputs,
    _register_dropout_hooks,
)
from walltopo_uq.config import load_config
from walltopo_uq.data import ManifestCrackDataset
from walltopo_uq.metrics import validate_batch
from walltopo_uq.models import build_model
from walltopo_uq.trainer import Trainer
from walltopo_uq.uncertainty import selective_metrics
from walltopo_uq.utils import resolve_device, write_json


CORRUPTION_LEVELS = {
    "noise": [0.02, 0.04, 0.08],
    "blur": [1.0, 2.0, 3.0],
    "jpeg": [70.0, 50.0, 30.0],
    "brightness": [0.10, 0.20, 0.30],
    "contrast": [0.80, 0.60, 0.40],
    "shadow": [0.30, 0.50, 0.70],
}


def _gaussian_noise(array: np.ndarray, level: float) -> np.ndarray:
    noise = np.random.normal(0.0, float(level), size=array.shape)
    return np.clip(array + noise, 0.0, 1.0)


def _gaussian_blur(array: np.ndarray, level: float) -> np.ndarray:
    sigma = float(level)
    return cv2.GaussianBlur(
        array,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _jpeg(array: np.ndarray, level: float) -> np.ndarray:
    quality = int(np.clip(level, 1.0, 100.0))
    encoded = cv2.imencode(
        ".jpg",
        (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG corruption failed.")
    return decoded.astype(np.float32) / 255.0


def _brightness(array: np.ndarray, level: float) -> np.ndarray:
    return np.clip(array + float(level), 0.0, 1.0)


def _contrast(array: np.ndarray, level: float) -> np.ndarray:
    return np.clip((array - 0.5) * float(level) + 0.5, 0.0, 1.0)


def _shadow(array: np.ndarray, level: float) -> np.ndarray:
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    center_x = width * 0.35
    center_y = height * 0.50
    sigma_x = max(width * 0.28, 1.0)
    sigma_y = max(height * 0.48, 1.0)
    shadow = 1.0 - float(level) * np.exp(
        -(
            ((xx - center_x) ** 2) / (2.0 * sigma_x**2)
            + ((yy - center_y) ** 2) / (2.0 * sigma_y**2)
        )
    )
    return np.clip(array * shadow[..., None], 0.0, 1.0)


def _apply_corruption(
    image: torch.Tensor,
    corruption: str,
    level: float,
) -> torch.Tensor:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    if corruption == "noise":
        array = _gaussian_noise(array, level)
    elif corruption == "blur":
        array = _gaussian_blur(array, level)
    elif corruption == "jpeg":
        array = _jpeg(array, level)
    elif corruption == "brightness":
        array = _brightness(array, level)
    elif corruption == "contrast":
        array = _contrast(array, level)
    elif corruption == "shadow":
        array = _shadow(array, level)
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()


class CorruptedDataset(Dataset):
    def __init__(
        self,
        base: ManifestCrackDataset,
        corruption: str,
        level: float,
    ) -> None:
        self.base = base
        self.corruption = corruption
        self.level = float(level)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sample = self.base[index]
        sample["image"] = _apply_corruption(
            sample["image"],
            self.corruption,
            self.level,
        )
        return sample


def _build_loader(
    manifest: str | Path,
    config: dict,
    corruption: str,
    level: float,
) -> DataLoader:
    base = ManifestCrackDataset(
        manifest=manifest,
        image_size=int(config["image_size"]),
        training=False,
        config=config["data"],
    )
    dataset = CorruptedDataset(base, corruption, level)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=int(config["num_workers"]) > 0,
    )


@torch.no_grad()
def _evaluate_external_method(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    method: str,
    threshold: float,
    coverage: float,
    passes: int,
    dropout_p: float,
    dropout_targets: list[str],
) -> dict[str, float]:
    per_image: list[dict[str, float]] = []
    start = time.time()
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if method == "mc_dropout":
        handles = _register_dropout_hooks(
            models[0],
            probability=float(dropout_p),
            targets=dropout_targets,
        )
    try:
        for batch in tqdm(loader, desc=method, leave=False):
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            if method == "softmax":
                models[0].eval()
                outputs = [_forward_once(models[0], batch["image"], True)]
            elif method == "mc_dropout":
                models[0].train()
                outputs = [
                    _forward_once(models[0], batch["image"], True)
                    for _ in range(max(2, int(passes)))
                ]
            elif method == "ensemble":
                outputs = []
                for model in models:
                    model.eval()
                    outputs.append(_forward_once(model, batch["image"], True))
            else:
                raise ValueError(f"Unsupported external method: {method}")

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
    finally:
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
            coverage=float(coverage),
        )
    )
    elapsed = time.time() - start
    metrics.update(
        {
            "seconds_per_image": elapsed / max(len(per_image), 1),
            "members": float(
                1
                if method in {"softmax", "mc_dropout"}
                else len(models)
            ),
            "stochastic_passes": float(
                max(2, int(passes)) if method == "mc_dropout" else 1
            ),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=sorted(CORRUPTION_LEVELS),
        default=sorted(CORRUPTION_LEVELS),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["eus", "softmax", "mc_dropout", "ensemble"],
        default=["eus", "softmax", "mc_dropout", "ensemble"],
    )
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument(
        "--dropout-targets",
        nargs="+",
        default=["ConvNormAct", "ResidualBlock"],
    )
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--threshold", type=float)
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

    trainer = Trainer(config, device)
    first_checkpoint = torch.load(
        args.checkpoints[0],
        map_location=device,
        weights_only=False,
    )
    trainer.model.load_state_dict(first_checkpoint["model"])
    trainer.mask_threshold = threshold

    results: dict[str, dict[str, dict[str, float]]] = {}
    for corruption in args.corruptions:
        levels = CORRUPTION_LEVELS[corruption]
        for level_index in args.levels:
            if level_index < 1 or level_index > len(levels):
                raise SystemExit(
                    f"Invalid level {level_index} for corruption={corruption}"
                )
            level = float(levels[level_index - 1])
            key = f"{corruption}_s{level_index}"
            loader = _build_loader(manifest, config, corruption, level)
            results[key] = {
                "corruption": corruption,
                "level_index": float(level_index),
                "level": level,
            }
            if "eus" in args.methods:
                results[key]["eus"] = trainer.validate(
                    loader,
                    use_tta=True,
                    threshold=threshold,
                )
            for method in ("softmax", "mc_dropout", "ensemble"):
                if method not in args.methods:
                    continue
                results[key][method] = _evaluate_external_method(
                    models=models,
                    loader=loader,
                    device=device,
                    method=method,
                    threshold=threshold,
                    coverage=float(args.coverage),
                    passes=int(args.passes),
                    dropout_p=float(args.dropout_p),
                    dropout_targets=list(args.dropout_targets),
                )
            print(f"{key}: {json.dumps(results[key], ensure_ascii=False)}")

    output = args.output or (
        args.checkpoints[0].parent.parent
        / f"uncertainty_corruptions_{args.split}.json"
    )
    write_json(output, results)
    print(f"Saved corruption uncertainty results to {output.resolve()}")


if __name__ == "__main__":
    main()
