"""Run WallTopo-UQ on unlabeled images and export visual reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.geometry import extract_geometry, visual_screening_score
from walltopo_uq.models import build_model
from walltopo_uq.uncertainty import (
    geometry_dispersion,
    normalized_mahalanobis,
    orientation_dispersion,
    report_uncertainty,
    topology_instability,
)
from walltopo_uq.utils import ensure_dir, imread_unicode, imwrite_unicode, resolve_device


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        item
        for item in path.glob(pattern)
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_rgb(path: Path) -> np.ndarray:
    image = imread_unicode(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image


def _restore_map(tensor: torch.Tensor, view: str) -> torch.Tensor:
    if view == "hflip":
        return tensor.flip(-1)
    if view == "vflip":
        return tensor.flip(-2)
    if view == "rot180":
        return tensor.flip(-1).flip(-2)
    return tensor


def _restore_orientation(tensor: torch.Tensor, view: str) -> torch.Tensor:
    restored = _restore_map(tensor, view).clone()
    if view in ("hflip", "rot180"):
        restored[:, 0] *= -1.0
    if view in ("vflip", "rot180"):
        restored[:, 1] *= -1.0
    return restored


def _views(image: torch.Tensor, count: int) -> list[tuple[str, torch.Tensor]]:
    available = {
        "identity": image,
        "hflip": image.flip(-1),
        "vflip": image.flip(-2),
        "rot180": image.flip(-1).flip(-2),
    }
    names = ["identity", "hflip", "vflip", "rot180"][: max(1, min(4, count))]
    return [(name, available[name]) for name in names]


@torch.inference_mode()
def _predict_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    tta_views: int,
    use_amp: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    average: dict[str, torch.Tensor] = {}
    raw: dict[str, list[torch.Tensor]] = {
        "mask_prob": [],
        "skeleton_prob": [],
        "width": [],
        "orientation": [],
    }
    count = 0
    for view, transformed in _views(image, tta_views):
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(transformed)
        mask_probability = _restore_map(torch.sigmoid(outputs["mask_logits"]), view)
        skeleton_probability = _restore_map(
            torch.sigmoid(outputs["skeleton_logits"]),
            view,
        )
        width = _restore_map(outputs["width"], view)
        orientation = _restore_orientation(outputs["orientation"], view)
        evidence = _restore_map(outputs["evidence"], view)
        raw["mask_prob"].append(mask_probability)
        raw["skeleton_prob"].append(skeleton_probability)
        raw["width"].append(width)
        raw["orientation"].append(orientation)
        for key in ("mask_logits", "skeleton_logits", "endpoint_logits", "junction_logits"):
            value = _restore_map(outputs[key], view)
            average[key] = average[key] + value if key in average else value
        for key, value in (
            ("width", width),
            ("orientation", orientation),
            ("evidence", evidence),
            ("severity_logits", outputs["severity_logits"]),
        ):
            average[key] = average[key] + value if key in average else value
        if "topology_tokens" in outputs:
            token = outputs["topology_tokens"].mean(dim=1)
        elif "topology_anchor_logits" in outputs:
            token = outputs["topology_anchor_logits"].mean(dim=(2, 3))
        else:
            token = outputs["mask_logits"].mean(dim=(2, 3)).expand(-1, 4)
        average["embedding"] = (
            average["embedding"] + token if "embedding" in average else token
        )
        count += 1
    for key, value in list(average.items()):
        average[key] = value / count
    average["mask_prob"] = torch.sigmoid(average["mask_logits"])
    return average, raw


def _geometry_uncertainty(
    skeleton_predictions: list[np.ndarray],
    width_values: list[np.ndarray],
    orientation_values: list[np.ndarray],
) -> tuple[float, float]:
    from walltopo_uq.geometry import skeleton_length

    topology_term = topology_instability(skeleton_predictions)
    lengths = [
        skeleton_length(prediction)
        for prediction in skeleton_predictions
    ]
    width_term = geometry_dispersion(
        [float(value.mean()) for value in width_values],
        [float(value) for value in lengths],
    )
    orientation_term = orientation_dispersion(
        orientation_values,
        skeleton_predictions[0],
    )
    geometry_term = float(np.clip(0.65 * width_term + 0.35 * orientation_term, 0.0, 1.0))
    return topology_term, geometry_term


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.asarray([1.0, 0.05, 0.02])
    overlay[skeleton] = np.asarray([0.0, 1.0, 0.15])
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (1.0, 0.85, 0.0), 1)
    return np.clip(overlay, 0.0, 1.0)


def _resize_for_panel(array: np.ndarray, width: int = 420) -> np.ndarray:
    height = max(1, int(round(array.shape[0] * width / array.shape[1])))
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)


def _load_calibration(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def _domain_shift(
    embedding: torch.Tensor,
    calibration: dict[str, np.ndarray] | None,
) -> float:
    if calibration is None or not all(
        key in calibration
        for key in ("feature_mean", "feature_std", "feature_scale")
    ):
        return 0.0
    return normalized_mahalanobis(
        embedding.detach().float().cpu().numpy()[0],
        calibration["feature_mean"],
        calibration["feature_std"],
        float(calibration["feature_scale"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "inference")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta-views", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = (
        load_config(args.config)
        if args.config is not None
        else dict(checkpoint["config"])
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_size = int(args.image_size or config["image_size"])
    tta_views = int(
        args.tta_views
        if args.tta_views is not None
        else config.get("uncertainty", {}).get("tta_views", 4)
    )
    calibration_path = args.calibration
    if calibration_path is None:
        calibration_path = args.checkpoint.parent.parent / "report_calibration.npz"
    calibration = _load_calibration(calibration_path)
    review_threshold = (
        float(calibration["review_threshold"])
        if calibration is not None and "review_threshold" in calibration
        else None
    )

    images = _collect_images(args.input.resolve(), args.recursive)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under: {args.input}")

    output_dir = ensure_dir(args.output.resolve())
    records: list[dict[str, object]] = []
    use_amp = device.type == "cuda"
    for image_path in tqdm(images, desc="infer"):
        image = _read_rgb(image_path)
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        outputs, raw = _predict_tta(model, tensor, tta_views, use_amp)

        mask_probability = outputs["mask_prob"][0, 0].float().cpu().numpy()
        skeleton_probability = torch.sigmoid(outputs["skeleton_logits"])[0, 0].float().cpu().numpy()
        width_map = outputs["width"][0, 0].float().cpu().numpy()
        orientation_map = outputs["orientation"][0].float().cpu().numpy()
        vacuity_map = (
            2.0 / outputs["evidence"].sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        )[0, 0].float().cpu().numpy()
        mask = mask_probability >= float(args.threshold)
        skeleton = skeleton_probability >= float(args.threshold)

        skeleton_predictions = [
            item[0, 0].float().cpu().numpy() >= float(args.threshold)
            for item in raw["skeleton_prob"]
        ]
        width_values = [item[0, 0].float().cpu().numpy() for item in raw["width"]]
        orientation_values = [item[0].float().cpu().numpy() for item in raw["orientation"]]
        topology_term, geometry_term = _geometry_uncertainty(
            skeleton_predictions,
            width_values,
            orientation_values,
        )
        shift_term = _domain_shift(outputs["embedding"], calibration)
        report_term = report_uncertainty(
            vacuity_map,
            skeleton,
            topology_term,
            geometry_term,
            shift_term,
        )
        manual_review = (
            report_term > review_threshold if review_threshold is not None else None
        )

        geometry = extract_geometry(
            mask,
            skeleton,
            width=width_map,
            orientation=orientation_map,
            pixels_per_unit=None,
        )
        geometry["visual_screening_score"] = visual_screening_score(geometry)
        overlay = _overlay(resized, mask, skeleton)
        mask_visual = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)
        skeleton_visual = np.repeat(skeleton[..., None], 3, axis=2).astype(np.float32)
        uncertainty_visual = cv2.applyColorMap(
            (np.clip(vacuity_map, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        uncertainty_visual = cv2.cvtColor(uncertainty_visual, cv2.COLOR_BGR2RGB).astype(
            np.float32
        ) / 255.0
        panel = np.concatenate(
            [
                _resize_for_panel(resized),
                _resize_for_panel(overlay),
                _resize_for_panel(mask_visual),
                _resize_for_panel(skeleton_visual),
                _resize_for_panel(uncertainty_visual),
            ],
            axis=1,
        )

        stem = image_path.stem
        imwrite_unicode(
            output_dir / f"{stem}_overlay.png",
            cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        imwrite_unicode(output_dir / f"{stem}_mask.png", (mask * 255).astype(np.uint8))
        imwrite_unicode(
            output_dir / f"{stem}_skeleton.png",
            (skeleton * 255).astype(np.uint8),
        )
        imwrite_unicode(
            output_dir / f"{stem}_uncertainty.png",
            cv2.cvtColor(
                (uncertainty_visual * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        imwrite_unicode(
            output_dir / f"{stem}_panel.png",
            cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        records.append(
            {
                "image": str(image_path),
                "original_size": [original_width, original_height],
                "inference_size": [image_size, image_size],
                "threshold": float(args.threshold),
                "mask_area_ratio": float(mask.mean()),
                "probability_mean": float(mask_probability.mean()),
                "mean_vacuity": float(vacuity_map.mean()),
                "topology_uncertainty": float(topology_term),
                "geometry_uncertainty": float(geometry_term),
                "domain_shift": float(shift_term),
                "report_uncertainty": float(report_term),
                "review_threshold": review_threshold,
                "manual_review": manual_review,
                **geometry,
            }
        )

    records.sort(
        key=lambda record: (
            bool(record["manual_review"]),
            float(record["visual_screening_score"]),
        ),
        reverse=True,
    )
    with (output_dir / "inference_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": records}, handle, indent=2, ensure_ascii=False)
    with (output_dir / "inference_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Processed {len(records)} images. Results: {output_dir}")


if __name__ == "__main__":
    main()
