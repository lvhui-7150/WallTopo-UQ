"""Build multi-model qualitative galleries close to the current Figure 4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.data import ManifestCrackDataset
from walltopo_uq.metrics import binary_scores
from walltopo_uq.models import build_model
from walltopo_uq.utils import resolve_device


DISPLAY_NAMES = {
    "walltopo_uq": "WallTopo-UQ",
    "walltopo_uq_v2": "WallTopo-UQ-v2",
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "smp_unetplusplus": "U-Net++",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def _resolve_run(
    dataset: str,
    model: str,
    run_prefix: str,
    seed: int,
) -> tuple[Path, Path, float] | None:
    generated = (
        ROOT
        / "configs"
        / "generated"
        / run_prefix
        / dataset
        / model
        / f"seed_{seed}.yaml"
    )
    run_dir = (
        ROOT
        / "runs"
        / run_prefix
        / dataset
        / model
        / f"seed_{seed}"
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if generated.exists() and checkpoint.exists():
        threshold_path = run_dir / "threshold.json"
        threshold = 0.5
        if threshold_path.exists():
            with threshold_path.open("r", encoding="utf-8") as handle:
                threshold = float(
                    json.load(handle).get("mask_threshold", threshold)
                )
        return generated, checkpoint, threshold

    if model == "walltopo_uq":
        legacy_config = ROOT / "configs" / f"{dataset}.yaml"
        legacy_checkpoint = (
            ROOT / "runs" / dataset / "checkpoints" / "best.pt"
        )
        if legacy_config.exists() and legacy_checkpoint.exists():
            return legacy_config, legacy_checkpoint, 0.5
    return None


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    views = [
        image,
        image.flip(-1),
        image.flip(-2),
        image.flip(-1).flip(-2),
    ]
    probabilities = []
    vacuuities = []
    skeletons = []
    for index, view in enumerate(views):
        outputs = model(view)
        probability = torch.sigmoid(outputs["mask_logits"])
        skeleton = torch.sigmoid(outputs["skeleton_logits"])
        vacuity = 2.0 / outputs["evidence"].sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        if index == 1:
            probability = probability.flip(-1)
            skeleton = skeleton.flip(-1)
            vacuity = vacuity.flip(-1)
        elif index == 2:
            probability = probability.flip(-2)
            skeleton = skeleton.flip(-2)
            vacuity = vacuity.flip(-2)
        elif index == 3:
            probability = probability.flip(-1).flip(-2)
            skeleton = skeleton.flip(-1).flip(-2)
            vacuity = vacuity.flip(-1).flip(-2)
        probabilities.append(probability)
        skeletons.append(skeleton)
        vacuuities.append(vacuity)
    probability = torch.stack(probabilities, dim=0).mean(dim=0)
    skeleton = torch.stack(skeletons, dim=0).mean(dim=0)
    vacuity = torch.stack(vacuuities, dim=0).mean(dim=0)
    return (
        probability[0, 0].detach().cpu().numpy(),
        skeleton[0, 0].detach().cpu().numpy(),
        vacuity[0, 0].detach().cpu().numpy(),
    )


def _error_map(
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    prediction = probability >= threshold
    output = image * 0.38
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    output[true_positive] = np.asarray([0.02, 0.80, 0.18])
    output[false_positive] = np.asarray([0.95, 0.06, 0.06])
    output[false_negative] = np.asarray([0.04, 0.28, 1.0])
    return np.clip(output, 0.0, 1.0)


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[mask] = 0.50 * output[mask] + 0.50 * np.asarray([1.0, 0.05, 0.03])
    return np.clip(output, 0.0, 1.0)


def _load_dataset_batch(
    dataset: str,
    model_specs: dict[str, tuple[Path, Path, float]],
    device: torch.device,
    proposed_name: str,
) -> tuple[list[dict[str, object]], list[str]]:
    proposed = model_specs[proposed_name]
    proposed_config = load_config(proposed[0])
    dataset_object = ManifestCrackDataset(
        proposed_config["data"]["test_manifest"],
        int(proposed_config["image_size"]),
        training=False,
        config=proposed_config["data"],
    )
    loader = DataLoader(dataset_object, batch_size=1, shuffle=False)

    proposed_model = build_model(proposed_config).to(device)
    proposed_model.load_state_dict(
        torch.load(proposed[1], map_location=device, weights_only=False)["model"]
    )
    proposed_model.eval()

    scored: list[tuple[float, dict[str, object]]] = []
    with torch.inference_mode():
        for batch in loader:
            probability, _, _ = _predict(
                proposed_model,
                batch["image"].to(device),
                device,
            )
            score = binary_scores(
                probability >= proposed[2],
                batch["mask"][0, 0].numpy() >= 0.5,
            )["dice"]
            scored.append((float(score), batch))
    scored.sort(key=lambda item: item[0])
    return [
        {"dataset": dataset, "rank": rank, "score": score, "batch": batch}
        for rank, (score, batch) in enumerate(scored)
    ], list(model_specs)


def _select_samples(
    scored: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    if not scored:
        return []
    indices = np.linspace(
        0,
        len(scored) - 1,
        max(1, int(count)),
        dtype=int,
    )
    return [scored[int(index)] for index in indices]


def _draw_gallery(
    samples: list[dict[str, object]],
    model_names: list[str],
    model_paths: dict[str, tuple[Path, Path, float]],
    device: torch.device,
    output_stem: Path,
    proposed_name: str,
) -> list[str]:
    column_titles = [
        "Input",
        "Ground truth",
        *[DISPLAY_NAMES.get(name, name) for name in model_names],
        "Ours probability",
        "Ours uncertainty",
    ]
    columns = len(column_titles)
    figure, axes = plt.subplots(
        len(samples),
        columns,
        figsize=(1.28 * columns, 1.40 * len(samples)),
        squeeze=False,
    )

    loaded_models: dict[str, torch.nn.Module] = {}
    for model_name in model_names:
        config_path, checkpoint_path, _ = model_paths[model_name]
        config = load_config(config_path)
        model = build_model(config).to(device)
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )["model"]
        )
        model.eval()
        loaded_models[model_name] = model

    for row, sample in enumerate(samples):
        batch = sample["batch"]
        image_tensor = batch["image"].to(device)
        image = batch["image"][0].permute(1, 2, 0).numpy()
        target = batch["mask"][0, 0].numpy() >= 0.5
        panels: list[tuple[np.ndarray, str]] = [
            (image, "Input"),
            (target.astype(np.float32), "Ground truth"),
        ]
        uncertainties: dict[str, np.ndarray] = {}
        probabilities: dict[str, np.ndarray] = {}
        for model_name in model_names:
            probability, _, uncertainty = _predict(
                loaded_models[model_name],
                image_tensor,
                device,
            )
            probabilities[model_name] = probability
            uncertainties[model_name] = uncertainty
            threshold = model_paths[model_name][2]
            dice = binary_scores(probability >= threshold, target)["dice"]
            panels.append(
                (
                    _error_map(image, target, probability, threshold),
                    f"{DISPLAY_NAMES.get(model_name, model_name)}\nDice={dice:.3f}",
                )
            )
        proposed = probabilities[proposed_name]
        panels.extend(
            [
                (proposed, "WallTopo-UQ\nprobability"),
                (
                    uncertainties[proposed_name],
                    f"{DISPLAY_NAMES.get(proposed_name, proposed_name)}\nuncertainty",
                ),
            ]
        )

        for column, (panel, title) in enumerate(panels):
            axis = axes[row, column]
            cmap = (
                "magma"
                if column == columns - 1
                else "gray"
                if column in {1, columns - 2}
                else None
            )
            axis.imshow(
                panel,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[column], pad=2)
            for spine in axis.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.35)
        dataset = str(sample["dataset"])
        rank = int(sample["rank"])
        dice = float(sample["score"])
        axes[row, 0].set_ylabel(
            f"{dataset.title()}\nrank {rank}\nDice={dice:.3f}",
            fontsize=6.5,
        )

    figure.subplots_adjust(wspace=0.035, hspace=0.05)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=600)
    figure.savefig(pdf)
    plt.close(figure)
    return [str(png), str(pdf)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default="qualitative")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dais", "deepcrack", "crack500", "crackforest"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "deeplabv3plus", "walltopo_uq"],
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "figures" / "qualitative",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    _style()

    proposed_names = [
        name
        for name in ("walltopo_uq", "walltopo_uq_v2")
        if name in args.models
    ]
    if not proposed_names:
        raise SystemExit(
            "walltopo_uq or walltopo_uq_v2 must be included for the gallery."
        )
    proposed_name = proposed_names[0]

    outputs: list[str] = []
    overview_samples: list[dict[str, object]] = []
    for dataset in args.datasets:
        model_paths = {
            model: _resolve_run(
                dataset,
                model,
                args.run_prefix,
                args.seed,
            )
            for model in args.models
        }
        missing = [
            model for model, value in model_paths.items() if value is None
        ]
        if missing:
            print(f"skip {dataset}: missing {missing}")
            continue
        model_paths = {
            model: value
            for model, value in model_paths.items()
            if value is not None
        }
        scored, model_names = _load_dataset_batch(
            dataset,
            model_paths,
            device,
            proposed_name,
        )
        selected = _select_samples(scored, int(args.samples))
        outputs.extend(
            _draw_gallery(
                selected,
                model_names,
                model_paths,
                device,
                args.output_dir / f"fig_comparison_{dataset}",
                proposed_name,
            )
        )
        if selected:
            overview_samples.append(selected[len(selected) // 2])

    if overview_samples:
        overview_paths = {
            dataset: {
                model: _resolve_run(
                    dataset,
                    model,
                    args.run_prefix,
                    args.seed,
                )
                for model in args.models
            }
            for dataset in args.datasets
        }
        first_dataset = str(overview_samples[0]["dataset"])
        output_paths = {
            model: value
            for model, value in overview_paths[first_dataset].items()
            if value is not None
        }
        # The overview figure uses each sample's own dataset configuration.
        combined: list[dict[str, object]] = []
        for sample in overview_samples:
            dataset = str(sample["dataset"])
            resolved = overview_paths[dataset]
            if any(value is None for value in resolved.values()):
                continue
            combined.append(sample)
        if combined:
            # Draw each dataset separately and concatenate visually by reusing
            # the gallery renderer per dataset.
            for dataset in sorted({str(item["dataset"]) for item in combined}):
                subset = [
                    item for item in combined if str(item["dataset"]) == dataset
                ]
                resolved = {
                    model: value
                    for model, value in overview_paths[dataset].items()
                    if value is not None
                }
                outputs.extend(
                    _draw_gallery(
                        subset,
                        args.models,
                        resolved,
                        device,
                        args.output_dir / f"fig_comparison_overview_{dataset}",
                        proposed_name,
                    )
                )

    manifest = args.output_dir / "qualitative_manifest.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} qualitative files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
