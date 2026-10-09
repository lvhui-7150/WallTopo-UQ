"""Create a publication-quality qualitative comparison across models."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _error_map(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    output = image * 0.38
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    output[true_positive] = np.asarray([0.0, 0.78, 0.18])
    output[false_positive] = np.asarray([0.95, 0.08, 0.08])
    output[false_negative] = np.asarray([0.05, 0.28, 1.0])
    return np.clip(output, 0.0, 1.0)


def _predict(
    config_path: Path,
    checkpoint_path: Path,
    image: torch.Tensor,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        if use_tta:
            views = [
                image,
                image.flip(-1),
                image.flip(-2),
                image.flip(-1).flip(-2),
            ]
            probabilities = []
            for index, view in enumerate(views):
                probability = torch.sigmoid(model(view)["mask_logits"])
                if index == 1:
                    probability = probability.flip(-1)
                elif index == 2:
                    probability = probability.flip(-2)
                elif index == 3:
                    probability = probability.flip(-1).flip(-2)
                probabilities.append(probability)
            probability = torch.stack(probabilities, dim=0).mean(dim=0)
        else:
            probability = torch.sigmoid(model(image)["mask_logits"])
    return probability[0, 0].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-prefix",
        default="benchmark",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dais", "deepcrack"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "deeplabv3plus", "walltopo_uq"],
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "figures" / "publication" / "fig_qualitative.png",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    _style()

    dataset_samples: list[dict[str, object]] = []
    for dataset in args.datasets:
        proposed_config = (
            ROOT
            / "configs"
            / "generated"
            / args.run_prefix
            / dataset
            / "walltopo_uq"
            / f"seed_{args.seed}.yaml"
        )
        proposed_checkpoint = (
            ROOT
            / "runs"
            / args.run_prefix
            / dataset
            / "walltopo_uq"
            / f"seed_{args.seed}"
            / "checkpoints"
            / "best.pt"
        )
        if not proposed_config.exists() or not proposed_checkpoint.exists():
            print(f"skip missing proposed run for {dataset}")
            continue
        config = load_config(proposed_config)
        dataset_object = ManifestCrackDataset(
            config["data"]["test_manifest"],
            int(config["image_size"]),
            training=False,
            config=config["data"],
        )
        loader = DataLoader(dataset_object, batch_size=1, shuffle=False)
        scorer = build_model(config).to(device)
        scorer.load_state_dict(
            torch.load(
                proposed_checkpoint,
                map_location=device,
                weights_only=False,
            )["model"]
        )
        scorer.eval()
        scored: list[tuple[float, int, dict[str, object]]] = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                probability = torch.sigmoid(
                    scorer(batch["image"].to(device))["mask_logits"]
                )
                score = binary_scores(
                    probability[0, 0].detach().cpu().numpy() >= args.threshold,
                    batch["mask"][0, 0].numpy() >= 0.5,
                )["dice"]
                scored.append((float(score), index, batch))
        scored.sort(key=lambda item: item[0])
        if not scored:
            continue
        requested = max(1, int(args.samples))
        indices = np.linspace(
            0,
            len(scored) - 1,
            requested,
            dtype=int,
        )
        for selected_index in indices:
            _, source_index, batch = scored[int(selected_index)]
            dataset_samples.append(
                {
                    "dataset": dataset,
                    "sample_index": source_index,
                    "batch": batch,
                }
            )

    if not dataset_samples:
        raise SystemExit("No complete proposed runs were found.")

    columns = 3 + len(args.models)
    figure, axes = plt.subplots(
        len(dataset_samples),
        columns,
        figsize=(1.55 * columns, 1.45 * len(dataset_samples)),
        squeeze=False,
    )
    for row, sample in enumerate(dataset_samples):
        dataset = str(sample["dataset"])
        batch = sample["batch"]
        image_tensor = batch["image"].to(device)
        image = batch["image"][0].permute(1, 2, 0).numpy()
        target = batch["mask"][0, 0].numpy() >= 0.5
        panels: list[tuple[np.ndarray, str, str | None]] = [
            (image, "Input", None),
            (target.astype(np.float32), "Ground truth", "gray"),
        ]
        proposed_probability = None
        for model_name in args.models:
            config_path = (
                ROOT
                / "configs"
                / "generated"
                / args.run_prefix
                / dataset
                / model_name
                / f"seed_{args.seed}.yaml"
            )
            checkpoint_path = (
                ROOT
                / "runs"
                / args.run_prefix
                / dataset
                / model_name
                / f"seed_{args.seed}"
                / "checkpoints"
                / "best.pt"
            )
            if not config_path.exists() or not checkpoint_path.exists():
                panels.append((np.zeros_like(image), f"{DISPLAY_NAMES.get(model_name, model_name)}\nmissing", None))
                continue
            probability = _predict(
                config_path,
                checkpoint_path,
                image_tensor,
                device,
            )
            prediction = probability >= args.threshold
            score = binary_scores(prediction, target)["dice"]
            if model_name == "walltopo_uq":
                proposed_probability = probability
                panel = _error_map(image, target, prediction)
                panels.append((panel, f"{DISPLAY_NAMES[model_name]}\nDice={score:.3f}", None))
            else:
                panels.append(
                    (
                        prediction.astype(np.float32),
                        f"{DISPLAY_NAMES.get(model_name, model_name)}\nDice={score:.3f}",
                        "gray",
                    )
                )
        if proposed_probability is not None:
            calibration = np.abs(proposed_probability - target.astype(np.float32))
            panels.append((calibration, "Ours error", "magma"))

        for column, (panel, title, colormap) in enumerate(panels[:columns]):
            axis = axes[row, column]
            axis.imshow(
                panel,
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(title, pad=2)
            for spine in axis.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#777777")
        axes[row, 0].set_ylabel(
            f"{dataset.title()}\n#{sample['sample_index']}",
            fontsize=7,
        )

    figure.subplots_adjust(wspace=0.035, hspace=0.06)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=600)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"Saved qualitative comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
