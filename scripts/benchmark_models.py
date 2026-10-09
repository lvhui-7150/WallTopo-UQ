"""Benchmark parameter count, training speed, and peak GPU memory."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.models import build_model
from walltopo_uq.utils import count_parameters, resolve_device


MODEL_OVERRIDES = {
    "walltopo_uq": {},
    "walltopo_uq_v2": {
        "model_name": "walltopo_uq",
        "use_segmentation_shortcut": True,
        "shortcut_gate_bias": -2.0,
    },
    "unet": {"model_name": "unet"},
    "deeplabv3plus": {
        "model_name": "deeplabv3plus",
        "pretrained_backbone": True,
    },
    "smp_unetplusplus": {
        "model_name": "smp_unetplusplus",
        "smp_arch": "unetplusplus",
        "encoder_name": "resnet34",
        "encoder_weights": None,
    },
}


def _measure(
    model_name: str,
    config: dict,
    device: torch.device,
    size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> dict[str, float | int | str]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    image = torch.randn(batch_size, 3, size, size, device=device)
    target = (torch.rand_like(image[:, :1]) > 0.97).float()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(image)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["mask_logits"],
                target,
            )
        loss.backward()
        optimizer.step()

    for _ in range(max(0, warmup)):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "model": model_name,
        "parameters": count_parameters(model),
        "image_size": size,
        "batch_size": batch_size,
        "iterations": iterations,
        "seconds_per_iteration": elapsed / max(1, iterations),
        "images_per_second": batch_size * iterations / max(elapsed, 1.0e-9),
        "peak_cuda_memory_gb": (
            float(torch.cuda.max_memory_allocated() / (1024**3))
            if device.type == "cuda"
            else 0.0
        ),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dais.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["walltopo_uq", "unet", "deeplabv3plus"],
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "benchmark" / "efficiency.csv",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        if model_name not in MODEL_OVERRIDES:
            print(f"skip unsupported benchmark model: {model_name}")
            continue
        config = load_config(args.config)
        config.update(MODEL_OVERRIDES[model_name])
        config["device"] = str(device)
        try:
            row = _measure(
                model_name=model_name,
                config=config,
                device=device,
                size=int(args.size),
                batch_size=int(args.batch_size),
                iterations=int(args.iterations),
                warmup=int(args.warmup),
            )
        except Exception as error:  # noqa: BLE001 - benchmark should record failures
            row = {
                "model": model_name,
                "parameters": 0,
                "image_size": int(args.size),
                "batch_size": int(args.batch_size),
                "iterations": int(args.iterations),
                "seconds_per_iteration": float("nan"),
                "images_per_second": float("nan"),
                "peak_cuda_memory_gb": float("nan"),
                "device": str(device),
                "error": str(error),
            }
        rows.append(row)
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "parameters",
        "image_size",
        "batch_size",
        "iterations",
        "seconds_per_iteration",
        "images_per_second",
        "peak_cuda_memory_gb",
        "device",
        "error",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with (args.output.with_suffix(".json")).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"Wrote efficiency results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
