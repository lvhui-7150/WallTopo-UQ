"""Measure model size, forward/backward time, and CUDA memory."""

from __future__ import annotations

import argparse
import time

import torch

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.models import WallTopoUQ
from walltopo_uq.utils import count_parameters, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, nargs=4, default=[24, 48, 80, 128])
    parser.add_argument("--mdsm-stage", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = WallTopoUQ(
        channels=args.channels,
        mdsm_stages=[args.mdsm_stage],
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    image = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    target = (torch.rand_like(image[:, :1]) > 0.97).float()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["mask_logits"],
            target,
        )
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"parameters: {count_parameters(model):,}")
    print(f"input: {tuple(image.shape)}")
    print(f"iterations: {args.iterations}")
    print(f"seconds_per_iteration: {elapsed / args.iterations:.4f}")
    if device.type == "cuda":
        print(
            "peak_cuda_memory_gb: "
            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}"
        )


if __name__ == "__main__":
    main()
