"""General utilities for reproducible experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import cv2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image on Windows even when the path contains non-ASCII text."""

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Write an image on Windows even when the path contains non-ASCII text."""

    target = Path(path)
    ensure_dir(target.parent)
    extension = target.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    encoded.tofile(str(target))
    return True


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def move_to_device(
    value: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_to_device(item, device, non_blocking) for item in value)
    return value


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


def detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    return value


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
