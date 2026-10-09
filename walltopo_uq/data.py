"""Synthetic and real folder datasets."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import derive_targets, geometry_severity
from .utils import imread_unicode


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = imread_unicode(path, flags)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if grayscale:
        return image.astype(np.float32) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_cached_targets(
    cache_path: Path,
    mask: np.ndarray,
    normalize_width: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        payload = np.load(cache_path)
        return {key: payload[key] for key in payload.files}
    targets = derive_targets(mask, normalize_width=normalize_width)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **targets)
    return targets


class ManifestCrackDataset(Dataset):
    """Dataset backed by a manifest with image, mask, and optional targets."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: int,
        training: bool,
        config: dict[str, Any],
        domain: str | None = None,
        group: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Generate synthetic data or build a real-data manifest first."
            )
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if domain is not None:
            self.rows = [
                row for row in self.rows if row.get("domain", "") == str(domain)
            ]
        if group is not None:
            self.rows = [
                row for row in self.rows if row.get("group", "") == str(group)
            ]
        if not self.rows:
            filters = []
            if domain is not None:
                filters.append(f"domain={domain!r}")
            if group is not None:
                filters.append(f"group={group!r}")
            suffix = f" after filtering {', '.join(filters)}" if filters else ""
            raise ValueError(f"Empty manifest: {self.manifest_path}{suffix}")
        self.base = self.manifest_path.parent
        self.image_size = image_size
        self.training = training
        self.config = config
        cache_setting = config.get("cache_derived", True)
        self.cache_dir = (
            Path(cache_setting).resolve()
            if isinstance(cache_setting, str) and cache_setting not in ("true", "false")
            else self.base / "derived_cache"
        )
        self.foreground_ratio = float(config.get("foreground_ratio", 0.65))
        self.max_rotation = float(config.get("max_rotation", 10.0)) if training else 0.0

    def __len__(self) -> int:
        return len(self.rows)

    def _optional_path(self, row: dict[str, str], key: str) -> Path | None:
        value = row.get(key, "")
        return _resolve_path(value, self.base) if value else None

    def _training_crop(
        self,
        image: np.ndarray,
        dense: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        height, width = image.shape[:2]
        crop_size = min(self.image_size, height, width)
        if not self.training or (height <= crop_size and width <= crop_size):
            return image, dense
        mask = dense["mask"]
        if np.random.random() < self.foreground_ratio and mask.any():
            foreground_y, foreground_x = np.argwhere(mask > 0)[
                np.random.randint(int(mask.sum()))
            ]
            center_y = int(foreground_y)
            center_x = int(foreground_x)
        else:
            center_y = int(np.random.randint(0, height))
            center_x = int(np.random.randint(0, width))
        half = crop_size // 2
        top = int(np.clip(center_y - half, 0, height - crop_size))
        left = int(np.clip(center_x - half, 0, width - crop_size))
        bottom = top + crop_size
        right = left + crop_size
        cropped_dense = {
            key: value[..., top:bottom, left:right]
            if value.ndim == 3 and key == "orientation"
            else value[top:bottom, left:right]
            for key, value in dense.items()
        }
        return image[top:bottom, left:right], cropped_dense

    def _augment(
        self,
        image: np.ndarray,
        dense: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        image, dense = self._training_crop(image, dense)
        height, width = image.shape[:2]
        if self.training:
            angle = float(np.random.uniform(-self.max_rotation, self.max_rotation))
            if abs(angle) > 1.0e-4:
                scale = float(np.random.uniform(0.92, 1.08))
                matrix = cv2.getRotationMatrix2D(
                    (width / 2.0, height / 2.0),
                    angle,
                    scale,
                )
                image = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for key, value in dense.items():
                    if key == "orientation":
                        components = [
                            cv2.warpAffine(
                                value[component],
                                matrix,
                                (width, height),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101,
                            )
                            for component in range(value.shape[0])
                        ]
                        dense[key] = np.stack(components, axis=0)
                    else:
                        interpolation = (
                            cv2.INTER_NEAREST if key != "width" else cv2.INTER_LINEAR
                        )
                        dense[key] = cv2.warpAffine(
                            value,
                            matrix,
                            (width, height),
                            flags=interpolation,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                orientation = dense["orientation"]
                angle_radians = np.deg2rad(angle)
                cosine = np.cos(angle_radians)
                sine = np.sin(angle_radians)
                x_component = cosine * orientation[0] - sine * orientation[1]
                y_component = sine * orientation[0] + cosine * orientation[1]
                dense["orientation"] = np.stack([x_component, y_component], axis=0)

            if np.random.random() < 0.5:
                image = image[:, ::-1]
                for key, value in dense.items():
                    dense[key] = value[:, ::-1]
                dense["orientation"][0] *= -1.0
            if np.random.random() < 0.5:
                image = image[::-1]
                for key, value in dense.items():
                    dense[key] = value[::-1]
                dense["orientation"][1] *= -1.0

            brightness = float(np.random.uniform(-0.12, 0.12))
            contrast = float(np.random.uniform(0.85, 1.15))
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            if np.random.random() < 0.35:
                gamma = float(np.random.uniform(0.75, 1.35))
                image = np.power(np.clip(image, 0.0, 1.0), gamma)
            if np.random.random() < 0.20:
                sigma = float(np.random.uniform(0.25, 1.15))
                image = cv2.GaussianBlur(
                    image,
                    ksize=(0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                )
            if np.random.random() < 0.25:
                noise = np.random.normal(
                    0.0,
                    float(np.random.uniform(0.008, 0.028)),
                    size=image.shape,
                )
                image = np.clip(image + noise, 0.0, 1.0)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        for key, value in dense.items():
            if key == "orientation":
                value = value.transpose(1, 2, 0)
                resized = cv2.resize(
                    value,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                dense[key] = resized.transpose(2, 0, 1)
            else:
                interpolation = (
                    cv2.INTER_NEAREST if key != "width" else cv2.INTER_LINEAR
                )
                dense[key] = cv2.resize(
                    value,
                    (self.image_size, self.image_size),
                    interpolation=interpolation,
                )
        return image, dense

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve_path(row["image"], self.base)
        mask_path = _resolve_path(row["mask"], self.base)
        image = _read_image(image_path, grayscale=False)
        mask = _read_image(mask_path, grayscale=True)
        mask = (mask > 0.5).astype(np.float32)

        cache_key = hashlib.sha1(
            f"{mask_path}:{mask_path.stat().st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:16]
        derived = _load_cached_targets(
            self.cache_dir / f"{cache_key}.npz",
            mask,
            normalize_width=True,
        )

        overrides = {
            "skeleton": self._optional_path(row, "skeleton"),
            "endpoint": self._optional_path(row, "endpoint"),
            "junction": self._optional_path(row, "junction"),
            "orientation": self._optional_path(row, "orientation"),
            "width": self._optional_path(row, "width"),
        }
        for key, path in overrides.items():
            if path is not None:
                if key == "orientation":
                    orientation = _read_image(path)
                    orientation = orientation[..., :2].transpose(2, 0, 1)
                    orientation = orientation * 2.0 - 1.0
                    norm = np.linalg.norm(orientation, axis=0, keepdims=True)
                    orientation = orientation / np.clip(norm, 1.0e-6, None)
                    derived[key] = orientation.astype(np.float32)
                elif key == "width":
                    derived[key] = _read_image(path, grayscale=True)
                else:
                    derived[key] = _read_image(path, grayscale=True)

        use_flags = {
            "skeleton": bool(self.config.get("use_skeleton", True)),
            "endpoint": bool(self.config.get("use_endpoints", True)),
            "junction": bool(self.config.get("use_junctions", True)),
            "orientation": bool(self.config.get("use_orientation", True)),
            "width": bool(self.config.get("use_width", True)),
        }
        dense = {"mask": mask, "skeleton": derived["skeleton"], "width": derived["width"]}
        dense["endpoint"] = derived["endpoint"] if use_flags["endpoint"] else np.zeros_like(mask)
        dense["junction"] = derived["junction"] if use_flags["junction"] else np.zeros_like(mask)
        dense["orientation"] = (
            derived["orientation"] if use_flags["orientation"] else np.zeros((2, *mask.shape), np.float32)
        )
        if not use_flags["skeleton"]:
            dense["skeleton"] = np.zeros_like(mask)
        if not use_flags["width"]:
            dense["width"] = np.zeros_like(mask)

        image, dense = self._augment(image, dense)
        severity_value = row.get("severity", "")
        if severity_value == "":
            severity = -1
        else:
            severity = int(float(severity_value))
        severity_valid = 1.0 if (
            bool(self.config.get("use_severity", False)) and severity >= 0
        ) else 0.0

        sample = {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(dense["mask"][None].copy()).float(),
            "skeleton": torch.from_numpy(dense["skeleton"][None].copy()).float(),
            "endpoint": torch.from_numpy(dense["endpoint"][None].copy()).float(),
            "junction": torch.from_numpy(dense["junction"][None].copy()).float(),
            "orientation": torch.from_numpy(dense["orientation"].copy()).float(),
            "width": torch.from_numpy(dense["width"][None].copy()).float(),
            "severity": torch.tensor(severity, dtype=torch.long),
            "severity_valid": torch.tensor(severity_valid, dtype=torch.float32),
            "meta": {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            },
        }
        return sample


class SyntheticWallDataset(Dataset):
    """Procedural wall images for smoke testing and code validation only."""

    def __init__(
        self,
        size: int,
        length: int,
        seed: int = 0,
        num_severity_classes: int = 5,
    ) -> None:
        self.size = size
        self.length = length
        self.seed = seed
        self.num_severity_classes = num_severity_classes

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _draw_wall(rng: np.random.Generator, size: int) -> np.ndarray:
        base_color = rng.integers(130, 205, size=3)
        image = np.ones((size, size, 3), dtype=np.float32) * base_color
        noise = rng.normal(0.0, 5.5, size=(size, size, 1))
        image = np.clip(image + noise, 0.0, 255.0)

        brick_height = int(rng.integers(max(16, size // 12), max(22, size // 7)))
        brick_width = int(rng.integers(max(24, size // 8), max(36, size // 5)))
        mortar = max(1, int(rng.integers(1, 4)))
        mortar_color = np.clip(base_color + rng.normal(0, 8, size=3), 90, 225)
        for row_index, y in enumerate(range(0, size, brick_height)):
            offset = 0 if row_index % 2 == 0 else brick_width // 2
            image[y : y + mortar] = mortar_color
            for x in range(-offset, size, brick_width):
                image[y : min(y + brick_height, size), x : x + mortar] = mortar_color

        stain_count = int(rng.integers(0, 4))
        for _ in range(stain_count):
            center = (
                int(rng.integers(0, size)),
                int(rng.integers(0, size)),
            )
            axes = (
                int(rng.integers(size // 20, size // 7)),
                int(rng.integers(size // 20, size // 8)),
            )
            angle = float(rng.uniform(0, 180))
            color = tuple(float(value) for value in rng.integers(60, 150, size=3))
            overlay = image.copy()
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
            image = 0.82 * image + 0.18 * overlay
        return np.clip(image, 0.0, 255.0)

    @staticmethod
    def _draw_crack(
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.uint8)
        start_x = int(rng.integers(0, size))
        start_y = 0 if rng.random() < 0.5 else size - 1
        vertical = abs(start_y - size // 2) > size // 4
        if vertical:
            x, y = start_x, 0
            direction = 1
            step = max(5, size // 28)
            thickness = int(rng.integers(1, max(2, size // 70 + 2)))
            while y < size:
                next_y = min(size - 1, y + step)
                next_x = int(np.clip(x + rng.integers(-step // 2, step // 2 + 1), 0, size - 1))
                cv2.line(mask, (x, y), (next_x, next_y), 1, thickness)
                if rng.random() < 0.18:
                    branch_x = int(np.clip(next_x + rng.integers(-step * 2, step * 2 + 1), 0, size - 1))
                    branch_y = min(size - 1, next_y + step * 2)
                    cv2.line(mask, (next_x, next_y), (branch_x, branch_y), 1, max(1, thickness - 1))
                if next_y == size - 1:
                    break
                x, y = next_x, next_y
                direction += 1
        else:
            x, y = 0, start_y
            step = max(5, size // 28)
            thickness = int(rng.integers(1, max(2, size // 70 + 2)))
            while x < size:
                next_x = min(size - 1, x + step)
                next_y = int(np.clip(y + rng.integers(-step // 2, step // 2 + 1), 0, size - 1))
                cv2.line(mask, (x, y), (next_x, next_y), 1, thickness)
                if rng.random() < 0.18:
                    branch_x = min(size - 1, next_x + step * 2)
                    branch_y = int(np.clip(next_y + rng.integers(-step * 2, step * 2 + 1), 0, size - 1))
                    cv2.line(mask, (next_x, next_y), (branch_x, branch_y), 1, max(1, thickness - 1))
                if next_x == size - 1:
                    break
                x, y = next_x, next_y
        return mask.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + index * 7919)
        image = self._draw_wall(rng, self.size)
        mask = self._draw_crack(rng, self.size)
        dark = np.where(mask > 0)
        image[dark] = np.clip(image[dark] * rng.uniform(0.25, 0.55), 0, 255)
        shadow_gradient = np.linspace(
            rng.uniform(0.75, 1.0),
            rng.uniform(0.75, 1.0),
            self.size,
            dtype=np.float32,
        )[None, :, None]
        image = np.clip(image * shadow_gradient, 0.0, 255.0)

        image = image.astype(np.float32) / 255.0
        targets = derive_targets(mask, normalize_width=True)
        severity = geometry_severity(mask, targets["skeleton"], self.num_severity_classes)
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(targets["mask"][None].copy()).float(),
            "skeleton": torch.from_numpy(targets["skeleton"][None].copy()).float(),
            "endpoint": torch.from_numpy(targets["endpoint"][None].copy()).float(),
            "junction": torch.from_numpy(targets["junction"][None].copy()).float(),
            "orientation": torch.from_numpy(targets["orientation"].copy()).float(),
            "width": torch.from_numpy(targets["width"][None].copy()).float(),
            "severity": torch.tensor(severity, dtype=torch.long),
            "severity_valid": torch.tensor(1.0, dtype=torch.float32),
            "meta": {"sample": index},
        }
