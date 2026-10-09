"""Training, validation, checkpointing, and prediction workflows."""

from __future__ import annotations

import csv
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ManifestCrackDataset
from .geometry import skeleton_length
from .losses import WallTopoLoss
from .metrics import composite_score, validate_batch
from .models import build_model
from .uncertainty import (
    geometry_dispersion,
    normalized_mahalanobis,
    orientation_dispersion,
    report_uncertainty,
    selective_metrics,
    topology_instability,
)
from .utils import (
    AverageMeter,
    append_jsonl,
    count_parameters,
    ensure_dir,
    move_to_device,
    set_seed,
    write_json,
    imwrite_unicode,
)


def build_dataloader(
    manifest: str | Path,
    config: dict[str, Any],
    training: bool,
    domain: str | None = None,
    group: str | None = None,
) -> DataLoader:
    dataset = ManifestCrackDataset(
        manifest=manifest,
        image_size=int(config["image_size"]),
        training=training,
        config=config["data"],
        domain=domain,
        group=group,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=training,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(torch.cuda.is_available()),
        drop_last=training and len(dataset) > int(config["batch_size"]),
        persistent_workers=int(config["num_workers"]) > 0,
    )


def consistency_loss(
    model: torch.nn.Module,
    image: torch.Tensor,
    outputs: torch.Tensor,
) -> torch.Tensor:
    flipped_image = image.flip(-1)
    flipped_outputs = model(flipped_image)
    mask = F.smooth_l1_loss(
        torch.sigmoid(outputs["mask_logits"]),
        torch.sigmoid(flipped_outputs["mask_logits"]).flip(-1),
    )
    skeleton = F.smooth_l1_loss(
        torch.sigmoid(outputs["skeleton_logits"]),
        torch.sigmoid(flipped_outputs["skeleton_logits"]).flip(-1),
    )
    width = F.smooth_l1_loss(
        outputs["width"],
        flipped_outputs["width"].flip(-1),
    )
    orientation_prediction = outputs["orientation"]
    orientation_flipped = flipped_outputs["orientation"].flip(-1)
    orientation_flipped = orientation_flipped.clone()
    orientation_flipped[:, 0] *= -1.0
    orientation = F.smooth_l1_loss(orientation_prediction, orientation_flipped)
    return mask + skeleton + width + orientation


class Trainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        set_seed(int(config["seed"]))
        self.output_dir = ensure_dir(config["output_dir"])
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.prediction_dir = ensure_dir(self.output_dir / "predictions")
        self.mask_threshold = float(
            config.get("evaluation", {}).get("threshold", 0.5)
        )
        threshold_path = self.output_dir / "threshold.json"
        if threshold_path.exists():
            with threshold_path.open("r", encoding="utf-8") as handle:
                threshold_payload = json.load(handle)
            self.mask_threshold = float(
                threshold_payload.get("mask_threshold", self.mask_threshold)
            )
        self.model = build_model(config).to(device)
        self.criterion = WallTopoLoss(config["loss"])
        self.uncertainty_config = dict(config.get("uncertainty", {}))
        self.report_calibration_path = self.output_dir / "report_calibration.npz"
        self.report_calibration = self._load_report_calibration()
        ema_config = dict(config.get("ema", {}))
        self.ema_enabled = bool(ema_config.get("enabled", False))
        self.ema_decay = float(ema_config.get("decay", 0.999))
        self.ema_start_epoch = int(ema_config.get("start_epoch", 10))
        self.ema_state: dict[str, torch.Tensor] | None = None
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler_name = str(config["scheduler"]["name"]).lower()
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(config["epochs"])),
                eta_min=float(config["scheduler"]["min_lr"]),
            )
        elif scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=max(1, int(config["validation"]["patience"]) // 3),
                min_lr=float(config["scheduler"]["min_lr"]),
            )
        else:
            self.scheduler = None

        self.use_amp = bool(config["amp"]) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.train_loader = build_dataloader(
            config["data"]["train_manifest"],
            config,
            training=True,
        )
        self.val_loader = build_dataloader(
            config["data"]["val_manifest"],
            config,
            training=False,
        )
        self.best_score = -np.inf
        self.patience_count = 0
        self.history_path = self.output_dir / "history.csv"
        self.events_path = self.output_dir / "events.jsonl"
        write_json(
            self.output_dir / "model_summary.json",
            {
                "parameters": count_parameters(self.model),
                "train_samples": len(self.train_loader.dataset),
                "validation_samples": len(self.val_loader.dataset),
                "device": str(device),
                "model_name": str(config.get("model_name", "walltopo_uq")),
                "report_calibrated": self.report_calibration is not None,
                "mask_threshold": self.mask_threshold,
                "ema_enabled": self.ema_enabled,
            },
        )

    def _load_report_calibration(self) -> dict[str, np.ndarray] | None:
        if not self.report_calibration_path.exists():
            return None
        payload = np.load(self.report_calibration_path)
        return {key: payload[key] for key in payload.files}

    @staticmethod
    def _restore_view(tensor: torch.Tensor, view: str) -> torch.Tensor:
        if view == "hflip":
            return tensor.flip(-1)
        if view == "vflip":
            return tensor.flip(-2)
        if view == "rot180":
            return tensor.flip(-1).flip(-2)
        return tensor

    @classmethod
    def _restore_orientation_view(cls, tensor: torch.Tensor, view: str) -> torch.Tensor:
        restored = cls._restore_view(tensor, view)
        if view in ("hflip", "rot180"):
            restored = restored.clone()
            restored[:, 0] *= -1.0
        if view in ("vflip", "rot180"):
            restored = restored.clone()
            restored[:, 1] *= -1.0
        return restored

    def _tta_views(self, image: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        available = {
            "identity": image,
            "hflip": image.flip(-1),
            "vflip": image.flip(-2),
            "rot180": image.flip(-1).flip(-2),
        }
        requested = int(self.uncertainty_config.get("tta_views", 4))
        names = ["identity", "hflip", "vflip", "rot180"][: max(1, min(4, requested))]
        return [(name, available[name]) for name in names]

    @torch.no_grad()
    def _forward_tta(
        self,
        image: torch.Tensor,
        outputs_for_auxiliary_heads: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Run geometric TTA and restore all maps to the original frame."""

        averaged: dict[str, torch.Tensor] = {}
        raw: dict[str, list[torch.Tensor]] = {
            "skeleton_prob": [],
            "width": [],
            "orientation": [],
            "mask_prob": [],
        }
        count = 0
        for view, transformed in self._tta_views(image):
            outputs = self.model(transformed)
            mask_prob = self._restore_view(torch.sigmoid(outputs["mask_logits"]), view)
            skeleton_prob = self._restore_view(
                torch.sigmoid(outputs["skeleton_logits"]),
                view,
            )
            width = self._restore_view(outputs["width"], view)
            orientation = self._restore_orientation_view(outputs["orientation"], view)
            evidence = self._restore_view(outputs["evidence"], view)
            tokens = outputs.get("topology_tokens")
            if tokens is not None:
                tokens = tokens.mean(dim=1)
                averaged["embedding"] = (
                    averaged["embedding"] + tokens
                    if "embedding" in averaged
                    else tokens
                )
            elif "topology_anchor_logits" in outputs:
                fallback = outputs["topology_anchor_logits"].mean(dim=(2, 3))
                averaged["embedding"] = (
                    averaged["embedding"] + fallback
                    if "embedding" in averaged
                    else fallback
                )
            raw["mask_prob"].append(mask_prob)
            raw["skeleton_prob"].append(skeleton_prob)
            raw["width"].append(width)
            raw["orientation"].append(orientation)
            for key in ("mask_logits", "skeleton_logits"):
                value = self._restore_view(outputs[key], view)
                averaged[key] = averaged[key] + value if key in averaged else value
            for key, value in (
                ("endpoint_logits", outputs["endpoint_logits"]),
                ("junction_logits", outputs["junction_logits"]),
                ("severity_logits", outputs["severity_logits"]),
            ):
                value = self._restore_view(value, view) if value.ndim == 4 else value
                averaged[key] = averaged[key] + value if key in averaged else value
            for key, value in (
                ("orientation", orientation),
                ("width", width),
                ("evidence", evidence),
            ):
                averaged[key] = averaged[key] + value if key in averaged else value
            count += 1

        for key, value in list(averaged.items()):
            averaged[key] = value / count
        averaged["mask_prob"] = torch.sigmoid(averaged["mask_logits"])
        if outputs_for_auxiliary_heads is not None:
            for key in ("topology_tokens", "topology_anchor_logits"):
                if key in outputs_for_auxiliary_heads:
                    averaged[key] = outputs_for_auxiliary_heads[key]
        return averaged, raw

    @staticmethod
    def _embedding_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if "topology_tokens" in outputs:
            return outputs["topology_tokens"].mean(dim=1)
        if "topology_anchor_logits" in outputs:
            return outputs["topology_anchor_logits"].mean(dim=(2, 3))
        return outputs["mask_logits"].mean(dim=(2, 3)).expand(-1, 4)

    def _domain_shift_score(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.report_calibration is None:
            return torch.zeros(embedding.shape[0], device=embedding.device)
        mean = self.report_calibration["feature_mean"]
        standard_deviation = self.report_calibration["feature_std"]
        scale = float(self.report_calibration["feature_scale"])
        embedding_np = embedding.detach().cpu().numpy()
        scores = [
            normalized_mahalanobis(vector, mean, standard_deviation, scale)
            for vector in embedding_np
        ]
        return torch.tensor(scores, dtype=embedding.dtype, device=embedding.device)

    def _report_uncertainty_batch(
        self,
        outputs: dict[str, torch.Tensor],
        raw: dict[str, list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_probability = outputs["mask_prob"]
        vacuity = 2.0 / outputs["evidence"].sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        skeleton_predictions = [
            item[:, 0].detach().cpu().numpy() >= self.mask_threshold
            for item in raw["skeleton_prob"]
        ]
        width_values = [
            item[:, 0].detach().cpu().numpy() for item in raw["width"]
        ]
        length_values: list[np.ndarray] = []
        for skeleton in skeleton_predictions:
            length_values.append(
                np.asarray(
                    [skeleton_length(item) for item in skeleton],
                    dtype=np.float64,
                )
            )
        orientation_values = [
            item.detach().cpu().numpy() for item in raw["orientation"]
        ]
        domain_shift = self._domain_shift_score(outputs["embedding"])

        report_values: list[float] = []
        topology_values: list[float] = []
        geometry_values: list[float] = []
        for index in range(mask_probability.shape[0]):
            topology_term = topology_instability(
                [prediction[index] for prediction in skeleton_predictions]
            )
            geometry_term = geometry_dispersion(
                [value[index].mean() for value in width_values],
                [float(value[index]) for value in length_values],
            )
            geometry_term = float(
                np.clip(
                    0.65 * geometry_term
                    + 0.35
                    * orientation_dispersion(
                        [value[index] for value in orientation_values],
                        skeleton_predictions[0][index],
                    ),
                    0.0,
                    1.0,
                )
            )
            report_values.append(
                report_uncertainty(
                    vacuity[index, 0].detach().cpu().numpy(),
                    skeleton_predictions[0][index],
                    topology_term,
                    geometry_term,
                    float(domain_shift[index].detach().cpu()),
                )
            )
            topology_values.append(topology_term)
            geometry_values.append(geometry_term)
        return (
            torch.tensor(report_values, dtype=mask_probability.dtype, device=self.device),
            torch.tensor(topology_values, dtype=mask_probability.dtype, device=self.device),
            torch.tensor(geometry_values, dtype=mask_probability.dtype, device=self.device),
        )

    @staticmethod
    def _loss_components_to_float(components: dict[str, torch.Tensor]) -> dict[str, float]:
        return {key: float(value.detach().cpu()) for key, value in components.items()}

    @torch.no_grad()
    def _update_ema(self, epoch: int) -> None:
        if not self.ema_enabled or epoch < self.ema_start_epoch:
            return
        current = self.model.state_dict()
        if self.ema_state is None:
            self.ema_state = {
                key: value.detach().clone() for key, value in current.items()
            }
            return
        for key, value in current.items():
            if value.is_floating_point():
                self.ema_state[key].mul_(self.ema_decay).add_(
                    value.detach(),
                    alpha=1.0 - self.ema_decay,
                )
            else:
                self.ema_state[key].copy_(value)

    def _ema_ready(self, epoch: int) -> bool:
        return (
            self.ema_enabled
            and self.ema_state is not None
            and epoch >= self.ema_start_epoch
        )

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        meter = AverageMeter()
        component_meter: dict[str, AverageMeter] = {}
        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        accumulation = max(1, int(self.config["grad_accum_steps"]))
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress):
            batch = move_to_device(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch["image"])
                loss, components = self.criterion(
                    outputs,
                    batch,
                    epoch_fraction=epoch / max(int(self.config["epochs"]) - 1, 1),
                )
                consistency_weight = float(self.config["loss"].get("consistency", 0.0))
                if consistency_weight > 0:
                    consistency = consistency_loss(self.model, batch["image"], outputs)
                    loss = loss + consistency_weight * consistency
                    components["loss_consistency"] = consistency.detach()
                loss = loss / accumulation

            self.scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._update_ema(epoch)

            batch_size = batch["image"].shape[0]
            meter.update(float(loss.detach().cpu()) * accumulation, batch_size)
            for key, value in self._loss_components_to_float(components).items():
                component_meter.setdefault(key, AverageMeter()).update(value, batch_size)
            progress.set_postfix(loss=f"{meter.average:.4f}")

        metrics = {"train_loss": meter.average}
        metrics.update(
            {f"train_{key}": value.average for key, value in component_meter.items()}
        )
        return metrics

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader | None = None,
        use_tta: bool = False,
        threshold: float | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        loader = loader or self.val_loader
        threshold = self.mask_threshold if threshold is None else float(threshold)
        per_image: list[dict[str, Any]] = []
        losses: list[float] = []
        for batch in tqdm(loader, desc="validate", leave=False):
            batch = move_to_device(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                if use_tta:
                    outputs, raw = self._forward_tta(batch["image"])
                else:
                    outputs = self.model(batch["image"])
                    raw = None
                loss, _ = self.criterion(outputs, batch, epoch_fraction=1.0)
                if raw is not None:
                    report, topology_term, geometry_term = self._report_uncertainty_batch(
                        outputs,
                        raw,
                    )
                else:
                    report = (
                        2.0
                        / outputs["evidence"].sum(dim=1, keepdim=True).clamp_min(1.0e-6)
                    ).mean(dim=(1, 2, 3))
                    topology_term = torch.zeros_like(report)
                    geometry_term = torch.zeros_like(report)
            losses.append(float(loss.detach().cpu()))
            _, image_metrics = validate_batch(
                outputs,
                batch,
                threshold=threshold,
            )
            for index, metrics in enumerate(image_metrics):
                metrics["report_uncertainty"] = float(report[index].detach().cpu())
                metrics["topology_uncertainty"] = float(
                    topology_term[index].detach().cpu()
                )
                metrics["geometry_uncertainty"] = float(
                    geometry_term[index].detach().cpu()
                )
            per_image.extend(image_metrics)

        keys = sorted({key for metrics in per_image for key in metrics})
        metrics = {
            key: float(np.mean([item.get(key, 0.0) for item in per_image]))
            for key in keys
        }
        metrics["val_loss"] = float(np.mean(losses))
        errors = np.asarray([item["image_error"] for item in per_image], dtype=np.float64)
        width_errors = np.asarray(
            [item.get("width_mae_px", 0.0) for item in per_image],
            dtype=np.float64,
        )
        uncertainty_key = "report_uncertainty" if use_tta else "mean_vacuity"
        uncertainty = np.asarray(
            [item[uncertainty_key] for item in per_image],
            dtype=np.float64,
        )
        metrics.update(
            selective_metrics(
                errors,
                uncertainty,
                secondary_error=width_errors,
                coverage=float(self.uncertainty_config.get("target_coverage", 0.8)),
            )
        )
        metrics["composite"] = composite_score(metrics)
        metrics["mask_threshold"] = float(threshold)
        self._last_validation_per_image = per_image
        return metrics

    @torch.no_grad()
    def _collect_embeddings(self, loader: DataLoader, max_samples: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        collected = 0
        for batch in tqdm(loader, desc="calibrate", leave=False):
            batch = move_to_device(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch["image"])
            embedding = self._embedding_from_outputs(outputs)
            vectors.append(embedding.detach().float().cpu().numpy())
            collected += embedding.shape[0]
            if collected >= max_samples:
                break
        if not vectors:
            raise RuntimeError("No samples were available for uncertainty calibration.")
        return np.concatenate(vectors, axis=0)[:max_samples]

    def calibrate_report_uncertainty(self) -> dict[str, float]:
        """Fit domain reference and selective-reporting threshold.

        The domain reference is estimated only on the training split. The
        review threshold is selected on validation images without using test
        labels.
        """

        max_samples = int(
            self.uncertainty_config.get("max_calibration_samples", 128)
        )
        train_features = self._collect_embeddings(self.train_loader, max_samples)
        feature_mean = train_features.mean(axis=0)
        feature_std = np.maximum(train_features.std(axis=0), 1.0e-4)
        distances = np.sqrt(
            np.mean(
                ((train_features - feature_mean) / feature_std) ** 2,
                axis=1,
            )
        )
        feature_scale = max(float(np.quantile(distances, 0.95)), 1.0e-4)

        checkpoint_path = self.checkpoint_dir / "best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model"])
        self.report_calibration = {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_scale": np.asarray(feature_scale),
            "review_threshold": np.asarray(1.0),
        }

        metrics = self.validate(self.val_loader, use_tta=True)
        report_values = np.asarray(
            [item["report_uncertainty"] for item in self._last_validation_per_image],
            dtype=np.float64,
        )
        errors = np.asarray(
            [item["image_error"] for item in self._last_validation_per_image],
            dtype=np.float64,
        )
        target_coverage = float(
            self.uncertainty_config.get("target_coverage", 0.8)
        )
        target_error = self.uncertainty_config.get("target_error")
        order = np.argsort(report_values)
        accepted_count = max(1, int(np.ceil(target_coverage * report_values.size)))
        threshold_without_feasible_target: float | None = None
        if target_error is not None:
            sorted_errors = errors[order]
            accepted_count = 0
            for candidate in range(1, errors.size + 1):
                if float(sorted_errors[:candidate].mean()) <= float(target_error):
                    accepted_count = candidate
            if accepted_count == 0:
                threshold_without_feasible_target = max(
                    0.0,
                    float(report_values.min()) - 1.0e-6,
                )
        accepted_count = min(max(accepted_count, 1), report_values.size)
        review_threshold = (
            float(
                np.quantile(report_values, accepted_count / report_values.size)
            )
            if threshold_without_feasible_target is None
            else threshold_without_feasible_target
        )
        self.report_calibration["review_threshold"] = np.asarray(review_threshold)
        np.savez_compressed(
            self.report_calibration_path,
            **self.report_calibration,
        )
        summary = {
            "review_threshold": review_threshold,
            "validation_coverage": float(accepted_count / report_values.size),
            "validation_selective_dice": float(1.0 - metrics["selective_error"]),
            "validation_aurc": float(metrics["aurc"]),
            "validation_error_detection_auroc": float(
                metrics["error_detection_auroc"]
            ),
            "feature_scale": feature_scale,
        }
        write_json(self.output_dir / "report_calibration.json", summary)
        return summary

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], name: str) -> Path:
        path = self.checkpoint_dir / f"{name}.pt"
        include_optimizer = name == "last"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": (
                    self.optimizer.state_dict() if include_optimizer else None
                ),
                "scheduler": (
                    self.scheduler.state_dict()
                    if include_optimizer and self.scheduler
                    else None
                ),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        return path

    def _append_history(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        exists = self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def fit(self) -> dict[str, float]:
        best_metrics: dict[str, float] = {}
        start = time.time()
        if self.history_path.exists():
            self.history_path.unlink()
        if self.events_path.exists():
            self.events_path.unlink()
        for epoch in range(int(self.config["epochs"])):
            train_metrics = self.train_epoch(epoch)
            validation_interval = max(1, int(self.config["validation"]["interval"]))
            if (epoch + 1) % validation_interval == 0 or epoch + 1 == int(self.config["epochs"]):
                training_state = None
                if self._ema_ready(epoch):
                    training_state = self._snapshot_model_state()
                    self.model.load_state_dict(self.ema_state)
                val_metrics = self.validate()
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["composite"])
                self._append_history(epoch, train_metrics, val_metrics)
                append_jsonl(
                    self.events_path,
                    {"epoch": epoch, **train_metrics, **val_metrics},
                )
                score = float(val_metrics[self.config["validation"]["metric"]])
                if score > self.best_score:
                    self.best_score = score
                    self.patience_count = 0
                    best_metrics = val_metrics
                    self.save_checkpoint(epoch, val_metrics, "best")
                else:
                    self.patience_count += 1
                print(
                    f"epoch={epoch:03d} train_loss={train_metrics['train_loss']:.4f} "
                    f"dice={val_metrics['dice']:.4f} cldice={val_metrics['cl_dice']:.4f} "
                    f"width_mae={val_metrics['width_mae_px']:.3f} composite={score:.4f}"
                )
                if self.patience_count >= int(self.config["validation"]["patience"]):
                    print("Early stopping triggered.")
                    if training_state is not None:
                        self.model.load_state_dict(training_state)
                    break
                if training_state is not None:
                    self.model.load_state_dict(training_state)
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
        self.save_checkpoint(
            epoch,
            best_metrics,
            "last",
        )
        summary = {
            "best_metrics": best_metrics,
            "elapsed_seconds": time.time() - start,
            "parameters": count_parameters(self.model),
        }
        if bool(self.uncertainty_config.get("calibrate", False)):
            summary["report_calibration"] = self.calibrate_report_uncertainty()
        write_json(self.output_dir / "training_summary.json", summary)
        return best_metrics

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, save_panels: bool = False) -> list[dict[str, Any]]:
        import cv2

        self.model.eval()
        records: list[dict[str, Any]] = []
        panel_dir = ensure_dir(self.prediction_dir / "panels")
        for batch_index, batch in enumerate(tqdm(loader, desc="predict", leave=False)):
            batch = move_to_device(batch, self.device)
            use_tta = int(self.uncertainty_config.get("tta_views", 4)) > 1
            if use_tta:
                outputs, raw = self._forward_tta(batch["image"])
                report, topology_term, geometry_term = self._report_uncertainty_batch(
                    outputs,
                    raw,
                )
                domain_shift = self._domain_shift_score(outputs["embedding"])
            else:
                outputs = self.model(batch["image"])
                raw = {
                    "skeleton_prob": [torch.sigmoid(outputs["skeleton_logits"])],
                }
                report = (
                    2.0 / outputs["evidence"].sum(dim=1, keepdim=True).clamp_min(1.0e-6)
                ).mean(dim=(1, 2, 3))
                topology_term = torch.zeros_like(report)
                geometry_term = torch.zeros_like(report)
                domain_shift = torch.zeros_like(report)
            probability = outputs["mask_prob"]
            skeleton_probability = torch.sigmoid(outputs["skeleton_logits"])
            vacuity = 2.0 / outputs["evidence"].sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            threshold = (
                float(self.report_calibration["review_threshold"])
                if self.report_calibration is not None
                and "review_threshold" in self.report_calibration
                else float("inf")
            )
            for index in range(batch["image"].shape[0]):
                image = batch["image"][index].detach().cpu().numpy().transpose(1, 2, 0)
                mask = (
                    probability[index, 0].detach().cpu().numpy()
                    >= self.mask_threshold
                )
                skeleton = (
                    skeleton_probability[index, 0].detach().cpu().numpy()
                    >= self.mask_threshold
                )
                uncertainty_map = vacuity[index, 0].detach().cpu().numpy()
                width = outputs["width"][index, 0].detach().cpu().numpy() * mask.shape[-1]
                orientation = outputs["orientation"][index].detach().cpu().numpy()
                report_value = float(report[index].detach().cpu())
                record = {
                    "batch": batch_index,
                    "index": index,
                    "image_path": batch["meta"]["image_path"][index],
                    "mask_area": float(mask.mean()),
                    "mean_vacuity": float(uncertainty_map.mean()),
                    "report_uncertainty": report_value,
                    "topology_uncertainty": float(
                        topology_term[index].detach().cpu()
                    ),
                    "geometry_uncertainty": float(
                        geometry_term[index].detach().cpu()
                    ),
                    "domain_shift": float(domain_shift[index].detach().cpu()),
                    "manual_review": bool(report_value > threshold),
                    "width_p90": float(np.quantile(width[mask], 0.9)) if mask.any() else 0.0,
                    "orientation_mean": float(np.abs(orientation[:, skeleton]).mean())
                    if skeleton.any()
                    else 0.0,
                }
                records.append(record)
                if save_panels:
                    uncertainty_visual = np.clip(uncertainty_map, 0.0, 1.0)
                    panel = np.concatenate(
                        [
                            image,
                            np.repeat(mask[..., None], 3, axis=2),
                            np.repeat(skeleton[..., None], 3, axis=2),
                            np.repeat(uncertainty_visual[..., None], 3, axis=2),
                        ],
                        axis=1,
                    )
                    panel = cv2.cvtColor((panel.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    imwrite_unicode(
                        panel_dir / f"panel_{batch_index:04d}_{index:02d}.png",
                        panel,
                    )
        write_json(self.prediction_dir / "predictions.json", {"records": records})
        return records
