"""Morphology-conditioned delayed state memory."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, LayerNorm2d, ResidualBlock


class MDSMBlock(nn.Module):
    """Four-direction delayed state propagation with morphology gates."""

    def __init__(
        self,
        channels: int,
        delays: Sequence[int] = (1, 2, 4, 8),
        use_morphology: bool = True,
    ) -> None:
        super().__init__()
        if not delays:
            raise ValueError("At least one delay is required")
        if list(delays) != sorted(delays):
            raise ValueError("Delays must be sorted")
        self.channels = channels
        self.delays = tuple(int(delay) for delay in delays)
        self.max_delay = max(self.delays)
        self.use_morphology = bool(use_morphology)

        hidden = channels * 2
        self.morphology_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 3, kernel_size=1),
        )
        self.gate_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_b = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.delay_logits = nn.Conv2d(channels * 4, len(self.delays), kernel_size=1)
        self.drive = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_drive = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.local_projection = nn.Conv2d(channels, channels, kernel_size=1)

        self.direction_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )
        self.output = nn.Sequential(
            LayerNorm2d(channels),
            ResidualBlock(channels),
            ConvNormAct(channels, channels, kernel_size=1),
        )
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _orient(tensor: torch.Tensor, direction: int) -> torch.Tensor:
        if direction == 1:
            tensor = tensor.flip(-1)
        elif direction == 2:
            tensor = tensor.permute(0, 1, 3, 2)
        elif direction == 3:
            tensor = tensor.permute(0, 1, 3, 2).flip(-1)
        return tensor.contiguous()

    @staticmethod
    def _restore(tensor: torch.Tensor, direction: int) -> torch.Tensor:
        if direction == 1:
            tensor = tensor.flip(-1)
        elif direction == 2:
            tensor = tensor.permute(0, 1, 3, 2)
        elif direction == 3:
            tensor = tensor.flip(-1)
            tensor = tensor.permute(0, 1, 3, 2)
        return tensor.contiguous()

    def _scan_one_direction(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        beta: torch.Tensor,
        drive: torch.Tensor,
        direction: int,
    ) -> torch.Tensor:
        a = self._orient(a, direction)
        b = self._orient(b, direction)
        beta = self._orient(beta, direction)
        drive = self._orient(drive, direction)

        length = a.shape[-1]
        history: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for position in range(length):
            a_t = a[..., position]
            b_t = b[..., position]
            drive_t = drive[..., position]
            beta_t = beta[..., position]

            previous = history[-1] if history else torch.zeros_like(drive_t)
            delayed = torch.zeros_like(drive_t)
            for delay_index, delay in enumerate(self.delays):
                if len(history) > delay:
                    weight = beta_t[:, delay_index : delay_index + 1]
                    delayed = delayed + weight * history[-(delay + 1)]

            state = a_t * previous + b_t * delayed + drive_t
            history.append(state)
            if len(history) > self.max_delay + 1:
                history.pop(0)
            outputs.append(state)

        result = torch.stack(outputs, dim=-1)
        return self._restore(result, direction)

    def forward(self, x: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        if local.shape[-2:] != x.shape[-2:]:
            local = F.interpolate(local, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.use_morphology:
            morphology = torch.sigmoid(
                self.morphology_gate(torch.cat([x, local], dim=1))
            )
        else:
            morphology = x.new_zeros(
                x.shape[0],
                self.channels * 3,
                *x.shape[-2:],
            )
        joined = torch.cat([x, morphology], dim=1)
        gate_a = torch.sigmoid(self.gate_a(x))
        gate_b = torch.sigmoid(self.gate_b(joined))
        beta = F.softmax(self.delay_logits(joined), dim=1)
        local_drive = torch.sigmoid(self.local_drive(torch.cat([x, local], dim=1)))
        drive = self.drive(x) + local_drive * self.local_projection(local)

        states = [
            self._scan_one_direction(gate_a, gate_b, beta, drive, direction)
            for direction in range(4)
        ]
        stacked = torch.stack(states, dim=1)
        direction_weights = F.softmax(
            self.direction_gate(stacked.flatten(start_dim=1, end_dim=2)),
            dim=1,
        )
        fused = (stacked * direction_weights[:, :, None]).sum(dim=1)
        return x + torch.sigmoid(self.output_gate) * self.output(fused)
