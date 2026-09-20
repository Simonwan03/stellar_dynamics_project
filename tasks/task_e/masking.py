"""Partial-observation wrappers for Task E forecasting windows.

The physical channels are masked in the already standardized coordinate system.
Hidden timestamps are filled with zero and one extra channel records whether a
timestamp was observed.  The target is never masked and keeps the original
physical channel count.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from tasks.task_f.masking import apply_observation_mask
from tasks.task_f.mixed_masking import fixed_mask, sample_training_mask


class MaskedForecastWindowDataset(Dataset):
    """Apply deterministic or per-epoch Task-F-compatible masks to contexts."""

    def __init__(
        self,
        base: Dataset,
        *,
        regime: str,
        seed: int,
        mask_ratio: float = 0.0,
        mask_kind: str = "random",
        append_mask_channel: bool = True,
    ) -> None:
        if regime not in {"clean_only", "random_only", "mixed", "fixed"}:
            raise ValueError(f"Unknown mask regime {regime!r}.")
        if not 0.0 <= float(mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        if mask_kind not in {"random", "single_block", "multi_block"}:
            raise ValueError(f"Unknown mask kind {mask_kind!r}.")
        self.base = base
        self.regime = str(regime)
        self.seed = int(seed)
        self.mask_ratio = float(mask_ratio)
        self.mask_kind = str(mask_kind)
        self.append_mask_channel = bool(append_mask_channel)
        self.epoch = 0
        # DynamicTrajectoryWindowDataset also changes its selected records per
        # epoch.  Expose the flag so DataLoader does not retain stale workers.
        self.dynamic_per_epoch = self.regime in {"random_only", "mixed"} or bool(
            getattr(base, "dynamic_per_epoch", False)
        )

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.base, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def _sample_id(self, item: dict[str, torch.Tensor]) -> int:
        trajectory = int(item["trajectory_index"])
        start = int(item["start"])
        # Keep trajectory and start in the seed while remaining compatible with
        # SeedSequence's 32-bit integer words.
        return trajectory * 1_000_003 + start

    def _observed_mask(self, num_steps: int, sample_id: int) -> tuple[np.ndarray, str, float]:
        if self.regime == "fixed":
            observed = fixed_mask(
                num_steps,
                self.mask_ratio,
                kind=self.mask_kind,
                seed=self.seed,
                sample_id=sample_id,
                epoch=0,
            )
            return observed, self.mask_kind, self.mask_ratio
        observed, kind, ratio = sample_training_mask(
            num_steps,
            seed=self.seed,
            sample_id=sample_id,
            epoch=self.epoch,
            regime=self.regime,
        )
        return observed, kind, ratio

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        context = item["context"].detach().cpu().numpy().astype(np.float32, copy=False)
        if context.ndim != 2:
            raise ValueError(f"Expected context [time, channels], got {context.shape}.")
        observed, kind, ratio = self._observed_mask(context.shape[0], self._sample_id(item))
        masked = apply_observation_mask(
            context,
            observed,
            append_mask_channel=self.append_mask_channel,
        )
        item["context"] = torch.from_numpy(masked)
        item["mask_observed"] = torch.from_numpy(observed)
        item["mask_kind"] = kind
        item["mask_ratio"] = torch.tensor(ratio, dtype=torch.float32)
        return item

    def window_record(self, index: int) -> tuple[int, int]:
        if hasattr(self.base, "window_record"):
            return self.base.window_record(index)
        raise AttributeError("The wrapped dataset does not expose window records.")

