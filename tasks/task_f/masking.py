"""Deterministic random masks for Task F."""
from __future__ import annotations

import numpy as np


DEFAULT_MASK_RATIOS = (0.25, 0.50, 0.75)


def validate_mask_ratio(mask_ratio: float) -> float:
    ratio = float(mask_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask_ratio must be between 0 and 1, got {mask_ratio}.")
    return ratio


def random_observation_mask(
    num_steps: int,
    mask_ratio: float,
    *,
    seed: int,
    sample_id: int,
    epoch: int = 0,
) -> np.ndarray:
    """Return a reproducible mask where True means observed.

    Exactly ``round(num_steps * mask_ratio)`` randomly chosen time positions are
    hidden. The seed depends only on explicit experiment state, never labels or
    trajectory completion times.
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}.")
    ratio = validate_mask_ratio(mask_ratio)
    num_hidden = int(round(num_steps * ratio))
    num_hidden = min(max(num_hidden, 1), num_steps - 1)
    sequence = np.random.SeedSequence(
        [int(seed), int(sample_id) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF]
    )
    rng = np.random.default_rng(sequence)
    hidden = rng.choice(num_steps, size=num_hidden, replace=False)
    observed = np.ones(num_steps, dtype=bool)
    observed[hidden] = False
    return observed


def apply_observation_mask(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    append_mask_channel: bool = True,
) -> np.ndarray:
    """Zero hidden time positions and optionally append the observed flag."""
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [time, channels], got {values.shape}.")
    if observed.shape != (values.shape[0],):
        raise ValueError(
            f"observed mask shape {observed.shape} does not match {values.shape[0]} steps."
        )
    masked = values.copy()
    masked[~observed] = 0.0
    if not append_mask_channel:
        return masked
    return np.concatenate([masked, observed[:, None].astype(np.float32)], axis=1)
