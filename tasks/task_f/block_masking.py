"""Contiguous observation masks for stricter Task-F missingness tests."""
from __future__ import annotations

import numpy as np


BLOCK_MASK_MODES = ("contiguous1", "contiguous2")


def contiguous_observation_mask(
    num_steps: int,
    mask_ratio: float,
    *,
    seed: int,
    sample_id: int,
    epoch: int = 0,
    num_blocks: int = 1,
) -> np.ndarray:
    """Return a deterministic mask with one or two contiguous hidden blocks.

    The hidden count is exactly ``round(num_steps * mask_ratio)``.  For two
    blocks, both blocks are non-empty and at least one observed position is
    retained between them.
    """
    num_steps = int(num_steps)
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}.")
    ratio = float(mask_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"mask_ratio must be between 0 (inclusive) and 1, got {mask_ratio}.")
    num_blocks = int(num_blocks)
    if num_blocks not in (1, 2):
        raise ValueError(f"num_blocks must be 1 or 2, got {num_blocks}.")
    if ratio == 0.0:
        return np.ones(num_steps, dtype=bool)

    num_hidden = int(round(num_steps * ratio))
    num_hidden = min(max(num_hidden, 1), num_steps - 1)
    if num_blocks == 2 and num_hidden < 2:
        raise ValueError("Two contiguous blocks require at least two hidden positions.")

    sequence = np.random.SeedSequence(
        [int(seed), int(sample_id) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF]
    )
    rng = np.random.default_rng(sequence)
    observed = np.ones(num_steps, dtype=bool)
    if num_blocks == 1:
        start = int(rng.integers(0, num_steps - num_hidden + 1))
        observed[start : start + num_hidden] = False
        return observed

    first_hidden = int(rng.integers(1, num_hidden))
    second_hidden = num_hidden - first_hidden
    observed_count = num_steps - num_hidden
    middle_gap = int(rng.integers(1, observed_count + 1))
    outer_observed = observed_count - middle_gap
    before = int(rng.integers(0, outer_observed + 1))
    first_start = before
    second_start = first_start + first_hidden + middle_gap
    observed[first_start : first_start + first_hidden] = False
    observed[second_start : second_start + second_hidden] = False
    return observed


def contiguous_mask_mode_observation(
    num_steps: int,
    mask_ratio: float,
    *,
    seed: int,
    sample_id: int,
    epoch: int = 0,
    mode: str,
) -> np.ndarray:
    """Dispatch ``contiguous1`` or ``contiguous2`` mask generation."""
    if mode not in BLOCK_MASK_MODES:
        raise ValueError(f"Unknown contiguous mask mode {mode!r}; choose from {BLOCK_MASK_MODES}.")
    return contiguous_observation_mask(
        num_steps,
        mask_ratio,
        seed=seed,
        sample_id=sample_id,
        epoch=epoch,
        num_blocks=1 if mode == "contiguous1" else 2,
    )
