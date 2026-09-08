"""Mixed-mask policies for the Task-F partial-observation benchmark."""
from __future__ import annotations

from typing import Literal

import numpy as np


MaskRegime = Literal["clean_only", "random_only", "mixed"]
MaskKind = Literal["random", "single_block", "multi_block"]

MASK_REGIMES = ("clean_only", "random_only", "mixed")
MASK_KINDS = ("random", "single_block", "multi_block")
DEFAULT_CLEAN_PROBABILITY = 0.10
DEFAULT_MAX_TRAIN_RATIO = 0.50
DEFAULT_BLOCK_MIN_LENGTH = 4
DEFAULT_BLOCK_MIN_GAP = 4


def _rng(seed: int, sample_id: int, epoch: int, view: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(sample_id) & 0xFFFFFFFF,
            int(epoch) & 0xFFFFFFFF,
            int(view) & 0xFFFFFFFF,
        ]
    )
    return np.random.default_rng(sequence)


def _random_mask(num_steps: int, num_hidden: int, rng: np.random.Generator) -> np.ndarray:
    observed = np.ones(num_steps, dtype=bool)
    if num_hidden:
        observed[rng.choice(num_steps, size=num_hidden, replace=False)] = False
    return observed


def _random_composition(
    total: int,
    parts: int,
    rng: np.random.Generator,
    *,
    minimum: int = 0,
) -> list[int]:
    if parts <= 0 or total < parts * minimum:
        raise ValueError(f"Cannot split total={total} into {parts} parts with minimum={minimum}.")
    result: list[int] = []
    remaining = int(total)
    for index in range(parts - 1):
        remaining_parts = parts - index - 1
        low = minimum
        high = remaining - minimum * remaining_parts
        value = int(rng.integers(low, high + 1))
        result.append(value)
        remaining -= value
    result.append(remaining)
    return result


def _block_mask(
    num_steps: int,
    num_hidden: int,
    rng: np.random.Generator,
    *,
    num_blocks: int,
    min_length: int = DEFAULT_BLOCK_MIN_LENGTH,
    min_gap: int = DEFAULT_BLOCK_MIN_GAP,
) -> np.ndarray:
    if num_hidden <= 0:
        return np.ones(num_steps, dtype=bool)
    observed_count = num_steps - num_hidden
    if observed_count < (num_blocks - 1) * min_gap:
        raise ValueError("Not enough observed positions to separate the requested blocks.")
    lengths = _random_composition(num_hidden, num_blocks, rng, minimum=min_length)
    residual_observed = observed_count - (num_blocks - 1) * min_gap
    gaps = _random_composition(residual_observed, num_blocks + 1, rng, minimum=0)

    observed = np.ones(num_steps, dtype=bool)
    cursor = gaps[0]
    for index, length in enumerate(lengths):
        observed[cursor : cursor + length] = False
        cursor += length
        if index < num_blocks - 1:
            cursor += min_gap + gaps[index + 1]
    if int((~observed).sum()) != num_hidden:
        raise AssertionError("Block mask did not preserve the requested hidden count.")
    return observed


def fixed_mask(
    num_steps: int,
    mask_ratio: float,
    *,
    kind: str,
    seed: int,
    sample_id: int,
    epoch: int = 0,
    view: int = 0,
    fixed_start: int | None = None,
    num_blocks: int | None = None,
) -> np.ndarray:
    """Create one deterministic mask with an exact hidden count.

    ``True`` means observed.  The random, single-block and multi-block modes
    all hide exactly ``round(num_steps * mask_ratio)`` timestamps.
    """
    num_steps = int(num_steps)
    ratio = float(mask_ratio)
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")
    if not 0.0 <= ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1).")
    if kind not in MASK_KINDS:
        raise ValueError(f"Unknown mask kind {kind!r}; choose from {MASK_KINDS}.")
    num_hidden = min(max(int(round(num_steps * ratio)), 0), num_steps - 1)
    if num_hidden == 0:
        return np.ones(num_steps, dtype=bool)
    rng = _rng(seed, sample_id, epoch, view)
    if kind == "random":
        return _random_mask(num_steps, num_hidden, rng)
    if kind == "single_block":
        start = (
            int(fixed_start)
            if fixed_start is not None
            else int(rng.integers(0, num_steps - num_hidden + 1))
        )
        if not 0 <= start <= num_steps - num_hidden:
            raise ValueError("fixed_start does not fit the requested block.")
        observed = np.ones(num_steps, dtype=bool)
        observed[start : start + num_hidden] = False
        return observed
    blocks = int(num_blocks) if num_blocks is not None else int(rng.integers(2, 5))
    blocks = min(blocks, max(2, num_hidden // DEFAULT_BLOCK_MIN_LENGTH))
    if num_hidden < blocks * DEFAULT_BLOCK_MIN_LENGTH:
        blocks = 2 if num_hidden >= 2 * DEFAULT_BLOCK_MIN_LENGTH else 1
    if blocks == 1:
        return fixed_mask(
            num_steps,
            ratio,
            kind="single_block",
            seed=seed,
            sample_id=sample_id,
            epoch=epoch,
            view=view,
            fixed_start=fixed_start,
        )
    return _block_mask(num_steps, num_hidden, rng, num_blocks=blocks)


def sample_training_mask(
    num_steps: int,
    *,
    seed: int,
    sample_id: int,
    epoch: int,
    regime: str,
    view: int = 0,
    clean_probability: float = DEFAULT_CLEAN_PROBABILITY,
    max_ratio: float = DEFAULT_MAX_TRAIN_RATIO,
) -> tuple[np.ndarray, str, float]:
    """Sample a reproducible training mask and return mask, kind, and ratio."""
    if regime not in MASK_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; choose from {MASK_REGIMES}.")
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must be in [0, 1].")
    if not 0.0 < max_ratio < 1.0:
        raise ValueError("max_ratio must be in (0, 1).")
    if regime == "clean_only":
        return np.ones(num_steps, dtype=bool), "clean", 0.0

    rng = _rng(seed, sample_id, epoch, view)
    if rng.random() < clean_probability:
        return np.ones(num_steps, dtype=bool), "clean", 0.0

    max_hidden = max(1, int(round(num_steps * max_ratio)))
    num_hidden = int(rng.integers(1, max_hidden + 1))
    ratio = num_hidden / num_steps
    if regime == "random_only":
        kind = "random"
    else:
        kind = ("single_block", "multi_block", "random")[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
    return (
        fixed_mask(
            num_steps,
            ratio,
            kind=kind,
            seed=seed,
            sample_id=sample_id,
            epoch=epoch,
            view=view,
        ),
        kind,
        ratio,
    )

