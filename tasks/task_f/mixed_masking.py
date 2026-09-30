"""Compatibility imports for the Task F mixed-mask API."""

from tasks.common.mixed_masking import (
    DEFAULT_BLOCK_MIN_GAP,
    DEFAULT_BLOCK_MIN_LENGTH,
    DEFAULT_CLEAN_PROBABILITY,
    DEFAULT_MAX_TRAIN_RATIO,
    MASK_KINDS,
    MASK_REGIMES,
    MaskKind,
    MaskRegime,
    fixed_mask,
    sample_training_mask,
)

__all__ = [
    "DEFAULT_BLOCK_MIN_GAP",
    "DEFAULT_BLOCK_MIN_LENGTH",
    "DEFAULT_CLEAN_PROBABILITY",
    "DEFAULT_MAX_TRAIN_RATIO",
    "MASK_KINDS",
    "MASK_REGIMES",
    "MaskKind",
    "MaskRegime",
    "fixed_mask",
    "sample_training_mask",
]
