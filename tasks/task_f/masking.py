"""Compatibility imports for the Task F observation-mask API."""

from tasks.common.observation_mask import (
    DEFAULT_MASK_RATIOS,
    apply_observation_mask,
    random_observation_mask,
    validate_mask_ratio,
)

__all__ = [
    "DEFAULT_MASK_RATIOS",
    "apply_observation_mask",
    "random_observation_mask",
    "validate_mask_ratio",
]
