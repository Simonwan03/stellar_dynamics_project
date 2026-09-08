"""Task-B dynamical summaries adapted to masked Task-F prefixes."""
from __future__ import annotations

import numpy as np


TASK_B_STAT_NAMES = ("initial", "final", "mean", "std", "slope", "min", "max", "delta")


def task_b_summary_feature_names(feature_names: tuple[str, ...] | list[str]) -> list[str]:
    """Return the Task-B channel-major names for the eight summary statistics."""
    return [f"{channel}__{stat}" for channel in feature_names for stat in TASK_B_STAT_NAMES]


def summarize_observed_prefix(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute Task-B's eight summaries using only observed time positions.

    The statistic order is exactly
    ``initial, final, mean, std, slope, min, max, delta`` and the output is
    flattened channel-major, matching ``task_b_spock_summary``.  For a masked
    sequence, ``initial`` and ``final`` mean the first and last observed values;
    the slope uses the original normalized time coordinates of observed points.
    """
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, time, channels], got {values.shape}.")
    if observed.shape != values.shape[:2]:
        raise ValueError(
            f"observed shape {observed.shape} does not match values {values.shape[:2]}."
        )
    if values.shape[1] <= 1:
        raise ValueError("The prefix must contain at least two time steps.")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite numbers.")

    batch_size, time_steps, channels = values.shape
    count = observed.sum(axis=1).astype(np.float32)
    if np.any(count <= 0.0):
        raise ValueError("Every sequence must contain at least one observed time step.")
    weights = observed.astype(np.float32)
    time = np.linspace(0.0, 1.0, time_steps, dtype=np.float32)
    observed_time_mean = (weights * time[None, :]).sum(axis=1) / count
    centered_time = time[None, :] - observed_time_mean[:, None]
    slope_denominator = (weights * np.square(centered_time)).sum(axis=1)
    if np.any(slope_denominator <= 0.0):
        raise ValueError("At least two distinct observed time positions are required.")

    first_index = np.argmax(observed, axis=1)
    last_index = time_steps - 1 - np.argmax(observed[:, ::-1], axis=1)
    row_index = np.arange(batch_size)
    initial = values[row_index, first_index]
    final = values[row_index, last_index]
    weighted_values = values * weights[:, :, None]
    mean = weighted_values.sum(axis=1) / count[:, None]
    centered_values = values - mean[:, None, :]
    std = np.sqrt((np.square(centered_values) * weights[:, :, None]).sum(axis=1) / count[:, None])
    minimum = np.where(observed[:, :, None], values, np.inf).min(axis=1)
    maximum = np.where(observed[:, :, None], values, -np.inf).max(axis=1)
    slope = (values * centered_time[:, :, None] * weights[:, :, None]).sum(axis=1)
    slope = slope / slope_denominator[:, None]
    delta = final - initial

    summary = np.stack(
        [initial, final, mean, std, slope, minimum, maximum, delta], axis=2
    ).reshape(batch_size, channels * len(TASK_B_STAT_NAMES))
    if not np.isfinite(summary).all():
        raise ValueError("Summary features contain non-finite values.")
    return summary.astype(np.float32, copy=False)
