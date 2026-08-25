from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import StandardScaler


def forecast_linear_extrapolation_batch(
    context_z: np.ndarray,
    pred_len: int,
    *,
    scaler: StandardScaler,
    feature_names: Sequence[str] | None = None,
    fit_window: int = 32,
    project_angle_pairs: bool = True,
) -> np.ndarray:
    """Forecast each channel with a least-squares local linear trend.

    Task E batches are represented in the train-only standardized transformed
    feature space.  The fit is therefore performed after undoing the scaler,
    and predictions are transformed back before being returned.  For orbital15
    angle channels, optional projection onto the sin/cos unit circle keeps the
    baseline physically valid in the angle representation.
    """
    context_z = np.asarray(context_z, dtype=np.float32)
    if context_z.ndim != 3:
        raise ValueError(f"Expected context [batch,time,channels], got {context_z.shape}.")
    if context_z.shape[1] <= 0 or context_z.shape[2] <= 0:
        raise ValueError(f"Context must have positive time/channels dimensions, got {context_z.shape}.")
    if int(pred_len) <= 0:
        raise ValueError(f"pred_len must be positive, got {pred_len}.")
    if int(fit_window) <= 0:
        raise ValueError(f"fit_window must be positive, got {fit_window}.")
    if context_z.shape[2] != scaler.mean.size:
        raise ValueError(
            "Context channel count does not match the scaler: "
            f"{context_z.shape[2]} vs {scaler.mean.size}."
        )
    if feature_names is not None and len(feature_names) != context_z.shape[2]:
        raise ValueError(
            "Feature-name count does not match the context channels: "
            f"{len(feature_names)} vs {context_z.shape[2]}."
        )
    if not np.isfinite(context_z).all():
        raise ValueError("Linear extrapolation received non-finite context values.")

    # Fit in the transformed (but unstandardized) coordinates.  This matters
    # for sin/cos channels, whose means/stds generally differ by channel.
    context_transformed = scaler.inverse_transform(context_z).astype(np.float64, copy=False)
    window = min(int(fit_window), context_transformed.shape[1])
    recent = context_transformed[:, -window:, :]
    time_fit = np.arange(window, dtype=np.float64)
    centered_time = time_fit - np.mean(time_fit)
    denominator = float(np.dot(centered_time, centered_time))
    recent_mean = np.mean(recent, axis=1)
    if denominator > 0.0:
        slope = np.sum(
            (recent - recent_mean[:, None, :]) * centered_time[None, :, None],
            axis=1,
        ) / denominator
    else:
        # A one-step fit is a persistence forecast.
        slope = np.zeros_like(recent_mean)
    future_time = np.arange(window, window + int(pred_len), dtype=np.float64)
    prediction_transformed = recent_mean[:, None, :] + slope[:, None, :] * (
        future_time[None, :, None] - np.mean(time_fit)
    )

    if project_angle_pairs and feature_names is not None:
        _project_angle_pairs(
            prediction_transformed,
            recent[:, -1, :],
            [str(name) for name in feature_names],
        )

    prediction_z = scaler.transform(prediction_transformed.astype(np.float32, copy=False))
    if not np.isfinite(prediction_z).all():
        raise ValueError("Linear extrapolation produced non-finite predictions.")
    return prediction_z.astype(np.float32, copy=False)


def _project_angle_pairs(
    prediction: np.ndarray,
    last_observation: np.ndarray,
    feature_names: list[str],
    *,
    epsilon: float = 1e-8,
) -> None:
    """Project orbital sin/cos pairs in-place, with a safe fallback direction."""
    lookup = {name: index for index, name in enumerate(feature_names)}
    for family in ("APs", "LANs"):
        for component in range(2):
            sin_index = lookup.get(f"{family}_sin_{component}")
            cos_index = lookup.get(f"{family}_cos_{component}")
            if sin_index is None or cos_index is None:
                continue
            pair_indices = (sin_index, cos_index)
            pair = prediction[:, :, pair_indices]
            norm = np.linalg.norm(pair, axis=-1, keepdims=True)
            normalized = pair / np.maximum(norm, epsilon)

            fallback = last_observation[:, pair_indices]
            fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
            fallback = fallback / np.maximum(fallback_norm, epsilon)
            invalid_fallback = fallback_norm[:, 0] < epsilon
            if np.any(invalid_fallback):
                fallback[invalid_fallback] = np.asarray([0.0, 1.0], dtype=np.float64)

            invalid_prediction = norm[:, :, 0] < epsilon
            if np.any(invalid_prediction):
                normalized = np.where(
                    invalid_prediction[:, :, None],
                    fallback[:, None, :],
                    normalized,
                )
            prediction[:, :, sin_index] = normalized[:, :, 0]
            prediction[:, :, cos_index] = normalized[:, :, 1]
