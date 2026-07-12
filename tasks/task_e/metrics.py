from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .data import StandardScaler


def compute_forecast_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
    distribution_bins: int = 64,
    distribution_epsilon: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pred_z = np.asarray(pred_z, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    if pred_z.shape != target_z.shape or pred_z.ndim != 3:
        raise ValueError(f"Expected matching [N,H,C] arrays, got {pred_z.shape} and {target_z.shape}")
    pred_raw = scaler.inverse_transform(pred_z)
    target_raw = scaler.inverse_transform(target_z)
    error_z = pred_z - target_z
    error_raw = pred_raw - target_raw

    metrics = {
        "mse_z": float(np.mean(np.square(error_z), dtype=np.float64)),
        "mae_z": float(np.mean(np.abs(error_z), dtype=np.float64)),
        "rmse_z": float(np.sqrt(np.mean(np.square(error_z), dtype=np.float64))),
        "smape_percent_z": symmetric_mean_absolute_percentage_error(pred_z, target_z),
        "spearman_z": spearman_correlation(pred_z, target_z),
        "mse_raw": float(np.mean(np.square(error_raw), dtype=np.float64)),
        "mae_raw": float(np.mean(np.abs(error_raw), dtype=np.float64)),
        "rmse_raw": float(np.sqrt(np.mean(np.square(error_raw), dtype=np.float64))),
        "smape_percent_raw": symmetric_mean_absolute_percentage_error(pred_raw, target_raw),
        "spearman_raw": spearman_correlation(pred_raw, target_raw),
        **distribution_metrics(pred_z, target_z, bins=distribution_bins, epsilon=distribution_epsilon, suffix="_z"),
        **distribution_metrics(pred_raw, target_raw, bins=distribution_bins, epsilon=distribution_epsilon, suffix="_raw"),
        "num_windows": int(pred_z.shape[0]),
        "pred_len": int(pred_z.shape[1]),
        "num_channels": int(pred_z.shape[2]),
        "distribution_bins": int(distribution_bins),
        "distribution_epsilon": float(distribution_epsilon),
    }
    metrics.update(_transformed_aliases(metrics))
    horizon_rows = []
    for horizon in range(pred_z.shape[1]):
        horizon_rows.append(
            _metric_row(
                pred_z[:, horizon],
                target_z[:, horizon],
                pred_raw[:, horizon],
                target_raw[:, horizon],
                bins=distribution_bins,
                epsilon=distribution_epsilon,
                horizon_step=horizon + 1,
            )
        )
    channel_rows = []
    for channel, feature_name in enumerate(feature_names):
        channel_rows.append(
            _metric_row(
                pred_z[:, :, channel],
                target_z[:, :, channel],
                pred_raw[:, :, channel],
                target_raw[:, :, channel],
                bins=distribution_bins,
                epsilon=distribution_epsilon,
                channel=channel,
                feature_name=feature_name,
            )
        )
    return metrics, horizon_rows, channel_rows


def _metric_row(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    *,
    bins: int,
    epsilon: float,
    **identity: Any,
) -> dict[str, Any]:
    error_z = pred_z - target_z
    error_raw = pred_raw - target_raw
    mse_z = float(np.mean(np.square(error_z), dtype=np.float64))
    mse_raw = float(np.mean(np.square(error_raw), dtype=np.float64))
    row = {
        **identity,
        "mse_z": mse_z,
        "mae_z": float(np.mean(np.abs(error_z), dtype=np.float64)),
        "rmse_z": float(np.sqrt(mse_z)),
        "smape_percent_z": symmetric_mean_absolute_percentage_error(pred_z, target_z),
        "spearman_z": spearman_correlation(pred_z, target_z),
        "mse_raw": mse_raw,
        "mae_raw": float(np.mean(np.abs(error_raw), dtype=np.float64)),
        "rmse_raw": float(np.sqrt(mse_raw)),
        "smape_percent_raw": symmetric_mean_absolute_percentage_error(pred_raw, target_raw),
        "spearman_raw": spearman_correlation(pred_raw, target_raw),
        **distribution_metrics(pred_z, target_z, bins=bins, epsilon=epsilon, suffix="_z"),
        **distribution_metrics(pred_raw, target_raw, bins=bins, epsilon=epsilon, suffix="_raw"),
    }
    row.update(_transformed_aliases(row))
    return row


def _transformed_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expose accurate names while retaining legacy *_raw fields for compatibility."""
    return {
        key.removesuffix("_raw") + "_transformed": value
        for key, value in metrics.items()
        if key.endswith("_raw")
    }


def symmetric_mean_absolute_percentage_error(
    pred: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Return sMAPE as a percentage in [0, 200] for finite paired values."""
    pred_values, target_values = _finite_pairs(pred, target)
    if pred_values.size == 0:
        return float("nan")
    denominator = np.abs(pred_values) + np.abs(target_values)
    terms = 2.0 * np.abs(pred_values - target_values) / np.maximum(denominator, epsilon)
    return float(100.0 * np.mean(terms, dtype=np.float64))


def spearman_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Return flattened paired Spearman rank correlation, or NaN if undefined."""
    pred_values, target_values = _finite_pairs(pred, target)
    if pred_values.size < 2 or np.ptp(pred_values) == 0.0 or np.ptp(target_values) == 0.0:
        return float("nan")
    result = spearmanr(pred_values, target_values)
    correlation = float(result.statistic)
    return correlation if np.isfinite(correlation) else float("nan")


def _finite_pairs(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_values = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred_values) & np.isfinite(target_values)
    return pred_values[finite], target_values[finite]


def distribution_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
    epsilon: float,
    suffix: str,
) -> dict[str, float]:
    pred_prob, target_prob = _empirical_probabilities(pred, target, bins=bins, epsilon=epsilon)
    kl_pred_target = float(np.sum(pred_prob * np.log(pred_prob / target_prob), dtype=np.float64))
    kl_target_pred = float(np.sum(target_prob * np.log(target_prob / pred_prob), dtype=np.float64))
    hellinger = float(np.sqrt(0.5 * np.sum(np.square(np.sqrt(pred_prob) - np.sqrt(target_prob)), dtype=np.float64)))
    return {
        f"kl_pred_target{suffix}": kl_pred_target,
        f"kl_target_pred{suffix}": kl_target_pred,
        f"hellinger{suffix}": hellinger,
    }


def _empirical_probabilities(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("distribution_bins must be at least 2.")
    pred_values = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred_values) & np.isfinite(target_values)
    pred_values = pred_values[finite]
    target_values = target_values[finite]
    if pred_values.size == 0:
        uniform = np.full(bins, 1.0 / bins, dtype=np.float64)
        return uniform, uniform

    pooled = np.concatenate([pred_values, target_values])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        low = float(np.min(pooled))
        high = float(np.max(pooled))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, bins + 1)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    pred_counts, _ = np.histogram(pred_values, bins=edges)
    target_counts, _ = np.histogram(target_values, bins=edges)
    pred_prob = pred_counts.astype(np.float64) + epsilon
    target_prob = target_counts.astype(np.float64) + epsilon
    pred_prob /= pred_prob.sum()
    target_prob /= target_prob.sum()
    return pred_prob, target_prob


def save_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
