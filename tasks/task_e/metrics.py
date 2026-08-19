from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .data import StandardScaler


def _hierarchical_reduce(
    values: np.ndarray,
    trajectory_ids: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce window values as window -> trajectory -> system -> dataset."""
    values = np.asarray(values, dtype=np.float64)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
    if values.ndim < 1 or values.shape[0] != trajectory_ids.size:
        raise ValueError(
            f"Values and trajectory IDs disagree: values={values.shape}, ids={trajectory_ids.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty set of windows.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    unique_trajectories, inverse = np.unique(trajectory_ids, return_inverse=True)
    trajectory_sums = np.zeros((unique_trajectories.size, *values.shape[1:]), dtype=np.float64)
    np.add.at(trajectory_sums, inverse, values)
    trajectory_counts = np.bincount(inverse, minlength=unique_trajectories.size).astype(np.float64)
    reshape = (trajectory_counts.size,) + (1,) * (values.ndim - 1)
    trajectory_means = trajectory_sums / trajectory_counts.reshape(reshape)

    system_ids_for_trajectory = unique_trajectories // int(group_size)
    unique_systems, system_inverse = np.unique(system_ids_for_trajectory, return_inverse=True)
    system_sums = np.zeros((unique_systems.size, *values.shape[1:]), dtype=np.float64)
    np.add.at(system_sums, system_inverse, trajectory_means)
    system_counts = np.bincount(system_inverse, minlength=unique_systems.size).astype(np.float64)
    system_means = system_sums / system_counts.reshape(
        (system_counts.size,) + (1,) * (values.ndim - 1)
    )
    overall = np.mean(system_means, axis=0, dtype=np.float64)
    return overall, trajectory_means, system_means, unique_systems


def _hierarchical_reduce_finite(
    values: np.ndarray,
    trajectory_ids: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        finite = np.isfinite(values)
    else:
        finite = np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    if not np.any(finite):
        nan = np.asarray(float("nan"), dtype=np.float64)
        return nan, np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    return _hierarchical_reduce(values[finite], np.asarray(trajectory_ids)[finite], group_size)


def _bootstrap_error_ci(
    system_mse: np.ndarray,
    system_mae: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap scalar z-space errors using systems as the resampling unit."""
    system_mse = np.asarray(system_mse, dtype=np.float64).reshape(-1)
    system_mae = np.asarray(system_mae, dtype=np.float64).reshape(-1)
    finite = np.isfinite(system_mse) & np.isfinite(system_mae)
    system_mse = system_mse[finite]
    system_mae = system_mae[finite]
    if system_mse.size == 0:
        empty = {"mean": None, "ci95_low": None, "ci95_high": None, "num_systems": 0}
        return {name: dict(empty) for name in ("mse_z", "mae_z", "rmse_z")}
    mse_mean = float(np.mean(system_mse, dtype=np.float64))
    mae_mean = float(np.mean(system_mae, dtype=np.float64))
    rmse_mean = float(np.sqrt(mse_mean))
    if samples <= 0 or system_mse.size == 1:
        return {
            "mse_z": {
                "mean": mse_mean,
                "ci95_low": mse_mean,
                "ci95_high": mse_mean,
                "num_systems": int(system_mse.size),
            },
            "mae_z": {
                "mean": mae_mean,
                "ci95_low": mae_mean,
                "ci95_high": mae_mean,
                "num_systems": int(system_mae.size),
            },
            "rmse_z": {
                "mean": rmse_mean,
                "ci95_low": rmse_mean,
                "ci95_high": rmse_mean,
                "num_systems": int(system_mse.size),
            },
        }
    rng = np.random.default_rng(seed)
    bootstrap_mse = np.empty(int(samples), dtype=np.float64)
    bootstrap_mae = np.empty(int(samples), dtype=np.float64)
    bootstrap_rmse = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draw = rng.integers(0, system_mse.size, size=system_mse.size)
        draw_mse = float(np.mean(system_mse[draw], dtype=np.float64))
        bootstrap_mse[index] = draw_mse
        bootstrap_mae[index] = np.mean(system_mae[draw], dtype=np.float64)
        bootstrap_rmse[index] = np.sqrt(draw_mse)
    return {
        "mse_z": {
            "mean": mse_mean,
            "ci95_low": float(np.quantile(bootstrap_mse, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_mse, 0.975)),
            "num_systems": int(system_mse.size),
        },
        "mae_z": {
            "mean": mae_mean,
            "ci95_low": float(np.quantile(bootstrap_mae, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_mae, 0.975)),
            "num_systems": int(system_mae.size),
        },
        "rmse_z": {
            "mean": rmse_mean,
            "ci95_low": float(np.quantile(bootstrap_rmse, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_rmse, 0.975)),
            "num_systems": int(system_mse.size),
        },
    }


def summarize_error_hierarchy(
    error_z: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    group_size: int = 32,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarize errors with both pooled-window and system-balanced reductions."""
    error_z = np.asarray(error_z, dtype=np.float64)
    if error_z.ndim != 3:
        raise ValueError(f"Expected error array [N,H,C], got {error_z.shape}.")
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
    if trajectory_ids.size != error_z.shape[0]:
        raise ValueError(
            f"Expected one trajectory ID per window, got {trajectory_ids.size} for {error_z.shape[0]} windows."
        )
    if not np.isfinite(error_z).all():
        raise ValueError("Cannot aggregate non-finite forecast errors.")

    window_mse = np.mean(np.square(error_z), axis=(1, 2), dtype=np.float64)
    window_mae = np.mean(np.abs(error_z), axis=(1, 2), dtype=np.float64)
    window_horizon_mse = np.mean(np.square(error_z), axis=2, dtype=np.float64)
    window_horizon_mae = np.mean(np.abs(error_z), axis=2, dtype=np.float64)
    mse, _, system_mse, systems = _hierarchical_reduce(window_mse, trajectory_ids, group_size)
    mae, _, system_mae, _ = _hierarchical_reduce(window_mae, trajectory_ids, group_size)
    horizon_mae, _, system_horizon_mae, _ = _hierarchical_reduce(
        window_horizon_mae,
        trajectory_ids,
        group_size,
    )
    horizon_mse, _, system_horizon_mse, _ = _hierarchical_reduce(
        window_horizon_mse,
        trajectory_ids,
        group_size,
    )
    window_mse_mean = float(np.mean(window_mse, dtype=np.float64))
    window_mae_mean = float(np.mean(window_mae, dtype=np.float64))
    system_mse_value = float(mse)
    system_mae_value = float(mae)
    summary = {
        "num_windows": int(error_z.shape[0]),
        "num_trajectories": int(np.unique(trajectory_ids).size),
        "num_systems": int(systems.size),
        "mse_z_window_weighted": window_mse_mean,
        "mae_z_window_weighted": window_mae_mean,
        "rmse_z_window_weighted": float(np.sqrt(window_mse_mean)),
        "mse_z_system_balanced": system_mse_value,
        "mae_z_system_balanced": system_mae_value,
        "rmse_z_system_balanced": float(np.sqrt(system_mse_value)),
        "mse_z": system_mse_value,
        "mae_z": system_mae_value,
        "rmse_z": float(np.sqrt(system_mse_value)),
        "horizon_mae_window_weighted": window_horizon_mae.mean(axis=0, dtype=np.float64).tolist(),
        "horizon_mae_system_balanced": np.asarray(horizon_mae, dtype=np.float64).tolist(),
        "horizon_mae": np.asarray(horizon_mae, dtype=np.float64).tolist(),
        "horizon_mse_window_weighted": window_horizon_mse.mean(axis=0, dtype=np.float64).tolist(),
        "horizon_mse_system_balanced": np.asarray(horizon_mse, dtype=np.float64).tolist(),
        "horizon_rmse_system_balanced": np.sqrt(np.asarray(horizon_mse, dtype=np.float64)).tolist(),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_ci95": _bootstrap_error_ci(
            system_mse,
            system_mae,
            bootstrap_samples,
            bootstrap_seed,
        ),
    }
    internals = {
        "systems": systems.astype(np.int64, copy=False),
        "system_mse": np.asarray(system_mse, dtype=np.float64),
        "system_mae": np.asarray(system_mae, dtype=np.float64),
        "system_horizon_mae": np.asarray(system_horizon_mae, dtype=np.float64),
        "system_horizon_mse": np.asarray(system_horizon_mse, dtype=np.float64),
    }
    return summary, internals


class HierarchicalErrorAccumulator:
    """Online window -> trajectory -> system error aggregation for E2."""

    def __init__(self, horizon: int, group_size: int = 32) -> None:
        if horizon <= 0 or group_size <= 0:
            raise ValueError("horizon and group_size must be positive.")
        self.horizon = int(horizon)
        self.group_size = int(group_size)
        self.num_channels: int | None = None
        self.num_windows = 0
        self.window_abs_sum = np.zeros(self.horizon, dtype=np.float64)
        self.window_sq_sum = 0.0
        self.trajectory_horizon_sums: dict[int, np.ndarray] = {}
        self.trajectory_mse_sums: dict[int, float] = {}
        self.trajectory_mae_sums: dict[int, float] = {}
        self.trajectory_counts: dict[int, int] = {}

    def add(self, error_z: np.ndarray, trajectory_ids: np.ndarray) -> None:
        error_z = np.asarray(error_z, dtype=np.float64)
        trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
        if error_z.ndim != 3 or error_z.shape[1] != self.horizon:
            raise ValueError(f"Expected error [N,{self.horizon},C], got {error_z.shape}.")
        if error_z.shape[0] != trajectory_ids.size:
            raise ValueError("One trajectory ID is required for every rolling window.")
        if not np.isfinite(error_z).all():
            raise ValueError("Cannot aggregate non-finite rolling errors.")
        if self.num_channels is None:
            self.num_channels = int(error_z.shape[2])
        elif self.num_channels != error_z.shape[2]:
            raise ValueError("Rolling error channel count changed between batches.")

        abs_error = np.abs(error_z)
        squared_error = np.square(error_z)
        self.window_abs_sum += np.sum(abs_error, axis=(0, 2), dtype=np.float64)
        self.window_sq_sum += float(np.sum(squared_error, dtype=np.float64))
        self.num_windows += int(error_z.shape[0])
        window_horizon_mae = np.mean(abs_error, axis=2, dtype=np.float64)
        window_mse = np.mean(squared_error, axis=(1, 2), dtype=np.float64)
        window_mae = np.mean(abs_error, axis=(1, 2), dtype=np.float64)
        for trajectory_id in np.unique(trajectory_ids).tolist():
            mask = trajectory_ids == trajectory_id
            trajectory_id = int(trajectory_id)
            self.trajectory_horizon_sums[trajectory_id] = (
                self.trajectory_horizon_sums.get(
                    trajectory_id,
                    np.zeros(self.horizon, dtype=np.float64),
                )
                + np.sum(window_horizon_mae[mask], axis=0, dtype=np.float64)
            )
            self.trajectory_mse_sums[trajectory_id] = self.trajectory_mse_sums.get(trajectory_id, 0.0) + float(
                np.sum(window_mse[mask], dtype=np.float64)
            )
            self.trajectory_mae_sums[trajectory_id] = self.trajectory_mae_sums.get(trajectory_id, 0.0) + float(
                np.sum(window_mae[mask], dtype=np.float64)
            )
            self.trajectory_counts[trajectory_id] = self.trajectory_counts.get(trajectory_id, 0) + int(mask.sum())

    def finalize(
        self,
        *,
        bootstrap_samples: int = 0,
        bootstrap_seed: int = 42,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self.num_windows <= 0 or self.num_channels is None:
            raise ValueError("Cannot finalize an empty rolling error accumulator.")
        trajectory_ids = np.asarray(sorted(self.trajectory_counts), dtype=np.int64)
        counts = np.asarray([self.trajectory_counts[int(value)] for value in trajectory_ids], dtype=np.float64)
        trajectory_horizon = np.stack(
            [self.trajectory_horizon_sums[int(value)] for value in trajectory_ids],
            axis=0,
        ) / counts[:, None]
        trajectory_mse = np.asarray(
            [self.trajectory_mse_sums[int(value)] for value in trajectory_ids],
            dtype=np.float64,
        ) / counts
        trajectory_mae = np.asarray(
            [self.trajectory_mae_sums[int(value)] for value in trajectory_ids],
            dtype=np.float64,
        ) / counts
        systems = trajectory_ids // self.group_size
        unique_systems, inverse = np.unique(systems, return_inverse=True)
        system_horizon = np.zeros((unique_systems.size, self.horizon), dtype=np.float64)
        system_mse = np.zeros(unique_systems.size, dtype=np.float64)
        system_mae = np.zeros(unique_systems.size, dtype=np.float64)
        np.add.at(system_horizon, inverse, trajectory_horizon)
        np.add.at(system_mse, inverse, trajectory_mse)
        np.add.at(system_mae, inverse, trajectory_mae)
        system_counts = np.bincount(inverse, minlength=unique_systems.size).astype(np.float64)
        system_horizon /= system_counts[:, None]
        system_mse /= system_counts
        system_mae /= system_counts
        horizon_mae = np.mean(system_horizon, axis=0, dtype=np.float64)
        window_horizon_mae = self.window_abs_sum / (self.num_windows * self.num_channels)
        window_mse = self.window_sq_sum / (self.num_windows * self.horizon * self.num_channels)
        mse = float(np.mean(system_mse, dtype=np.float64))
        mae = float(np.mean(system_mae, dtype=np.float64))
        summary = {
            "num_windows": int(self.num_windows),
            "num_trajectories": int(trajectory_ids.size),
            "num_systems": int(unique_systems.size),
            "mse_z_window_weighted": float(window_mse),
            "mae_z_window_weighted": float(self.window_abs_sum.sum() / (self.num_windows * self.num_channels * self.horizon)),
            "rmse_z_window_weighted": float(np.sqrt(window_mse)),
            "mse_z_system_balanced": mse,
            "mae_z_system_balanced": mae,
            "rmse_z_system_balanced": float(np.sqrt(mse)),
            "mse_z": mse,
            "mae_z": mae,
            "rmse_z": float(np.sqrt(mse)),
            "rolling_horizon_mae_window_weighted": window_horizon_mae.tolist(),
            "rolling_horizon_mae_system_balanced": horizon_mae.tolist(),
            "rolling_horizon_mae": horizon_mae.tolist(),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_ci95": _bootstrap_error_ci(
                system_mse,
                system_mae,
                bootstrap_samples,
                bootstrap_seed,
            ),
        }
        return summary, {
            "systems": unique_systems.astype(np.int64, copy=False),
            "system_mse": system_mse,
            "system_mae": system_mae,
            "system_horizon_mae": system_horizon,
        }


def bootstrap_block_ci(
    system_block_mae: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ratio_epsilon: float = 1e-12,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap E2 block means and block-4/block-1 using system resampling."""
    values = np.asarray(system_block_mae, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"Expected system block MAE [systems,blocks], got {values.shape}.")
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if values.shape[0] == 0:
        return {}
    point = np.mean(values, axis=0, dtype=np.float64)
    ratio = _stable_ratio(float(point[0]), float(point[-1]), ratio_epsilon)
    result: dict[str, dict[str, float | int | None]] = {
        f"block_{index + 1}_mae": {
            "mean": float(point[index]),
            "ci95_low": float(point[index]),
            "ci95_high": float(point[index]),
            "num_systems": int(values.shape[0]),
        }
        for index in range(values.shape[1])
    }
    result["error_amplification_ratio"] = {
        "mean": ratio,
        "ci95_low": ratio,
        "ci95_high": ratio,
        "num_systems": int(values.shape[0]),
    }
    if bootstrap_samples <= 0 or values.shape[0] == 1:
        return result
    rng = np.random.default_rng(bootstrap_seed)
    block_boot = np.empty((int(bootstrap_samples), values.shape[1]), dtype=np.float64)
    ratio_boot = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        draw = rng.integers(0, values.shape[0], size=values.shape[0])
        means = np.mean(values[draw], axis=0, dtype=np.float64)
        block_boot[index] = means
        ratio_value = _stable_ratio(float(means[0]), float(means[-1]), ratio_epsilon)
        ratio_boot[index] = np.nan if ratio_value is None else ratio_value
    for index in range(values.shape[1]):
        result[f"block_{index + 1}_mae"] = {
            "mean": float(point[index]),
            "ci95_low": float(np.quantile(block_boot[:, index], 0.025)),
            "ci95_high": float(np.quantile(block_boot[:, index], 0.975)),
            "num_systems": int(values.shape[0]),
        }
    finite_ratio = ratio_boot[np.isfinite(ratio_boot)]
    result["error_amplification_ratio"] = {
        "mean": ratio,
        "ci95_low": float(np.quantile(finite_ratio, 0.025)) if finite_ratio.size else None,
        "ci95_high": float(np.quantile(finite_ratio, 0.975)) if finite_ratio.size else None,
        "num_systems": int(values.shape[0]),
    }
    return result


def _stable_ratio(block_1: float, block_last: float, epsilon: float) -> float | None:
    if not np.isfinite(block_1) or not np.isfinite(block_last):
        return None
    if block_1 > epsilon:
        return float(block_last / block_1)
    if abs(block_last) <= epsilon:
        return 1.0
    return None


def compute_forecast_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
    distribution_bins: int = 64,
    distribution_epsilon: float = 1e-12,
    trajectory_ids: np.ndarray | None = None,
    group_size: int = 32,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pred_z = np.asarray(pred_z, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    if pred_z.shape != target_z.shape or pred_z.ndim != 3:
        raise ValueError(f"Expected matching [N,H,C] arrays, got {pred_z.shape} and {target_z.shape}")
    pred_raw = scaler.inverse_transform(pred_z)
    target_raw = scaler.inverse_transform(target_z)
    error_z = pred_z - target_z
    error_raw = pred_raw - target_raw

    if trajectory_ids is None:
        trajectory_ids = np.arange(pred_z.shape[0], dtype=np.int64)
        effective_group_size = 1
    else:
        trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
        effective_group_size = int(group_size)
    hierarchy, _ = summarize_error_hierarchy(
        error_z,
        trajectory_ids,
        group_size=effective_group_size,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    raw_hierarchy, _ = summarize_error_hierarchy(
        error_raw,
        trajectory_ids,
        group_size=effective_group_size,
        bootstrap_samples=0,
        bootstrap_seed=bootstrap_seed,
    )
    window_pearson_z = _per_window_pearson(pred_z, target_z)
    window_pearson_raw = _per_window_pearson(pred_raw, target_raw)
    balanced_pearson_z, _, _, _ = _hierarchical_reduce_finite(
        window_pearson_z,
        trajectory_ids,
        effective_group_size,
    )
    balanced_pearson_raw, _, _, _ = _hierarchical_reduce_finite(
        window_pearson_raw,
        trajectory_ids,
        effective_group_size,
    )
    window_smape_z = _per_window_smape(pred_z, target_z, distribution_epsilon)
    window_smape_raw = _per_window_smape(pred_raw, target_raw, distribution_epsilon)
    balanced_smape_z, _, _, _ = _hierarchical_reduce_finite(
        window_smape_z,
        trajectory_ids,
        effective_group_size,
    )
    balanced_smape_raw, _, _, _ = _hierarchical_reduce_finite(
        window_smape_raw,
        trajectory_ids,
        effective_group_size,
    )

    metrics = {
        **hierarchy,
        "mse_raw_window_weighted": raw_hierarchy["mse_z_window_weighted"],
        "mae_raw_window_weighted": raw_hierarchy["mae_z_window_weighted"],
        "rmse_raw_window_weighted": raw_hierarchy["rmse_z_window_weighted"],
        "mse_raw_system_balanced": raw_hierarchy["mse_z_system_balanced"],
        "mae_raw_system_balanced": raw_hierarchy["mae_z_system_balanced"],
        "rmse_raw_system_balanced": raw_hierarchy["rmse_z_system_balanced"],
        "mse_raw": raw_hierarchy["mse_z"],
        "mae_raw": raw_hierarchy["mae_z"],
        "rmse_raw": raw_hierarchy["rmse_z"],
        "pearson_z_window_weighted": pearson_correlation(pred_z, target_z),
        "pearson_z_system_balanced": float(balanced_pearson_z) if np.isfinite(balanced_pearson_z) else None,
        "smape_percent_z_window_weighted": symmetric_mean_absolute_percentage_error(pred_z, target_z),
        "smape_percent_z_system_balanced": float(balanced_smape_z) if np.isfinite(balanced_smape_z) else None,
        "smape_percent_z": float(balanced_smape_z) if np.isfinite(balanced_smape_z) else None,
        "spearman_z": spearman_correlation(pred_z, target_z),
        "pearson_raw_window_weighted": pearson_correlation(pred_raw, target_raw),
        "pearson_raw_system_balanced": float(balanced_pearson_raw) if np.isfinite(balanced_pearson_raw) else None,
        "smape_percent_raw_window_weighted": symmetric_mean_absolute_percentage_error(pred_raw, target_raw),
        "smape_percent_raw_system_balanced": float(balanced_smape_raw) if np.isfinite(balanced_smape_raw) else None,
        "smape_percent_raw": float(balanced_smape_raw) if np.isfinite(balanced_smape_raw) else None,
        "spearman_raw": spearman_correlation(pred_raw, target_raw),
        **distribution_metrics(pred_z, target_z, bins=distribution_bins, epsilon=distribution_epsilon, suffix="_z"),
        **distribution_metrics(pred_raw, target_raw, bins=distribution_bins, epsilon=distribution_epsilon, suffix="_raw"),
        "pred_len": int(pred_z.shape[1]),
        "num_channels": int(pred_z.shape[2]),
        "distribution_bins": int(distribution_bins),
        "distribution_epsilon": float(distribution_epsilon),
    }
    metrics.update(_transformed_aliases(metrics))
    metrics["pearson_z"] = metrics["pearson_z_system_balanced"]
    metrics["pearson_raw"] = metrics["pearson_raw_system_balanced"]
    metrics["pearson"] = metrics["pearson_z"]
    metrics["smape_percent_transformed"] = metrics["smape_percent_raw"]
    unit_circle_overall, unit_circle_by_angle = compute_unit_circle_error(pred_raw, feature_names)
    unit_window_values, unit_window_by_angle = _per_window_unit_circle_error(pred_raw, feature_names)
    balanced_unit, _, _, _ = _hierarchical_reduce_finite(
        unit_window_values,
        trajectory_ids,
        effective_group_size,
    )
    balanced_unit_by_angle: dict[str, float | None] = {}
    for key, values in unit_window_by_angle.items():
        reduced, _, _, _ = _hierarchical_reduce_finite(
            values,
            trajectory_ids,
            effective_group_size,
        )
        balanced_unit_by_angle[key] = float(reduced) if np.isfinite(reduced) else None
    metrics["unit_circle_error_window_weighted"] = unit_circle_overall
    metrics["unit_circle_error_system_balanced"] = float(balanced_unit) if np.isfinite(balanced_unit) else None
    metrics["unit_circle_error"] = metrics["unit_circle_error_system_balanced"]
    metrics["unit_circle_error_by_angle_window_weighted"] = unit_circle_by_angle
    metrics["unit_circle_error_by_angle"] = balanced_unit_by_angle
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
    weighted_horizon = np.asarray(hierarchy["horizon_mae_window_weighted"], dtype=np.float64)
    balanced_horizon = np.asarray(hierarchy["horizon_mae_system_balanced"], dtype=np.float64)
    weighted_horizon_mse = np.asarray(hierarchy["horizon_mse_window_weighted"], dtype=np.float64)
    balanced_horizon_mse = np.asarray(hierarchy["horizon_mse_system_balanced"], dtype=np.float64)
    raw_weighted_horizon = np.asarray(raw_hierarchy["horizon_mae_window_weighted"], dtype=np.float64)
    raw_balanced_horizon = np.asarray(raw_hierarchy["horizon_mae_system_balanced"], dtype=np.float64)
    raw_weighted_horizon_mse = np.asarray(raw_hierarchy["horizon_mse_window_weighted"], dtype=np.float64)
    raw_balanced_horizon_mse = np.asarray(raw_hierarchy["horizon_mse_system_balanced"], dtype=np.float64)
    for index, row in enumerate(horizon_rows):
        row["mae_z_window_weighted"] = float(row["mae_z"])
        row["mae_z_system_balanced"] = float(balanced_horizon[index])
        row["mae_z"] = float(balanced_horizon[index])
        row["mse_z_window_weighted"] = float(row["mse_z"])
        row["mse_z_system_balanced"] = float(balanced_horizon_mse[index])
        row["mse_z"] = float(balanced_horizon_mse[index])
        row["rmse_z"] = float(np.sqrt(balanced_horizon_mse[index]))
        row["mae_raw_window_weighted"] = float(row["mae_raw"])
        row["mae_raw_system_balanced"] = float(raw_balanced_horizon[index])
        row["mae_raw"] = float(raw_balanced_horizon[index])
        row["mse_raw_window_weighted"] = float(row["mse_raw"])
        row["mse_raw_system_balanced"] = float(raw_balanced_horizon_mse[index])
        row["mse_raw"] = float(raw_balanced_horizon_mse[index])
        row["rmse_raw"] = float(np.sqrt(raw_balanced_horizon_mse[index]))
        row["mae_transformed"] = float(raw_balanced_horizon[index])
        row["horizon_mae_window_weighted"] = float(weighted_horizon[index])
        row["horizon_mae_system_balanced"] = float(balanced_horizon[index])
    metrics["horizon_mae"] = balanced_horizon.tolist()
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



def summarize_horizon_trends(horizon_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether per-step forecast error increases with horizon."""
    if not horizon_rows:
        return {"num_horizons": 0, "horizon_steps": [], "metrics": {}}

    ordered = sorted(horizon_rows, key=lambda row: int(row["horizon_step"]))
    steps = np.asarray([int(row["horizon_step"]) for row in ordered], dtype=np.float64)
    result: dict[str, Any] = {
        "num_horizons": int(steps.size),
        "horizon_steps": steps.astype(np.int64).tolist(),
        "adjacent_tolerance_factor": 1e-8,
        "metrics": {},
    }
    metric_names = ("mse_z", "mae_z", "rmse_z", "mse_raw", "mae_raw", "rmse_raw")
    for metric_name in metric_names:
        values = np.asarray([row.get(metric_name, np.nan) for row in ordered], dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(steps)
        if int(finite.sum()) < 2:
            result["metrics"][metric_name] = {
                "finite_horizons": int(finite.sum()),
                "is_nondecreasing": None,
                "overall_increased": None,
                "nondecreasing_adjacent_fraction": None,
                "strictly_increasing_adjacent_fraction": None,
                "spearman_horizon_error": None,
                "ols_slope_per_step": None,
                "first_error": None,
                "last_error": None,
                "last_minus_first": None,
                "last_over_first": None,
            }
            continue

        x = steps[finite]
        y = values[finite]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        differences = np.diff(y)
        tolerance = 1e-8 * max(1.0, float(np.max(np.abs(y))))
        adjacent_count = int(differences.size)
        nondecreasing_count = int(np.count_nonzero(differences >= -tolerance))
        strictly_increasing_count = int(np.count_nonzero(differences > tolerance))
        centered_x = x - float(np.mean(x))
        centered_y = y - float(np.mean(y))
        denominator = float(np.sum(np.square(centered_x), dtype=np.float64))
        slope = float(np.sum(centered_x * centered_y, dtype=np.float64) / denominator) if denominator else 0.0
        correlation = spearmanr(x, y).statistic
        correlation = float(correlation) if np.isfinite(correlation) else None
        first_error = float(y[0])
        last_error = float(y[-1])
        last_over_first = last_error / first_error if abs(first_error) > tolerance else None
        result["metrics"][metric_name] = {
            "finite_horizons": int(y.size),
            "is_nondecreasing": bool(np.all(differences >= -tolerance)),
            "overall_increased": bool(last_error > first_error + tolerance),
            "nondecreasing_adjacent_fraction": float(nondecreasing_count / max(adjacent_count, 1)),
            "strictly_increasing_adjacent_fraction": float(strictly_increasing_count / max(adjacent_count, 1)),
            "nondecreasing_adjacent_count": nondecreasing_count,
            "adjacent_pair_count": adjacent_count,
            "spearman_horizon_error": correlation,
            "ols_slope_per_step": slope,
            "first_error": first_error,
            "last_error": last_error,
            "last_minus_first": float(last_error - first_error),
            "last_over_first": float(last_over_first) if last_over_first is not None else None,
        }
    return result


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
        "pearson_z": pearson_correlation(pred_z, target_z),
        "smape_percent_z": symmetric_mean_absolute_percentage_error(pred_z, target_z),
        "spearman_z": spearman_correlation(pred_z, target_z),
        "mse_raw": mse_raw,
        "mae_raw": float(np.mean(np.abs(error_raw), dtype=np.float64)),
        "rmse_raw": float(np.sqrt(mse_raw)),
        "pearson_raw": pearson_correlation(pred_raw, target_raw),
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


def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Return flattened paired Pearson correlation, or NaN if undefined."""
    pred_values, target_values = _finite_pairs(pred, target)
    if pred_values.size < 2 or np.ptp(pred_values) == 0.0 or np.ptp(target_values) == 0.0:
        return float("nan")
    result = pearsonr(pred_values, target_values)
    correlation = float(result.statistic)
    return correlation if np.isfinite(correlation) else float("nan")


def compute_unit_circle_error(
    prediction_transformed: np.ndarray,
    feature_names: list[str],
) -> tuple[float | None, dict[str, float | None]]:
    """Measure how closely predicted angle sin/cos pairs stay on the unit circle."""
    prediction_transformed = np.asarray(prediction_transformed, dtype=np.float64)
    if prediction_transformed.ndim != 3:
        raise ValueError(
            f"Expected transformed predictions with shape [N,H,C], got {prediction_transformed.shape}."
        )
    channel_lookup = {name: index for index, name in enumerate(feature_names)}
    values: list[np.ndarray] = []
    by_angle: dict[str, float | None] = {}
    for angle_family in ("APs", "LANs"):
        for component in range(2):
            key = f"{angle_family}_{component}"
            sin_index = channel_lookup.get(f"{angle_family}_sin_{component}")
            cos_index = channel_lookup.get(f"{angle_family}_cos_{component}")
            if sin_index is None or cos_index is None:
                by_angle[key] = None
                continue
            circle_error = np.abs(
                np.square(prediction_transformed[:, :, sin_index])
                + np.square(prediction_transformed[:, :, cos_index])
                - 1.0
            )
            finite = circle_error[np.isfinite(circle_error)]
            by_angle[key] = float(np.mean(finite, dtype=np.float64)) if finite.size else None
            if finite.size:
                values.append(finite)
    if not values:
        return None, by_angle
    return float(np.mean(np.concatenate(values), dtype=np.float64)), by_angle


def _per_window_unit_circle_error(
    prediction_transformed: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prediction_transformed = np.asarray(prediction_transformed, dtype=np.float64)
    channel_lookup = {name: index for index, name in enumerate(feature_names)}
    by_angle: dict[str, np.ndarray] = {}
    for angle_family in ("APs", "LANs"):
        for component in range(2):
            key = f"{angle_family}_{component}"
            sin_index = channel_lookup.get(f"{angle_family}_sin_{component}")
            cos_index = channel_lookup.get(f"{angle_family}_cos_{component}")
            if sin_index is None or cos_index is None:
                continue
            circle_error = np.abs(
                np.square(prediction_transformed[:, :, sin_index])
                + np.square(prediction_transformed[:, :, cos_index])
                - 1.0
            )
            by_angle[key] = np.mean(circle_error, axis=1, dtype=np.float64)
    if not by_angle:
        return np.full(prediction_transformed.shape[0], np.nan, dtype=np.float64), {}
    values = np.stack(list(by_angle.values()), axis=1)
    return np.mean(values, axis=1, dtype=np.float64), by_angle


def _per_window_smape(pred: np.ndarray, target: np.ndarray, epsilon: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = np.abs(pred) + np.abs(target)
    terms = 2.0 * np.abs(pred - target) / np.maximum(denominator, max(float(epsilon), 1e-12))
    return 100.0 * np.mean(terms, axis=(1, 2), dtype=np.float64)


def _per_window_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1)
    target = np.asarray(target, dtype=np.float64).reshape(target.shape[0], -1)
    pred_centered = pred - np.mean(pred, axis=1, keepdims=True, dtype=np.float64)
    target_centered = target - np.mean(target, axis=1, keepdims=True, dtype=np.float64)
    numerator = np.sum(pred_centered * target_centered, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(np.square(pred_centered), axis=1, dtype=np.float64)
        * np.sum(np.square(target_centered), axis=1, dtype=np.float64)
    )
    result = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


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
