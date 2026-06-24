from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import StandardScaler


def compute_forecast_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
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
        "mse_raw": float(np.mean(np.square(error_raw), dtype=np.float64)),
        "mae_raw": float(np.mean(np.abs(error_raw), dtype=np.float64)),
        "rmse_raw": float(np.sqrt(np.mean(np.square(error_raw), dtype=np.float64))),
        "num_windows": int(pred_z.shape[0]),
        "pred_len": int(pred_z.shape[1]),
        "num_channels": int(pred_z.shape[2]),
    }
    horizon_rows = []
    for horizon in range(pred_z.shape[1]):
        z = error_z[:, horizon]
        raw = error_raw[:, horizon]
        horizon_rows.append(_metric_row(z, raw, horizon_step=horizon + 1))
    channel_rows = []
    for channel, feature_name in enumerate(feature_names):
        z = error_z[:, :, channel]
        raw = error_raw[:, :, channel]
        channel_rows.append(_metric_row(z, raw, channel=channel, feature_name=feature_name))
    return metrics, horizon_rows, channel_rows


def _metric_row(error_z: np.ndarray, error_raw: np.ndarray, **identity: Any) -> dict[str, Any]:
    mse_z = float(np.mean(np.square(error_z), dtype=np.float64))
    mse_raw = float(np.mean(np.square(error_raw), dtype=np.float64))
    return {
        **identity,
        "mse_z": mse_z,
        "mae_z": float(np.mean(np.abs(error_z), dtype=np.float64)),
        "rmse_z": float(np.sqrt(mse_z)),
        "mse_raw": mse_raw,
        "mae_raw": float(np.mean(np.abs(error_raw), dtype=np.float64)),
        "rmse_raw": float(np.sqrt(mse_raw)),
    }


def save_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
