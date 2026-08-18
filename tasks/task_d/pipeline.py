from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
LABEL_VERSION = 6
DEFAULT_GROUP_SIZE = 32
DEFAULT_VALID_PAIRS = tuple((i, i + 16) for i in range(16))
LEGACY_VALID_PAIRS_8 = ((0, 4), (1, 5), (2, 6), (3, 7))
ORBITAL15_FEATURE_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
)
INITIAL_FEATURE_KEYS = (
    "masses_complete",
    "mratios_complete",
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "aratios_complete",
    "energy_complete",
    "angMomentum_complete",
    "periods_complete",
)
INITIAL_ANGLE_KEYS = frozenset({"APs_complete", "LANs_complete"})
PREFIX_FEATURE_KEYS = (
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "aratios_complete",
    "energy_complete",
    "angMomentum_complete",
)
PREFIX_EXTRA_STATE_KEYS = ("positions_complete", "velocities_complete")
INITIAL_PREFIX_TRAJECTORY_KEYS = (
    "aratios_complete",
    "eccs_complete",
    "incs_complete",
)
INVALID_PAIR_CHECKS = ((0, 1), (0, 2), (0, 4))


@dataclass
class PipelineConfig:
    task_setting: str = "initial"
    target_mode: str = "lyapunov_proxy"
    label_observable: str = "smax"
    lyapunov_aggregation: str = "max"
    dataset_dir: str | Path = "./dataset_v3"
    sampling_id: int = 2
    group_size: int = DEFAULT_GROUP_SIZE
    valid_pairs: tuple[tuple[int, int], ...] = DEFAULT_VALID_PAIRS
    threshold: float = 1e4
    growth_lower: float = 1.0
    growth_upper: float = 100.0
    growth_sensitivity_uppers: tuple[float, ...] = (10.0, 30.0, 100.0)
    min_growth_points: int = 5
    lambda_r2_min: float = 0.0
    min_valid_lambda_pairs: int = 1
    max_pair_log_mad: float = 2.0
    require_pair_consistency: bool = False
    aggregation: str = "min"
    censoring_rule: str = "all_pairs"
    initial_representation: str = "orbital15"
    feature_aggregation: str = "first"
    exclude_censored: bool = False
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str | Path = "./results/task_d_initial_lyapunov_proxy"
    require_complete_valid_group: bool = True
    completion_table_name: str = "completed_systems_sampling_2.txt"
    max_shards: int | None = None
    max_systems: int | None = None
    prefix_len: int = 200
    prefix_time_stride: int = 1
    sequence_aggregation: str = "mean_std"
    run_mlp: bool = True
    run_cnn_orbital: bool = True
    run_cnn_phase_space: bool = True
    run_set_cnn: bool = True
    mlp_epochs: int = 80
    cnn_epochs: int = 80
    nn_batch_size: int = 128
    nn_lr: float = 1e-3
    nn_weight_decay: float = 1e-4
    nn_dropout: float = 0.1
    nn_val_fraction: float = 0.1
    nn_patience: int = 12
    loss: str = "huber"
    huber_beta: float = 0.5
    device: str | None = None
    force_recompute_labels: bool = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def discover_result_paths(dataset_dir: str | Path, sampling_id: int) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    result_paths = sorted(dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if not result_paths:
        raise FileNotFoundError(
            f"No result shards found under {dataset_dir} for sampling {sampling_id}."
        )
    return result_paths


def extract_batch_id(path: str | Path) -> int:
    match = RESULT_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def load_completion_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "aux_value", "diagnosis"],
    )
    df = df.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float64",
            "aux_value": "float64",
            "diagnosis": "int64",
        }
    )
    df["valid"] = (df["time_unbound"] >= 0.0) & (df["diagnosis"] == 0)
    return df


def summarize_series(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": float(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _make_pair_column(i: int, j: int, prefix: str = "tau") -> str:
    return f"{prefix}_pair_{i}_{j}"


def _flatten_feature_row(arr: np.ndarray, prefix: str) -> tuple[list[str], np.ndarray]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    names = [f"{prefix}_{idx}" for idx in range(flat.size)]
    return names, flat


def validate_array_shapes(
    smaxes: np.ndarray,
    times: np.ndarray,
    group_size: int,
) -> None:
    assert smaxes.shape[0] % group_size == 0, (smaxes.shape, group_size)
    assert times.shape[0] == smaxes.shape[0], (times.shape, smaxes.shape)
    assert smaxes.shape[1] == times.shape[1], (smaxes.shape, times.shape)


def _center_of_mass_frame(states: np.ndarray, masses: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(f"Expected states with shape (T, bodies, 3), got {states.shape}")
    if masses.ndim == 1:
        masses = np.broadcast_to(masses[None, :], states.shape[:2])
    if masses.shape != states.shape[:2]:
        raise ValueError(f"Mass/state shape mismatch: {masses.shape} vs {states.shape}")
    mass_sum = np.sum(masses, axis=1, keepdims=True)
    safe_mass_sum = np.where(np.abs(mass_sum) > eps, mass_sum, 1.0)
    center = np.sum(states * masses[..., None], axis=1, keepdims=True) / safe_mass_sum[..., None]
    return states - center


def _initial_state_scale(states_i: np.ndarray, states_j: np.ndarray, eps: float = 1e-12) -> float:
    initial = np.concatenate([states_i[0].reshape(-1), states_j[0].reshape(-1)])
    scale = float(np.sqrt(np.mean(initial**2)))
    return max(scale, eps) if np.isfinite(scale) else 1.0


def _fit_log_growth_rate(
    ratio: np.ndarray,
    times: np.ndarray,
    lower: float,
    upper: float,
    min_points: int,
) -> tuple[float, float, int]:
    ratio = np.asarray(ratio, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    mask = (
        np.isfinite(ratio)
        & np.isfinite(times)
        & (ratio > lower)
        & (ratio < upper)
        & (ratio > 0.0)
    )
    indices = np.flatnonzero(mask)
    if indices.size < min_points:
        return math.nan, math.nan, int(indices.size)
    x = times[indices] - times[indices[0]]
    y = np.log(ratio[indices])
    if np.ptp(x) <= 0.0:
        return math.nan, math.nan, int(indices.size)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else math.nan
    return float(slope), float(r2), int(indices.size)


def _lambda_quality_mask(
    lambdas: Sequence[float],
    r2_values: Sequence[float],
    point_counts: Sequence[int],
    r2_min: float,
    min_points: int,
) -> np.ndarray:
    lambdas_arr = np.asarray(lambdas, dtype=np.float64)
    r2_arr = np.asarray(r2_values, dtype=np.float64)
    points_arr = np.asarray(point_counts, dtype=np.int64)
    return (
        np.isfinite(lambdas_arr)
        & (lambdas_arr > 0.0)
        & np.isfinite(r2_arr)
        & (r2_arr >= r2_min)
        & (points_arr >= min_points)
    )


def _pair_log_mad(lambdas: Sequence[float]) -> float:
    arr = np.asarray(lambdas, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 2:
        return math.nan
    logs = np.log(arr)
    center = np.median(logs)
    return float(np.median(np.abs(logs - center)))


def _growth_range_name(lower: float, upper: float) -> str:
    def clean(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return f"{clean(lower)}_{clean(upper)}"


def compute_pair_phase_space_divergence(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    times: np.ndarray,
    idx_i: int,
    idx_j: int,
    threshold: float = 1e3,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    min_growth_points: int = 5,
    eps: float = 1e-12,
) -> dict[str, Any]:
    pos_i = _center_of_mass_frame(positions[idx_i], masses[idx_i], eps=eps)
    pos_j = _center_of_mass_frame(positions[idx_j], masses[idx_j], eps=eps)
    vel_i = _center_of_mass_frame(velocities[idx_i], masses[idx_i], eps=eps)
    vel_j = _center_of_mass_frame(velocities[idx_j], masses[idx_j], eps=eps)
    pair_times = np.asarray(times[idx_i], dtype=np.float64)

    length_scale = _initial_state_scale(pos_i, pos_j, eps=eps)
    velocity_scale = _initial_state_scale(vel_i, vel_j, eps=eps)
    delta_position = np.linalg.norm((pos_i - pos_j).reshape(pos_i.shape[0], -1), axis=1) / length_scale
    delta_velocity = np.linalg.norm((vel_i - vel_j).reshape(vel_i.shape[0], -1), axis=1) / velocity_scale
    distance = np.sqrt(delta_position**2 + delta_velocity**2)
    distance[~np.isfinite(distance)] = np.nan

    distance0_raw = float(distance[0]) if distance.size else math.nan
    distance0 = max(distance0_raw, eps) if np.isfinite(distance0_raw) else eps
    ratio = distance / distance0
    finite = np.isfinite(ratio) & np.isfinite(pair_times)
    crossing = np.flatnonzero(finite & (ratio > threshold))
    if crossing.size:
        tau = float(pair_times[int(crossing[0])])
        censored = False
    else:
        finite_times = np.flatnonzero(np.isfinite(pair_times))
        tau = float(pair_times[int(finite_times[-1])]) if finite_times.size else math.nan
        censored = True

    growth_rate, growth_r2, growth_points = _fit_log_growth_rate(
        ratio=ratio,
        times=pair_times,
        lower=growth_lower,
        upper=growth_upper,
        min_points=min_growth_points,
    )
    return {
        "tau": tau,
        "censored": censored,
        "ratio": ratio,
        "distance": distance,
        "distance0": distance0_raw,
        "length_scale": length_scale,
        "velocity_scale": velocity_scale,
        "lambda_phase_space": growth_rate,
        "lambda_fit_r2": growth_r2,
        "lambda_fit_points": growth_points,
    }


def compute_pair_aratio_divergence(
    aratios: np.ndarray,
    times: np.ndarray,
    idx_i: int,
    idx_j: int,
    threshold: float = 1e3,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    min_growth_points: int = 5,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute a finite-time divergence label from ``aratios_complete`` only.

    ``aratios_complete`` contains one scalar observable per trajectory and time
    step.  The pair separation is the absolute observable difference,
    normalized by its value at the first stored time.  The fitted slope is an
    observable-based finite-time growth rate, not a strict variational-equation
    maximum Lyapunov exponent.
    """
    aratios = np.asarray(aratios, dtype=np.float64)
    if aratios.ndim not in {2, 3}:
        raise ValueError(f"Expected aratios with shape (N, T) or (N, T, 1), got {aratios.shape}")
    if aratios.ndim == 3 and aratios.shape[-1] != 1:
        raise ValueError(
            "The aratio label expects one scalar channel in aratios_complete; "
            f"got shape {aratios.shape}."
        )
    series_i = aratios[idx_i].reshape(aratios.shape[1], -1)
    series_j = aratios[idx_j].reshape(aratios.shape[1], -1)
    if series_i.shape[1] != 1 or series_j.shape[1] != 1:
        raise ValueError(
            "The aratio label expects one scalar channel in aratios_complete; "
            f"got pair shapes {series_i.shape} and {series_j.shape}."
        )
    pair_times = np.asarray(times[idx_i], dtype=np.float64)
    if idx_i != idx_j:
        other_times = np.asarray(times[idx_j], dtype=np.float64)
        if pair_times.shape != other_times.shape or not np.allclose(
            np.nan_to_num(pair_times, nan=-1.0),
            np.nan_to_num(other_times, nan=-1.0),
            rtol=1e-10,
            atol=1e-10,
        ):
            warnings.warn(
                f"Time grids differ for pair ({idx_i}, {idx_j}); using times[{idx_i}] as reference.",
                RuntimeWarning,
                stacklevel=2,
            )
    tau, censored, ratio, delta = _compute_pair_divergence_from_series(
        a_i=series_i[:, 0],
        a_j=series_j[:, 0],
        times=pair_times,
        threshold=threshold,
        eps=eps,
    )
    growth_rate, growth_r2, growth_points = _fit_log_growth_rate(
        ratio=ratio,
        times=pair_times,
        lower=growth_lower,
        upper=growth_upper,
        min_points=min_growth_points,
    )
    return {
        "tau": tau,
        "censored": censored,
        "ratio": ratio,
        "distance": delta,
        "distance0": float(delta[0]) if delta.size and np.isfinite(delta[0]) else math.nan,
        "lambda_aratio": growth_rate,
        "lambda_fit_r2": growth_r2,
        "lambda_fit_points": growth_points,
    }


def compute_pair_divergence_time(
    smaxes: np.ndarray,
    times: np.ndarray,
    idx_i: int,
    idx_j: int,
    threshold: float = 1e3,
    eps: float = 1e-12,
    inner_index: int = 0,
) -> tuple[float, bool, np.ndarray, np.ndarray]:
    """
    Compute the semimajor-axis divergence time for one valid pair.

    Parameters
    ----------
    smaxes : np.ndarray
        Shape (N, T, 2). Semimajor axes.
    times : np.ndarray
        Shape (N, T). Time grid.
    idx_i, idx_j : int
        Global trajectory indices.
    threshold : float
        Divergence threshold for R_a(t).
    eps : float
        Small value to avoid division by zero.
    inner_index : int
        Which semimajor axis to use. Default 0 for inner orbit.

    Returns
    -------
    tau : float
        First time when normalized divergence reaches or exceeds threshold.
    censored : bool
        Whether the pair never crossed the threshold.
    R_a : np.ndarray
        Normalized divergence curve.
    delta_a : np.ndarray
        Absolute semimajor-axis difference curve.
    """
    a_i = np.asarray(smaxes[idx_i, :, inner_index], dtype=np.float64)
    a_j = np.asarray(smaxes[idx_j, :, inner_index], dtype=np.float64)
    t_i = np.asarray(times[idx_i], dtype=np.float64)
    if idx_i != idx_j:
        t_j = np.asarray(times[idx_j], dtype=np.float64)
        if t_i.shape != t_j.shape or not np.allclose(
            np.nan_to_num(t_i, nan=-1.0),
            np.nan_to_num(t_j, nan=-1.0),
            rtol=1e-10,
            atol=1e-10,
        ):
            warnings.warn(
                f"Time grids differ for pair ({idx_i}, {idx_j}); using times[{idx_i}] as reference.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _compute_pair_divergence_from_series(
        a_i=a_i,
        a_j=a_j,
        times=t_i,
        threshold=threshold,
        eps=eps,
    )


def _compute_pair_divergence_from_series(
    a_i: np.ndarray,
    a_j: np.ndarray,
    times: np.ndarray,
    threshold: float,
    eps: float,
) -> tuple[float, bool, np.ndarray, np.ndarray]:
    delta_a = np.abs(np.asarray(a_i, dtype=np.float64) - np.asarray(a_j, dtype=np.float64))
    times = np.asarray(times, dtype=np.float64)
    if delta_a.shape != times.shape:
        raise ValueError(f"delta_a and times shape mismatch: {delta_a.shape} vs {times.shape}")

    delta_a = delta_a.astype(np.float64, copy=False)
    delta_a[~np.isfinite(delta_a)] = np.nan

    delta0_raw = float(delta_a[0]) if delta_a.size else math.nan
    delta0 = max(delta0_raw, eps) if np.isfinite(delta0_raw) else eps
    R_a = delta_a / delta0
    finite_mask = np.isfinite(R_a) & np.isfinite(times)
    crossing_idx = np.flatnonzero(finite_mask & (R_a >= threshold))
    if crossing_idx.size > 0:
        idx = int(crossing_idx[0])
        return float(times[idx]), False, R_a, delta_a

    finite_time_idx = np.flatnonzero(np.isfinite(times))
    if finite_time_idx.size == 0:
        return math.nan, True, R_a, delta_a
    final_idx = int(finite_time_idx[-1])
    return float(times[final_idx]), True, R_a, delta_a


def _aggregate_tau(
    pair_taus: Sequence[float],
    aggregation: str = "median",
) -> float:
    arr = np.asarray(pair_taus, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan
    if aggregation == "median":
        return float(np.median(finite))
    if aggregation == "mean":
        return float(np.mean(finite))
    if aggregation == "min":
        return float(np.min(finite))
    if aggregation == "max":
        return float(np.max(finite))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _aggregate_censoring_flag(
    pair_taus: Sequence[float],
    pair_censored: Sequence[bool],
    tau_system: float,
    censoring_rule: str,
) -> bool:
    flags = np.asarray(pair_censored, dtype=bool)
    if flags.size == 0:
        return True
    if censoring_rule == "all_pairs":
        return bool(np.all(flags))
    if censoring_rule == "median_pair":
        pair_taus_arr = np.asarray(pair_taus, dtype=np.float64)
        diffs = np.abs(pair_taus_arr - tau_system)
        diffs[~np.isfinite(diffs)] = np.inf
        chosen_idx = int(np.argmin(diffs))
        return bool(flags[chosen_idx])
    raise ValueError(f"Unsupported censoring_rule: {censoring_rule}")


def _compute_group_divergence_label(
    positions_group: np.ndarray | None,
    velocities_group: np.ndarray | None,
    masses_group: np.ndarray | None,
    times_group: np.ndarray,
    system_id: int,
    valid_pairs: Sequence[tuple[int, int]],
    threshold: float,
    growth_lower: float,
    growth_upper: float,
    min_growth_points: int,
    aggregation: str,
    censoring_rule: str,
    growth_sensitivity_uppers: Sequence[float] = (10.0, 30.0, 100.0),
    lambda_r2_min: float = 0.8,
    min_valid_lambda_pairs: int = 8,
    max_pair_log_mad: float = 0.75,
    eps: float = 1e-12,
    aratios_group: np.ndarray | None = None,
    smaxes_group: np.ndarray | None = None,
    label_observable: str = "phase_space",
    lyapunov_aggregation: str = "median",
    require_pair_consistency: bool = True,
    label_start_index: int = 0,
) -> dict[str, Any]:
    if label_observable not in {"smax", "aratio", "phase_space"}:
        raise ValueError(f"Unsupported label_observable: {label_observable}")
    if lyapunov_aggregation not in {"median", "mean", "max"}:
        raise ValueError(f"Unsupported lyapunov_aggregation: {lyapunov_aggregation}")
    if label_observable == "aratio" and aratios_group is None:
        raise ValueError("aratios_group is required when label_observable='aratio'.")
    if label_observable == "smax" and smaxes_group is None:
        raise ValueError("smaxes_group is required when label_observable='smax'.")
    if label_observable == "phase_space" and (
        positions_group is None or velocities_group is None or masses_group is None
    ):
        raise ValueError(
            "positions_group, velocities_group, and masses_group are required "
            "when label_observable='phase_space'."
        )

    pair_taus: list[float] = []
    pair_censored: list[bool] = []
    pair_distance0s: list[float] = []
    pair_lambdas: list[float] = []
    pair_lambda_r2s: list[float] = []
    pair_lambda_points: list[int] = []
    sensitivity_uppers = tuple(sorted(set(float(value) for value in growth_sensitivity_uppers)))
    sensitivity_fits: dict[float, list[tuple[float, float, int]]] = {
        upper: [] for upper in sensitivity_uppers if upper > growth_lower
    }
    for idx_i, idx_j in valid_pairs:
        if label_observable == "smax":
            tau, censored, ratio, delta = compute_pair_divergence_time(
                smaxes=smaxes_group,
                times=times_group,
                idx_i=idx_i,
                idx_j=idx_j,
                threshold=threshold,
                eps=eps,
                inner_index=0,
            )
            growth_rate, growth_r2, growth_points = _fit_log_growth_rate(
                ratio=ratio,
                times=np.asarray(times_group[idx_i], dtype=np.float64),
                lower=growth_lower,
                upper=growth_upper,
                min_points=min_growth_points,
            )
            pair_result = {
                "tau": tau,
                "censored": censored,
                "ratio": ratio,
                "distance": delta,
                "distance0": (
                    float(delta[0]) if delta.size and np.isfinite(delta[0]) else math.nan
                ),
                "lambda_smax": growth_rate,
                "lambda_fit_r2": growth_r2,
                "lambda_fit_points": growth_points,
            }
        elif label_observable == "aratio":
            pair_result = compute_pair_aratio_divergence(
                aratios=aratios_group,
                times=times_group,
                idx_i=idx_i,
                idx_j=idx_j,
                threshold=threshold,
                growth_lower=growth_lower,
                growth_upper=growth_upper,
                min_growth_points=min_growth_points,
                eps=eps,
            )
        else:
            pair_result = compute_pair_phase_space_divergence(
                positions=positions_group,
                velocities=velocities_group,
                masses=masses_group,
                times=times_group,
                idx_i=idx_i,
                idx_j=idx_j,
                threshold=threshold,
                growth_lower=growth_lower,
                growth_upper=growth_upper,
                min_growth_points=min_growth_points,
                eps=eps,
            )
        pair_taus.append(float(pair_result["tau"]))
        pair_censored.append(bool(pair_result["censored"]))
        pair_distance0s.append(float(pair_result["distance0"]))
        lambda_key = {
            "smax": "lambda_smax",
            "aratio": "lambda_aratio",
            "phase_space": "lambda_phase_space",
        }[label_observable]
        pair_lambdas.append(float(pair_result[lambda_key]))
        pair_lambda_r2s.append(float(pair_result["lambda_fit_r2"]))
        pair_lambda_points.append(int(pair_result["lambda_fit_points"]))
        for upper in sensitivity_fits:
            sensitivity_fits[upper].append(
                _fit_log_growth_rate(
                    ratio=pair_result["ratio"],
                    times=np.asarray(times_group[idx_i], dtype=np.float64),
                    lower=growth_lower,
                    upper=upper,
                    min_points=min_growth_points,
                )
            )

    tau_system = _aggregate_tau(pair_taus, aggregation=aggregation)
    lambda_system_unfiltered = _aggregate_tau(pair_lambdas, aggregation=lyapunov_aggregation)
    pair_lambda_quality = _lambda_quality_mask(
        pair_lambdas,
        pair_lambda_r2s,
        pair_lambda_points,
        r2_min=lambda_r2_min,
        min_points=min_growth_points,
    )
    valid_pair_lambdas = np.asarray(pair_lambdas, dtype=np.float64)[pair_lambda_quality]
    lambda_system = _aggregate_tau(valid_pair_lambdas, aggregation=lyapunov_aggregation)
    lambda_fast = _aggregate_tau(valid_pair_lambdas, aggregation="max")
    log_lambda_fast = (
        float(np.log(lambda_fast)) if np.isfinite(lambda_fast) and lambda_fast > 0.0 else math.nan
    )
    pair_log_mad = _pair_log_mad(valid_pair_lambdas)
    valid_lambda_pair_count = int(pair_lambda_quality.sum())
    enough_valid_pairs = valid_lambda_pair_count >= min_valid_lambda_pairs
    pair_consistency_pass = (
        bool(np.isfinite(pair_log_mad) and pair_log_mad <= max_pair_log_mad)
        if require_pair_consistency
        else True
    )
    lyapunov_label_valid = bool(
        enough_valid_pairs
        and pair_consistency_pass
        and np.isfinite(lambda_system)
        and lambda_system > 0.0
    )
    lyapunov_time_proxy = (
        float(1.0 / lambda_system) if lyapunov_label_valid else math.nan
    )
    log_lyapunov_time_proxy = (
        float(np.log(lyapunov_time_proxy))
        if np.isfinite(lyapunov_time_proxy) and lyapunov_time_proxy > 0.0
        else math.nan
    )

    sensitivity_summary: dict[str, dict[str, float | int | bool]] = {}
    for upper, fits in sensitivity_fits.items():
        range_lambdas = [fit[0] for fit in fits]
        range_r2s = [fit[1] for fit in fits]
        range_points = [fit[2] for fit in fits]
        range_quality = _lambda_quality_mask(
            range_lambdas, range_r2s, range_points, lambda_r2_min, min_growth_points
        )
        filtered = np.asarray(range_lambdas, dtype=np.float64)[range_quality]
        system_value = _aggregate_tau(filtered, aggregation=lyapunov_aggregation)
        range_log_mad = _pair_log_mad(filtered)
        range_valid = bool(
            int(range_quality.sum()) >= min_valid_lambda_pairs
            and (
                not require_pair_consistency
                or (np.isfinite(range_log_mad) and range_log_mad <= max_pair_log_mad)
            )
            and np.isfinite(system_value)
            and system_value > 0.0
        )
        sensitivity_summary[_growth_range_name(growth_lower, upper)] = {
            "lambda": float(system_value),
            "lyapunov_time": float(1.0 / system_value) if range_valid else math.nan,
            "valid_pair_count": int(range_quality.sum()),
            "pair_log_mad": float(range_log_mad),
            "valid": range_valid,
        }
    system_censored = _aggregate_censoring_flag(
        pair_taus=pair_taus,
        pair_censored=pair_censored,
        tau_system=tau_system,
        censoring_rule=censoring_rule,
    )
    lambda_tau_ratio = (
        float(lambda_system * tau_system / np.log(threshold))
        if np.isfinite(lambda_system) and np.isfinite(tau_system) and threshold > 1.0
        else math.nan
    )
    return {
        "system_id": system_id,
        "label_observable": label_observable,
        "lyapunov_aggregation": lyapunov_aggregation,
        "require_pair_consistency": require_pair_consistency,
        "divergence_observable": (
            "smaxes_complete inner-semimajor-axis difference"
            if label_observable == "smax"
            else (
                "aratios_complete scalar difference"
                if label_observable == "aratio"
                else "center-of-mass position+velocity phase-space distance"
            )
        ),
        "tau_system": tau_system,
        "lyapunov_proxy_time": tau_system,
        "log_lyapunov_proxy_time": (
            float(np.log(tau_system))
            if np.isfinite(tau_system) and tau_system > 0.0
            else math.nan
        ),
        "phase_space_divergence_time": tau_system,
        "lyapunov_like_divergence_time": tau_system,
        "tau_div": tau_system,
        "log_tau_system": float(np.log1p(tau_system)) if np.isfinite(tau_system) else math.nan,
        "lambda_phase_space_unfiltered": (
            lambda_system_unfiltered if label_observable == "phase_space" else math.nan
        ),
        "lambda_phase_space": lambda_system if label_observable == "phase_space" else math.nan,
        "lambda_smax": lambda_system if label_observable == "smax" else math.nan,
        "lambda_aratio": lambda_system if label_observable == "aratio" else math.nan,
        "lambda_observable": lambda_system,
        "lambda_fast": lambda_fast,
        "log_lambda_fast": log_lambda_fast,
        "lyapunov_time_proxy": lyapunov_time_proxy,
        "log_lyapunov_time_proxy": log_lyapunov_time_proxy,
        "lyapunov_time_label": lyapunov_time_proxy,
        "log_lyapunov_time_label": log_lyapunov_time_proxy,
        "lyapunov_label_valid": lyapunov_label_valid,
        "valid_lambda_pair_count": valid_lambda_pair_count,
        "valid_lambda_pair_fraction": float(valid_lambda_pair_count / max(len(valid_pairs), 1)),
        "pair_lambda_log_mad": pair_log_mad,
        "pair_consistency_pass": pair_consistency_pass,
        "lambda_tau_ratio": lambda_tau_ratio,
        "system_censored": bool(system_censored),
        "lambda_sensitivity": sensitivity_summary,
        "pair_taus": pair_taus,
        "pair_censored": pair_censored,
        "pair_distance0s": pair_distance0s,
        "pair_lambdas": pair_lambdas,
        "pair_lambda_r2s": pair_lambda_r2s,
        "pair_lambda_points": pair_lambda_points,
        "pair_lambda_quality": pair_lambda_quality.tolist(),
    }


def compute_system_divergence_label(
    positions: np.ndarray | None,
    velocities: np.ndarray | None,
    masses: np.ndarray | None,
    times: np.ndarray,
    system_id: int,
    group_size: int = 8,
    valid_pairs: Sequence[tuple[int, int]] = LEGACY_VALID_PAIRS_8,
    threshold: float = 1e3,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    min_growth_points: int = 5,
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
    aratios: np.ndarray | None = None,
    label_observable: str = "phase_space",
    lyapunov_aggregation: str = "median",
    require_pair_consistency: bool = True,
) -> dict[str, Any]:
    """Compute a system-level divergence label from near-identical pairs.

    Set ``label_observable='aratio'`` and pass ``aratios`` to compute the label
    from the scalar observable. The default preserves the legacy
    position/velocity implementation.
    """
    if label_observable not in {"aratio", "phase_space"}:
        raise ValueError(f"Unsupported label_observable: {label_observable}")
    if label_observable == "aratio":
        if aratios is None:
            raise ValueError("aratios is required when label_observable='aratio'.")
        if aratios.shape[0] % group_size != 0:
            raise ValueError(
                f"Aratio row count {aratios.shape[0]} is not divisible by group_size={group_size}"
            )
        if aratios.shape[:2] != times.shape:
            raise ValueError(f"Aratio/time shape mismatch: {aratios.shape} vs {times.shape}")
    else:
        if positions is None or velocities is None or masses is None:
            raise ValueError("State arrays are required when label_observable='phase_space'.")
        if positions.shape[0] % group_size != 0:
            raise ValueError(
                f"Position row count {positions.shape[0]} is not divisible by group_size={group_size}"
            )
        if positions.shape != velocities.shape:
            raise ValueError(f"Position/velocity shape mismatch: {positions.shape} vs {velocities.shape}")
        if positions.shape[:2] != times.shape:
            raise ValueError(f"State/time shape mismatch: {positions.shape} vs {times.shape}")
    start = system_id * group_size
    stop = start + group_size
    row_count = aratios.shape[0] if label_observable == "aratio" else positions.shape[0]
    if stop > row_count:
        raise IndexError(f"system_id {system_id} exceeds available rows")
    return _compute_group_divergence_label(
        positions_group=None if positions is None else positions[start:stop],
        velocities_group=None if velocities is None else velocities[start:stop],
        masses_group=None if masses is None else masses[start:stop],
        times_group=times[start:stop],
        system_id=system_id,
        valid_pairs=valid_pairs,
        threshold=threshold,
        growth_lower=growth_lower,
        growth_upper=growth_upper,
        min_growth_points=min_growth_points,
        aggregation=aggregation,
        censoring_rule=censoring_rule,
        aratios_group=None if aratios is None else aratios[start:stop],
        label_observable=label_observable,
        lyapunov_aggregation=lyapunov_aggregation,
        require_pair_consistency=require_pair_consistency,
    )


def build_task_c_labels(
    positions: np.ndarray | None,
    velocities: np.ndarray | None,
    masses: np.ndarray | None,
    times: np.ndarray,
    group_size: int = 8,
    threshold: float = 1e3,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    min_growth_points: int = 5,
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
    valid_pairs: Sequence[tuple[int, int]] = LEGACY_VALID_PAIRS_8,
    aratios: np.ndarray | None = None,
    label_observable: str = "phase_space",
    lyapunov_aggregation: str = "median",
    require_pair_consistency: bool = True,
) -> pd.DataFrame:
    """
    Build labels for all systems from in-memory arrays.

    Returns a DataFrame with one row per system.
    """
    rows: list[dict[str, Any]] = []
    n_rows = aratios.shape[0] if label_observable == "aratio" else positions.shape[0]
    n_systems = n_rows // group_size
    for system_id in range(n_systems):
        label = compute_system_divergence_label(
            positions=positions,
            velocities=velocities,
            masses=masses,
            times=times,
            system_id=system_id,
            group_size=group_size,
            valid_pairs=valid_pairs,
            threshold=threshold,
            growth_lower=growth_lower,
            growth_upper=growth_upper,
            min_growth_points=min_growth_points,
            aggregation=aggregation,
            censoring_rule=censoring_rule,
            aratios=aratios,
            label_observable=label_observable,
            lyapunov_aggregation=lyapunov_aggregation,
            require_pair_consistency=require_pair_consistency,
        )
        row = {
            "system_id": label["system_id"],
                "tau_system": label["tau_system"],
                "phase_space_divergence_time": label["tau_system"],
                "lyapunov_like_divergence_time": label["tau_system"],
                "tau_div": label["tau_system"],
                "log_tau_system": label["log_tau_system"],
            "lambda_phase_space": label["lambda_phase_space"],
            "lambda_aratio": label["lambda_aratio"],
            "lambda_observable": label["lambda_observable"],
            "lambda_fast": label["lambda_fast"],
            "log_lambda_fast": label["log_lambda_fast"],
            "lambda_tau_ratio": label["lambda_tau_ratio"],
            "lyapunov_time_label": label["lyapunov_time_label"],
            "log_lyapunov_time_label": label["log_lyapunov_time_label"],
            "system_censored": label["system_censored"],
        }
        for (idx_i, idx_j), tau in zip(valid_pairs, label["pair_taus"], strict=True):
            row[_make_pair_column(idx_i, idx_j, prefix="tau")] = tau
        for (idx_i, idx_j), flag in zip(valid_pairs, label["pair_censored"], strict=True):
            row[_make_pair_column(idx_i, idx_j, prefix="censored")] = bool(flag)
        for (idx_i, idx_j), value in zip(valid_pairs, label["pair_lambdas"], strict=True):
            row[_make_pair_column(idx_i, idx_j, prefix="lambda")] = value
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_group_structure(
    result_paths: Sequence[str | Path],
    group_size: int = DEFAULT_GROUP_SIZE,
    valid_pairs: Sequence[tuple[int, int]] = DEFAULT_VALID_PAIRS,
    legacy_pairs_8: Sequence[tuple[int, int]] = LEGACY_VALID_PAIRS_8,
    invalid_pairs: Sequence[tuple[int, int]] = INVALID_PAIR_CHECKS,
) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("No result paths supplied")

    delta_stats = {
        "configured_valid_pairs": [],
        "legacy_pairs_8": [],
        "invalid_pairs": [],
    }
    sample_rows: list[dict[str, float]] = []
    pair_state_diffs: list[dict[str, float]] = []
    first_shard = Path(result_paths[0])

    for path_idx, result_path in enumerate(result_paths):
        with h5py.File(result_path, "r") as f:
            a0 = np.asarray(f["smaxes_complete"][:, 0, 0], dtype=np.float64)
            rows = int(a0.shape[0])
            if rows % group_size != 0:
                raise ValueError(f"{result_path} row count {rows} is not divisible by group_size={group_size}")

            if path_idx == 0:
                take = min(group_size, rows)
                inner = np.asarray(f["tAnos_complete"][:take, 0, 0], dtype=np.float64)
                outer = np.asarray(f["tAnos_complete"][:take, 0, 1], dtype=np.float64)
                pos0 = np.asarray(f["positions_complete"][:take, 0], dtype=np.float64)
                for idx in range(take):
                    sample_rows.append(
                        {
                            "trajectory_index": idx,
                            "a_in_0": float(a0[idx]),
                            "inner_tAno_0": float(inner[idx]) if np.isfinite(inner[idx]) else math.nan,
                            "outer_tAno_0": float(outer[idx]) if np.isfinite(outer[idx]) else math.nan,
                        }
                    )

                pair_state_diffs = []
                for idx_i, idx_j in list(valid_pairs[:3]) + list(legacy_pairs_8[:3]):
                    pair_state_diffs.append(
                        {
                            "pair": f"{idx_i}_{idx_j}",
                            "position_l2_at_t0": float(np.linalg.norm(pos0[idx_i] - pos0[idx_j])),
                        }
                    )
            for start in range(0, rows, group_size):
                group_a0 = a0[start : start + group_size]
                for idx_i, idx_j in valid_pairs:
                    delta_stats["configured_valid_pairs"].append(abs(group_a0[idx_i] - group_a0[idx_j]))
                if group_size >= 8:
                    for idx_i, idx_j in legacy_pairs_8:
                        delta_stats["legacy_pairs_8"].append(abs(group_a0[idx_i] - group_a0[idx_j]))
                for idx_i, idx_j in invalid_pairs:
                    if idx_j < group_size:
                        delta_stats["invalid_pairs"].append(abs(group_a0[idx_i] - group_a0[idx_j]))

    configured_summary = summarize_series(delta_stats["configured_valid_pairs"])
    legacy_summary = summarize_series(delta_stats["legacy_pairs_8"])
    invalid_summary = summarize_series(delta_stats["invalid_pairs"])

    diagnostics = {
        "first_shard": str(first_shard),
        "group_size": group_size,
        "valid_pairs": [list(pair) for pair in valid_pairs],
        "legacy_pairs_8": [list(pair) for pair in legacy_pairs_8],
        "configured_valid_pairs_delta0": configured_summary,
        "legacy_pairs_8_delta0": legacy_summary,
        "invalid_pairs_delta0": invalid_summary,
        "fraction_configured_pairs_near_1e6": float(
            np.mean(np.isclose(delta_stats["configured_valid_pairs"], 1e-6, rtol=1e-3, atol=1e-9))
        ),
        "fraction_legacy_pairs_near_zero": float(
            np.mean(np.isclose(delta_stats["legacy_pairs_8"], 0.0, atol=1e-10))
        )
        if delta_stats["legacy_pairs_8"]
        else math.nan,
        "sample_first_group": sample_rows,
        "sample_pair_state_diffs": pair_state_diffs,
    }
    if configured_summary["max"] > 5e-6 or configured_summary["min"] < 5e-7:
        warnings.warn(
            "Configured valid pair initial delta_a values are not tightly clustered near 1e-6.",
            RuntimeWarning,
            stacklevel=2,
        )
    return diagnostics


def build_labels_from_result_shards(
    result_paths: Sequence[str | Path],
    group_size: int,
    valid_pairs: Sequence[tuple[int, int]],
    threshold: float,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    growth_sensitivity_uppers: Sequence[float] = (10.0, 30.0, 100.0),
    min_growth_points: int = 5,
    lambda_r2_min: float = 0.8,
    min_valid_lambda_pairs: int = 8,
    max_pair_log_mad: float = 0.75,
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
    completion_df: pd.DataFrame | None = None,
    require_complete_valid_group: bool = True,
    max_systems: int | None = None,
    label_observable: str = "smax",
    lyapunov_aggregation: str = "median",
    require_pair_consistency: bool = True,
    label_start_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if label_observable not in {"smax", "aratio", "phase_space"}:
        raise ValueError(f"Unsupported label_observable: {label_observable}")
    if lyapunov_aggregation not in {"median", "mean", "max"}:
        raise ValueError(f"Unsupported lyapunov_aggregation: {lyapunov_aggregation}")
    if label_start_index < 0:
        raise ValueError(f"label_start_index must be non-negative, got {label_start_index}")
    rows: list[dict[str, Any]] = []
    dropped_invalid_groups = 0
    dropped_nan_labels = 0
    global_row_offset = 0
    system_counter = 0
    valid_pair_distance0s: list[float] = []
    valid_pair_lambdas: list[float] = []
    valid_pair_lambda_r2s: list[float] = []

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        with h5py.File(result_path, "r") as f:
            position_ds = f["positions_complete"] if label_observable == "phase_space" else None
            velocity_ds = f["velocities_complete"] if label_observable == "phase_space" else None
            mass_ds = f["masses_complete"] if label_observable == "phase_space" else None
            aratio_ds = f["aratios_complete"] if label_observable == "aratio" else None
            smax_ds = f["smaxes_complete"] if label_observable == "smax" else None
            time_ds = f["times_complete"]
            row_count, time_steps = time_ds.shape[:2]
            if label_observable == "phase_space":
                if velocity_ds.shape != position_ds.shape:
                    raise ValueError(
                        f"{result_path} position/velocity shape mismatch: "
                        f"{position_ds.shape} vs {velocity_ds.shape}"
                    )
                if mass_ds.shape != position_ds.shape[:3]:
                    raise ValueError(
                        f"{result_path} mass/state shape mismatch: {mass_ds.shape} vs {position_ds.shape}"
                    )
                if position_ds.shape[:2] != time_ds.shape:
                    raise ValueError(f"{result_path} state/time shape mismatch: {position_ds.shape} vs {time_ds.shape}")
            elif label_observable == "aratio":
                if aratio_ds.shape[:2] != time_ds.shape:
                    raise ValueError(
                        f"{result_path} aratio/time shape mismatch: {aratio_ds.shape} vs {time_ds.shape}"
                    )
                if aratio_ds.ndim != 3 or aratio_ds.shape[-1] != 1:
                    raise ValueError(
                        "The aratio label expects aratios_complete with shape (N, T, 1); "
                        f"got {aratio_ds.shape} in {result_path}."
                    )
            else:
                if smax_ds.shape[:2] != time_ds.shape:
                    raise ValueError(
                        f"{result_path} semimajor-axis/time shape mismatch: "
                        f"{smax_ds.shape} vs {time_ds.shape}"
                    )
                if smax_ds.ndim != 3 or smax_ds.shape[-1] < 1:
                    raise ValueError(
                        "The smax label expects smaxes_complete with shape (N, T, orbit); "
                        f"got {smax_ds.shape} in {result_path}."
                    )
            if row_count % group_size != 0:
                raise ValueError(f"{result_path} has {row_count} rows, not divisible by group_size={group_size}")

            batch_valid = None
            if completion_df is not None:
                batch_df = completion_df.loc[completion_df["num_batch"] == batch_id].sort_values("index")
                if len(batch_df) != row_count:
                    raise ValueError(
                        f"Completion table for batch {batch_id} has {len(batch_df)} rows, expected {row_count}"
                    )
                batch_valid = batch_df["valid"].to_numpy(dtype=bool)

            group_count = row_count // group_size
            if label_start_index >= time_steps:
                raise ValueError(
                    f"label_start_index={label_start_index} is outside the {time_steps}-step trajectory"
                )
            for local_group_id in range(group_count):
                row_start = local_group_id * group_size
                row_stop = row_start + group_size
                raw_system_id = (global_row_offset + row_start) // group_size
                if batch_valid is not None and require_complete_valid_group:
                    if not bool(np.all(batch_valid[row_start:row_stop])):
                        dropped_invalid_groups += 1
                        continue

                label = _compute_group_divergence_label(
                    positions_group=(
                        np.asarray(position_ds[row_start:row_stop], dtype=np.float64)
                        if position_ds is not None
                        else None
                    ),
                    velocities_group=(
                        np.asarray(velocity_ds[row_start:row_stop], dtype=np.float64)
                        if velocity_ds is not None
                        else None
                    ),
                    masses_group=(
                        np.asarray(mass_ds[row_start:row_stop], dtype=np.float64)
                        if mass_ds is not None
                        else None
                    ),
                    times_group=np.asarray(
                        time_ds[row_start:row_stop, label_start_index:], dtype=np.float64
                    ),
                    system_id=raw_system_id,
                    valid_pairs=valid_pairs,
                    threshold=threshold,
                    growth_lower=growth_lower,
                    growth_upper=growth_upper,
                    min_growth_points=min_growth_points,
                    aggregation=aggregation,
                    censoring_rule=censoring_rule,
                    growth_sensitivity_uppers=growth_sensitivity_uppers,
                    lambda_r2_min=lambda_r2_min,
                    min_valid_lambda_pairs=min_valid_lambda_pairs,
                    max_pair_log_mad=max_pair_log_mad,
                    aratios_group=(
                        np.asarray(
                            aratio_ds[row_start:row_stop, label_start_index:], dtype=np.float64
                        )
                        if aratio_ds is not None
                        else None
                    ),
                    smaxes_group=(
                        np.asarray(
                            smax_ds[row_start:row_stop, label_start_index:], dtype=np.float64
                        )
                        if smax_ds is not None
                        else None
                    ),
                    label_observable=label_observable,
                    lyapunov_aggregation=lyapunov_aggregation,
                    require_pair_consistency=require_pair_consistency,
                    label_start_index=label_start_index,
                )
                if not np.isfinite(label["tau_system"]):
                    dropped_nan_labels += 1
                    continue

                row: dict[str, Any] = {
                    "system_id": raw_system_id,
                    "batch_id": batch_id,
                    "local_group_id": local_group_id,
                    "raw_row_start": global_row_offset + row_start,
                    "rows_per_batch": row_count,
                    "time_steps": time_steps,
                    "label_start_index": int(label_start_index),
                    "label_time_steps": int(time_steps - label_start_index),
                    "label_observable": label_observable,
                    "tau_system": label["tau_system"],
                    "log_tau_system": label["log_tau_system"],
                    "lambda_phase_space_unfiltered": label["lambda_phase_space_unfiltered"],
                    "lambda_phase_space": label["lambda_phase_space"],
                    "lambda_smax": label["lambda_smax"],
                    "lambda_aratio": label["lambda_aratio"],
                    "lyapunov_proxy_time": label["lyapunov_proxy_time"],
                    "log_lyapunov_proxy_time": label["log_lyapunov_proxy_time"],
                    "lambda_observable": label["lambda_observable"],
                    "lambda_fast": label["lambda_fast"],
                    "log_lambda_fast": label["log_lambda_fast"],
                    "lyapunov_time_proxy": label["lyapunov_time_proxy"],
                    "log_lyapunov_time_proxy": label["log_lyapunov_time_proxy"],
                    "lyapunov_time_label": label["lyapunov_time_label"],
                    "log_lyapunov_time_label": label["log_lyapunov_time_label"],
                    "lyapunov_label_valid": label["lyapunov_label_valid"],
                    "valid_lambda_pair_count": label["valid_lambda_pair_count"],
                    "valid_lambda_pair_fraction": label["valid_lambda_pair_fraction"],
                    "pair_lambda_log_mad": label["pair_lambda_log_mad"],
                    "pair_consistency_pass": label["pair_consistency_pass"],
                    "lambda_tau_ratio": label["lambda_tau_ratio"],
                    "system_censored": label["system_censored"],
                }
                for range_name, range_payload in label["lambda_sensitivity"].items():
                    row[f"lambda_range_{range_name}"] = range_payload["lambda"]
                    row[f"lyapunov_time_range_{range_name}"] = range_payload["lyapunov_time"]
                    row[f"valid_pairs_range_{range_name}"] = range_payload["valid_pair_count"]
                    row[f"pair_log_mad_range_{range_name}"] = range_payload["pair_log_mad"]
                    row[f"valid_range_{range_name}"] = range_payload["valid"]
                for (idx_i, idx_j), tau, flag, distance0, pair_lambda, pair_r2, pair_points, pair_quality in zip(
                    valid_pairs,
                    label["pair_taus"],
                    label["pair_censored"],
                    label["pair_distance0s"],
                    label["pair_lambdas"],
                    label["pair_lambda_r2s"],
                    label["pair_lambda_points"],
                    label["pair_lambda_quality"],
                    strict=True,
                ):
                    row[_make_pair_column(idx_i, idx_j, prefix="tau")] = tau
                    row[_make_pair_column(idx_i, idx_j, prefix="censored")] = bool(flag)
                    row[_make_pair_column(idx_i, idx_j, prefix="distance0")] = distance0
                    row[_make_pair_column(idx_i, idx_j, prefix="lambda")] = pair_lambda
                    row[_make_pair_column(idx_i, idx_j, prefix="lambda_r2")] = pair_r2
                    row[_make_pair_column(idx_i, idx_j, prefix="lambda_points")] = pair_points
                    row[_make_pair_column(idx_i, idx_j, prefix="lambda_quality")] = bool(pair_quality)
                    valid_pair_distance0s.append(distance0)
                    valid_pair_lambdas.append(pair_lambda)
                    valid_pair_lambda_r2s.append(pair_r2)
                rows.append(row)
                system_counter += 1
                if max_systems is not None and system_counter >= max_systems:
                    break
        global_row_offset += row_count
        if max_systems is not None and system_counter >= max_systems:
            break

    labels_df = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    valid_label_count = int(labels_df["lyapunov_label_valid"].sum()) if len(labels_df) else 0
    sensitivity_metadata: dict[str, Any] = {}
    for upper in sorted(set(float(value) for value in growth_sensitivity_uppers)):
        range_name = _growth_range_name(growth_lower, upper)
        lambda_col = f"lambda_range_{range_name}"
        time_col = f"lyapunov_time_range_{range_name}"
        valid_col = f"valid_range_{range_name}"
        if lambda_col in labels_df:
            sensitivity_metadata[range_name] = {
                "valid_system_count": int(labels_df[valid_col].sum()),
                "valid_system_fraction": float(labels_df[valid_col].mean()),
                "lambda_summary": summarize_series(labels_df[lambda_col].to_numpy(dtype=np.float64)),
                "lyapunov_time_summary": summarize_series(
                    labels_df[time_col].to_numpy(dtype=np.float64)
                ),
            }
    metadata = {
        "label_version": LABEL_VERSION,
        "target": (
            "log_min_instability_time_from_aratio"
            if label_observable == "aratio" and lyapunov_aggregation == "max"
            else (
                "log_finite_time_lyapunov_time_proxy_from_aratio"
                if label_observable == "aratio"
                else "log_finite_time_lyapunov_time_proxy"
            )
        ),
        "label_observable": label_observable,
        "lyapunov_aggregation": lyapunov_aggregation,
        "require_pair_consistency": require_pair_consistency,
        "label_start_index": int(label_start_index),
        "label_window": (
            "trajectory after the prefix"
            if label_start_index > 0
            else "full stored trajectory"
        ),
        "num_systems": int(len(labels_df)),
        "num_valid_lyapunov_labels": valid_label_count,
        "num_rejected_lyapunov_labels": int(len(labels_df) - valid_label_count),
        "dropped_invalid_groups": int(dropped_invalid_groups),
        "dropped_nan_labels": int(dropped_nan_labels),
        "pair_phase_distance0_summary": summarize_series(valid_pair_distance0s)
        if label_observable == "phase_space"
        else summarize_series([]),
        "pair_label_distance0_summary": summarize_series(valid_pair_distance0s),
        "pair_lambda_summary": summarize_series(valid_pair_lambdas),
        "lambda_fast_summary": summarize_series(labels_df["lambda_fast"].to_numpy(dtype=np.float64)),
        "pair_lambda_r2_summary": summarize_series(valid_pair_lambda_r2s),
        "valid_lambda_pair_count_summary": summarize_series(
            labels_df["valid_lambda_pair_count"].to_numpy(dtype=np.float64)
        ),
        "pair_lambda_log_mad_summary": summarize_series(
            labels_df["pair_lambda_log_mad"].to_numpy(dtype=np.float64)
        ),
        "growth_range_sensitivity": sensitivity_metadata,
        "valid_pairs": [list(pair) for pair in valid_pairs],
        "group_size": int(group_size),
        "threshold": float(threshold),
        "growth_lower": float(growth_lower),
        "growth_upper": float(growth_upper),
        "growth_sensitivity_uppers": [float(value) for value in growth_sensitivity_uppers],
        "min_growth_points": int(min_growth_points),
        "lambda_r2_min": float(lambda_r2_min),
        "min_valid_lambda_pairs": int(min_valid_lambda_pairs),
        "max_pair_log_mad": float(max_pair_log_mad),
        "divergence_observable": (
            "aratios_complete scalar difference"
            if label_observable == "aratio"
            else "center-of-mass position+velocity phase-space distance"
        ),
        "aggregation": aggregation,
        "censoring_rule": censoring_rule,
    }
    return labels_df, metadata


def build_initial_condition_features(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    group_size: int,
    feature_aggregation: str = "first",
    feature_keys: Sequence[str] = ORBITAL15_FEATURE_KEYS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """
    Build one feature vector per system from t=0 orbital/system features.

    Returns X, y, metadata, feature_df.
    """
    if feature_aggregation not in {"first", "mean_std"}:
        raise ValueError(f"Unsupported feature_aggregation: {feature_aggregation}")
    if labels_df.empty:
        raise ValueError("labels_df is empty")

    label_records = labels_df[["system_id", "batch_id", "local_group_id", "log_tau_system"]].copy()
    labels_by_batch: dict[int, pd.DataFrame] = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in label_records.groupby("batch_id", sort=True)
    }
    feature_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue

        with h5py.File(result_path, "r") as f:
            shard = {key: np.asarray(f[key][:, 0, ...], dtype=np.float64) for key in feature_keys}
        for key in feature_keys:
            if key in INITIAL_ANGLE_KEYS:
                shard[key] = np.concatenate([np.sin(shard[key]), np.cos(shard[key])], axis=-1)

        for row in labels_by_batch[batch_id].itertuples(index=False):
            start = int(row.local_group_id) * group_size
            stop = start + group_size
            traj_feature_blocks: list[np.ndarray] = []
            traj_feature_names: list[str] = []
            for key in feature_keys:
                names, flat = _flatten_feature_row(shard[key][start], prefix=key)
                traj_feature_names.extend(names)
                traj_block = np.asarray(shard[key][start:stop], dtype=np.float64).reshape(group_size, -1)
                traj_feature_blocks.append(traj_block)

            traj_matrix = np.concatenate(traj_feature_blocks, axis=1)
            if feature_names is None:
                feature_names = traj_feature_names

            if feature_aggregation == "first":
                feature_vector = traj_matrix[0]
                names_out = feature_names
            else:
                mean_vec = np.nanmean(traj_matrix, axis=0)
                std_vec = np.nanstd(traj_matrix, axis=0)
                feature_vector = np.concatenate([mean_vec, std_vec], axis=0)
                names_out = [f"mean__{name}" for name in feature_names] + [f"std__{name}" for name in feature_names]

            feature_row = {"system_id": int(row.system_id)}
            for name, value in zip(names_out, feature_vector, strict=True):
                feature_row[name] = float(value) if np.isfinite(value) else math.nan
            feature_rows.append(feature_row)

    feature_df = pd.DataFrame(feature_rows).sort_values("system_id").reset_index(drop=True)
    merged = labels_df[["system_id", "log_tau_system"]].merge(feature_df, on="system_id", how="inner")
    if len(merged) != len(labels_df):
        raise ValueError(
            f"Feature/label mismatch: merged {len(merged)} rows vs labels {len(labels_df)} rows"
        )
    feature_columns = [col for col in merged.columns if col not in {"system_id", "log_tau_system"}]
    X = merged[feature_columns].to_numpy(dtype=np.float64)
    y = merged["log_tau_system"].to_numpy(dtype=np.float64)
    metadata = {
        "feature_columns": feature_columns,
        "feature_aggregation": feature_aggregation,
        "feature_keys": list(feature_keys),
        "angle_features_encoded_as_sin_cos": sorted(
            set(feature_keys).intersection(INITIAL_ANGLE_KEYS)
        ),
        "num_features": int(len(feature_columns)),
    }
    return X, y, metadata, merged


def build_initial_prefix_features(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    prefix_len: int = 100,
    time_stride: int = 1,
    group_size: int = DEFAULT_GROUP_SIZE,
    feature_aggregation: str = "first",
    valid_pairs: Sequence[tuple[int, int]] = DEFAULT_VALID_PAIRS,
    growth_lower: float = 1.0,
    growth_upper: float = 100.0,
    min_growth_points: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Build initial orbital15 plus an early-trajectory summary per system.

    The initial vector is the 15-dimensional orbital representation at t=0.
    The prefix part contains group-mean/std trajectories for aratio/ecc/inc,
    pair-wise log R(t) summaries, and early pair-growth-rate summaries. The
    latter are computed only from the observed prefix and are therefore valid
    covariates for a future-only label.
    """
    if prefix_len < 2 or time_stride < 1:
        raise ValueError(
            f"prefix_len must be at least 2 and time_stride positive, got {prefix_len}, {time_stride}"
        )
    if feature_aggregation not in {"first", "mean_std"}:
        raise ValueError(f"Unsupported feature_aggregation: {feature_aggregation}")
    if labels_df.empty:
        raise ValueError("labels_df is empty")

    label_records = labels_df[["system_id", "batch_id", "local_group_id", "log_tau_system"]].copy()
    labels_by_batch: dict[int, pd.DataFrame] = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in label_records.groupby("batch_id", sort=True)
    }
    feature_rows: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue
        with h5py.File(result_path, "r") as f:
            time_steps = int(f["times_complete"].shape[1])
            stop = min(prefix_len * time_stride, time_steps)
            if stop < 2:
                raise ValueError(f"No usable prefix in {result_path}: {time_steps} time steps")
            initial_shard = {
                key: np.asarray(f[key][:, 0, ...], dtype=np.float64)
                for key in ORBITAL15_FEATURE_KEYS
            }
            trajectory_shard = {
                key: np.asarray(f[key][:, :stop:time_stride, ...], dtype=np.float64)
                for key in INITIAL_PREFIX_TRAJECTORY_KEYS
            }
            times = np.asarray(f["times_complete"][:, :stop:time_stride], dtype=np.float64)

        prefix_points = int(times.shape[1])
        sampled_points = np.unique(
            np.linspace(0, prefix_points - 1, min(20, prefix_points), dtype=np.int64)
        )
        for row in labels_by_batch[batch_id].itertuples(index=False):
            start = int(row.local_group_id) * group_size
            stop_row = start + group_size
            feature_row: dict[str, Any] = {"system_id": int(row.system_id)}
            names_local: list[str] = []

            initial_blocks: list[np.ndarray] = []
            initial_names: list[str] = []
            for key in ORBITAL15_FEATURE_KEYS:
                data0 = initial_shard[key][start:stop_row, ...].reshape(group_size, -1)
                if key in INITIAL_ANGLE_KEYS:
                    data0 = np.concatenate([np.sin(data0), np.cos(data0)], axis=-1)
                if feature_aggregation == "first":
                    vector = data0[0]
                    prefix = f"initial__{key}"
                    names = [f"{prefix}_{idx}" for idx in range(vector.size)]
                else:
                    mean_vector = np.nanmean(data0, axis=0)
                    std_vector = np.nanstd(data0, axis=0)
                    vector = np.concatenate([mean_vector, std_vector])
                    names = (
                        [f"initial_mean__{key}_{idx}" for idx in range(mean_vector.size)]
                        + [f"initial_std__{key}_{idx}" for idx in range(std_vector.size)]
                    )
                initial_blocks.append(np.asarray(vector, dtype=np.float64))
                initial_names.extend(names)
            initial_vector = np.concatenate(initial_blocks)
            for name, value in zip(initial_names, initial_vector, strict=True):
                feature_row[name] = float(value) if np.isfinite(value) else math.nan
            names_local.extend(initial_names)

            for key in INITIAL_PREFIX_TRAJECTORY_KEYS:
                data = trajectory_shard[key][start:stop_row].reshape(group_size, prefix_points, -1)
                mean_series = np.nanmean(data, axis=0)
                std_series = np.nanstd(data, axis=0)
                for stat_name, values in (("mean", mean_series), ("std", std_series)):
                    flat = np.asarray(values[sampled_points], dtype=np.float64).reshape(-1)
                    names = [
                        f"trajectory_{stat_name}__{key}_{idx}"
                        for idx in range(flat.size)
                    ]
                    for name, value in zip(names, flat, strict=True):
                        feature_row[name] = float(value) if np.isfinite(value) else math.nan
                    names_local.extend(names)
                for stat_name, values in (
                    ("last", mean_series[-1]),
                    ("delta", mean_series[-1] - mean_series[0]),
                    ("time_mean", np.nanmean(mean_series, axis=0)),
                    ("time_std", np.nanstd(mean_series, axis=0)),
                ):
                    flat = np.asarray(values, dtype=np.float64).reshape(-1)
                    names = [f"trajectory_{stat_name}__{key}_{idx}" for idx in range(flat.size)]
                    for name, value in zip(names, flat, strict=True):
                        feature_row[name] = float(value) if np.isfinite(value) else math.nan
                    names_local.extend(names)

            aratio = trajectory_shard["aratios_complete"][start:stop_row].reshape(
                group_size, prefix_points
            )
            time_grid = times[start:stop_row]
            log_ratio_rows: list[np.ndarray] = []
            prefix_slopes: list[float] = []
            for idx_i, idx_j in valid_pairs:
                if idx_i >= group_size or idx_j >= group_size:
                    continue
                delta = np.abs(aratio[idx_i] - aratio[idx_j])
                delta0 = delta[0] if np.isfinite(delta[0]) else math.nan
                scale = max(float(delta0), 1e-12) if np.isfinite(delta0) else 1e-12
                ratio = delta / scale
                log_ratio_rows.append(
                    np.log(np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, np.nan))
                )
                slope, _, _ = _fit_log_growth_rate(
                    ratio=ratio,
                    times=time_grid[idx_i],
                    lower=growth_lower,
                    upper=growth_upper,
                    min_points=min_growth_points,
                )
                if np.isfinite(slope) and slope > 0.0:
                    prefix_slopes.append(float(slope))

            if log_ratio_rows:
                log_ratio = np.stack(log_ratio_rows, axis=0)
                ratio_stats = {
                    "mean": np.nanmean(log_ratio, axis=0),
                    "std": np.nanstd(log_ratio, axis=0),
                    "q90": np.nanpercentile(log_ratio, 90, axis=0),
                    "max": np.nanmax(log_ratio, axis=0),
                }
            else:
                ratio_stats = {
                    name: np.full(prefix_points, np.nan, dtype=np.float64)
                    for name in ("mean", "std", "q90", "max")
                }
            for stat_name, values in ratio_stats.items():
                sampled_values = np.asarray(values, dtype=np.float64)[sampled_points]
                names = [f"early_log_R_{stat_name}_{idx}" for idx in range(len(sampled_values))]
                for name, value in zip(names, sampled_values, strict=True):
                    feature_row[name] = float(value) if np.isfinite(value) else math.nan
                names_local.extend(names)

            slope_summary = {
                "median": float(np.median(prefix_slopes)) if prefix_slopes else math.nan,
                "mean": float(np.mean(prefix_slopes)) if prefix_slopes else math.nan,
                "max": float(np.max(prefix_slopes)) if prefix_slopes else math.nan,
                "std": float(np.std(prefix_slopes)) if prefix_slopes else math.nan,
                "count": float(len(prefix_slopes)),
            }
            for stat_name, value in slope_summary.items():
                name = f"early_lambda_{stat_name}"
                feature_row[name] = value if np.isfinite(value) else math.nan
                names_local.append(name)

            if feature_columns is None:
                feature_columns = names_local
            feature_rows.append(feature_row)

    feature_df = pd.DataFrame(feature_rows).sort_values("system_id").reset_index(drop=True)
    merged = labels_df[["system_id", "log_tau_system"]].merge(feature_df, on="system_id", how="inner")
    if len(merged) != len(labels_df):
        raise ValueError(
            f"Initial-prefix feature/label mismatch: merged {len(merged)} rows vs labels {len(labels_df)}"
        )
    feature_cols = [col for col in merged.columns if col not in {"system_id", "log_tau_system"}]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    y = merged["log_tau_system"].to_numpy(dtype=np.float64)
    metadata = {
        "feature_columns": feature_cols,
        "feature_aggregation": feature_aggregation,
        "initial_representation": "orbital15",
        "initial_feature_count": 15 if feature_aggregation == "first" else 30,
        "trajectory_feature_keys": list(INITIAL_PREFIX_TRAJECTORY_KEYS),
        "prefix_len": int(prefix_len),
        "time_stride": int(time_stride),
        "prefix_points": int(prefix_points),
        "feature_sample_points": int(len(sampled_points)),
        "valid_pairs": [list(pair) for pair in valid_pairs],
        "num_features": int(len(feature_cols)),
    }
    return X, y, metadata, merged


def build_prefix_summary_features(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    prefix_len: int = 200,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_positions_velocities: bool = False,
    feature_keys: Sequence[str] = PREFIX_FEATURE_KEYS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """
    Build one system-level feature vector from an early trajectory prefix.

    Returns X, y, metadata, feature_df.
    """
    keys = list(feature_keys)
    if include_positions_velocities:
        keys.extend(PREFIX_EXTRA_STATE_KEYS)

    label_records = labels_df[["system_id", "batch_id", "local_group_id", "log_tau_system"]].copy()
    labels_by_batch = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in label_records.groupby("batch_id", sort=True)
    }
    rows: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue
        with h5py.File(result_path, "r") as f:
            shard = {
                key: np.asarray(f[key][:, :prefix_len, ...], dtype=np.float64)
                for key in keys
            }

        for row in labels_by_batch[batch_id].itertuples(index=False):
            start = int(row.local_group_id) * group_size
            stop = start + group_size
            feature_row = {"system_id": int(row.system_id)}
            names_local: list[str] = []
            for key in keys:
                data = shard[key][start:stop].reshape(group_size, prefix_len, -1)
                first = np.nanmean(data[:, 0, :], axis=0)
                last = np.nanmean(data[:, -1, :], axis=0)
                summaries = {
                    "mean": np.nanmean(data, axis=(0, 1)),
                    "std": np.nanstd(data, axis=(0, 1)),
                    "min": np.nanmin(data, axis=(0, 1)),
                    "max": np.nanmax(data, axis=(0, 1)),
                    "last": last,
                    "delta": last - first,
                }
                for stat_name, stat_values in summaries.items():
                    for idx, value in enumerate(np.asarray(stat_values).reshape(-1)):
                        col = f"{stat_name}__{key}_{idx}"
                        feature_row[col] = float(value) if np.isfinite(value) else math.nan
                        names_local.append(col)
            if feature_columns is None:
                feature_columns = names_local
            rows.append(feature_row)

    feature_df = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    merged = labels_df[["system_id", "log_tau_system"]].merge(feature_df, on="system_id", how="inner")
    if len(merged) != len(labels_df):
        raise ValueError(
            f"Prefix feature/label mismatch: merged {len(merged)} rows vs labels {len(labels_df)} rows"
        )
    feature_cols = [col for col in merged.columns if col not in {"system_id", "log_tau_system"}]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    y = merged["log_tau_system"].to_numpy(dtype=np.float64)
    metadata = {
        "feature_columns": feature_cols,
        "prefix_len": int(prefix_len),
        "include_positions_velocities": bool(include_positions_velocities),
        "feature_keys": keys,
        "num_features": int(len(feature_cols)),
    }
    return X, y, metadata, merged


def build_prefix_sequence_features(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    prefix_len: int = 200,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_positions_velocities: bool = False,
    feature_keys: Sequence[str] = PREFIX_FEATURE_KEYS,
    sequence_aggregation: str = "mean_std",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build one system-level early-prefix sequence per system.

    Returns
    -------
    X_seq : np.ndarray
        Shape (N_systems, prefix_len, C).
    y : np.ndarray
        Shape (N_systems,).
    metadata : dict
        Sequence channel names and configuration.
    """
    if sequence_aggregation not in {"mean", "mean_std"}:
        raise ValueError(f"Unsupported sequence_aggregation: {sequence_aggregation}")

    keys = list(feature_keys)
    if include_positions_velocities:
        keys.extend(PREFIX_EXTRA_STATE_KEYS)

    label_records = labels_df[["system_id", "batch_id", "local_group_id", "log_tau_system"]].copy()
    labels_by_batch = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in label_records.groupby("batch_id", sort=True)
    }
    sequence_rows: list[np.ndarray] = []
    sequence_system_ids: list[int] = []
    channel_names: list[str] | None = None

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue
        with h5py.File(result_path, "r") as f:
            shard = {
                key: np.asarray(f[key][:, :prefix_len, ...], dtype=np.float64)
                for key in keys
            }

        for row in labels_by_batch[batch_id].itertuples(index=False):
            start = int(row.local_group_id) * group_size
            stop = start + group_size
            block_list: list[np.ndarray] = []
            names_local: list[str] = []
            for key in keys:
                data = shard[key][start:stop].reshape(group_size, prefix_len, -1)
                mean_block = np.nanmean(data, axis=0)
                block_list.append(mean_block)
                names_local.extend([f"mean__{key}_{idx}" for idx in range(mean_block.shape[-1])])
                if sequence_aggregation == "mean_std":
                    std_block = np.nanstd(data, axis=0)
                    block_list.append(std_block)
                    names_local.extend([f"std__{key}_{idx}" for idx in range(std_block.shape[-1])])
            seq = np.concatenate(block_list, axis=-1)
            if channel_names is None:
                channel_names = names_local
            sequence_rows.append(seq.astype(np.float64))
            sequence_system_ids.append(int(row.system_id))

    if not sequence_rows:
        raise ValueError("No sequence rows were built.")
    X_seq = np.stack(sequence_rows, axis=0)
    seq_df = pd.DataFrame({"system_id": sequence_system_ids})
    seq_df = labels_df[["system_id", "log_tau_system"]].merge(seq_df, on="system_id", how="inner")
    if len(seq_df) != len(labels_df):
        raise ValueError(f"Sequence/label mismatch: {len(seq_df)} vs {len(labels_df)}")
    y = seq_df["log_tau_system"].to_numpy(dtype=np.float64)
    metadata = {
        "prefix_len": int(prefix_len),
        "include_positions_velocities": bool(include_positions_velocities),
        "feature_keys": keys,
        "sequence_aggregation": sequence_aggregation,
        "channel_names": channel_names or [],
        "num_channels": int(X_seq.shape[-1]),
    }
    return X_seq, y, metadata


def compute_tau_diagnostics(labels_df: pd.DataFrame, t_obs: float | None = None) -> dict[str, Any]:
    tau = labels_df["tau_system"].to_numpy(dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "num_systems_total": int(tau.size),
        "fraction_tau_le_50": float(np.mean(tau <= 50.0)),
        "fraction_tau_le_100": float(np.mean(tau <= 100.0)),
        "fraction_tau_le_200": float(np.mean(tau <= 200.0)),
        "median_tau": float(np.median(tau)),
        "p10_tau": float(np.percentile(tau, 10)),
        "p25_tau": float(np.percentile(tau, 25)),
        "p75_tau": float(np.percentile(tau, 75)),
        "p90_tau": float(np.percentile(tau, 90)),
    }
    diagnostics["t_obs"] = None if t_obs is None else float(t_obs)
    diagnostics["fraction_tau_le_t_obs"] = None if t_obs is None else float(np.mean(tau <= t_obs))
    return diagnostics


def compute_prefix_observation_times(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    prefix_len: int,
    time_stride: int,
    group_size: int,
) -> np.ndarray:
    if prefix_len < 1 or time_stride < 1:
        raise ValueError(f"prefix_len and time_stride must be positive, got {prefix_len}, {time_stride}")
    labels_by_batch = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in labels_df.groupby("batch_id", sort=True)
    }
    observations: dict[int, float] = {}
    stop = prefix_len * time_stride
    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue
        with h5py.File(result_path, "r") as f:
            time_ds = f["times_complete"]
            for row in labels_by_batch[batch_id].itertuples(index=False):
                row_index = int(row.local_group_id) * group_size
                selected = np.asarray(time_ds[row_index, :stop:time_stride], dtype=np.float64)
                finite = selected[np.isfinite(selected)]
                observations[int(row.system_id)] = float(finite[-1]) if finite.size else math.nan
    return np.asarray([observations[int(system_id)] for system_id in labels_df["system_id"]], dtype=np.float64)


def _center_group_state_prefix(states: np.ndarray, masses: np.ndarray) -> np.ndarray:
    mass_sum = np.sum(masses, axis=2, keepdims=True)
    safe_mass_sum = np.where(np.abs(mass_sum) > 1e-12, mass_sum, 1.0)
    center = np.sum(states * masses[..., None], axis=2, keepdims=True) / safe_mass_sum[..., None]
    return states - center


def build_prefix_model_inputs(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    prefix_len: int,
    time_stride: int,
    group_size: int,
    sequence_aggregation: str = "mean_std",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sequence_aggregation not in {"mean", "mean_std"}:
        raise ValueError(f"Unsupported sequence_aggregation: {sequence_aggregation}")
    labels_by_batch = {
        int(batch_id): batch_df.sort_values("local_group_id")
        for batch_id, batch_df in labels_df.groupby("batch_id", sort=True)
    }
    orbital_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    ensemble_rows: list[np.ndarray] = []
    system_ids: list[int] = []
    orbital_names: list[str] | None = None
    phase_names: list[str] | None = None
    stop = prefix_len * time_stride

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        if batch_id not in labels_by_batch:
            continue
        with h5py.File(result_path, "r") as f:
            for row in labels_by_batch[batch_id].itertuples(index=False):
                start = int(row.local_group_id) * group_size
                stop_row = start + group_size
                orbital_blocks: list[np.ndarray] = []
                names_local: list[str] = []
                for key in PREFIX_FEATURE_KEYS:
                    data = np.asarray(f[key][start:stop_row, :stop:time_stride, ...], dtype=np.float32)
                    data = data.reshape(group_size, data.shape[1], -1)
                    orbital_blocks.append(data)
                    names_local.extend([f"{key}_{idx}" for idx in range(data.shape[-1])])
                orbital = np.concatenate(orbital_blocks, axis=-1)

                masses = np.asarray(f["masses_complete"][start:stop_row, :stop:time_stride, :], dtype=np.float32)
                positions = np.asarray(f["positions_complete"][start:stop_row, :stop:time_stride, :, :], dtype=np.float32)
                velocities = np.asarray(f["velocities_complete"][start:stop_row, :stop:time_stride, :, :], dtype=np.float32)
                positions = _center_group_state_prefix(positions, masses).reshape(group_size, orbital.shape[1], -1)
                velocities = _center_group_state_prefix(velocities, masses).reshape(group_size, orbital.shape[1], -1)
                phase = np.concatenate([orbital, positions, velocities], axis=-1)

                def aggregate(data: np.ndarray) -> np.ndarray:
                    blocks = [np.nanmean(data, axis=0)]
                    if sequence_aggregation == "mean_std":
                        blocks.append(np.nanstd(data, axis=0))
                    return np.concatenate(blocks, axis=-1).astype(np.float32)

                orbital_rows.append(aggregate(orbital))
                phase_rows.append(aggregate(phase))
                ensemble_rows.append(orbital.astype(np.float32))
                system_ids.append(int(row.system_id))
                if orbital_names is None:
                    stat_names = ["mean"] if sequence_aggregation == "mean" else ["mean", "std"]
                    orbital_names = [f"{stat}__{name}" for stat in stat_names for name in names_local]
                    raw_phase_names = names_local + [f"position_com_{i}" for i in range(9)] + [f"velocity_com_{i}" for i in range(9)]
                    phase_names = [f"{stat}__{name}" for stat in stat_names for name in raw_phase_names]

    if not system_ids:
        raise ValueError("No prefix model inputs were built.")
    expected_ids = labels_df["system_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.asarray(system_ids, dtype=np.int64), expected_ids):
        raise ValueError("Prefix feature order does not match labels_df system order.")
    arrays = {
        "cnn_orbital": np.stack(orbital_rows).astype(np.float32),
        "cnn_phase_space": np.stack(phase_rows).astype(np.float32),
        "set_cnn": np.stack(ensemble_rows).astype(np.float32),
    }
    metadata = {
        "prefix_len": int(prefix_len),
        "time_stride": int(time_stride),
        "sequence_aggregation": sequence_aggregation,
        "cnn_orbital_shape": list(arrays["cnn_orbital"].shape),
        "cnn_phase_space_shape": list(arrays["cnn_phase_space"].shape),
        "set_cnn_shape": list(arrays["set_cnn"].shape),
        "orbital_channel_names": orbital_names or [],
        "phase_space_channel_names": phase_names or [],
        "set_cnn_representation": "full 32-variation orbital prefix",
        "phase_space_reference_frame": "per-trajectory center of mass",
    }
    estimated_gb = sum(array.nbytes for array in arrays.values()) / (1024**3)
    metadata["in_memory_size_gb"] = float(estimated_gb)
    if estimated_gb > 2.0:
        warnings.warn(
            f"Prefix arrays use approximately {estimated_gb:.2f} GiB; consider --max-systems or a larger time stride.",
            RuntimeWarning,
            stacklevel=2,
        )
    return arrays, metadata


def build_survival_targets(labels_df: pd.DataFrame, horizons: Sequence[float]) -> np.ndarray:
    """
    Returns Y with shape (n_systems, n_horizons).
    Y[k, h] = 1 if tau_system > horizon else 0.
    """
    tau = labels_df["tau_system"].to_numpy(dtype=np.float64)[:, None]
    horizons_arr = np.asarray(horizons, dtype=np.float64)[None, :]
    return (tau > horizons_arr).astype(np.int64)


def _compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    if spearmanr is not None:
        value = spearmanr(y_true, y_pred).statistic
        return float(value) if value is not None else math.nan
    return float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank(), method="pearson"))


def _compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _subset_regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    if y_true_log.size == 0:
        return {
            "count": 0.0,
            "mae_log_tau": math.nan,
            "rmse_log_tau": math.nan,
            "mae_tau": math.nan,
            "spearman": math.nan,
            "pearson": math.nan,
            "factor_2_accuracy": math.nan,
            "factor_3_accuracy": math.nan,
            "median_absolute_log_error": math.nan,
        }
    y_true_tau = np.exp(y_true_log)
    y_pred_tau = np.exp(y_pred_log)
    abs_log_error = np.abs(y_pred_log - y_true_log)
    return {
        "count": float(y_true_log.size),
        "mae_log_tau": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log_tau": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_tau": float(mean_absolute_error(y_true_tau, y_pred_tau)),
        "spearman": _compute_spearman(y_true_log, y_pred_log),
        "pearson": _compute_pearson(y_true_log, y_pred_log),
        "factor_2_accuracy": float(np.mean(abs_log_error < np.log(2.0))),
        "factor_3_accuracy": float(np.mean(abs_log_error < np.log(3.0))),
        "median_absolute_log_error": float(np.median(abs_log_error)),
    }


def evaluate_regression_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    censored_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    censored_mask = np.asarray(censored_mask, dtype=bool)
    uncensored_mask = ~censored_mask
    return {
        "all": _subset_regression_metrics(y_true_log, y_pred_log),
        "uncensored_only": _subset_regression_metrics(y_true_log[uncensored_mask], y_pred_log[uncensored_mask]),
        "censored_only": _subset_regression_metrics(y_true_log[censored_mask], y_pred_log[censored_mask]),
    }


def make_regression_split(
    labels_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    exclude_censored: bool = False,
) -> dict[str, np.ndarray | float | int | bool]:
    n = len(labels_df)
    censored = labels_df["system_censored"].to_numpy(dtype=bool)
    train_idx, test_idx = train_test_split(
        np.arange(n),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    fit_train_idx = train_idx[~censored[train_idx]] if exclude_censored else train_idx
    if fit_train_idx.size == 0:
        raise ValueError("No training samples remain after exclude_censored filtering.")
    return {
        "train_idx": train_idx,
        "fit_train_idx": fit_train_idx,
        "test_idx": test_idx,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "exclude_censored": bool(exclude_censored),
    }


def _prepare_tabular_features(
    X: np.ndarray,
    fit_train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X[fit_train_idx])
    X_test = imputer.transform(X[test_idx])
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, imputer, scaler


def _prepare_sequence_features(
    X_seq: np.ndarray,
    fit_train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    X_seq = np.asarray(X_seq, dtype=np.float32)
    train_flat = X_seq[fit_train_idx].reshape(-1, X_seq.shape[-1])
    channel_median = np.nanmedian(train_flat, axis=0)
    channel_median = np.where(np.isfinite(channel_median), channel_median, 0.0)

    def _impute(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32).copy()
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(channel_median, np.where(bad)[-1])
        return out

    X_train = _impute(X_seq[fit_train_idx])
    X_test = _impute(X_seq[test_idx])

    train_flat = X_train.reshape(-1, X_train.shape[-1])
    channel_mean = train_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    channel_std = train_flat.std(axis=0, dtype=np.float64).astype(np.float32)
    channel_std = np.where(channel_std > 1e-8, channel_std, 1.0)
    broadcast_shape = (1,) * (X_train.ndim - 1) + (X_train.shape[-1],)
    X_train = (X_train - channel_mean.reshape(broadcast_shape)) / channel_std.reshape(broadcast_shape)
    X_test = (X_test - channel_mean.reshape(broadcast_shape)) / channel_std.reshape(broadcast_shape)
    stats = {
        "channel_median": channel_median,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }
    return X_train, X_test, stats


def _split_train_val_indices(
    fit_train_idx: np.ndarray,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit_train_idx = np.asarray(fit_train_idx, dtype=np.int64)
    if fit_train_idx.size < 10 or val_fraction <= 0.0:
        return fit_train_idx, fit_train_idx
    train_sub_idx, val_idx = train_test_split(
        fit_train_idx,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
    )
    if train_sub_idx.size == 0 or val_idx.size == 0:
        return fit_train_idx, fit_train_idx
    return train_sub_idx, val_idx


def _resolve_device(device_name: str | None = None) -> str:
    if torch is None:
        raise ImportError("PyTorch is required for the MLP and CNN baselines.")
    if device_name:
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_torch(seed: int) -> None:
    """Reset neural initialization and minibatch order for comparable runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNNRegressor(nn.Module):
    def __init__(self, input_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return self.head(x).squeeze(-1)


class SetCNNRegressor(nn.Module):
    def __init__(self, input_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, group, time, channels = x.shape
        encoded = self.temporal(x.reshape(batch * group, time, channels).transpose(1, 2)).squeeze(-1)
        pooled = encoded.reshape(batch, group, -1).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def _make_regression_loss(loss_name: str, huber_beta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=huber_beta)
    raise ValueError(f"Unsupported regression loss: {loss_name}")


def _fit_torch_regressor(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device_name: str,
    loss_name: str,
    huber_beta: float,
) -> nn.Module:
    device = torch.device(device_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = _make_regression_loss(loss_name, huber_beta)

    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(
        torch.from_numpy(np.asarray(X_val, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = math.inf
    wait = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss_sum += float(loss_fn(model(xb), yb).item()) * xb.size(0)
        val_loss = val_loss_sum / max(len(val_ds), 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch_regressor(
    model: nn.Module,
    X: np.ndarray,
    device_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def run_random_forest_baseline(
    X: np.ndarray,
    labels_df: pd.DataFrame,
    feature_columns: Sequence[str],
    split: dict[str, np.ndarray | float | int | bool] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    exclude_censored: bool = False,
) -> dict[str, Any]:
    system_ids = labels_df["system_id"].to_numpy(dtype=np.int64)
    y = labels_df["log_tau_system"].to_numpy(dtype=np.float64)
    censored = labels_df["system_censored"].to_numpy(dtype=bool)
    split = split or make_regression_split(
        labels_df=labels_df,
        test_size=test_size,
        random_state=random_state,
        exclude_censored=exclude_censored,
    )
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    fit_train_idx = np.asarray(split["fit_train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)

    X_train, X_test, imputer, _ = _prepare_tabular_features(
        X=X,
        fit_train_idx=fit_train_idx,
        test_idx=test_idx,
    )
    y_train = y[fit_train_idx]
    y_test = y[test_idx]
    censored_test = censored[test_idx]

    constant_value = float(np.median(y_train))
    y_pred_constant = np.full_like(y_test, fill_value=constant_value, dtype=np.float64)
    constant_metrics = evaluate_regression_predictions(y_true_log=y_test, y_pred_log=y_pred_constant, censored_mask=censored_test)

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    rf_metrics = evaluate_regression_predictions(y_true_log=y_test, y_pred_log=y_pred_rf, censored_mask=censored_test)

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    predictions = pd.DataFrame(
        {
            "system_id": system_ids[test_idx],
            "y_true_log_tau": y_test,
            "y_true_tau": np.exp(y_test),
            "system_censored": censored_test,
            "constant_pred_log_tau": y_pred_constant,
            "constant_pred_tau": np.exp(y_pred_constant),
            "constant_residual_log_tau": y_pred_constant - y_test,
            "rf_pred_log_tau": y_pred_rf,
            "rf_pred_tau": np.exp(y_pred_rf),
            "rf_residual_log_tau": y_pred_rf - y_test,
        }
    )
    return {
        "metrics": {
            "constant_median": constant_metrics,
            "random_forest": rf_metrics,
        },
        "predictions": predictions,
        "feature_importances": importance_df,
        "split": {
            "train_count": int(train_idx.size),
            "fit_train_count": int(fit_train_idx.size),
            "test_count": int(test_idx.size),
            "exclude_censored": bool(split["exclude_censored"]),
            "test_size": float(split["test_size"]),
            "random_state": int(split["random_state"]),
        },
    }


def run_mlp_baseline(
    X: np.ndarray,
    labels_df: pd.DataFrame,
    split: dict[str, np.ndarray | float | int | bool],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    val_fraction: float,
    patience: int,
    loss_name: str,
    huber_beta: float,
    device_name: str | None = None,
) -> dict[str, Any]:
    device_name = _resolve_device(device_name)
    _seed_torch(int(split["random_state"]))
    system_ids = labels_df["system_id"].to_numpy(dtype=np.int64)
    y = labels_df["log_tau_system"].to_numpy(dtype=np.float64)
    censored = labels_df["system_censored"].to_numpy(dtype=bool)
    fit_train_idx = np.asarray(split["fit_train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    train_sub_idx, val_idx = _split_train_val_indices(
        fit_train_idx=fit_train_idx,
        val_fraction=val_fraction,
        random_state=int(split["random_state"]),
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X[train_sub_idx])
    X_val = imputer.transform(X[val_idx])
    X_test = imputer.transform(X[test_idx])
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(input_dim=X_train.shape[1], dropout=dropout)
    model = _fit_torch_regressor(
        model=model,
        X_train=X_train,
        y_train=y[train_sub_idx],
        X_val=X_val,
        y_val=y[val_idx],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device_name=device_name,
        loss_name=loss_name,
        huber_beta=huber_beta,
    )
    y_pred = _predict_torch_regressor(model, X_test, device_name=device_name)
    metrics = evaluate_regression_predictions(
        y_true_log=y[test_idx],
        y_pred_log=y_pred,
        censored_mask=censored[test_idx],
    )
    predictions = pd.DataFrame(
        {
            "system_id": system_ids[test_idx],
            "y_true_log_tau": y[test_idx],
            "y_true_tau": np.exp(y[test_idx]),
            "system_censored": censored[test_idx],
            "mlp_pred_log_tau": y_pred,
            "mlp_pred_tau": np.exp(y_pred),
            "mlp_residual_log_tau": y_pred - y[test_idx],
        }
    )
    return {"metrics": {"mlp": metrics}, "predictions": predictions}


def run_cnn_baseline(
    X_seq: np.ndarray,
    labels_df: pd.DataFrame,
    split: dict[str, np.ndarray | float | int | bool],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    val_fraction: float,
    patience: int,
    model_name: str,
    loss_name: str,
    huber_beta: float,
    device_name: str | None = None,
) -> dict[str, Any]:
    device_name = _resolve_device(device_name)
    _seed_torch(int(split["random_state"]))
    system_ids = labels_df["system_id"].to_numpy(dtype=np.int64)
    y = labels_df["log_tau_system"].to_numpy(dtype=np.float64)
    censored = labels_df["system_censored"].to_numpy(dtype=bool)
    fit_train_idx = np.asarray(split["fit_train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    train_sub_idx, val_idx = _split_train_val_indices(
        fit_train_idx=fit_train_idx,
        val_fraction=val_fraction,
        random_state=int(split["random_state"]),
    )

    X_train_full, X_test, _ = _prepare_sequence_features(X_seq=X_seq, fit_train_idx=fit_train_idx, test_idx=test_idx)
    train_pos = {idx: pos for pos, idx in enumerate(fit_train_idx.tolist())}
    X_train = np.stack([X_train_full[train_pos[idx]] for idx in train_sub_idx], axis=0)
    X_val = np.stack([X_train_full[train_pos[idx]] for idx in val_idx], axis=0)

    model = CNNRegressor(input_channels=X_train.shape[-1], dropout=dropout)
    model = _fit_torch_regressor(
        model=model,
        X_train=X_train,
        y_train=y[train_sub_idx],
        X_val=X_val,
        y_val=y[val_idx],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device_name=device_name,
        loss_name=loss_name,
        huber_beta=huber_beta,
    )
    y_pred = _predict_torch_regressor(model, X_test, device_name=device_name)
    metrics = evaluate_regression_predictions(
        y_true_log=y[test_idx],
        y_pred_log=y_pred,
        censored_mask=censored[test_idx],
    )
    predictions = pd.DataFrame(
        {
            "system_id": system_ids[test_idx],
            "y_true_log_tau": y[test_idx],
            "y_true_tau": np.exp(y[test_idx]),
            "system_censored": censored[test_idx],
            f"{model_name}_pred_log_tau": y_pred,
            f"{model_name}_pred_tau": np.exp(y_pred),
            f"{model_name}_residual_log_tau": y_pred - y[test_idx],
        }
    )
    return {"metrics": {model_name: metrics}, "predictions": predictions}


def run_set_cnn_baseline(
    X_ensemble: np.ndarray,
    labels_df: pd.DataFrame,
    split: dict[str, np.ndarray | float | int | bool],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    val_fraction: float,
    patience: int,
    loss_name: str,
    huber_beta: float,
    device_name: str | None = None,
) -> dict[str, Any]:
    device_name = _resolve_device(device_name)
    _seed_torch(int(split["random_state"]))
    system_ids = labels_df["system_id"].to_numpy(dtype=np.int64)
    y = labels_df["log_tau_system"].to_numpy(dtype=np.float64)
    censored = labels_df["system_censored"].to_numpy(dtype=bool)
    fit_train_idx = np.asarray(split["fit_train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    train_sub_idx, val_idx = _split_train_val_indices(
        fit_train_idx=fit_train_idx,
        val_fraction=val_fraction,
        random_state=int(split["random_state"]),
    )

    X_train_full, X_test, _ = _prepare_sequence_features(
        X_seq=X_ensemble,
        fit_train_idx=fit_train_idx,
        test_idx=test_idx,
    )
    train_pos = {idx: pos for pos, idx in enumerate(fit_train_idx.tolist())}
    X_train = np.stack([X_train_full[train_pos[idx]] for idx in train_sub_idx], axis=0)
    X_val = np.stack([X_train_full[train_pos[idx]] for idx in val_idx], axis=0)

    model = SetCNNRegressor(input_channels=X_train.shape[-1], dropout=dropout)
    model = _fit_torch_regressor(
        model=model,
        X_train=X_train,
        y_train=y[train_sub_idx],
        X_val=X_val,
        y_val=y[val_idx],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device_name=device_name,
        loss_name=loss_name,
        huber_beta=huber_beta,
    )
    y_pred = _predict_torch_regressor(model, X_test, device_name=device_name)
    metrics = evaluate_regression_predictions(
        y_true_log=y[test_idx],
        y_pred_log=y_pred,
        censored_mask=censored[test_idx],
    )
    predictions = pd.DataFrame(
        {
            "system_id": system_ids[test_idx],
            "y_true_log_tau": y[test_idx],
            "y_true_tau": np.exp(y[test_idx]),
            "system_censored": censored[test_idx],
            "set_cnn_pred_log_tau": y_pred,
            "set_cnn_pred_tau": np.exp(y_pred),
            "set_cnn_residual_log_tau": y_pred - y[test_idx],
        }
    )
    return {"metrics": {"set_cnn": metrics}, "predictions": predictions}


def save_label_plots(
    labels_df: pd.DataFrame,
    output_dir: str | Path,
    valid_pairs: Sequence[tuple[int, int]],
) -> None:
    output_dir = ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(labels_df["tau_system"], bins=40, color="#4472C4", alpha=0.85)
    ax.set_title("Tau System Distribution")
    ax.set_xlabel("tau_system")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_tau_system.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(labels_df["log_tau_system"], bins=40, color="#ED7D31", alpha=0.85)
    ax.set_title("log1p(Tau System) Distribution")
    ax.set_xlabel("log1p(tau_system)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_log_tau_system.png", dpi=160)
    plt.close(fig)

    lambda_column = "lambda_observable" if "lambda_observable" in labels_df.columns else "lambda_phase_space"
    finite_lambda = labels_df[lambda_column].to_numpy(dtype=np.float64)
    finite_lambda = finite_lambda[np.isfinite(finite_lambda)]
    if finite_lambda.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(finite_lambda, bins=40, color="#5B9BD5", alpha=0.85)
        ax.set_title("Observable Growth Rate Distribution")
        ax.set_xlabel(lambda_column)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / "hist_lambda_observable.png", dpi=160)
        plt.close(fig)

    relation = labels_df[["tau_system", lambda_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if not relation.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(relation["tau_system"], relation[lambda_column], s=16, alpha=0.55, color="#4472C4")
        ax.set_title("Divergence Time vs Observable Growth Rate")
        ax.set_xlabel("tau_system")
        ax.set_ylabel(lambda_column)
        fig.tight_layout()
        fig.savefig(output_dir / "scatter_tau_vs_lambda.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    fraction = float(labels_df["system_censored"].mean()) if len(labels_df) else math.nan
    ax.bar(["uncensored", "censored"], [1.0 - fraction, fraction], color=["#70AD47", "#C00000"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("System Censoring Fraction")
    ax.set_ylabel("fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "censored_fraction.png", dpi=160)
    plt.close(fig)

    melted_frames = []
    for idx_i, idx_j in valid_pairs:
        col = _make_pair_column(idx_i, idx_j, prefix="tau")
        if col in labels_df.columns:
            melted_frames.append(
                pd.DataFrame(
                    {
                        "pair": f"{idx_i}_{idx_j}",
                        "tau": labels_df[col].to_numpy(dtype=np.float64),
                    }
                )
            )
    if melted_frames:
        pair_df = pd.concat(melted_frames, axis=0, ignore_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        order = list(dict.fromkeys(pair_df["pair"].tolist()))
        data = [pair_df.loc[pair_df["pair"] == pair, "tau"].to_numpy() for pair in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title("Pair-Level Tau Distribution")
        ax.set_xlabel("pair")
        ax.set_ylabel("tau")
        fig.tight_layout()
        fig.savefig(output_dir / "pair_tau_distribution.png", dpi=160)
        plt.close(fig)


def save_prediction_plots(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    model_prefix: str = "rf",
) -> None:
    output_dir = ensure_dir(output_dir)
    true_log = predictions_df["y_true_log_tau"].to_numpy(dtype=np.float64)
    pred_log = predictions_df[f"{model_prefix}_pred_log_tau"].to_numpy(dtype=np.float64)
    true_tau = predictions_df["y_true_tau"].to_numpy(dtype=np.float64)
    pred_tau = predictions_df[f"{model_prefix}_pred_tau"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_log, pred_log, s=18, alpha=0.65, color="#5B9BD5")
    lo = float(min(true_log.min(), pred_log.min()))
    hi = float(max(true_log.max(), pred_log.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_title("True vs Predicted log_tau_system")
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_true_vs_pred_{model_prefix}_log_tau.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_tau, pred_tau, s=18, alpha=0.65, color="#A5A5A5")
    lo = float(min(true_tau.min(), pred_tau.min()))
    hi = float(max(true_tau.max(), pred_tau.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_title("True vs Predicted tau_system")
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_true_vs_pred_{model_prefix}_tau.png", dpi=160)
    plt.close(fig)

    residual = pred_log - true_log
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(true_log, residual, s=18, alpha=0.6, color="#ED7D31")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Residual vs True Log Time: {model_prefix}")
    ax.set_xlabel("true log time")
    ax.set_ylabel("predicted - true")
    fig.tight_layout()
    fig.savefig(output_dir / f"residual_vs_true_{model_prefix}.png", dpi=160)
    plt.close(fig)

    quantile_count = min(5, max(1, np.unique(true_log).size))
    try:
        bins = pd.qcut(true_log, q=quantile_count, duplicates="drop")
        error_df = pd.DataFrame({"bin": bins, "absolute_error": np.abs(residual)})
        grouped = error_df.groupby("bin", observed=True)["absolute_error"].mean()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(np.arange(len(grouped)), grouped.to_numpy(), color="#70AD47")
        ax.set_xticks(np.arange(len(grouped)), labels=[str(item) for item in grouped.index], rotation=30, ha="right")
        ax.set_title(f"Absolute Log Error by True-Time Quantile: {model_prefix}")
        ax.set_xlabel("true log-time quantile")
        ax.set_ylabel("mean absolute log error")
        fig.tight_layout()
        fig.savefig(output_dir / f"error_by_tau_quantile_{model_prefix}.png", dpi=160)
        plt.close(fig)
    except ValueError:
        pass

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(true_log.min(), pred_log.min()), max(true_log.max(), pred_log.max()), 35)
    ax.hist(true_log, bins=bins, alpha=0.55, density=True, label="true", color="#4472C4")
    ax.hist(pred_log, bins=bins, alpha=0.55, density=True, label="predicted", color="#ED7D31")
    ax.set_title(f"Prediction and Target Distributions: {model_prefix}")
    ax.set_xlabel("log time")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"prediction_distribution_{model_prefix}.png", dpi=160)
    plt.close(fig)


def save_feature_correlation_heatmap(
    feature_df: pd.DataFrame,
    output_dir: str | Path,
    target_column: str = "log_tau_system",
    top_k: int = 40,
) -> None:
    output_dir = ensure_dir(output_dir)
    feature_cols = [col for col in feature_df.columns if col not in {"system_id", target_column}]
    if not feature_cols:
        return
    # Compute feature-target correlations directly. Building the full
    # feature-by-feature matrix is quadratic in the number of features and is
    # prohibitively expensive for initial_prefix inputs.
    numeric = feature_df[feature_cols + [target_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = numeric[feature_cols].corrwith(numeric[target_column])
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).head(top_k).index)

    fig_h = max(4, 0.25 * len(corr))
    fig, ax = plt.subplots(figsize=(6, fig_h))
    arr = corr.to_numpy(dtype=np.float64)[:, None]
    im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(corr)))
    ax.set_yticklabels(corr.index.tolist(), fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels([target_column])
    ax.set_title("Top Feature Correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def print_pipeline_summary(
    labels_df: pd.DataFrame,
    label_meta: dict[str, Any],
    diagnostics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    group_size: int,
    threshold: float,
) -> None:
    tau_summary = summarize_series(labels_df["tau_system"].to_numpy(dtype=np.float64))
    lambda_column = "lambda_observable" if "lambda_observable" in labels_df.columns else "lambda_phase_space"
    lambda_summary = summarize_series(labels_df[lambda_column].to_numpy(dtype=np.float64))
    print("Task D divergence label construction complete.")
    print(f"Number of trajectories used: {len(labels_df) * group_size}")
    print(f"Number of systems: {len(labels_df)}")
    print(f"Group size: {group_size}")
    print(f"Valid pairs per system: {len(diagnostics['valid_pairs'])}")
    print(f"Threshold: {threshold}")
    print()
    print("Configured valid pair delta0 summary:")
    print(diagnostics["configured_valid_pairs_delta0"])
    print("Legacy 8-way pair delta0 summary:")
    print(diagnostics["legacy_pairs_8_delta0"])
    print()
    print("Tau_system summary:")
    print(f"  mean:   {tau_summary['mean']:.6g}")
    print(f"  median: {tau_summary['median']:.6g}")
    print(f"  min:    {tau_summary['min']:.6g}")
    print(f"  max:    {tau_summary['max']:.6g}")
    print(f"  censored fraction: {labels_df['system_censored'].mean():.4f}")
    print(f"{lambda_column} summary:")
    print(f"  finite count: {int(lambda_summary['count'])}")
    print(f"  median:       {lambda_summary['median']:.6g}")
    print(f"  min:          {lambda_summary['min']:.6g}")
    print(f"  max:          {lambda_summary['max']:.6g}")
    print(f"  dropped invalid groups: {label_meta['dropped_invalid_groups']}")
    print(f"  dropped nan labels: {label_meta['dropped_nan_labels']}")
    print()
    print("Model results:")
    for model_name, metrics in baseline_metrics.items():
        print(f"{model_name}:")
        print(f"  all: {metrics['all']}")
        print(f"  uncensored_only: {metrics['uncensored_only']}")
        print(f"  censored_only: {metrics['censored_only']}")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    if config.task_setting not in {"initial", "initial_prefix", "prefix"}:
        raise ValueError(f"Unsupported task_setting: {config.task_setting}")
    if config.target_mode not in {
        "lyapunov_proxy",
        "crossing_upper",
        "fitted_lyapunov_time",
        "min_instability",
        "lambda_fast",
    }:
        raise ValueError(f"Unsupported target_mode: {config.target_mode}")
    if config.label_observable not in {"smax", "aratio", "phase_space"}:
        raise ValueError(f"Unsupported label_observable: {config.label_observable}")
    if config.initial_representation not in {"orbital15", "full"}:
        raise ValueError(
            f"Unsupported initial_representation: {config.initial_representation}"
        )
    if config.task_setting == "initial_prefix" and config.label_observable not in {"smax", "aratio"}:
        raise ValueError("initial_prefix currently requires --label-observable smax or aratio")
    if config.lyapunov_aggregation not in {"median", "mean", "max"}:
        raise ValueError(f"Unsupported lyapunov_aggregation: {config.lyapunov_aggregation}")
    if not 0.0 < config.growth_lower < config.growth_upper:
        raise ValueError(
            f"Expected 0 < growth_lower < growth_upper, got {config.growth_lower} and {config.growth_upper}."
        )
    if config.threshold <= 1.0:
        raise ValueError(f"threshold must exceed 1 for normalized divergence, got {config.threshold}.")
    if config.min_growth_points < 2:
        raise ValueError(f"min_growth_points must be at least 2, got {config.min_growth_points}.")
    if not 0.0 <= config.lambda_r2_min <= 1.0:
        raise ValueError(f"lambda_r2_min must be in [0, 1], got {config.lambda_r2_min}.")
    if not 1 <= config.min_valid_lambda_pairs <= len(config.valid_pairs):
        raise ValueError(
            "min_valid_lambda_pairs must be between 1 and the configured pair count "
            f"({len(config.valid_pairs)}), got {config.min_valid_lambda_pairs}."
        )
    if config.max_pair_log_mad < 0.0:
        raise ValueError(f"max_pair_log_mad must be non-negative, got {config.max_pair_log_mad}.")
    crossing_proxy_mode = config.target_mode in {"lyapunov_proxy", "crossing_upper"}
    if crossing_proxy_mode and config.aggregation not in {"min", "median", "mean"}:
        raise ValueError(
            "lyapunov_proxy target requires aggregation to be one of min, median, or mean."
        )
    invalid_sensitivity_uppers = [
        value for value in config.growth_sensitivity_uppers if value <= config.growth_lower
    ]
    if invalid_sensitivity_uppers:
        raise ValueError(
            "All growth_sensitivity_uppers must exceed growth_lower; invalid values: "
            f"{invalid_sensitivity_uppers}."
        )
    output_dir = ensure_dir(config.output_dir)
    result_paths = discover_result_paths(config.dataset_dir, config.sampling_id)
    if config.max_shards is not None:
        result_paths = result_paths[: config.max_shards]

    completion_df = load_completion_table(Path(config.dataset_dir) / config.completion_table_name)
    diagnostics = diagnose_group_structure(
        result_paths=result_paths,
        group_size=config.group_size,
        valid_pairs=config.valid_pairs,
    )
    labels_path = output_dir / "task_d_labels.csv"
    labels_meta_path = output_dir / "task_d_labels_meta.json"
    label_start_index = (
        config.prefix_len * config.prefix_time_stride
        if config.task_setting == "initial_prefix"
        else 0
    )
    expected_label_config = {
        "label_version": LABEL_VERSION,
        "label_observable": config.label_observable,
        "lyapunov_aggregation": config.lyapunov_aggregation,
        "threshold": float(config.threshold),
        "group_size": int(config.group_size),
        "aggregation": config.aggregation,
        "growth_lower": float(config.growth_lower),
        "growth_upper": float(config.growth_upper),
        "growth_sensitivity_uppers": [float(value) for value in config.growth_sensitivity_uppers],
        "min_growth_points": int(config.min_growth_points),
        "lambda_r2_min": float(config.lambda_r2_min),
        "min_valid_lambda_pairs": int(config.min_valid_lambda_pairs),
        "max_pair_log_mad": float(config.max_pair_log_mad),
        "require_pair_consistency": bool(config.require_pair_consistency),
        "label_start_index": int(label_start_index),
    }
    use_cached_labels = False
    if labels_path.exists() and labels_meta_path.exists() and not config.force_recompute_labels:
        with open(labels_meta_path, "r", encoding="utf-8") as fh:
            cached_meta = json.load(fh)
        use_cached_labels = all(cached_meta.get(key) == value for key, value in expected_label_config.items())
        if not use_cached_labels:
            print("Cached labels use an older or different Lyapunov-label configuration; rebuilding ...")
    if use_cached_labels:
        print(f"Loading cached labels from {labels_path}")
        labels_df = pd.read_csv(labels_path)
        label_meta = cached_meta
    else:
        if config.force_recompute_labels and labels_path.exists():
            print("--force-recompute set, rebuilding labels ...")
        labels_df, label_meta = build_labels_from_result_shards(
            result_paths=result_paths,
            group_size=config.group_size,
            valid_pairs=config.valid_pairs,
            threshold=config.threshold,
            growth_lower=config.growth_lower,
            growth_upper=config.growth_upper,
            growth_sensitivity_uppers=config.growth_sensitivity_uppers,
            min_growth_points=config.min_growth_points,
            lambda_r2_min=config.lambda_r2_min,
            min_valid_lambda_pairs=config.min_valid_lambda_pairs,
            max_pair_log_mad=config.max_pair_log_mad,
            aggregation=config.aggregation,
            censoring_rule=config.censoring_rule,
            completion_df=completion_df,
            require_complete_valid_group=config.require_complete_valid_group,
            max_systems=config.max_systems,
            label_observable=config.label_observable,
            lyapunov_aggregation=config.lyapunov_aggregation,
            require_pair_consistency=config.require_pair_consistency,
            label_start_index=label_start_index,
        )
        labels_df.to_csv(labels_path, index=False)
        write_json(labels_meta_path, label_meta)
    plots_dir = ensure_dir(output_dir / "plots")
    save_label_plots(labels_df=labels_df, output_dir=plots_dir, valid_pairs=config.valid_pairs)

    full_tau_diagnostics = compute_tau_diagnostics(labels_df)
    setting_diagnostics: dict[str, Any] = dict(full_tau_diagnostics)
    feature_meta: dict[str, Any] | None = None
    sequence_meta: dict[str, Any] | None = None
    feature_importances = pd.DataFrame()
    feature_joined: pd.DataFrame | None = None
    all_metrics: dict[str, Any] = {}
    combined_predictions: pd.DataFrame
    split_payload: dict[str, Any]
    target_name: str
    if crossing_proxy_mode:
        # The first sampled threshold crossing is an upper bound on the true
        # continuous crossing time. A trajectory group that never crosses is
        # right-censored, not a valid finite upper-bound target.
        label_quality_mask = (
            ~labels_df["system_censored"].astype(bool).to_numpy()
            & np.isfinite(labels_df["tau_system"].to_numpy(dtype=np.float64))
            & (labels_df["tau_system"].to_numpy(dtype=np.float64) > 0.0)
        )
        target_filter_name = "uncensored_threshold_crossing"
    elif config.target_mode == "lambda_fast":
        label_quality_mask = (
            labels_df["lyapunov_label_valid"].astype(bool).to_numpy()
            & np.isfinite(labels_df["log_lambda_fast"].to_numpy(dtype=np.float64))
        )
        target_filter_name = "lambda_fast_label_quality"
    else:
        label_quality_mask = (
            labels_df["lyapunov_label_valid"].astype(bool).to_numpy()
            & np.isfinite(labels_df["log_lyapunov_time_label"].to_numpy(dtype=np.float64))
        )
        target_filter_name = "lyapunov_label_quality"
    working_labels = labels_df.loc[label_quality_mask].copy().reset_index(drop=True)
    if working_labels.empty:
        if crossing_proxy_mode:
            raise ValueError(
                "No systems with an observed threshold crossing survived; lower --threshold "
                "or use a denser trajectory output."
            )
        raise ValueError(
            "No systems passed the Lyapunov-label quality filters; relax --lambda-r2-min, "
            "--min-valid-lambda-pairs, or --max-pair-log-mad."
        )
    working_labels["tau_cross_system"] = working_labels["tau_system"]
    working_labels["crossing_time_censored"] = working_labels["system_censored"]
    if config.target_mode == "lambda_fast":
        working_labels["tau_system"] = working_labels["lambda_fast"]
        working_labels["log_tau_system"] = working_labels["log_lambda_fast"]
    else:
        working_labels["tau_system"] = working_labels["lyapunov_time_label"]
        working_labels["log_tau_system"] = working_labels["log_lyapunov_time_label"]
    working_labels["system_censored"] = False
    setting_diagnostics.update(
        {
            "target_mode": config.target_mode,
            "target_filter": target_filter_name,
            "num_labels_before_quality_filter": int(len(labels_df)),
            "num_labels_after_quality_filter": int(len(working_labels)),
            "num_labels_rejected_by_quality_filter": int(len(labels_df) - len(working_labels)),
            "censored_fraction_before_target_filter": float(labels_df["system_censored"].mean()),
            "lyapunov_target": (
                "log(first sampled time with R >= threshold)"
                if crossing_proxy_mode
                else (
                    f"log(max_pair_lambda_{config.label_observable})"
                    if config.target_mode == "lambda_fast"
                    else (
                        f"log(1 / max_pair_lambda_{config.label_observable})"
                        if config.target_mode == "min_instability"
                        else f"log(1 / lambda_{config.label_observable})"
                    )
                )
            ),
            "label_observable": config.label_observable,
            "lyapunov_aggregation": config.lyapunov_aggregation,
            "initial_representation": config.initial_representation,
            "require_pair_consistency": config.require_pair_consistency,
            "lyapunov_proxy_definition": "log(first sampled time with R >= threshold)",
        }
    )

    if crossing_proxy_mode:
        target_name = "log_lyapunov_proxy_time"
        working_labels["tau_system"] = working_labels["tau_cross_system"]
        working_labels["log_tau_system"] = np.log(working_labels["tau_system"].to_numpy(dtype=np.float64))
        # Censored systems were removed above, so the remaining regression
        # target is an ordinary finite point target.
        working_labels["system_censored"] = False
    if config.task_setting in {"initial", "initial_prefix"}:
        target_name = (
            "log_lyapunov_proxy_time"
            if crossing_proxy_mode
            else (
                "log_lambda_fast"
                if config.target_mode == "lambda_fast"
                else (
                "log_min_instability_time_from_aratio"
                if config.target_mode == "min_instability" and config.label_observable == "aratio"
                else (
                    "log_min_instability_time_from_phase_space"
                    if config.target_mode == "min_instability"
                    else (
                        "log_finite_time_lyapunov_time_proxy_from_aratio"
                        if config.label_observable == "aratio"
                        else "log_finite_time_lyapunov_time_proxy"
                    )
                )
                )
            )
        )
        if config.task_setting == "initial_prefix":
            X, _, feature_meta, feature_df = build_initial_prefix_features(
                result_paths=result_paths,
                labels_df=working_labels,
                prefix_len=config.prefix_len,
                time_stride=config.prefix_time_stride,
                group_size=config.group_size,
                feature_aggregation=config.feature_aggregation,
                valid_pairs=config.valid_pairs,
                growth_lower=config.growth_lower,
                growth_upper=config.growth_upper,
                min_growth_points=config.min_growth_points,
            )
        else:
            X, _, feature_meta, feature_df = build_initial_condition_features(
                result_paths=result_paths,
                labels_df=working_labels,
                group_size=config.group_size,
                feature_aggregation=config.feature_aggregation,
                feature_keys=(
                    ORBITAL15_FEATURE_KEYS
                    if config.initial_representation == "orbital15"
                    else INITIAL_FEATURE_KEYS
                ),
            )
        feature_joined = working_labels[["system_id", "log_tau_system"]].merge(
            feature_df.drop(columns=["log_tau_system"], errors="ignore"),
            on="system_id",
            how="left",
        )
        split = make_regression_split(
            labels_df=working_labels,
            test_size=config.test_size,
            random_state=config.random_state,
            exclude_censored=config.exclude_censored,
        )
        baseline = run_random_forest_baseline(
            X=X,
            labels_df=working_labels,
            feature_columns=feature_meta["feature_columns"],
            split=split,
        )
        all_metrics.update(baseline["metrics"])
        combined_predictions = baseline["predictions"].copy()
        feature_importances = baseline["feature_importances"]
        feature_importances.to_csv(output_dir / "feature_importance.csv", index=False)
        split_payload = baseline["split"]
        save_prediction_plots(baseline["predictions"], plots_dir, model_prefix="rf")
        save_prediction_plots(
            baseline["predictions"].rename(
                columns={
                    "constant_pred_log_tau": "constant_median_pred_log_tau",
                    "constant_pred_tau": "constant_median_pred_tau",
                }
            ),
            plots_dir,
            model_prefix="constant_median",
        )

        if config.run_mlp:
            mlp_result = run_mlp_baseline(
                X=X,
                labels_df=working_labels,
                split=split,
                epochs=config.mlp_epochs,
                batch_size=config.nn_batch_size,
                lr=config.nn_lr,
                weight_decay=config.nn_weight_decay,
                dropout=config.nn_dropout,
                val_fraction=config.nn_val_fraction,
                patience=config.nn_patience,
                loss_name=config.loss,
                huber_beta=config.huber_beta,
                device_name=config.device,
            )
            all_metrics.update(mlp_result["metrics"])
            combined_predictions = combined_predictions.merge(
                mlp_result["predictions"].drop(
                    columns=["y_true_log_tau", "y_true_tau", "system_censored"],
                    errors="ignore",
                ),
                on="system_id",
                how="left",
            )
            save_prediction_plots(mlp_result["predictions"], plots_dir, model_prefix="mlp")
        save_feature_correlation_heatmap(feature_df=feature_joined, output_dir=plots_dir)
    else:
        observation_times = compute_prefix_observation_times(
            result_paths=result_paths,
            labels_df=working_labels,
            prefix_len=config.prefix_len,
            time_stride=config.prefix_time_stride,
            group_size=config.group_size,
        )
        finite_obs = observation_times[np.isfinite(observation_times)]
        if finite_obs.size == 0:
            raise ValueError("No finite prefix observation times were found.")
        representative_t_obs = float(np.median(finite_obs))
        working_labels["t_obs"] = observation_times
        setting_diagnostics.update(
            {
                "t_obs": representative_t_obs,
                "t_obs_min": float(finite_obs.min()),
                "t_obs_max": float(finite_obs.max()),
                "prefix_target_overlap_note": (
                    "The finite-time pair-growth target is fitted from the stored trajectory, including "
                    "the prefix interval; use a future-only fit for a strict forecasting study."
                ),
            }
        )
        working_labels.to_csv(output_dir / "task_d_prefix_labels.csv", index=False)
        target_name = (
            "log_lyapunov_proxy_time"
            if crossing_proxy_mode
            else (
                "log_lambda_fast"
                if config.target_mode == "lambda_fast"
                else (
                "log_min_instability_time_from_aratio"
                if config.target_mode == "min_instability" and config.label_observable == "aratio"
                else (
                    "log_min_instability_time_from_phase_space"
                    if config.target_mode == "min_instability"
                    else (
                        "log_finite_time_lyapunov_time_proxy_from_aratio"
                        if config.label_observable == "aratio"
                        else "log_finite_time_lyapunov_time_proxy"
                    )
                )
                )
            )
        )

        prefix_arrays, sequence_meta = build_prefix_model_inputs(
            result_paths=result_paths,
            labels_df=working_labels,
            prefix_len=config.prefix_len,
            time_stride=config.prefix_time_stride,
            group_size=config.group_size,
            sequence_aggregation=config.sequence_aggregation,
        )
        split = make_regression_split(
            labels_df=working_labels,
            test_size=config.test_size,
            random_state=config.random_state,
            exclude_censored=config.exclude_censored,
        )
        split_payload = {
            "train_count": int(np.asarray(split["train_idx"]).size),
            "fit_train_count": int(np.asarray(split["fit_train_idx"]).size),
            "test_count": int(np.asarray(split["test_idx"]).size),
            "exclude_censored": bool(split["exclude_censored"]),
            "test_size": float(split["test_size"]),
            "random_state": int(split["random_state"]),
        }
        combined_predictions = pd.DataFrame(
            {
                "system_id": working_labels.iloc[np.asarray(split["test_idx"], dtype=np.int64)]["system_id"].to_numpy(),
            }
        )
        prefix_models: list[tuple[str, np.ndarray, bool]] = []
        if config.run_cnn_orbital:
            prefix_models.append(("cnn_orbital", prefix_arrays["cnn_orbital"], False))
        if config.run_cnn_phase_space:
            prefix_models.append(("cnn_phase_space", prefix_arrays["cnn_phase_space"], False))
        if config.run_set_cnn:
            prefix_models.append(("set_cnn", prefix_arrays["set_cnn"], True))
        if not prefix_models:
            raise ValueError("No prefix models are enabled.")

        for model_name, model_input, is_set_model in prefix_models:
            if is_set_model:
                result = run_set_cnn_baseline(
                    X_ensemble=model_input,
                    labels_df=working_labels,
                    split=split,
                    epochs=config.cnn_epochs,
                    batch_size=config.nn_batch_size,
                    lr=config.nn_lr,
                    weight_decay=config.nn_weight_decay,
                    dropout=config.nn_dropout,
                    val_fraction=config.nn_val_fraction,
                    patience=config.nn_patience,
                    loss_name=config.loss,
                    huber_beta=config.huber_beta,
                    device_name=config.device,
                )
            else:
                result = run_cnn_baseline(
                    X_seq=model_input,
                    labels_df=working_labels,
                    split=split,
                    epochs=config.cnn_epochs,
                    batch_size=config.nn_batch_size,
                    lr=config.nn_lr,
                    weight_decay=config.nn_weight_decay,
                    dropout=config.nn_dropout,
                    val_fraction=config.nn_val_fraction,
                    patience=config.nn_patience,
                    model_name=model_name,
                    loss_name=config.loss,
                    huber_beta=config.huber_beta,
                    device_name=config.device,
                )
            all_metrics.update(result["metrics"])
            combined_predictions = combined_predictions.merge(
                result["predictions"].drop(
                    columns=["y_true_log_tau", "y_true_tau", "system_censored"],
                    errors="ignore",
                ),
                on="system_id",
                how="left",
            )
            save_prediction_plots(result["predictions"], plots_dir, model_prefix=model_name)

        test_rows = working_labels.iloc[np.asarray(split["test_idx"], dtype=np.int64)]
        truth_columns = [
            "system_id",
            "t_obs",
            "tau_cross_system",
            "crossing_time_censored",
            "tau_system",
            "log_tau_system",
        ]
        combined_predictions = test_rows[truth_columns].rename(
            columns={"log_tau_system": "true_log_tau", "tau_system": "true_tau"}
        ).merge(
            combined_predictions,
            on="system_id",
            how="left",
        )

    if config.task_setting in {"initial", "initial_prefix"}:
        combined_predictions["true_log_tau"] = combined_predictions["y_true_log_tau"]
        combined_predictions["true_tau"] = combined_predictions["y_true_tau"]
    combined_predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "diagnostics.json", setting_diagnostics)

    model_summaries = []
    summary_train_idx, summary_val_idx = _split_train_val_indices(
        np.asarray(split["fit_train_idx"], dtype=np.int64),
        config.nn_val_fraction,
        config.random_state,
    )
    for model_name, payload in all_metrics.items():
        is_neural = model_name in {"mlp", "cnn_orbital", "cnn_phase_space", "set_cnn"}
        summary = {
            "task_setting": config.task_setting,
            "model_name": model_name,
            "target": target_name,
            "t_obs": setting_diagnostics.get("t_obs"),
            "num_train": int(summary_train_idx.size) if is_neural else int(split_payload["fit_train_count"]),
            "num_val": int(summary_val_idx.size) if is_neural else 0,
            "num_test": int(split_payload["test_count"]),
            "filtered_tau_le_t_obs": int(setting_diagnostics.get("num_filtered_due_to_tau_le_t_obs", 0)),
        }
        summary.update(payload["all"])
        model_summaries.append(summary)

    metrics = {
        "task_setting": config.task_setting,
        "target_mode": config.target_mode,
        "label_observable": config.label_observable,
        "target": target_name,
        "target_value_name": (
            "lambda_fast"
            if config.target_mode == "lambda_fast"
            else (
                "min_instability_time"
                if config.target_mode == "min_instability"
                else ("lyapunov_proxy_time" if crossing_proxy_mode else "time")
            )
        ),
        "t_obs": setting_diagnostics.get("t_obs"),
        "config": asdict(config),
        "diagnostics": diagnostics,
        "setting_diagnostics": setting_diagnostics,
        "label_metadata": label_meta,
        "feature_metadata": feature_meta,
        "sequence_metadata": sequence_meta,
        "model_metrics": all_metrics,
        "model_summaries": model_summaries,
        "split": split_payload,
        "tau_cross_system_summary": summarize_series(labels_df["tau_system"].to_numpy(dtype=np.float64)),
        "log_tau_cross_system_summary": summarize_series(labels_df["log_tau_system"].to_numpy(dtype=np.float64)),
        "lyapunov_proxy_time_summary": summarize_series(
            labels_df["lyapunov_proxy_time"].to_numpy(dtype=np.float64)
        ),
        "log_lyapunov_proxy_time_summary": summarize_series(
            labels_df["log_lyapunov_proxy_time"].to_numpy(dtype=np.float64)
        ),
        "lambda_observable_summary": summarize_series(
            labels_df["lambda_observable"].to_numpy(dtype=np.float64)
        ),
        "lambda_phase_space_summary": summarize_series(
            labels_df["lambda_phase_space"].to_numpy(dtype=np.float64)
        ),
        "lambda_smax_summary": summarize_series(
            labels_df["lambda_smax"].to_numpy(dtype=np.float64)
        ),
        "lambda_aratio_summary": summarize_series(
            labels_df["lambda_aratio"].to_numpy(dtype=np.float64)
        ),
        "lyapunov_time_proxy_summary": summarize_series(
            labels_df["lyapunov_time_proxy"].to_numpy(dtype=np.float64)
        ),
        "log_lyapunov_time_proxy_summary": summarize_series(
            labels_df["log_lyapunov_time_proxy"].to_numpy(dtype=np.float64)
        ),
        "target_time_summary": summarize_series(working_labels["tau_system"].to_numpy(dtype=np.float64)),
        "target_log_time_summary": summarize_series(working_labels["log_tau_system"].to_numpy(dtype=np.float64)),
        "lambda_tau_ratio_summary": summarize_series(labels_df["lambda_tau_ratio"].to_numpy(dtype=np.float64)),
        "system_censored_fraction": float(labels_df["system_censored"].mean()),
        "labels_csv": str(labels_path),
    }
    write_json(output_dir / "metrics.json", metrics)

    print_pipeline_summary(
        labels_df=working_labels,
        label_meta=label_meta,
        diagnostics=diagnostics,
        baseline_metrics=all_metrics,
        group_size=config.group_size,
        threshold=config.threshold,
    )
    return {
        "labels_df": labels_df,
        "feature_df": feature_joined,
        "metrics": metrics,
        "predictions": combined_predictions,
        "feature_importances": feature_importances,
    }
