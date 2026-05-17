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
DEFAULT_GROUP_SIZE = 32
DEFAULT_VALID_PAIRS = tuple((i, i + 16) for i in range(16))
LEGACY_VALID_PAIRS_8 = ((0, 4), (1, 5), (2, 6), (3, 7))
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
PREFIX_FEATURE_KEYS = (
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "aratios_complete",
    "energy_complete",
    "angMomentum_complete",
)
PREFIX_EXTRA_STATE_KEYS = ("positions_complete", "velocities_complete")
INVALID_PAIR_CHECKS = ((0, 1), (0, 2), (0, 4))


@dataclass
class PipelineConfig:
    dataset_dir: str | Path = "./dataset_v3"
    sampling_id: int = 2
    group_size: int = DEFAULT_GROUP_SIZE
    valid_pairs: tuple[tuple[int, int], ...] = DEFAULT_VALID_PAIRS
    threshold: float = 1e3
    aggregation: str = "median"
    censoring_rule: str = "median_pair"
    feature_aggregation: str = "first"
    exclude_censored: bool = False
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str | Path = "./outputs/task_c_lyapunov"
    require_complete_valid_group: bool = True
    completion_table_name: str = "completed_systems_sampling_2.txt"
    max_shards: int | None = None
    max_systems: int | None = None
    include_prefix_features: bool = False
    prefix_len: int = 200
    include_positions_velocities: bool = False
    sequence_aggregation: str = "mean_std"
    run_mlp: bool = True
    run_cnn: bool = True
    mlp_epochs: int = 80
    cnn_epochs: int = 80
    nn_batch_size: int = 128
    nn_lr: float = 1e-3
    nn_weight_decay: float = 1e-4
    nn_dropout: float = 0.1
    nn_val_fraction: float = 0.1
    nn_patience: int = 12
    device: str | None = None


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
        First time when normalized divergence exceeds threshold.
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
    crossing_idx = np.flatnonzero(finite_mask & (R_a > threshold))
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
    smaxes_group: np.ndarray,
    times_group: np.ndarray,
    system_id: int,
    valid_pairs: Sequence[tuple[int, int]],
    threshold: float,
    aggregation: str,
    censoring_rule: str,
    eps: float = 1e-12,
    inner_index: int = 0,
) -> dict[str, Any]:
    pair_taus: list[float] = []
    pair_censored: list[bool] = []
    pair_delta0s: list[float] = []
    for idx_i, idx_j in valid_pairs:
        tau, censored, _, delta_a = compute_pair_divergence_time(
            smaxes=smaxes_group,
            times=times_group,
            idx_i=idx_i,
            idx_j=idx_j,
            threshold=threshold,
            eps=eps,
            inner_index=inner_index,
        )
        delta0_raw = float(delta_a[0]) if delta_a.size else math.nan
        pair_taus.append(tau)
        pair_censored.append(censored)
        pair_delta0s.append(delta0_raw)

    tau_system = _aggregate_tau(pair_taus, aggregation=aggregation)
    system_censored = _aggregate_censoring_flag(
        pair_taus=pair_taus,
        pair_censored=pair_censored,
        tau_system=tau_system,
        censoring_rule=censoring_rule,
    )
    return {
        "system_id": system_id,
        "tau_system": tau_system,
        "log_tau_system": float(np.log1p(tau_system)) if np.isfinite(tau_system) else math.nan,
        "system_censored": bool(system_censored),
        "pair_taus": pair_taus,
        "pair_censored": pair_censored,
        "pair_delta0s": pair_delta0s,
    }


def compute_system_divergence_label(
    smaxes: np.ndarray,
    times: np.ndarray,
    system_id: int,
    group_size: int = 8,
    valid_pairs: Sequence[tuple[int, int]] = LEGACY_VALID_PAIRS_8,
    threshold: float = 1e3,
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
) -> dict[str, Any]:
    """
    Compute system-level divergence label from the valid near-identical pairs.
    """
    validate_array_shapes(smaxes=smaxes, times=times, group_size=group_size)
    start = system_id * group_size
    stop = start + group_size
    if stop > smaxes.shape[0]:
        raise IndexError(f"system_id {system_id} exceeds available rows")
    group_smaxes = smaxes[start:stop]
    group_times = times[start:stop]
    return _compute_group_divergence_label(
        smaxes_group=group_smaxes,
        times_group=group_times,
        system_id=system_id,
        valid_pairs=valid_pairs,
        threshold=threshold,
        aggregation=aggregation,
        censoring_rule=censoring_rule,
    )


def build_task_c_labels(
    smaxes: np.ndarray,
    times: np.ndarray,
    group_size: int = 8,
    threshold: float = 1e3,
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
    valid_pairs: Sequence[tuple[int, int]] = LEGACY_VALID_PAIRS_8,
) -> pd.DataFrame:
    """
    Build labels for all systems from in-memory arrays.

    Returns a DataFrame with one row per system.
    """
    validate_array_shapes(smaxes=smaxes, times=times, group_size=group_size)
    rows: list[dict[str, Any]] = []
    n_systems = smaxes.shape[0] // group_size
    for system_id in range(n_systems):
        label = compute_system_divergence_label(
            smaxes=smaxes,
            times=times,
            system_id=system_id,
            group_size=group_size,
            valid_pairs=valid_pairs,
            threshold=threshold,
            aggregation=aggregation,
            censoring_rule=censoring_rule,
        )
        row = {
            "system_id": label["system_id"],
            "tau_system": label["tau_system"],
            "log_tau_system": label["log_tau_system"],
            "system_censored": label["system_censored"],
        }
        for (idx_i, idx_j), tau in zip(valid_pairs, label["pair_taus"], strict=True):
            row[_make_pair_column(idx_i, idx_j, prefix="tau")] = tau
        for (idx_i, idx_j), flag in zip(valid_pairs, label["pair_censored"], strict=True):
            row[_make_pair_column(idx_i, idx_j, prefix="censored")] = bool(flag)
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
    aggregation: str = "median",
    censoring_rule: str = "median_pair",
    completion_df: pd.DataFrame | None = None,
    require_complete_valid_group: bool = True,
    max_systems: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped_invalid_groups = 0
    dropped_nan_labels = 0
    global_row_offset = 0
    system_counter = 0
    valid_pair_delta0s: list[float] = []

    for result_path in result_paths:
        batch_id = extract_batch_id(result_path)
        with h5py.File(result_path, "r") as f:
            inner_smaxes = np.asarray(f["smaxes_complete"][:, :, 0], dtype=np.float64)
            times = np.asarray(f["times_complete"][:, :], dtype=np.float64)
            row_count = inner_smaxes.shape[0]
            time_steps = inner_smaxes.shape[1]
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
        for local_group_id in range(group_count):
            row_start = local_group_id * group_size
            row_stop = row_start + group_size
            raw_system_id = (global_row_offset + row_start) // group_size
            if batch_valid is not None and require_complete_valid_group:
                if not bool(np.all(batch_valid[row_start:row_stop])):
                    dropped_invalid_groups += 1
                    continue

            group_smaxes = inner_smaxes[row_start:row_stop]
            group_times = times[row_start:row_stop]
            group_smaxes_full = group_smaxes[:, :, None]
            label = _compute_group_divergence_label(
                smaxes_group=group_smaxes_full,
                times_group=group_times,
                system_id=raw_system_id,
                valid_pairs=valid_pairs,
                threshold=threshold,
                aggregation=aggregation,
                censoring_rule=censoring_rule,
                inner_index=0,
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
                "tau_system": label["tau_system"],
                "log_tau_system": label["log_tau_system"],
                "system_censored": label["system_censored"],
            }
            for (idx_i, idx_j), tau, flag, delta0 in zip(
                valid_pairs,
                label["pair_taus"],
                label["pair_censored"],
                label["pair_delta0s"],
                strict=True,
            ):
                row[_make_pair_column(idx_i, idx_j, prefix="tau")] = tau
                row[_make_pair_column(idx_i, idx_j, prefix="censored")] = bool(flag)
                row[_make_pair_column(idx_i, idx_j, prefix="delta0")] = delta0
                valid_pair_delta0s.append(delta0)
            rows.append(row)
            system_counter += 1
            if max_systems is not None and system_counter >= max_systems:
                break
        global_row_offset += row_count
        if max_systems is not None and system_counter >= max_systems:
            break

    labels_df = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    metadata = {
        "num_systems": int(len(labels_df)),
        "dropped_invalid_groups": int(dropped_invalid_groups),
        "dropped_nan_labels": int(dropped_nan_labels),
        "pair_delta0_summary": summarize_series(valid_pair_delta0s),
        "valid_pairs": [list(pair) for pair in valid_pairs],
        "group_size": int(group_size),
        "threshold": float(threshold),
        "aggregation": aggregation,
        "censoring_rule": censoring_rule,
    }
    return labels_df, metadata


def build_initial_condition_features(
    result_paths: Sequence[str | Path],
    labels_df: pd.DataFrame,
    group_size: int,
    feature_aggregation: str = "first",
    feature_keys: Sequence[str] = INITIAL_FEATURE_KEYS,
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
        "num_features": int(len(feature_columns)),
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


def _subset_regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    if y_true_log.size == 0:
        return {
            "count": 0.0,
            "mae_log_tau": math.nan,
            "rmse_log_tau": math.nan,
            "mae_tau": math.nan,
            "spearman": math.nan,
        }
    y_true_tau = np.expm1(y_true_log)
    y_pred_tau = np.expm1(y_pred_log)
    return {
        "count": float(y_true_log.size),
        "mae_log_tau": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log_tau": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_tau": float(mean_absolute_error(y_true_tau, y_pred_tau)),
        "spearman": _compute_spearman(y_true_log, y_pred_log),
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
    X_seq = np.asarray(X_seq, dtype=np.float64)
    train_flat = X_seq[fit_train_idx].reshape(-1, X_seq.shape[-1])
    channel_median = np.nanmedian(train_flat, axis=0)
    channel_median = np.where(np.isfinite(channel_median), channel_median, 0.0)

    def _impute(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64).copy()
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(channel_median, np.where(bad)[-1])
        return out

    X_train = _impute(X_seq[fit_train_idx])
    X_test = _impute(X_seq[test_idx])

    train_flat = X_train.reshape(-1, X_train.shape[-1])
    channel_mean = train_flat.mean(axis=0)
    channel_std = train_flat.std(axis=0)
    channel_std = np.where(channel_std > 1e-8, channel_std, 1.0)
    X_train = (X_train - channel_mean[None, None, :]) / channel_std[None, None, :]
    X_test = (X_test - channel_mean[None, None, :]) / channel_std[None, None, :]
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
) -> nn.Module:
    device = torch.device(device_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

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
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())
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
) -> np.ndarray:
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device))
    return pred.detach().cpu().numpy().astype(np.float64)


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
            "y_true_tau": np.expm1(y_test),
            "system_censored": censored_test,
            "constant_pred_log_tau": y_pred_constant,
            "constant_pred_tau": np.expm1(y_pred_constant),
            "rf_pred_log_tau": y_pred_rf,
            "rf_pred_tau": np.expm1(y_pred_rf),
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
    device_name: str | None = None,
) -> dict[str, Any]:
    device_name = _resolve_device(device_name)
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
            "y_true_tau": np.expm1(y[test_idx]),
            "system_censored": censored[test_idx],
            "mlp_pred_log_tau": y_pred,
            "mlp_pred_tau": np.expm1(y_pred),
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
    device_name: str | None = None,
) -> dict[str, Any]:
    device_name = _resolve_device(device_name)
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
            "y_true_tau": np.expm1(y[test_idx]),
            "system_censored": censored[test_idx],
            "cnn_pred_log_tau": y_pred,
            "cnn_pred_tau": np.expm1(y_pred),
        }
    )
    return {"metrics": {"cnn": metrics}, "predictions": predictions}


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
        ax.boxplot(data, labels=order, showfliers=False)
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
    corr = feature_df[feature_cols + [target_column]].corr(numeric_only=True)[target_column].drop(target_column)
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
    print("Lyapunov divergence label construction complete.")
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
    print(f"  dropped invalid groups: {label_meta['dropped_invalid_groups']}")
    print(f"  dropped nan labels: {label_meta['dropped_nan_labels']}")
    print()
    print("Baseline results:")
    for model_name, metrics in baseline_metrics.items():
        print(f"{model_name}:")
        print(f"  all: {metrics['all']}")
        print(f"  uncensored_only: {metrics['uncensored_only']}")
        print(f"  censored_only: {metrics['censored_only']}")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
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
    labels_df, label_meta = build_labels_from_result_shards(
        result_paths=result_paths,
        group_size=config.group_size,
        valid_pairs=config.valid_pairs,
        threshold=config.threshold,
        aggregation=config.aggregation,
        censoring_rule=config.censoring_rule,
        completion_df=completion_df,
        require_complete_valid_group=config.require_complete_valid_group,
        max_systems=config.max_systems,
    )
    labels_path = output_dir / "task_c_labels.csv"
    labels_df.to_csv(labels_path, index=False)

    X, y, feature_meta, feature_df = build_initial_condition_features(
        result_paths=result_paths,
        labels_df=labels_df,
        group_size=config.group_size,
        feature_aggregation=config.feature_aggregation,
    )
    feature_joined = labels_df[["system_id", "log_tau_system"]].merge(
        feature_df.drop(columns=["log_tau_system"], errors="ignore"),
        on="system_id",
        how="left",
    )

    split = make_regression_split(
        labels_df=labels_df,
        test_size=config.test_size,
        random_state=config.random_state,
        exclude_censored=config.exclude_censored,
    )

    baseline = run_random_forest_baseline(
        X=X,
        labels_df=labels_df,
        feature_columns=feature_meta["feature_columns"],
        split=split,
    )
    all_metrics = dict(baseline["metrics"])
    combined_predictions = baseline["predictions"].copy()
    baseline["feature_importances"].to_csv(output_dir / "feature_importances.csv", index=False)

    save_label_plots(labels_df=labels_df, output_dir=output_dir, valid_pairs=config.valid_pairs)
    save_prediction_plots(
        predictions_df=baseline["predictions"],
        output_dir=output_dir,
        model_prefix="rf",
    )

    mlp_result = None
    if config.run_mlp:
        mlp_result = run_mlp_baseline(
            X=X,
            labels_df=labels_df,
            split=split,
            epochs=config.mlp_epochs,
            batch_size=config.nn_batch_size,
            lr=config.nn_lr,
            weight_decay=config.nn_weight_decay,
            dropout=config.nn_dropout,
            val_fraction=config.nn_val_fraction,
            patience=config.nn_patience,
            device_name=config.device,
        )
        all_metrics.update(mlp_result["metrics"])
        combined_predictions = combined_predictions.merge(
            mlp_result["predictions"][["system_id", "mlp_pred_log_tau", "mlp_pred_tau"]],
            on="system_id",
            how="left",
        )
        save_prediction_plots(
            predictions_df=mlp_result["predictions"],
            output_dir=output_dir,
            model_prefix="mlp",
        )

    cnn_result = None
    sequence_meta = None
    if config.run_cnn:
        X_seq, _, sequence_meta = build_prefix_sequence_features(
            result_paths=result_paths,
            labels_df=labels_df,
            prefix_len=config.prefix_len,
            group_size=config.group_size,
            include_positions_velocities=config.include_positions_velocities,
            sequence_aggregation=config.sequence_aggregation,
        )
        cnn_result = run_cnn_baseline(
            X_seq=X_seq,
            labels_df=labels_df,
            split=split,
            epochs=config.cnn_epochs,
            batch_size=config.nn_batch_size,
            lr=config.nn_lr,
            weight_decay=config.nn_weight_decay,
            dropout=config.nn_dropout,
            val_fraction=config.nn_val_fraction,
            patience=config.nn_patience,
            device_name=config.device,
        )
        all_metrics.update(cnn_result["metrics"])
        combined_predictions = combined_predictions.merge(
            cnn_result["predictions"][["system_id", "cnn_pred_log_tau", "cnn_pred_tau"]],
            on="system_id",
            how="left",
        )
        save_prediction_plots(
            predictions_df=cnn_result["predictions"],
            output_dir=output_dir,
            model_prefix="cnn",
        )

    combined_predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_feature_correlation_heatmap(feature_df=feature_joined, output_dir=output_dir)

    metrics = {
        "config": asdict(config),
        "diagnostics": diagnostics,
        "label_metadata": label_meta,
        "feature_metadata": feature_meta,
        "sequence_metadata": sequence_meta,
        "baseline_metrics": all_metrics,
        "split": baseline["split"],
        "tau_system_summary": summarize_series(labels_df["tau_system"].to_numpy(dtype=np.float64)),
        "log_tau_system_summary": summarize_series(labels_df["log_tau_system"].to_numpy(dtype=np.float64)),
        "system_censored_fraction": float(labels_df["system_censored"].mean()),
        "labels_csv": str(labels_path),
    }
    write_json(output_dir / "metrics.json", metrics)

    print_pipeline_summary(
        labels_df=labels_df,
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
        "feature_importances": baseline["feature_importances"],
        "mlp_predictions": None if mlp_result is None else mlp_result["predictions"],
        "cnn_predictions": None if cnn_result is None else cnn_result["predictions"],
    }
