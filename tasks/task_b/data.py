"""Trajectory prefix loading, feature construction, and case grouping for Task B/C."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from tasks.common.orbital_schema import ANGLE_RESULT_KEYS


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
RUNS_PER_BATCH = 8000
GROUP_SIZE = 32


def batch_id_from_path(path: str | Path) -> int:
    match = RESULT_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse result batch id from {path}")
    return int(match.group(1))


def discover_result_paths(dataset_dir: str | Path, sampling_id: int, max_shards: int | None = None) -> list[Path]:
    paths = sorted(Path(dataset_dir).glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No result shards found in {dataset_dir} for sampling {sampling_id}.")
    return paths


def sincos(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


def pairwise_geom(pos: np.ndarray, vel: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pairs = ((0, 1), (0, 2), (1, 2))
    channels: list[np.ndarray] = []
    for i, j in pairs:
        dr = pos[:, :, i, :] - pos[:, :, j, :]
        dv = vel[:, :, i, :] - vel[:, :, j, :]
        dist = np.linalg.norm(dr, axis=-1, keepdims=True)
        speed = np.linalg.norm(dv, axis=-1, keepdims=True)
        radial = (dr * dv).sum(axis=-1, keepdims=True) / np.maximum(dist, eps)
        channels.extend([dist, speed, radial])
    return np.concatenate(channels, axis=-1).astype(np.float32)


def load_result_prefixes(
    result_paths: list[Path],
    result_keys: tuple[str, ...],
    prefix_len: int,
    time_stride: int,
    rows_per_shard: int | None,
    include_geom: bool,
    add_time: bool,
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    data_parts: dict[str, list[np.ndarray]] = {key: [] for key in result_keys}
    if include_geom:
        data_parts["positions_complete"] = []
        data_parts["velocities_complete"] = []
    if add_time:
        data_parts["times_complete"] = []
    batch_ids: list[int] = []
    rows_loaded: list[int] = []
    time_sel = slice(0, prefix_len, time_stride)

    for path in result_paths:
        batch_id = batch_id_from_path(path)
        with h5py.File(path, "r") as handle:
            n = handle["times_complete"].shape[0]
            if rows_per_shard is not None:
                n = min(n, rows_per_shard)
            row_sel = slice(0, n)
            for key in data_parts:
                if key == "times_complete":
                    arr = np.asarray(handle[key][row_sel, time_sel], dtype=np.float32)
                else:
                    arr = np.asarray(handle[key][row_sel, time_sel, ...], dtype=np.float32)
                data_parts[key].append(arr)
        batch_ids.append(batch_id)
        rows_loaded.append(n)

    data = {key: np.concatenate(parts, axis=0) for key, parts in data_parts.items()}
    return data, batch_ids, rows_loaded


def load_aligned_labels(
    labels_path: str | Path,
    batch_ids: list[int],
    rows_loaded: list[int],
    runs_per_batch: int = RUNS_PER_BATCH,
) -> pd.DataFrame:
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    labels = labels.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float32",
            "wall_time": "float32",
            "diagnosis": "int64",
        }
    )
    # The completion table already records the output batch.  Derive the row
    # order within each batch from the global index so shards with a different
    # row count (for example sampling_3: 4000 instead of 8000) align correctly.
    labels["batch_id"] = labels["num_batch"]
    labels = labels.sort_values(["batch_id", "index"]).reset_index(drop=True)
    labels["row_in_batch"] = labels.groupby("batch_id", sort=False).cumcount()
    frames = []
    for batch_id, row_count in zip(batch_ids, rows_loaded):
        frame = labels[(labels["batch_id"] == batch_id) & (labels["row_in_batch"] < row_count)].copy()
        frame = frame.sort_values("row_in_batch")
        if len(frame) != row_count:
            raise ValueError(f"Expected {row_count} labels for batch {batch_id}, got {len(frame)}.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_prefix_features(
    data: dict[str, np.ndarray],
    result_keys: tuple[str, ...],
    add_time: bool = True,
    include_geom: bool = True,
) -> tuple[np.ndarray, list[str]]:
    channels: list[np.ndarray] = []
    names: list[str] = []
    for key in result_keys:
        arr = data[key]
        if key in ANGLE_RESULT_KEYS:
            arr = sincos(arr)
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1).astype(np.float32)
        channels.append(flat)
        names.extend([f"{key}_{i}" for i in range(flat.shape[-1])])
    if include_geom:
        geom = pairwise_geom(data["positions_complete"], data["velocities_complete"])
        channels.append(geom)
        names.extend([f"geom_{i}" for i in range(geom.shape[-1])])
    if add_time:
        time = data["times_complete"].astype(np.float32)
        denom = max(float(np.nanmax(time)), 1.0)
        time = (time / denom)[..., None]
        channels.append(time)
        names.append("time_norm")
    X = np.concatenate(channels, axis=-1).astype(np.float32)
    return X, names


def filter_valid_samples(
    X: np.ndarray,
    labels: pd.DataFrame,
    data: dict[str, np.ndarray],
    time_horizon: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    valid_label = (labels["time_unbound"] >= 0.0) & (labels["diagnosis"] == 0)
    finite_prefix = np.isfinite(X).all(axis=(1, 2))
    mask = valid_label.to_numpy() & finite_prefix
    X = X[mask]
    labels = labels.loc[mask].reset_index(drop=True)
    if time_horizon is None:
        time_horizon = float(labels["time_unbound"].max())
    y = (labels["time_unbound"].to_numpy(dtype=np.float32) >= float(time_horizon)).astype(np.float32)
    return X, y, labels, mask


def build_group_fraction_samples(
    X: np.ndarray,
    y: np.ndarray,
    labels: pd.DataFrame,
    group_size: int = GROUP_SIZE,
    aggregation: str = "concat",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels = labels.copy()
    labels["raw_case_id"] = labels["index"] // group_size
    labels["variation_id"] = labels["index"] % group_size
    labels["_row"] = np.arange(len(labels), dtype=np.int64)
    group_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for case_id, frame in labels.groupby("raw_case_id", sort=True):
        if len(frame) != group_size:
            continue
        frame = frame.sort_values("variation_id")
        if not np.array_equal(frame["variation_id"].to_numpy(dtype=np.int64), np.arange(group_size)):
            continue
        rows = frame["_row"].to_numpy(dtype=np.int64)
        if not np.isfinite(X[rows]).all():
            continue
        group_rows.append(rows)
        stable = y[rows].astype(np.float32)
        meta_rows.append(
            {
                "raw_case_id": int(case_id),
                "stable_count": int(stable.sum()),
                "stable_fraction": float(stable.mean()),
                "valid_variations": int(group_size),
                "min_time_unbound": float(frame["time_unbound"].min()),
                "median_time_unbound": float(frame["time_unbound"].median()),
                "max_time_unbound": float(frame["time_unbound"].max()),
            }
        )
    if not group_rows:
        raise ValueError("No complete finite trajectory groups were found for group_fraction mode.")
    rows_arr = np.stack(group_rows, axis=0)
    X_group = X[rows_arr]  # [G, group, T, C]
    X_group = X_group.transpose(0, 2, 1, 3)  # [G, T, group, C]
    if aggregation == "concat":
        X_group = X_group.reshape(X_group.shape[0], X_group.shape[1], -1)
    elif aggregation == "set_attention":
        pass
    else:
        raise ValueError(f"Unsupported group aggregation: {aggregation}")
    y_group = np.asarray([row["stable_fraction"] for row in meta_rows], dtype=np.float32)
    return X_group.astype(np.float32), y_group, pd.DataFrame(meta_rows)


def grouped_case_split_indices(
    y: np.ndarray,
    sample_case_ids: np.ndarray,
    test_size: int | float = 0.2,
    val_size: int | float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1).astype(np.float32)
    sample_case_ids = np.asarray(sample_case_ids).reshape(-1).astype(np.int64)
    if y.size != sample_case_ids.size:
        raise ValueError(f"y and sample_case_ids must have the same length, got {y.size} and {sample_case_ids.size}.")

    unique_case_ids, inverse = np.unique(sample_case_ids, return_inverse=True)
    n_cases = unique_case_ids.size
    if n_cases < 3:
        raise ValueError(f"Need at least 3 cases for train/val/test split, got {n_cases}.")

    def resolve_size(size: int | float, base: int) -> int:
        if isinstance(size, float) and 0.0 < size < 1.0:
            return int(round(base * size))
        return int(size)

    case_scores = np.zeros(n_cases, dtype=np.float32)
    for case_idx in range(n_cases):
        rows = inverse == case_idx
        case_scores[case_idx] = float(y[rows].mean())

    # Coarse buckets keep split balance while avoiding sparse 33-class stratification on stable_count.
    case_bins = np.zeros(n_cases, dtype=np.int64)
    case_bins[(case_scores > 0.0) & (case_scores < 0.5)] = 1
    case_bins[(case_scores >= 0.5) & (case_scores < 1.0)] = 2
    case_bins[case_scores >= 1.0] = 3

    rng = np.random.default_rng(seed)
    all_case_idx = np.arange(n_cases)

    def sample_stratified_cases(pool: np.ndarray, target_size: int) -> np.ndarray:
        target_size = max(0, min(int(target_size), pool.size - 1))
        if target_size == 0:
            return np.array([], dtype=np.int64)
        classes = np.unique(case_bins[pool])
        selected: list[np.ndarray] = []
        remaining = target_size
        for cls in classes[:-1]:
            cls_idx = pool[case_bins[pool] == cls]
            take = int(round(target_size * cls_idx.size / pool.size))
            take = min(take, cls_idx.size)
            if take > 0:
                selected.append(rng.choice(cls_idx, size=take, replace=False))
                remaining -= take
        cls_idx = pool[case_bins[pool] == classes[-1]]
        take = min(remaining, cls_idx.size)
        if take > 0:
            selected.append(rng.choice(cls_idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
        if out.size < target_size:
            extras = np.setdiff1d(pool, out, assume_unique=False)
            out = np.concatenate([out, rng.choice(extras, size=target_size - out.size, replace=False)])
        rng.shuffle(out)
        return out.astype(np.int64)

    test_case_count = resolve_size(test_size, n_cases)
    test_case_idx = sample_stratified_cases(all_case_idx, test_case_count)
    train_val_case_idx = np.setdiff1d(all_case_idx, test_case_idx, assume_unique=False)
    val_case_count = resolve_size(val_size, train_val_case_idx.size)
    val_case_idx = sample_stratified_cases(train_val_case_idx, val_case_count)
    train_case_idx = np.setdiff1d(train_val_case_idx, val_case_idx, assume_unique=False)
    rng.shuffle(train_case_idx)

    def expand(case_idx: np.ndarray) -> np.ndarray:
        mask = np.isin(inverse, case_idx)
        rows = np.flatnonzero(mask).astype(np.int64)
        rng.shuffle(rows)
        return rows

    return expand(train_case_idx), expand(val_case_idx), expand(test_case_idx)


def standardize_sequence_train(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((X_train - mu) / std).astype(np.float32),
        [((arr - mu) / std).astype(np.float32) for arr in arrays],
        {"mean": mu.astype(np.float32), "std": std.astype(np.float32)},
    )


