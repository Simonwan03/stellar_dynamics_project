from __future__ import annotations

import glob
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ORBITAL_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
LOG_FEATURES = {"aratios", "mratios"}
ORBITAL15_ANGLES = {"APs", "LANs"}
TRAJECTORY_ALIASES = {
    "positions": ("positions_complete", "positions"),
    "velocities": ("velocities_complete", "velocities"),
}


@dataclass(frozen=True)
class SequenceSource:
    path: Path
    row_count: int
    time_steps: int
    resolved_keys: dict[str, str]
    step_shapes: dict[str, tuple[int, ...]]


@dataclass
class PreparedForecastData:
    train_dataset: "ForecastWindowDataset"
    val_dataset: "ForecastWindowDataset"
    test_dataset: "ForecastWindowDataset"
    scaler: "StandardScaler"
    feature_metadata: dict[str, Any]
    split_indices: dict[str, np.ndarray]
    source_paths: list[str]
    trajectory_count: int
    time_steps: int


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError(f"Scaler mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}")
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((np.asarray(array, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return (np.asarray(array, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path, feature_names: list[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            feature_names=np.asarray(feature_names, dtype=object),
        )


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[SequenceSource],
        source_ids: np.ndarray,
        local_ids: np.ndarray,
        trajectory_indices: np.ndarray,
        representation: str,
        feature_metadata: dict[str, Any],
        scaler: StandardScaler,
        context_len: int,
        pred_len: int,
        stride: int,
        max_windows_per_trajectory: int,
        window_sampling: str,
        seed: int,
        drop_nan_windows: bool,
    ) -> None:
        self.sources = sources
        self.source_ids = source_ids
        self.local_ids = local_ids
        self.trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
        self.representation = representation
        self.feature_metadata = feature_metadata
        self.scaler = scaler
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.total_len = self.context_len + self.pred_len
        self.stride = int(stride)
        self.drop_nan_windows = bool(drop_nan_windows)
        self._handles: dict[int, h5py.File] = {}

        all_starts = np.arange(
            0,
            sources[0].time_steps - self.total_len + 1,
            self.stride,
            dtype=np.int64,
        )
        if all_starts.size == 0:
            raise ValueError(
                f"No windows fit T={sources[0].time_steps}, context={context_len}, pred={pred_len}."
            )
        if max_windows_per_trajectory > 0 and all_starts.size > max_windows_per_trajectory:
            if window_sampling == "sequential":
                all_starts = all_starts[:max_windows_per_trajectory]
            else:
                rng = np.random.default_rng(seed)
                all_starts = np.sort(
                    rng.choice(all_starts, size=max_windows_per_trajectory, replace=False)
                )
        self.starts = all_starts

    def __len__(self) -> int:
        return int(self.trajectory_indices.size * self.starts.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory_offset = index // self.starts.size
        start_offset = index % self.starts.size
        global_idx = int(self.trajectory_indices[trajectory_offset])

        for probe in range(self.starts.size):
            start = int(self.starts[(start_offset + probe) % self.starts.size])
            raw = self._load_window(global_idx, start)
            finite = np.isfinite(raw).all()
            if finite or not self.drop_nan_windows:
                if not finite:
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                normalized = self.scaler.transform(raw)
                _assert_window(normalized, self.context_len, self.pred_len)
                return {
                    "context": torch.from_numpy(normalized[: self.context_len]),
                    "target": torch.from_numpy(normalized[self.context_len :]),
                    "trajectory_index": torch.tensor(global_idx, dtype=torch.long),
                    "start": torch.tensor(start, dtype=torch.long),
                }
        raise ValueError(f"Trajectory {global_idx} has no finite windows among selected starts.")

    def _load_window(self, global_idx: int, start: int) -> np.ndarray:
        source_id = int(self.source_ids[global_idx])
        local_id = int(self.local_ids[global_idx])
        source = self.sources[source_id]
        handle = self._get_handle(source_id)
        stop = start + self.total_len
        blocks = []
        for group in self.feature_metadata["groups"]:
            raw = np.asarray(handle[group["source_key"]][local_id, start:stop], dtype=np.float32)
            blocks.append(transform_group(raw, group["name"], self.representation))
        window = np.concatenate([block.reshape(self.total_len, -1) for block in blocks], axis=-1)
        if window.shape != (self.total_len, self.feature_metadata["num_features"]):
            raise ValueError(
                f"Constructed window has shape {window.shape}, expected "
                f"{(self.total_len, self.feature_metadata['num_features'])}."
            )
        return window.astype(np.float32, copy=False)

    def _get_handle(self, source_id: int) -> h5py.File:
        handle = self._handles.get(source_id)
        if handle is None:
            handle = h5py.File(self.sources[source_id].path, "r")
            self._handles[source_id] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover
        self.close()


def prepare_forecasting_data(
    data_glob: str,
    representation: str,
    context_len: int,
    pred_len: int,
    stride: int,
    split_seed: int,
    train_frac: float,
    val_frac: float,
    max_trajectories: int = 0,
    max_windows_per_trajectory: int = 0,
    window_sampling: str = "sequential",
    drop_nan_windows: bool = False,
    max_scaler_trajectories: int = 0,
) -> PreparedForecastData:
    representation = normalize_representation(representation)
    if context_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("context_len, pred_len, and stride must be positive.")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Expected train_frac > 0, val_frac > 0, and train_frac + val_frac < 1.")
    if window_sampling not in {"random", "sequential"}:
        raise ValueError("window_sampling must be random or sequential.")

    paths = [Path(path) for path in sorted(glob.glob(data_glob))]
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched --data_glob={data_glob!r}")
    sources = inspect_sources(paths, representation)
    feature_metadata = build_feature_metadata(sources[0], representation)
    source_ids, local_ids = build_source_lookup(sources)
    trajectory_count = int(source_ids.size)

    if 0 < max_trajectories < trajectory_count:
        rng = np.random.default_rng(split_seed + 17)
        selected = np.sort(rng.choice(trajectory_count, size=max_trajectories, replace=False))
    else:
        selected = np.arange(trajectory_count, dtype=np.int64)
    split_indices = split_trajectory_indices(selected, split_seed, train_frac, val_frac)

    scaler_indices = split_indices["train"]
    if 0 < max_scaler_trajectories < scaler_indices.size:
        rng = np.random.default_rng(split_seed + 23)
        scaler_indices = np.sort(
            rng.choice(scaler_indices, size=max_scaler_trajectories, replace=False)
        )
    scaler = fit_scaler(
        sources,
        source_ids,
        local_ids,
        scaler_indices,
        feature_metadata,
        representation,
    )

    common = dict(
        sources=sources,
        source_ids=source_ids,
        local_ids=local_ids,
        representation=representation,
        feature_metadata=feature_metadata,
        scaler=scaler,
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        max_windows_per_trajectory=max_windows_per_trajectory,
        window_sampling=window_sampling,
        drop_nan_windows=drop_nan_windows,
    )
    train_dataset = ForecastWindowDataset(
        trajectory_indices=split_indices["train"], seed=split_seed + 101, **common
    )
    val_dataset = ForecastWindowDataset(
        trajectory_indices=split_indices["val"], seed=split_seed + 102, **common
    )
    test_dataset = ForecastWindowDataset(
        trajectory_indices=split_indices["test"], seed=split_seed + 103, **common
    )
    feature_metadata["representation"] = representation
    feature_metadata["actual_source_keys"] = {
        str(source.path): source.resolved_keys for source in sources
    }
    return PreparedForecastData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        scaler=scaler,
        feature_metadata=feature_metadata,
        split_indices=split_indices,
        source_paths=[str(source.path) for source in sources],
        trajectory_count=int(selected.size),
        time_steps=sources[0].time_steps,
    )


def normalize_representation(value: str) -> str:
    aliases = {"raw_state18": "trajectory", "raw_state": "trajectory"}
    normalized = aliases.get(value, value)
    if normalized not in {"orbital15", "trajectory"}:
        raise ValueError(f"Unsupported representation: {value}")
    return normalized


def inspect_sources(paths: list[Path], representation: str) -> list[SequenceSource]:
    logical_keys = list(ORBITAL_FEATURES) if representation != "trajectory" else ["positions", "velocities"]
    sources = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            resolved: dict[str, str] = {}
            shapes: dict[str, tuple[int, ...]] = {}
            row_count = None
            time_steps = None
            for logical_key in logical_keys:
                key = resolve_key(handle, logical_key)
                dataset = handle[key]
                if dataset.ndim < 3:
                    raise ValueError(f"Expected {key} in {path} to be [N,T,...], got {dataset.shape}")
                if row_count is None:
                    row_count, time_steps = int(dataset.shape[0]), int(dataset.shape[1])
                elif (row_count, time_steps) != (int(dataset.shape[0]), int(dataset.shape[1])):
                    raise ValueError(f"Inconsistent sequence shape in {path}: {key}={dataset.shape}")
                resolved[logical_key] = key
                shapes[logical_key] = tuple(int(dim) for dim in dataset.shape[2:])
            assert row_count is not None and time_steps is not None
            sources.append(SequenceSource(path, row_count, time_steps, resolved, shapes))
    reference = sources[0]
    for source in sources[1:]:
        if source.time_steps != reference.time_steps or source.step_shapes != reference.step_shapes:
            raise ValueError(f"Incompatible HDF5 shard shapes: {reference.path} and {source.path}")
    return sources


def resolve_key(handle: h5py.File, logical_key: str) -> str:
    if logical_key in TRAJECTORY_ALIASES:
        candidates = TRAJECTORY_ALIASES[logical_key]
    else:
        candidates = (logical_key, f"{logical_key}_complete")
    for candidate in candidates:
        if candidate in handle:
            return candidate
    raise KeyError(f"Missing {logical_key}; tried {candidates}. Available: {sorted(handle.keys())}")


def build_feature_metadata(source: SequenceSource, representation: str) -> dict[str, Any]:
    groups = []
    channel_names: list[str] = []
    start = 0
    logical_keys = list(ORBITAL_FEATURES) if representation != "trajectory" else ["positions", "velocities"]
    for name in logical_keys:
        raw_dim = int(np.prod(source.step_shapes[name]))
        transformed_dim = raw_dim * 2 if name in ORBITAL15_ANGLES else raw_dim
        if name in LOG_FEATURES:
            transform = "log1p"
            names = [f"{name}_{idx}" for idx in range(raw_dim)]
        elif name in ORBITAL15_ANGLES:
            transform = "sin_cos"
            names = [f"{name}_sin_{idx}" for idx in range(raw_dim)]
            names += [f"{name}_cos_{idx}" for idx in range(raw_dim)]
        elif representation == "trajectory":
            transform = "flatten"
            prefix = "pos" if name == "positions" else "vel"
            names = [
                f"{prefix}_body{body}_{axis}"
                for body in range(3)
                for axis in ("x", "y", "z")
            ]
        else:
            transform = "identity"
            names = [f"{name}_{idx}" for idx in range(raw_dim)]
        if len(names) != transformed_dim:
            names = [f"{name}_{idx}" for idx in range(transformed_dim)]
        groups.append(
            {
                "name": name,
                "source_key": source.resolved_keys[name],
                "source_shape": list(source.step_shapes[name]),
                "transform": transform,
                "channel_start": start,
                "channel_end": start + transformed_dim,
                "channel_names": names,
            }
        )
        channel_names.extend(names)
        start += transformed_dim
    if representation == "trajectory" and start != 18:
        raise ValueError(f"trajectory/raw_state18 must produce 18 channels, got {start}")
    return {
        "representation": representation,
        "num_features": start,
        "channel_names": channel_names,
        "groups": groups,
    }


def transform_group(raw: np.ndarray, name: str, representation: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    flat = raw.reshape(raw.shape[0], -1)
    if name in LOG_FEATURES:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log1p(flat).astype(np.float32)
    if name in ORBITAL15_ANGLES:
        return np.concatenate([np.sin(flat), np.cos(flat)], axis=-1).astype(np.float32)
    return flat.astype(np.float32, copy=False)


def split_trajectory_indices(
    selected_indices: np.ndarray,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, np.ndarray]:
    if selected_indices.size < 3:
        raise ValueError("Need at least three trajectories for train/val/test.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(selected_indices)
    train_count = max(1, int(order.size * train_frac))
    val_count = max(1, int(order.size * val_frac))
    if train_count + val_count >= order.size:
        train_count, val_count = order.size - 2, 1
    return {
        "train": np.sort(order[:train_count]),
        "val": np.sort(order[train_count : train_count + val_count]),
        "test": np.sort(order[train_count + val_count :]),
    }


def build_source_lookup(sources: list[SequenceSource]) -> tuple[np.ndarray, np.ndarray]:
    source_ids = np.concatenate(
        [np.full(source.row_count, idx, dtype=np.int64) for idx, source in enumerate(sources)]
    )
    local_ids = np.concatenate(
        [np.arange(source.row_count, dtype=np.int64) for source in sources]
    )
    return source_ids, local_ids


def fit_scaler(
    sources: list[SequenceSource],
    source_ids: np.ndarray,
    local_ids: np.ndarray,
    train_indices: np.ndarray,
    feature_metadata: dict[str, Any],
    representation: str,
    chunk_size: int = 64,
) -> StandardScaler:
    channels = int(feature_metadata["num_features"])
    sums = np.zeros(channels, dtype=np.float64)
    sumsq = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    for source_id, source in enumerate(sources):
        rows = np.sort(local_ids[train_indices[source_ids[train_indices] == source_id]])
        if rows.size == 0:
            continue
        with h5py.File(source.path, "r") as handle:
            for offset in range(0, rows.size, chunk_size):
                batch_rows = rows[offset : offset + chunk_size]
                blocks = []
                for group in feature_metadata["groups"]:
                    raw = np.asarray(handle[group["source_key"]][batch_rows], dtype=np.float32)
                    flat = raw.reshape(-1, *raw.shape[2:])
                    transformed = transform_group(flat, group["name"], representation)
                    blocks.append(transformed.reshape(batch_rows.size, source.time_steps, -1))
                values = np.concatenate(blocks, axis=-1).reshape(-1, channels)
                finite = np.isfinite(values)
                safe = np.where(finite, values, 0.0)
                sums += safe.sum(axis=0, dtype=np.float64)
                sumsq += np.square(safe, dtype=np.float64).sum(axis=0, dtype=np.float64)
                counts += finite.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("At least one feature channel has no finite training values.")
    if np.any(counts < counts.max()):
        warnings.warn("Non-finite training values were excluded from scaler statistics.", RuntimeWarning)
    mean = sums / counts
    variance = np.maximum(sumsq / counts - np.square(mean), 1e-12)
    return StandardScaler(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))


def _assert_window(window: np.ndarray, context_len: int, pred_len: int) -> None:
    if window.ndim != 2:
        raise ValueError(f"Expected [T,C] window, got {window.shape}")
    context = window[:context_len]
    target = window[context_len : context_len + pred_len]
    assert context.ndim == 2 and target.ndim == 2
    assert context.shape[-1] == target.shape[-1]
    if not np.isfinite(window).all():
        raise ValueError("NaN/inf remained after feature construction and scaling.")
