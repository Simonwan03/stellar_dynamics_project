from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_TRAJECTORY_KEYS = ("positions_complete", "velocities_complete")
ICS_RE = re.compile(r"ICs_sampling_(\d+)(?:_groupof\d+)?\.hdf5$")


@dataclass(frozen=True)
class TrajectorySource:
    path: Path
    row_count: int
    time_steps: int
    keys: tuple[str, ...]
    per_step_shapes: dict[str, tuple[int, ...]]


@dataclass
class PreparedTrajectoryData:
    train_dataset: "TrajectoryWindowDataset"
    val_dataset: "TrajectoryWindowDataset"
    test_dataset: "TrajectoryWindowDataset"
    feature_metadata: dict[str, Any]
    normalization_stats: dict[str, np.ndarray]
    split_indices: dict[str, np.ndarray]
    source_paths: list[str]
    trajectory_count: int
    time_steps: int


def parse_trajectory_keys(keys: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if keys is None:
        return list(DEFAULT_TRAJECTORY_KEYS)
    if isinstance(keys, str):
        parsed = [item.strip() for item in keys.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in keys if str(item).strip()]
    if not parsed:
        raise ValueError("No trajectory keys were selected. Pass at least one HDF5 sequence key.")
    return parsed


def resolve_trajectory_sources(data_path: str | Path, trajectory_keys: list[str]) -> list[TrajectorySource]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if data_path.is_dir():
        candidate_paths = sorted(
            path
            for path in data_path.glob("*.hdf5")
            if path.is_file() and _file_contains_trajectories(path, trajectory_keys)
        )
        if not candidate_paths:
            raise FileNotFoundError(f"No HDF5 sequence files with keys {trajectory_keys} were found in {data_path}")
    else:
        candidate_paths = _resolve_single_path(data_path, trajectory_keys)

    sources = [_inspect_source(path, trajectory_keys) for path in candidate_paths]
    _validate_compatible_sources(sources)
    return sources


def prepare_trajectory_datasets(
    data_path: str | Path,
    trajectory_keys: list[str],
    context_len: int,
    pred_len: int,
    stride: int,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_normalization_trajectories: int = 0,
) -> PreparedTrajectoryData:
    if context_len <= 0 or pred_len <= 0:
        raise ValueError(f"context_len and pred_len must be positive, got {context_len=} {pred_len=}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    sources = resolve_trajectory_sources(data_path, trajectory_keys)
    time_steps = sources[0].time_steps
    total_window = context_len + pred_len
    if time_steps < total_window:
        raise ValueError(
            f"Trajectory length {time_steps} is shorter than context_len + pred_len = {total_window}."
        )

    feature_metadata = build_feature_metadata(sources[0])
    trajectory_count = sum(source.row_count for source in sources)
    split_indices = split_trajectory_indices(
        trajectory_count=trajectory_count,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    normalization_indices = split_indices["train"]
    if 0 < max_normalization_trajectories < normalization_indices.size:
        rng = np.random.default_rng(seed + 3)
        normalization_indices = np.sort(
            rng.choice(normalization_indices, size=max_normalization_trajectories, replace=False)
        )

    normalization_stats = compute_normalization_stats(
        sources=sources,
        train_indices=normalization_indices,
        feature_metadata=feature_metadata,
    )
    normalization_stats["trajectory_count"] = np.int64(normalization_indices.size)
    source_lookup = build_source_lookup(sources)

    dataset_kwargs = {
        "sources": sources,
        "source_lookup": source_lookup,
        "context_len": context_len,
        "pred_len": pred_len,
        "stride": stride,
        "feature_metadata": feature_metadata,
        "normalization_stats": normalization_stats,
    }
    train_dataset = TrajectoryWindowDataset(trajectory_indices=split_indices["train"], **dataset_kwargs)
    val_dataset = TrajectoryWindowDataset(trajectory_indices=split_indices["val"], **dataset_kwargs)
    test_dataset = TrajectoryWindowDataset(trajectory_indices=split_indices["test"], **dataset_kwargs)

    return PreparedTrajectoryData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        feature_metadata=feature_metadata,
        normalization_stats=normalization_stats,
        split_indices=split_indices,
        source_paths=[str(source.path) for source in sources],
        trajectory_count=trajectory_count,
        time_steps=time_steps,
    )


def save_feature_metadata(path: str | Path, feature_metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(feature_metadata), handle, indent=2)


def save_normalization_stats(path: str | Path, normalization_stats: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=normalization_stats["mean"],
        std=normalization_stats["std"],
        channel_names=np.asarray(normalization_stats["channel_names"], dtype=object),
        group_names=np.asarray(normalization_stats["group_names"], dtype=object),
        trajectory_count=np.asarray(normalization_stats.get("trajectory_count", -1), dtype=np.int64),
    )


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[TrajectorySource],
        source_lookup: dict[str, np.ndarray],
        trajectory_indices: np.ndarray,
        context_len: int,
        pred_len: int,
        stride: int,
        feature_metadata: dict[str, Any],
        normalization_stats: dict[str, np.ndarray],
    ) -> None:
        super().__init__()
        self.sources = sources
        self.trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.total_window = self.context_len + self.pred_len
        self.feature_metadata = feature_metadata
        self.normalization_stats = normalization_stats
        self.channel_mean = normalization_stats["mean"].astype(np.float32)
        self.channel_std = normalization_stats["std"].astype(np.float32)
        self.num_features = int(feature_metadata["num_features"])
        self.source_ids = source_lookup["source_ids"]
        self.local_ids = source_lookup["local_ids"]
        self.time_steps = int(source_lookup["time_steps"])
        self.windows_per_trajectory = ((self.time_steps - self.total_window) // self.stride) + 1
        if self.windows_per_trajectory <= 0:
            raise ValueError(
                f"Trajectory length {self.time_steps} is too short for context={self.context_len}, pred={self.pred_len}."
            )
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return int(self.trajectory_indices.size * self.windows_per_trajectory)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")

        traj_offset = index // self.windows_per_trajectory
        window_offset = index % self.windows_per_trajectory
        global_traj_idx = int(self.trajectory_indices[traj_offset])
        source_id = int(self.source_ids[global_traj_idx])
        local_id = int(self.local_ids[global_traj_idx])
        start = int(window_offset * self.stride)
        stop = start + self.total_window

        source = self.sources[source_id]
        handle = self._get_handle(source_id)
        blocks = []
        for group in self.feature_metadata["groups"]:
            key = group["source_key"]
            raw = np.asarray(handle[key][local_id, start:stop], dtype=np.float32)
            flat = raw.reshape(self.total_window, group["transformed_dim"])
            blocks.append(sanitize_array(flat, feature_name=key))
        flattened = np.concatenate(blocks, axis=-1)
        normalized = normalize_feature_block(flattened, self.channel_mean, self.channel_std)

        context = torch.from_numpy(normalized[: self.context_len])
        target = torch.from_numpy(normalized[self.context_len :])
        return {"context": context, "target": target}

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


def build_feature_metadata(source: TrajectorySource) -> dict[str, Any]:
    coord_labels = ["x", "y", "z"]
    groups: list[dict[str, Any]] = []
    channel_names: list[str] = []
    channel_to_group: list[str] = []
    channel_start = 0

    for key in source.keys:
        per_step_shape = source.per_step_shapes[key]
        flattened_dim = int(np.prod(per_step_shape))
        body_count = per_step_shape[0] if len(per_step_shape) >= 1 else 1
        coord_count = per_step_shape[1] if len(per_step_shape) >= 2 else 1
        group_channel_names = []
        prefix = "pos" if key == "positions_complete" else "vel" if key == "velocities_complete" else key
        for body_idx in range(body_count):
            for coord_idx in range(coord_count):
                suffix = coord_labels[coord_idx] if coord_idx < len(coord_labels) else str(coord_idx)
                group_channel_names.append(f"{prefix}_body{body_idx}_{suffix}")
        groups.append(
            {
                "name": key,
                "source_key": key,
                "raw_dim": flattened_dim,
                "transformed_dim": flattened_dim,
                "transform": "flatten_then_standardize",
                "channel_start": channel_start,
                "channel_end": channel_start + flattened_dim,
                "channel_names": group_channel_names,
                "per_step_shape": list(per_step_shape),
            }
        )
        channel_names.extend(group_channel_names)
        channel_to_group.extend([key] * flattened_dim)
        channel_start += flattened_dim

    return {
        "selected_features": list(source.keys),
        "num_features": channel_start,
        "trajectory_keys": list(source.keys),
        "per_step_shapes": {key: list(shape) for key, shape in source.per_step_shapes.items()},
        "groups": groups,
        "channel_names": channel_names,
        "channel_to_group": channel_to_group,
    }


def split_trajectory_indices(
    trajectory_count: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if trajectory_count < 3:
        raise ValueError(f"Need at least 3 trajectories to create train/val/test splits, got {trajectory_count}.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trajectory_count)
    train_count = int(trajectory_count * train_ratio)
    val_count = int(trajectory_count * val_ratio)
    if train_count <= 0 or val_count <= 0 or trajectory_count - train_count - val_count <= 0:
        raise ValueError(
            "Split ratios produced an empty split. "
            f"Got counts train={train_count}, val={val_count}, test={trajectory_count - train_count - val_count}."
        )

    train_idx = np.sort(shuffled[:train_count])
    val_idx = np.sort(shuffled[train_count : train_count + val_count])
    test_idx = np.sort(shuffled[train_count + val_count :])
    return {"train": train_idx.astype(np.int64), "val": val_idx.astype(np.int64), "test": test_idx.astype(np.int64)}


def compute_normalization_stats(
    sources: list[TrajectorySource],
    train_indices: np.ndarray,
    feature_metadata: dict[str, Any],
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    num_features = int(feature_metadata["num_features"])
    sums = np.zeros(num_features, dtype=np.float64)
    sumsq = np.zeros(num_features, dtype=np.float64)
    total_count = 0
    source_lookup = build_source_lookup(sources)
    source_ids = source_lookup["source_ids"][train_indices]
    local_ids = source_lookup["local_ids"][train_indices]

    for source_id, source in enumerate(sources):
        source_local_ids = np.sort(local_ids[source_ids == source_id])
        if source_local_ids.size == 0:
            continue

        with h5py.File(source.path, "r") as handle:
            for start_idx in range(0, source_local_ids.size, chunk_size):
                batch_rows = source_local_ids[start_idx : start_idx + chunk_size]
                blocks = []
                for group in feature_metadata["groups"]:
                    key = group["source_key"]
                    raw = np.asarray(handle[key][batch_rows, :, ...], dtype=np.float32)
                    flat = raw.reshape(batch_rows.size, source.time_steps, group["transformed_dim"])
                    blocks.append(sanitize_array(flat, feature_name=key, source_path=source.path))
                flattened = np.concatenate(blocks, axis=-1)
                if flattened.shape[-1] != num_features:
                    raise ValueError(f"Expected {num_features} channels, got {flattened.shape[-1]}.")
                sums += flattened.sum(axis=(0, 1), dtype=np.float64)
                sumsq += np.square(flattened, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
                total_count += int(batch_rows.size * source.time_steps)

    if total_count <= 0:
        raise ValueError("Training split is empty after source resolution; could not compute normalization stats.")

    mean = sums / total_count
    var = np.maximum((sumsq / total_count) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "channel_names": np.asarray(feature_metadata["channel_names"], dtype=object),
        "group_names": np.asarray([group["name"] for group in feature_metadata["groups"]], dtype=object),
    }


def build_source_lookup(sources: list[TrajectorySource]) -> dict[str, np.ndarray]:
    source_ids = []
    local_ids = []
    total = 0
    for source_id, source in enumerate(sources):
        total += source.row_count
        source_ids.append(np.full(source.row_count, source_id, dtype=np.int64))
        local_ids.append(np.arange(source.row_count, dtype=np.int64))
    if total == 0:
        raise ValueError("No trajectories were discovered in the provided data source(s).")
    return {
        "source_ids": np.concatenate(source_ids, axis=0),
        "local_ids": np.concatenate(local_ids, axis=0),
        "time_steps": np.int64(sources[0].time_steps),
    }


def normalize_feature_block(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (array - mean) / std
    return sanitize_array(normalized, feature_name="normalized_trajectory")


def sanitize_array(array: np.ndarray, feature_name: str, source_path: str | Path | None = None) -> np.ndarray:
    if np.isfinite(array).all():
        return array
    location = f" in {source_path}" if source_path is not None else ""
    warnings.warn(
        f"Detected NaN or inf values for feature '{feature_name}'{location}; applying np.nan_to_num.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_single_path(data_path: Path, trajectory_keys: list[str]) -> list[Path]:
    if _file_contains_trajectories(data_path, trajectory_keys):
        return [data_path]

    match = ICS_RE.fullmatch(data_path.name)
    if match is not None:
        sampling_id = int(match.group(1))
        shard_paths = sorted(data_path.parent.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
        shard_paths = [path for path in shard_paths if _file_contains_trajectories(path, trajectory_keys)]
        if shard_paths:
            return shard_paths

    raise ValueError(f"{data_path} does not contain keys {trajectory_keys}, and no matching result shards were found.")


def _file_contains_trajectories(path: Path, trajectory_keys: list[str]) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            for trajectory_key in trajectory_keys:
                if trajectory_key not in handle or handle[trajectory_key].ndim < 3:
                    return False
            return True
    except OSError as exc:
        raise OSError(f"Could not open HDF5 file: {path}") from exc


def _inspect_source(path: Path, trajectory_keys: list[str]) -> TrajectorySource:
    with h5py.File(path, "r") as handle:
        row_count = None
        time_steps = None
        per_step_shapes: dict[str, tuple[int, ...]] = {}
        for trajectory_key in trajectory_keys:
            if trajectory_key not in handle:
                raise KeyError(f"Could not find trajectory key '{trajectory_key}' in {path}")
            dataset = handle[trajectory_key]
            if dataset.ndim < 3:
                raise ValueError(f"Expected '{trajectory_key}' in {path} to have shape [N, T, ...], got {dataset.shape}.")
            if row_count is None:
                row_count = int(dataset.shape[0])
                time_steps = int(dataset.shape[1])
            elif row_count != int(dataset.shape[0]) or time_steps != int(dataset.shape[1]):
                raise ValueError(
                    f"Inconsistent shape for '{trajectory_key}' in {path}. "
                    f"Expected [N={row_count}, T={time_steps}, ...], got {dataset.shape}."
                )
            per_step_shapes[trajectory_key] = tuple(int(dim) for dim in dataset.shape[2:])
        assert row_count is not None and time_steps is not None
        return TrajectorySource(
            path=path,
            row_count=row_count,
            time_steps=time_steps,
            keys=tuple(trajectory_keys),
            per_step_shapes=per_step_shapes,
        )


def _validate_compatible_sources(sources: list[TrajectorySource]) -> None:
    if not sources:
        raise ValueError("No trajectory sources were resolved from the provided data path.")
    time_steps = sources[0].time_steps
    per_step_shapes = sources[0].per_step_shapes
    keys = sources[0].keys
    for source in sources[1:]:
        if source.time_steps != time_steps:
            raise ValueError(f"All sources must have the same time length. Found {time_steps} and {source.time_steps}.")
        if source.keys != keys:
            raise ValueError(f"All sources must use the same trajectory keys. Found {keys} and {source.keys}.")
        for key in keys:
            if source.per_step_shapes[key] != per_step_shapes[key]:
                raise ValueError(
                    f"Trajectory key '{key}' has inconsistent per-step shapes: "
                    f"{per_step_shapes[key]} vs {source.per_step_shapes[key]}."
                )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
