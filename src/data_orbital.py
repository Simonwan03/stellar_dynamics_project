from __future__ import annotations

import json
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

DEFAULT_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
ANGLE_FEATURES = {"incs", "APs", "LANs"}
LOG1P_FEATURES = {"aratios", "mratios"}
ICS_RE = re.compile(r"ICs_sampling_(\d+)(?:_groupof\d+)?\.hdf5$")


@dataclass(frozen=True)
class SequenceSource:
    path: Path
    row_count: int
    time_steps: int
    resolved_keys: dict[str, str]
    raw_dims: dict[str, int]


@dataclass
class PreparedOrbitalData:
    train_dataset: "OrbitalWindowDataset"
    val_dataset: "OrbitalWindowDataset"
    test_dataset: "OrbitalWindowDataset"
    feature_metadata: dict[str, Any]
    normalization_stats: dict[str, np.ndarray]
    split_indices: dict[str, np.ndarray]
    source_paths: list[str]
    trajectory_count: int
    time_steps: int


def parse_feature_list(features: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if features is None:
        return list(DEFAULT_FEATURES)
    if isinstance(features, str):
        parsed = [item.strip() for item in features.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in features if str(item).strip()]
    if not parsed:
        raise ValueError("No features were selected. Pass at least one orbital feature.")
    unknown = sorted(set(parsed) - set(DEFAULT_FEATURES))
    if unknown:
        raise ValueError(f"Unsupported feature(s): {unknown}. Supported: {list(DEFAULT_FEATURES)}")
    return parsed


def resolve_sequence_sources(data_path: str | Path, selected_features: list[str]) -> list[SequenceSource]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    candidate_paths: list[Path]
    if data_path.is_dir():
        candidate_paths = sorted(
            path
            for path in data_path.glob("*.hdf5")
            if path.is_file() and _file_contains_sequence_features(path, selected_features)
        )
        if not candidate_paths:
            raise FileNotFoundError(
                f"No sequence HDF5 files with shape [N, T, d] were found under directory: {data_path}"
            )
    else:
        candidate_paths = _resolve_single_path(data_path, selected_features)

    sources = [_inspect_source(path, selected_features) for path in candidate_paths]
    _validate_compatible_sources(sources, selected_features)
    return sources


def prepare_orbital_datasets(
    data_path: str | Path,
    selected_features: list[str],
    context_len: int,
    pred_len: int,
    stride: int,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> PreparedOrbitalData:
    if context_len <= 0 or pred_len <= 0:
        raise ValueError(f"context_len and pred_len must be positive, got {context_len=} {pred_len=}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    sources = resolve_sequence_sources(data_path, selected_features)
    time_steps = sources[0].time_steps
    total_window = context_len + pred_len
    if time_steps < total_window:
        raise ValueError(
            f"Trajectory length {time_steps} is shorter than context_len + pred_len = {total_window}."
        )

    feature_metadata = build_feature_metadata(sources[0], selected_features)
    trajectory_count = sum(source.row_count for source in sources)
    split_indices = split_trajectory_indices(
        trajectory_count=trajectory_count,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    normalization_stats = compute_normalization_stats(
        sources=sources,
        train_indices=split_indices["train"],
        feature_metadata=feature_metadata,
    )
    source_lookup = build_source_lookup(sources)

    train_dataset = OrbitalWindowDataset(
        sources=sources,
        source_lookup=source_lookup,
        trajectory_indices=split_indices["train"],
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        feature_metadata=feature_metadata,
        normalization_stats=normalization_stats,
    )
    val_dataset = OrbitalWindowDataset(
        sources=sources,
        source_lookup=source_lookup,
        trajectory_indices=split_indices["val"],
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        feature_metadata=feature_metadata,
        normalization_stats=normalization_stats,
    )
    test_dataset = OrbitalWindowDataset(
        sources=sources,
        source_lookup=source_lookup,
        trajectory_indices=split_indices["test"],
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        feature_metadata=feature_metadata,
        normalization_stats=normalization_stats,
    )

    return PreparedOrbitalData(
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
    )


class OrbitalWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[SequenceSource],
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
        window = np.empty((self.total_window, self.num_features), dtype=np.float32)
        for group in self.feature_metadata["groups"]:
            raw = np.asarray(
                handle[source.resolved_keys[group["name"]]][local_id, start:stop, :],
                dtype=np.float32,
            )
            transformed = sanitize_array(transform_feature_array(group["name"], raw), feature_name=group["name"])
            normalized = normalize_feature_block(
                transformed,
                self.channel_mean[group["channel_start"] : group["channel_end"]],
                self.channel_std[group["channel_start"] : group["channel_end"]],
            )
            window[:, group["channel_start"] : group["channel_end"]] = normalized

        context = torch.from_numpy(window[: self.context_len])
        target = torch.from_numpy(window[self.context_len :])
        return {"context": context, "target": target}

    def sample_pair(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        item = self[index]
        return item["context"].cpu().numpy(), item["target"].cpu().numpy()

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


def build_feature_metadata(source: SequenceSource, selected_features: list[str]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    channel_names: list[str] = []
    channel_to_group: list[str] = []
    channel_start = 0

    for feature_name in selected_features:
        raw_dim = int(source.raw_dims[feature_name])
        transformed_dim = raw_dim * 2 if feature_name in ANGLE_FEATURES else raw_dim
        if feature_name in LOG1P_FEATURES:
            transform = "log1p_then_standardize"
            names = [f"{feature_name}_{idx}" for idx in range(raw_dim)]
        elif feature_name in ANGLE_FEATURES:
            transform = "sincos_then_standardize"
            names = [f"{feature_name}_sin_{idx}" for idx in range(raw_dim)] + [
                f"{feature_name}_cos_{idx}" for idx in range(raw_dim)
            ]
        else:
            transform = "standardize"
            names = [f"{feature_name}_{idx}" for idx in range(raw_dim)]

        group = {
            "name": feature_name,
            "source_key": source.resolved_keys[feature_name],
            "raw_dim": raw_dim,
            "transformed_dim": transformed_dim,
            "transform": transform,
            "channel_start": channel_start,
            "channel_end": channel_start + transformed_dim,
            "channel_names": names,
        }
        groups.append(group)
        channel_names.extend(names)
        channel_to_group.extend([feature_name] * transformed_dim)
        channel_start += transformed_dim

    return {
        "selected_features": list(selected_features),
        "num_features": channel_start,
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
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_normalization_stats(
    sources: list[SequenceSource],
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
                for group in feature_metadata["groups"]:
                    raw = np.asarray(
                        handle[source.resolved_keys[group["name"]]][batch_rows, :, :],
                        dtype=np.float32,
                    )
                    transformed = transform_feature_array(group["name"], raw)
                    transformed = sanitize_array(transformed, feature_name=group["name"], source_path=source.path)
                    start = group["channel_start"]
                    end = group["channel_end"]
                    sums[start:end] += transformed.sum(axis=(0, 1), dtype=np.float64)
                    sumsq[start:end] += np.square(transformed, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
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


def build_source_lookup(sources: list[SequenceSource]) -> dict[str, np.ndarray]:
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


def transform_feature_array(feature_name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if feature_name in LOG1P_FEATURES:
        with np.errstate(invalid="ignore"):
            transformed = np.log1p(array)
    elif feature_name in ANGLE_FEATURES:
        transformed = np.concatenate([np.sin(array), np.cos(array)], axis=-1)
    else:
        transformed = array
    return np.asarray(transformed, dtype=np.float32)


def normalize_feature_block(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (array - mean) / std
    return sanitize_array(normalized, feature_name="normalized_feature")


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


def _resolve_single_path(data_path: Path, selected_features: list[str]) -> list[Path]:
    if _file_contains_sequence_features(data_path, selected_features):
        return [data_path]

    match = ICS_RE.fullmatch(data_path.name)
    if match is not None:
        sampling_id = int(match.group(1))
        shard_paths = sorted(data_path.parent.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
        if shard_paths:
            return shard_paths

    raise ValueError(
        f"{data_path} does not contain sequence arrays with shape [N, T, d], and no matching result shards were found."
    )


def _file_contains_sequence_features(path: Path, selected_features: list[str]) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            for feature_name in selected_features:
                key = _resolve_feature_key(handle, feature_name, raise_on_missing=False)
                if key is None:
                    return False
                dataset = handle[key]
                if dataset.ndim != 3:
                    return False
            return True
    except OSError as exc:
        raise OSError(f"Could not open HDF5 file: {path}") from exc


def _inspect_source(path: Path, selected_features: list[str]) -> SequenceSource:
    with h5py.File(path, "r") as handle:
        resolved_keys: dict[str, str] = {}
        raw_dims: dict[str, int] = {}
        row_count = None
        time_steps = None
        for feature_name in selected_features:
            key = _resolve_feature_key(handle, feature_name, raise_on_missing=True)
            dataset = handle[key]
            if dataset.ndim != 3:
                raise ValueError(
                    f"Expected dataset '{key}' in {path} to have shape [N, T, d], got {dataset.shape}."
                )
            if row_count is None:
                row_count = int(dataset.shape[0])
                time_steps = int(dataset.shape[1])
            elif row_count != int(dataset.shape[0]) or time_steps != int(dataset.shape[1]):
                raise ValueError(
                    f"Inconsistent shape for feature '{key}' in {path}. Expected [N={row_count}, T={time_steps}, d], "
                    f"got {dataset.shape}."
                )
            resolved_keys[feature_name] = key
            raw_dims[feature_name] = int(dataset.shape[2])

        assert row_count is not None and time_steps is not None
        return SequenceSource(
            path=path,
            row_count=row_count,
            time_steps=time_steps,
            resolved_keys=resolved_keys,
            raw_dims=raw_dims,
        )


def _validate_compatible_sources(sources: list[SequenceSource], selected_features: list[str]) -> None:
    if not sources:
        raise ValueError("No sequence sources were resolved from the provided data path.")
    time_steps = sources[0].time_steps
    reference_dims = sources[0].raw_dims
    for source in sources[1:]:
        if source.time_steps != time_steps:
            raise ValueError(
                f"All sources must have the same time length. Found {time_steps} and {source.time_steps}."
            )
        for feature_name in selected_features:
            if source.raw_dims[feature_name] != reference_dims[feature_name]:
                raise ValueError(
                    f"Feature '{feature_name}' has inconsistent raw dims across sources: "
                    f"{reference_dims[feature_name]} vs {source.raw_dims[feature_name]}."
                )


def _resolve_feature_key(
    handle: h5py.File,
    feature_name: str,
    raise_on_missing: bool,
) -> str | None:
    candidates = (feature_name, f"{feature_name}_complete")
    for key in candidates:
        if key in handle:
            return key

    if not raise_on_missing:
        return None

    available_keys = sorted(handle.keys())
    raise KeyError(
        f"Could not find feature '{feature_name}' in HDF5 file. "
        f"Tried keys {list(candidates)}. Available keys: {available_keys}"
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
