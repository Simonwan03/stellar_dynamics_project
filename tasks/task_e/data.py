from __future__ import annotations

import glob
import re
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
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")


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
        valid_lengths: np.ndarray,
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
        self.valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
        record_trajectories: list[np.ndarray] = []
        record_starts: list[np.ndarray] = []
        for global_idx in self.trajectory_indices:
            available = min(int(self.valid_lengths[int(global_idx)]), sources[0].time_steps)
            starts = np.arange(0, available - self.total_len + 1, self.stride, dtype=np.int64)
            if max_windows_per_trajectory > 0 and starts.size > max_windows_per_trajectory:
                if window_sampling == "sequential":
                    starts = starts[:max_windows_per_trajectory]
                else:
                    rng = np.random.default_rng(seed + int(global_idx))
                    starts = np.sort(rng.choice(starts, size=max_windows_per_trajectory, replace=False))
            if starts.size:
                record_trajectories.append(np.full(starts.size, int(global_idx), dtype=np.int64))
                record_starts.append(starts)
        if not record_starts:
            raise ValueError("No valid forecasting windows remain after completion-time filtering.")
        self.record_trajectories = np.concatenate(record_trajectories)
        self.record_starts = np.concatenate(record_starts)
        if self.drop_nan_windows:
            finite_mask = np.fromiter(
                (
                    np.isfinite(self._load_window(int(global_idx), int(start))).all()
                    for global_idx, start in zip(self.record_trajectories, self.record_starts)
                ),
                dtype=bool,
                count=self.record_starts.size,
            )
            self.record_trajectories = self.record_trajectories[finite_mask]
            self.record_starts = self.record_starts[finite_mask]
            if not self.record_starts.size:
                raise ValueError("No finite forecasting windows remain after completion-time filtering.")

    def __len__(self) -> int:
        return int(self.record_starts.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        global_idx = int(self.record_trajectories[index])
        start = int(self.record_starts[index])
        raw = self._load_window(global_idx, start)
        if not np.isfinite(raw).all():
            if self.drop_nan_windows:
                raise ValueError(
                    f"Non-finite values found in completion-filtered window trajectory={global_idx}, start={start}."
                )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = self.scaler.transform(raw)
        _assert_window(normalized, self.context_len, self.pred_len)
        return {
            "context": torch.from_numpy(normalized[: self.context_len]),
            "target": torch.from_numpy(normalized[self.context_len :]),
            "trajectory_index": torch.tensor(global_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }

    def load_raw_window(self, index: int) -> np.ndarray:
        return self._load_window(int(self.record_trajectories[index]), int(self.record_starts[index]))

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
    group_size: int = 32,
    completion_table: str | Path | None = None,
    require_complete_systems: bool = True,
    max_trajectories: int = 0,
    max_variations_per_system: int = 0,
    max_train_variations_per_system: int = 0,
    max_windows_per_trajectory: int = 0,
    window_sampling: str = "sequential",
    drop_nan_windows: bool = False,
    max_scaler_trajectories: int = 0,
    verbose: bool = True,
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
    if verbose:
        print(f"[data] matched {len(paths)} HDF5 shards for {data_glob!r}", flush=True)
    sources = inspect_sources(paths, representation)
    if verbose:
        total_rows = sum(source.row_count for source in sources)
        print(
            f"[data] inspected {len(sources)} usable shards: trajectories={total_rows}, "
            f"T={sources[0].time_steps}, representation={representation}",
            flush=True,
        )
    feature_metadata = build_feature_metadata(sources[0], representation)
    if verbose:
        print(
            f"[data] built feature metadata: channels={feature_metadata['num_features']}",
            flush=True,
        )
    source_ids, local_ids = build_source_lookup(sources)
    trajectory_count = int(source_ids.size)
    valid_mask, valid_lengths, completion_meta = load_completion_metadata(
        completion_table, sources, group_size, require_complete_systems
    )
    eligible = np.flatnonzero(valid_mask & (valid_lengths >= context_len + pred_len))
    eligible = cap_variations_per_system(
        eligible,
        max_variations_per_system=max_variations_per_system,
        group_size=group_size,
        seed=split_seed,
    )
    split_indices = split_grouped_trajectory_indices(
        eligible, group_size, split_seed, train_frac, val_frac
    )
    split_indices = cap_train_variations_per_system(
        split_indices,
        max_train_variations_per_system=max_train_variations_per_system,
        group_size=group_size,
        seed=split_seed,
    )
    split_indices = cap_split_trajectories(split_indices, max_trajectories, split_seed)
    selected = np.sort(np.concatenate(list(split_indices.values())))
    if verbose:
        print(
            "[data] split trajectories: "
            f"train={split_indices['train'].size}, val={split_indices['val'].size}, "
            f"test={split_indices['test'].size}",
            flush=True,
        )

    identity_scaler = StandardScaler(
        np.zeros(feature_metadata["num_features"], dtype=np.float32),
        np.ones(feature_metadata["num_features"], dtype=np.float32),
    )
    common = dict(
        sources=sources,
        source_ids=source_ids,
        local_ids=local_ids,
        representation=representation,
        feature_metadata=feature_metadata,
        scaler=identity_scaler,
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        max_windows_per_trajectory=max_windows_per_trajectory,
        window_sampling=window_sampling,
        drop_nan_windows=drop_nan_windows,
        valid_lengths=valid_lengths,
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
    scaler = fit_scaler_from_windows(
        train_dataset,
        max_trajectories=max_scaler_trajectories,
        seed=split_seed + 23,
        verbose=verbose,
    )
    train_dataset.scaler = scaler
    val_dataset.scaler = scaler
    test_dataset.scaler = scaler
    if verbose:
        print(
            f"[data] dataset windows: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test={len(test_dataset)}",
            flush=True,
        )
    feature_metadata["representation"] = representation
    feature_metadata["actual_source_keys"] = {
        str(source.path): source.resolved_keys for source in sources
    }
    feature_metadata["completion_filter"] = completion_meta
    feature_metadata["split_unit"] = f"system_group_of_{group_size}"
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


def load_completion_metadata(
    path: str | Path | None,
    sources: list[SequenceSource],
    group_size: int,
    require_complete_systems: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total = sum(source.row_count for source in sources)
    if path is None:
        return (
            np.ones(total, dtype=bool),
            np.full(total, sources[0].time_steps, dtype=np.int64),
            {"enabled": False, "reason": "completion_table_not_configured"},
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Completion table not found: {path}")
    table = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if table.shape[1] != 5:
        raise ValueError(f"Expected five completion columns, got {table.shape} from {path}")

    valid_mask = np.zeros(total, dtype=bool)
    valid_lengths = np.zeros(total, dtype=np.int64)
    global_offset = 0
    completion_horizon = float(np.nanmax(table[:, 2]))
    for source in sources:
        match = RESULT_RE.search(source.path.name)
        if match is None:
            raise ValueError(f"Cannot extract batch id from {source.path}")
        batch_id = int(match.group(1))
        batch = table[table[:, 1].astype(np.int64) == batch_id]
        order = np.argsort(batch[:, 0].astype(np.int64))
        batch = batch[order]
        if batch.shape[0] != source.row_count:
            raise ValueError(
                f"Completion rows for batch {batch_id}: {batch.shape[0]}, expected {source.row_count}"
            )
        valid = (batch[:, 2] >= 0.0) & (batch[:, 4] == 0.0)
        if require_complete_systems:
            if source.row_count % group_size:
                raise ValueError(f"Batch size {source.row_count} is not divisible by group_size={group_size}")
            complete = valid.reshape(-1, group_size).all(axis=1)
            valid &= np.repeat(complete, group_size)

        with h5py.File(source.path, "r") as handle:
            time_key = "times_complete" if "times_complete" in handle else "times"
            time_grid = np.asarray(handle[time_key][0], dtype=np.float64)
        finite_time = time_grid[np.isfinite(time_grid)]
        if finite_time.size == 0:
            raise ValueError(f"No finite time grid in {source.path}")
        time_scale = float(finite_time[-1] / completion_horizon) if completion_horizon > 0 else 1.0
        event_times = batch[:, 2] * time_scale
        lengths = np.searchsorted(time_grid, event_times, side="right").astype(np.int64)
        lengths = np.clip(lengths, 0, source.time_steps)
        sl = slice(global_offset, global_offset + source.row_count)
        valid_mask[sl] = valid
        valid_lengths[sl] = np.where(valid, lengths, 0)
        global_offset += source.row_count

    valid_systems = np.unique(np.flatnonzero(valid_mask) // group_size)
    return valid_mask, valid_lengths, {
        "enabled": True,
        "path": str(path),
        "require_complete_systems": bool(require_complete_systems),
        "valid_trajectories": int(valid_mask.sum()),
        "valid_systems": int(valid_systems.size),
        "completion_horizon": completion_horizon,
    }


def split_grouped_trajectory_indices(
    eligible_indices: np.ndarray,
    group_size: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, np.ndarray]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    system_ids = np.unique(np.asarray(eligible_indices, dtype=np.int64) // group_size)
    if system_ids.size < 3:
        raise ValueError("Need at least three eligible systems for train/val/test.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(system_ids)
    train_count = max(1, int(order.size * train_frac))
    val_count = max(1, int(order.size * val_frac))
    if train_count + val_count >= order.size:
        train_count, val_count = order.size - 2, 1
    system_splits = {
        "train": order[:train_count],
        "val": order[train_count : train_count + val_count],
        "test": order[train_count + val_count :],
    }
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    eligible_systems = eligible // group_size
    return {
        name: np.sort(eligible[np.isin(eligible_systems, split_systems)])
        for name, split_systems in system_splits.items()
    }


def cap_split_trajectories(
    splits: dict[str, np.ndarray],
    max_trajectories: int,
    seed: int,
) -> dict[str, np.ndarray]:
    total = sum(indices.size for indices in splits.values())
    if max_trajectories <= 0 or total <= max_trajectories:
        return splits
    fractions = {name: indices.size / total for name, indices in splits.items()}
    counts = {name: max(1, int(max_trajectories * frac)) for name, frac in fractions.items()}
    while sum(counts.values()) > max_trajectories:
        candidate = max((name for name in counts if counts[name] > 1), key=lambda name: counts[name])
        counts[candidate] -= 1
    while sum(counts.values()) < max_trajectories:
        candidate = max(
            (name for name in counts if counts[name] < splits[name].size),
            key=lambda name: fractions[name],
        )
        counts[candidate] += 1
    capped: dict[str, np.ndarray] = {}
    for offset, (name, indices) in enumerate(splits.items()):
        rng = np.random.default_rng(seed + 1000 + offset)
        capped[name] = np.sort(rng.choice(indices, size=counts[name], replace=False))
    return capped


def cap_variations_per_system(
    indices: np.ndarray,
    max_variations_per_system: int,
    group_size: int,
    seed: int,
) -> np.ndarray:
    """Keep a reproducible random subset from every system before splitting.

    Applying this before grouped splitting keeps the selected variations of each
    system in the same train/validation/test split.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if max_variations_per_system <= 0 or not indices.size:
        return indices
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    rng = np.random.default_rng(seed + 3000)
    selected: list[np.ndarray] = []
    for system_id in np.unique(indices // group_size):
        system_indices = indices[(indices // group_size) == system_id]
        if system_indices.size > max_variations_per_system:
            system_indices = rng.choice(
                system_indices,
                size=max_variations_per_system,
                replace=False,
            )
        selected.append(np.asarray(system_indices, dtype=np.int64))
    return np.sort(np.concatenate(selected))


def cap_train_variations_per_system(
    splits: dict[str, np.ndarray],
    max_train_variations_per_system: int,
    group_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Keep a reproducible random subset of variations in the training split only.

    System assignment to train/validation/test is performed before this function,
    so selecting training variations here cannot move a system across splits. A
    value of one selects one trajectory from every eligible training system while
    leaving validation and test trajectories unchanged.
    """
    if max_train_variations_per_system <= 0:
        return splits
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    train_indices = np.asarray(splits["train"], dtype=np.int64)
    if not train_indices.size:
        return {**splits, "train": train_indices.copy()}

    rng = np.random.default_rng(seed + 2000)
    selected: list[np.ndarray] = []
    system_ids = np.unique(train_indices // group_size)
    for system_id in system_ids:
        system_indices = train_indices[(train_indices // group_size) == system_id]
        if system_indices.size > max_train_variations_per_system:
            system_indices = rng.choice(
                system_indices,
                size=max_train_variations_per_system,
                replace=False,
            )
        selected.append(np.asarray(system_indices, dtype=np.int64))

    return {**splits, "train": np.sort(np.concatenate(selected))}


def build_source_lookup(sources: list[SequenceSource]) -> tuple[np.ndarray, np.ndarray]:
    source_ids = np.concatenate(
        [np.full(source.row_count, idx, dtype=np.int64) for idx, source in enumerate(sources)]
    )
    local_ids = np.concatenate(
        [np.arange(source.row_count, dtype=np.int64) for source in sources]
    )
    return source_ids, local_ids


def fit_scaler_from_windows(
    dataset: ForecastWindowDataset,
    max_trajectories: int,
    seed: int,
    verbose: bool = True,
) -> StandardScaler:
    trajectories = np.unique(dataset.record_trajectories)
    if 0 < max_trajectories < trajectories.size:
        rng = np.random.default_rng(seed)
        trajectories = np.sort(rng.choice(trajectories, size=max_trajectories, replace=False))
    selected_records = np.flatnonzero(np.isin(dataset.record_trajectories, trajectories))
    channels = int(dataset.feature_metadata["num_features"])
    sums = np.zeros(channels, dtype=np.float64)
    sumsq = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    dropped_windows = 0
    if verbose:
        print(
            f"[data] fitting scaler on {trajectories.size} train trajectories and "
            f"{selected_records.size} selected valid windows",
            flush=True,
        )
    for record_index in selected_records:
        values = dataset.load_raw_window(int(record_index)).reshape(-1, channels)
        if not np.isfinite(values).all():
            dropped_windows += 1
            continue
        sums += values.sum(axis=0, dtype=np.float64)
        sumsq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        counts += values.shape[0]
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Scaler has no finite values for channels {missing}.")
    if dropped_windows:
        warnings.warn(
            f"Dropped {dropped_windows} non-finite training windows while fitting scaler.",
            RuntimeWarning,
            stacklevel=2,
        )
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
