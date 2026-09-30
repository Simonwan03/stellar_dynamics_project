from __future__ import annotations

import glob
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


ORBITAL_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
LOG_FEATURES = {"aratios", "mratios"}
ORBITAL15_ANGLES = {"APs", "LANs"}
TRAJECTORY_ALIASES = {
    "positions": ("positions_complete", "positions"),
    "velocities": ("velocities_complete", "velocities"),
}
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")


CACHE_SCHEMA_VERSION = 1
CACHE_FILES = ("manifest.json", "feature_metadata.json", "scaler.npz", "records.npz")


def _work_chunks(indices: np.ndarray, num_workers: int) -> list[np.ndarray]:
    """Split record indices into a small number of ordered worker chunks."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if not indices.size:
        return []
    workers = max(1, min(int(num_workers), int(indices.size)))
    chunk_count = max(workers, workers * 4)
    chunk_size = max(1, (indices.size + chunk_count - 1) // chunk_count)
    return [
        indices[start : start + chunk_size]
        for start in range(0, indices.size, chunk_size)
    ]

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

    @classmethod
    def load(cls, path: str | Path) -> tuple["StandardScaler", list[str]]:
        """Load a scaler and its channel names from a cache/output ``.npz`` file."""
        with np.load(Path(path), allow_pickle=True) as payload:
            if "mean" not in payload or "std" not in payload:
                raise ValueError(f"Scaler file is missing mean/std arrays: {path}")
            feature_names = payload.get("feature_names")
            names = [] if feature_names is None else [str(name) for name in feature_names.tolist()]
            return cls(payload["mean"], payload["std"]), names


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
        num_workers: int = 1,
        cached_record_trajectories: np.ndarray | None = None,
        cached_record_starts: np.ndarray | None = None,
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
        self.num_workers = int(num_workers)
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}.")
        self._thread_local = threading.local()
        self._thread_handles: dict[int, dict[int, h5py.File]] = {}
        self._thread_handles_lock = threading.Lock()
        self.valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
        if (cached_record_trajectories is None) != (cached_record_starts is None):
            raise ValueError("cached_record_trajectories and cached_record_starts must be provided together.")
        if cached_record_trajectories is not None and cached_record_starts is not None:
            cached_trajectories = np.asarray(cached_record_trajectories, dtype=np.int64).reshape(-1)
            cached_starts = np.asarray(cached_record_starts, dtype=np.int64).reshape(-1)
            if cached_trajectories.shape != cached_starts.shape:
                raise ValueError(
                    "Cached window trajectory/start arrays must have the same shape: "
                    f"{cached_trajectories.shape} vs {cached_starts.shape}"
                )
            if not cached_starts.size:
                raise ValueError("Cached window records are empty.")
            if np.any(cached_trajectories < 0) or np.any(cached_trajectories >= self.valid_lengths.size):
                raise ValueError("Cached window records contain invalid trajectory indices.")
            if np.any(cached_starts < 0):
                raise ValueError("Cached window records contain negative starts.")
            available = np.minimum(self.valid_lengths[cached_trajectories], sources[0].time_steps)
            if np.any(cached_starts + self.total_len > available):
                raise ValueError("Cached window records exceed the valid trajectory prefixes.")
            # Cached records have already passed the optional finite-value scan.
            self.record_trajectories = cached_trajectories
            self.record_starts = cached_starts
            return
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
            record_indices = np.arange(self.record_starts.size, dtype=np.int64)

            def scan_chunk(chunk: np.ndarray) -> np.ndarray:
                return np.fromiter(
                    (
                        np.isfinite(
                            self._load_window(
                                int(self.record_trajectories[index]),
                                int(self.record_starts[index]),
                            )
                        ).all()
                        for index in chunk
                    ),
                    dtype=bool,
                    count=chunk.size,
                )

            chunks = _work_chunks(record_indices, self.num_workers)
            if len(chunks) <= 1:
                finite_mask = scan_chunk(chunks[0])
            else:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    masks = list(executor.map(scan_chunk, chunks))
                finite_mask = np.concatenate(masks)
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

    def window_record(self, index: int) -> tuple[int, int]:
        """Return the global trajectory index and start for a dataset window."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(self.record_trajectories[index]), int(self.record_starts[index])

    def load_raw_window(self, index: int, total_len: int | None = None) -> np.ndarray:
        """Load a window, optionally extending it for rolling evaluation."""
        global_idx, start = self.window_record(index)
        requested_len = self.total_len if total_len is None else int(total_len)
        if requested_len <= 0:
            raise ValueError(f"total_len must be positive, got {requested_len}.")
        available = min(int(self.valid_lengths[global_idx]), self.sources[0].time_steps)
        if start + requested_len > available:
            raise ValueError(
                f"Requested window exceeds the valid trajectory prefix: "
                f"trajectory={global_idx}, start={start}, length={requested_len}, available={available}."
            )
        return self._load_window(global_idx, start, requested_len)

    def load_transformed_window(self, index: int, total_len: int | None = None) -> np.ndarray:
        raw = self.load_raw_window(index, total_len=total_len)
        if not np.isfinite(raw).all():
            raise ValueError(
                f"Non-finite values found in forecasting window "
                f"trajectory={self.window_record(index)[0]}, start={self.window_record(index)[1]}."
            )
        return self.scaler.transform(raw)

    def _load_window(self, global_idx: int, start: int, total_len: int | None = None) -> np.ndarray:
        source_id = int(self.source_ids[global_idx])
        local_id = int(self.local_ids[global_idx])
        source = self.sources[source_id]
        handle = self._get_handle(source_id)
        requested_len = self.total_len if total_len is None else int(total_len)
        stop = start + requested_len
        blocks = []
        for group in self.feature_metadata["groups"]:
            raw = np.asarray(handle[group["source_key"]][local_id, start:stop], dtype=np.float32)
            blocks.append(transform_group(raw, group["name"], self.representation))
        window = np.concatenate([block.reshape(requested_len, -1) for block in blocks], axis=-1)
        if window.shape != (requested_len, self.feature_metadata["num_features"]):
            raise ValueError(
                f"Constructed window has shape {window.shape}, expected "
                f"{(requested_len, self.feature_metadata['num_features'])}."
            )
        return window.astype(np.float32, copy=False)

    def _get_handle(self, source_id: int) -> h5py.File:
        handles = getattr(self._thread_local, "handles", None)
        if handles is None:
            handles = {}
            self._thread_local.handles = handles
            with self._thread_handles_lock:
                self._thread_handles[threading.get_ident()] = handles
        handle = handles.get(source_id)
        if handle is None:
            handle = h5py.File(self.sources[source_id].path, "r")
            handles[source_id] = handle
        return handle

    def close(self) -> None:
        # Construction can fail before thread-local handles are initialized.
        if not hasattr(self, "_thread_handles_lock"):
            return
        with self._thread_handles_lock:
            handle_sets = list(self._thread_handles.values())
            self._thread_handles.clear()
        for handles in handle_sets:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()


def _unwrap_forecast_dataset(dataset: Dataset, index: int) -> tuple[ForecastWindowDataset, int]:
    """Resolve a possibly capped Subset back to its source forecasting dataset."""
    current: Dataset = dataset
    current_index = int(index)
    while isinstance(current, Subset):
        if current_index < 0 or current_index >= len(current):
            raise IndexError(current_index)
        current_index = int(current.indices[current_index])
        current = current.dataset
    if not isinstance(current, ForecastWindowDataset):
        raise TypeError(
            "Rolling evaluation requires a ForecastWindowDataset or a Subset of one; "
            f"got {type(current).__name__}."
        )
    return current, current_index


def _dataset_record_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return trajectory IDs and starts in the index space of ``dataset``.

    The normal path is a ``ForecastWindowDataset`` and therefore avoids any
    Python-level iteration over all records.  The Subset fallback keeps the
    legacy caps compatible with the sampling helpers.
    """
    if isinstance(dataset, ForecastWindowDataset):
        return dataset.record_trajectories, dataset.record_starts
    if isinstance(dataset, Subset):
        trajectory_ids: list[int] = []
        starts: list[int] = []
        for index in range(len(dataset)):
            base, base_index = _unwrap_forecast_dataset(dataset, index)
            trajectory_id, start = base.window_record(base_index)
            trajectory_ids.append(trajectory_id)
            starts.append(start)
        return np.asarray(trajectory_ids, dtype=np.int64), np.asarray(starts, dtype=np.int64)
    if isinstance(dataset, DynamicTrajectoryWindowDataset):
        return _dataset_record_arrays(dataset.base)
    raise TypeError(
        "Window sampling requires a ForecastWindowDataset or a compatible Subset, "
        f"got {type(dataset).__name__}."
    )


def _select_position_indices(
    count: int,
    requested: int,
    target_positions: tuple[float, ...] | list[float] | None,
) -> np.ndarray:
    """Select deterministic approximately-uniform positions from a sorted list."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if requested <= 0 or requested >= count:
        return np.arange(count, dtype=np.int64)
    positions = np.asarray(
        target_positions if target_positions is not None else (),
        dtype=np.float64,
    ).reshape(-1)
    if positions.size != requested:
        positions = np.linspace(0.0, 1.0, requested, dtype=np.float64)
    positions = np.clip(positions, 0.0, 1.0)
    raw = np.rint(positions * (count - 1)).astype(np.int64)
    chosen: list[int] = []
    used: set[int] = set()
    for index in raw.tolist():
        if index not in used:
            chosen.append(index)
            used.add(index)
    if len(chosen) < requested:
        for index in range(count):
            if index not in used:
                chosen.append(index)
                used.add(index)
                if len(chosen) == requested:
                    break
    return np.asarray(sorted(chosen), dtype=np.int64)


def _record_indices_by_trajectory(trajectory_ids: np.ndarray) -> list[np.ndarray]:
    """Group record indices by trajectory using one global stable sort."""
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
    if not trajectory_ids.size:
        return []
    order = np.argsort(trajectory_ids, kind="stable")
    sorted_ids = trajectory_ids[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]).astype(np.int64) + 1,
            np.asarray([sorted_ids.size], dtype=np.int64),
        )
    )
    return [order[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]


def select_deterministic_windows(
    dataset: Dataset,
    windows_per_trajectory: int,
    target_positions: tuple[float, ...] = (0.10, 0.35, 0.65, 0.90),
) -> Subset | Dataset:
    """Select fixed coverage windows from every trajectory.

    A value of zero preserves all candidate windows for backwards-compatible
    full evaluation.  Candidate records are sorted by their start time, and
    the target positions are interpreted over that sorted candidate list.
    """
    if windows_per_trajectory < 0:
        raise ValueError("windows_per_trajectory must be non-negative.")
    if windows_per_trajectory == 0:
        return dataset
    trajectory_ids, starts = _dataset_record_arrays(dataset)
    selected: list[np.ndarray] = []
    for record_indices in _record_indices_by_trajectory(trajectory_ids):
        order = np.argsort(starts[record_indices], kind="stable")
        ordered_indices = record_indices[order]
        positions = _select_position_indices(
            ordered_indices.size,
            windows_per_trajectory,
            target_positions,
        )
        selected.append(ordered_indices[positions])
    if not selected:
        raise ValueError("No windows are available for deterministic trajectory sampling.")
    indices = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    return Subset(dataset, indices.tolist())


class DynamicTrajectoryWindowDataset(Dataset):
    """Per-epoch random windows with a fixed maximum per trajectory.

    This is intentionally trajectory-balanced rather than fully hierarchical:
    each eligible trajectory contributes at most ``windows_per_trajectory``
    windows in an epoch.  Because every trajectory belongs to exactly one base
    system, this removes the dominant long-trajectory weighting while keeping
    the existing dataset and system split unchanged.
    """

    def __init__(
        self,
        base: Dataset,
        windows_per_trajectory: int,
        seed: int,
    ) -> None:
        if windows_per_trajectory <= 0:
            raise ValueError("Dynamic sampling requires windows_per_trajectory > 0.")
        self.base = base
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        trajectory_ids, _ = _dataset_record_arrays(base)
        self._trajectory_groups = _record_indices_by_trajectory(trajectory_ids)
        if not self._trajectory_groups:
            raise ValueError("No candidate windows are available for dynamic sampling.")
        self.selected_indices = np.empty(0, dtype=np.int64)
        self.epoch = -1
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.epoch)
        selected: list[np.ndarray] = []
        for candidate_indices in self._trajectory_groups:
            count = min(self.windows_per_trajectory, candidate_indices.size)
            if count == candidate_indices.size:
                chosen = candidate_indices
            else:
                chosen = np.sort(rng.choice(candidate_indices, size=count, replace=False))
            selected.append(np.asarray(chosen, dtype=np.int64))
        self.selected_indices = np.concatenate(selected).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.selected_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.base[int(self.selected_indices[index])]


def _rolling_record_candidates(
    dataset: Dataset,
    context_len: int,
    block_len: int,
    num_blocks: int,
    windows_per_trajectory: int,
    target_positions: tuple[float, ...],
) -> list[tuple[ForecastWindowDataset, int, int]]:
    if context_len <= 0 or block_len <= 0 or num_blocks <= 0:
        raise ValueError("Rolling lengths and num_blocks must be positive.")
    trajectory_ids, _ = _dataset_record_arrays(dataset)
    base_by_trajectory: dict[int, ForecastWindowDataset] = {}
    for index, trajectory_id in enumerate(trajectory_ids.tolist()):
        if trajectory_id not in base_by_trajectory:
            base, _ = _unwrap_forecast_dataset(dataset, index)
            base_by_trajectory[int(trajectory_id)] = base

    total_len = int(context_len + block_len * num_blocks)
    records: list[tuple[ForecastWindowDataset, int, int]] = []
    for trajectory_id in sorted(base_by_trajectory):
        base = base_by_trajectory[trajectory_id]
        available = min(int(base.valid_lengths[trajectory_id]), base.sources[0].time_steps)
        stride = max(int(getattr(base, "stride", 1)), 1)
        starts = np.arange(0, available - total_len + 1, stride, dtype=np.int64)
        positions = _select_position_indices(
            starts.size,
            windows_per_trajectory,
            target_positions,
        )
        for start in starts[positions].tolist():
            records.append((base, int(trajectory_id), int(start)))
    return records


def count_rolling_window_candidates(
    dataset: Dataset,
    context_len: int,
    block_len: int,
    num_blocks: int,
    windows_per_trajectory: int,
    target_positions: tuple[float, ...] = (0.25, 0.75),
) -> int:
    """Count rolling starts before the optional finite-value check."""
    return len(
        _rolling_record_candidates(
            dataset,
            context_len,
            block_len,
            num_blocks,
            windows_per_trajectory,
            target_positions,
        )
    )


def _load_rolling_raw(
    base: ForecastWindowDataset,
    global_idx: int,
    start: int,
    total_len: int,
) -> np.ndarray:
    # ForecastWindowDataset can load arbitrary starts through its private
    # low-level reader.  Small synthetic test doubles may only implement the
    # public record-based loader, so retain that fallback.
    if "groups" in getattr(base, "feature_metadata", {}) and hasattr(base, "source_ids"):
        return base._load_window(global_idx, start, total_len)
    return base.load_raw_window(0, total_len=total_len)


class RollingWindowDataset(Dataset):
    """Finite test windows with enough future for recursive block rollout.

    Rolling starts are generated from the already prepared test trajectory
    pool. This keeps the system split and train-only scaler identical to E1,
    while allowing E2 to use its own deterministic 25%/75% start positions.
    Starts that cannot provide the complete requested future prefix or contain
    non-finite values are removed.
    """

    def __init__(
        self,
        dataset: Dataset,
        context_len: int,
        block_len: int,
        num_blocks: int,
        windows_per_trajectory: int = 0,
        target_positions: tuple[float, ...] = (0.25, 0.75),
    ) -> None:
        self.context_len = int(context_len)
        self.block_len = int(block_len)
        self.num_blocks = int(num_blocks)
        if self.context_len <= 0 or self.block_len <= 0 or self.num_blocks <= 0:
            raise ValueError(
                "context_len, block_len, and num_blocks must be positive for rolling evaluation."
            )
        self.total_len = self.context_len + self.block_len * self.num_blocks
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.target_positions = tuple(float(value) for value in target_positions)
        if self.windows_per_trajectory < 0:
            raise ValueError("windows_per_trajectory must be non-negative.")
        candidate_records = _rolling_record_candidates(
            dataset,
            context_len,
            block_len,
            num_blocks,
            self.windows_per_trajectory,
            self.target_positions,
        )
        self.candidate_count = len(candidate_records)
        self.records: list[tuple[ForecastWindowDataset, int, int]] = []
        for base, global_idx, start in candidate_records:
            available = min(int(base.valid_lengths[global_idx]), base.sources[0].time_steps)
            if start + self.total_len > available:
                continue
            raw = _load_rolling_raw(base, global_idx, start, self.total_len)
            if np.isfinite(raw).all():
                self.records.append((base, global_idx, start))
        if not self.records:
            raise ValueError(
                "No test windows have a complete finite future for the requested rolling horizon "
                f"({self.block_len * self.num_blocks} steps)."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base, global_idx, start = self.records[index]
        raw = _load_rolling_raw(base, global_idx, start, self.total_len)
        normalized = base.scaler.transform(raw)
        context = normalized[: self.context_len]
        target = normalized[self.context_len :]
        if context.shape != (self.context_len, base.feature_metadata["num_features"]):
            raise ValueError(f"Unexpected rolling context shape {context.shape}.")
        if target.shape != (self.block_len * self.num_blocks, base.feature_metadata["num_features"]):
            raise ValueError(f"Unexpected rolling target shape {target.shape}.")
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "trajectory_index": torch.tensor(global_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


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
    num_workers: int = 1,
    verbose: bool = True,
) -> PreparedForecastData:
    representation = normalize_representation(representation)
    if context_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("context_len, pred_len, and stride must be positive.")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Expected train_frac > 0, val_frac > 0, and train_frac + val_frac < 1.")
    if window_sampling not in {"random", "sequential"}:
        raise ValueError("window_sampling must be random or sequential.")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")

    paths = [Path(path) for path in sorted(glob.glob(data_glob))]
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched --data_glob={data_glob!r}")
    if verbose:
        print(f"[data] matched {len(paths)} HDF5 shards for {data_glob!r}", flush=True)
        if num_workers > 1:
            print(f"[data] using {num_workers} worker threads for window preparation", flush=True)
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
        num_workers=num_workers,
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
        num_workers=num_workers,
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
            source_time_steps = int(handle.attrs.get("source_time_steps", time_grid.size))
            source_time_end_attr = handle.attrs.get("source_time_end")
        finite_time = time_grid[np.isfinite(time_grid)]
        if finite_time.size == 0:
            raise ValueError(f"No finite time grid in {source.path}")
        prefix_time_end = float(finite_time[-1])
        source_time_end = prefix_time_end
        source_time_end_is_valid = False
        if source_time_end_attr is not None:
            candidate_end = float(np.asarray(source_time_end_attr).reshape(-1)[0])
            # A full-source endpoint cannot precede the endpoint already
            # present in a truncated prefix.  Older sampling_3 prefix files
            # contain a corrupt value here because the downloader used the
            # wrong byte stride; reject it and reconstruct the endpoint below.
            if np.isfinite(candidate_end) and candidate_end >= prefix_time_end:
                source_time_end = candidate_end
                source_time_end_is_valid = True
        if not source_time_end_is_valid and source_time_steps > time_grid.size and finite_time.size >= 2:
            # Prefix files retain the original number of time steps in an
            # attribute; reconstruct the full endpoint instead of scaling from
            # the truncated prefix endpoint.
            deltas = np.diff(finite_time)
            finite_deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
            if finite_deltas.size:
                source_time_end = float(finite_time[0] + np.median(finite_deltas) * (source_time_steps - 1))
        time_scale = float(source_time_end / completion_horizon) if completion_horizon > 0 else 1.0
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
    num_workers: int = 1,
    verbose: bool = True,
) -> StandardScaler:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")
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
    def accumulate_chunk(record_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        chunk_sums = np.zeros(channels, dtype=np.float64)
        chunk_sumsq = np.zeros(channels, dtype=np.float64)
        chunk_counts = np.zeros(channels, dtype=np.int64)
        chunk_dropped = 0
        for record_index in record_indices:
            values = dataset.load_raw_window(int(record_index)).reshape(-1, channels)
            if not np.isfinite(values).all():
                chunk_dropped += 1
                continue
            chunk_sums += values.sum(axis=0, dtype=np.float64)
            chunk_sumsq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
            chunk_counts += values.shape[0]
        return chunk_sums, chunk_sumsq, chunk_counts, chunk_dropped

    chunks = _work_chunks(selected_records, num_workers)
    if len(chunks) <= 1:
        partials = [accumulate_chunk(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            partials = list(executor.map(accumulate_chunk, chunks))
    for chunk_sums, chunk_sumsq, chunk_counts, chunk_dropped in partials:
        sums += chunk_sums
        sumsq += chunk_sumsq
        counts += chunk_counts
        dropped_windows += chunk_dropped
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
