from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from tasks.task_b.data import (
    build_prefix_features,
    discover_result_paths,
    load_aligned_labels,
    load_result_prefixes,
)


CACHE_VERSION = 1


def _metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def _cache_files(cache_dir: Path) -> tuple[Path, ...]:
    return (
        cache_dir / "features.npy",
        cache_dir / "prefix_times.npy",
        cache_dir / "label_index.npy",
        cache_dir / "label_time_unbound.npy",
        cache_dir / "label_diagnosis.npy",
    )


def _num_steps(prefix_len: int, time_stride: int) -> int:
    if prefix_len < 1:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}.")
    if time_stride < 1:
        raise ValueError(f"time_stride must be positive, got {time_stride}.")
    return len(range(0, prefix_len, time_stride))


def _cache_matches(
    metadata: dict[str, Any],
    *,
    result_paths: list[Path],
    sampling_id: int,
    max_prefix_len: int,
    time_stride: int,
    result_keys: tuple[str, ...],
    include_geom: bool,
    rows_per_shard: int | None,
) -> bool:
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and int(metadata.get("sampling_id", -1)) == sampling_id
        and int(metadata.get("max_prefix_len", 0)) >= max_prefix_len
        and int(metadata.get("time_stride", 0)) == time_stride
        and tuple(metadata.get("result_keys", ())) == tuple(result_keys)
        and bool(metadata.get("include_geom", False)) == include_geom
        and metadata.get("rows_per_shard") == rows_per_shard
        and metadata.get("result_paths", []) == [str(path) for path in result_paths]
        and all(path.exists() for path in _cache_files(Path(metadata["cache_dir"])))
    )


def prepare_prefix_cache(
    cache_dir: str | Path,
    *,
    dataset_dir: str | Path,
    sampling_id: int,
    max_prefix_len: int,
    time_stride: int,
    result_keys: tuple[str, ...],
    include_geom: bool,
    max_shards: int | None = None,
    rows_per_shard: int | None = None,
    force: bool = False,
) -> Path:
    """Materialize one shared prefix cache for a whole ablation grid.

    The cache deliberately stores features without the optional normalized time
    channel.  Raw prefix times are stored separately so the caller can append
    a time channel with the normalization appropriate to each prefix length.
    """
    cache_dir = Path(cache_dir)
    dataset_dir = Path(dataset_dir)
    result_paths = discover_result_paths(dataset_dir, sampling_id, max_shards)
    metadata_path = _metadata_path(cache_dir)
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        metadata["cache_dir"] = str(cache_dir)
        if _cache_matches(
            metadata,
            result_paths=result_paths,
            sampling_id=sampling_id,
            max_prefix_len=max_prefix_len,
            time_stride=time_stride,
            result_keys=result_keys,
            include_geom=include_geom,
            rows_per_shard=rows_per_shard,
        ):
            print(f"Using prefix cache: {cache_dir}")
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    n_steps = _num_steps(max_prefix_len, time_stride)

    rows_per_path: list[int] = []
    for path in result_paths:
        with h5py.File(path, "r") as handle:
            row_count = int(handle["times_complete"].shape[0])
        if rows_per_shard is not None:
            row_count = min(row_count, rows_per_shard)
        rows_per_path.append(row_count)
    total_rows = sum(rows_per_path)
    if total_rows == 0:
        raise ValueError("The selected result shards contain no rows.")

    feature_memmap: np.memmap | None = None
    time_memmap = np.lib.format.open_memmap(
        cache_dir / "prefix_times.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, n_steps),
    )
    batch_ids: list[int] = []
    rows_loaded: list[int] = []
    feature_names: list[str] | None = None
    offset = 0

    for path, expected_rows in zip(result_paths, rows_per_path):
        data, shard_batch_ids, shard_rows_loaded = load_result_prefixes(
            [path],
            result_keys=result_keys,
            prefix_len=max_prefix_len,
            time_stride=time_stride,
            rows_per_shard=rows_per_shard,
            include_geom=include_geom,
            add_time=True,
        )
        X_shard, shard_feature_names = build_prefix_features(
            data,
            result_keys=result_keys,
            add_time=False,
            include_geom=include_geom,
        )
        if X_shard.shape[0] != expected_rows:
            raise ValueError(f"Cache row-count mismatch for {path}: expected {expected_rows}, got {X_shard.shape[0]}.")
        if feature_memmap is None:
            feature_memmap = np.lib.format.open_memmap(
                cache_dir / "features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, X_shard.shape[1], X_shard.shape[2]),
            )
            feature_names = shard_feature_names
        elif shard_feature_names != feature_names or X_shard.shape[1:] != feature_memmap.shape[1:]:
            raise ValueError(f"Inconsistent feature shape or names in result shard {path}.")

        end = offset + expected_rows
        feature_memmap[offset:end] = X_shard
        time_memmap[offset:end] = data["times_complete"]
        offset = end
        batch_ids.extend(shard_batch_ids)
        rows_loaded.extend(shard_rows_loaded)
        del data, X_shard
        gc.collect()

    if feature_memmap is None or feature_names is None:
        raise ValueError("Could not create prefix feature cache.")
    feature_memmap.flush()
    time_memmap.flush()
    del feature_memmap, time_memmap

    labels = load_aligned_labels(
        dataset_dir / f"completed_systems_sampling_{sampling_id}.txt",
        batch_ids,
        rows_loaded,
    )
    np.save(cache_dir / "label_index.npy", labels["index"].to_numpy(dtype=np.int64))
    np.save(cache_dir / "label_time_unbound.npy", labels["time_unbound"].to_numpy(dtype=np.float32))
    np.save(cache_dir / "label_diagnosis.npy", labels["diagnosis"].to_numpy(dtype=np.int64))

    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_dir": str(cache_dir),
        "dataset_dir": str(dataset_dir),
        "sampling_id": sampling_id,
        "result_paths": [str(path) for path in result_paths],
        "max_prefix_len": max_prefix_len,
        "time_stride": time_stride,
        "num_steps": n_steps,
        "result_keys": list(result_keys),
        "include_geom": include_geom,
        "rows_per_shard": rows_per_shard,
        "num_rows": total_rows,
        "feature_names": feature_names,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Built prefix cache: {cache_dir} ({total_rows} rows, {n_steps} steps)")
    return cache_dir


def load_cached_prefixes(
    cache_dir: str | Path,
    *,
    prefix_len: int,
    time_stride: int,
    add_time: bool,
    result_keys: tuple[str, ...],
    include_geom: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[Path], np.ndarray, list[str]]:
    """Load one prefix view from a materialized cache."""
    cache_dir = Path(cache_dir)
    metadata = json.loads(_metadata_path(cache_dir).read_text())
    if not all(path.exists() for path in _cache_files(cache_dir)):
        raise FileNotFoundError(f"Prefix cache is incomplete: {cache_dir}")
    if int(metadata.get("time_stride", 0)) != time_stride:
        raise ValueError("The cache time_stride does not match the current experiment.")
    if tuple(metadata.get("result_keys", ())) != tuple(result_keys):
        raise ValueError("The cache result representation does not match the current experiment.")
    if bool(metadata.get("include_geom", False)) != include_geom:
        raise ValueError("The cache geometry setting does not match the current experiment.")
    if int(metadata.get("max_prefix_len", 0)) < prefix_len:
        raise ValueError(
            f"Prefix cache only covers up to {metadata.get('max_prefix_len')} steps, requested {prefix_len}."
        )

    n_steps = _num_steps(prefix_len, time_stride)
    features = np.load(cache_dir / "features.npy", mmap_mode="r")[:, :n_steps]
    prefix_times = np.load(cache_dir / "prefix_times.npy", mmap_mode="r")[:, :n_steps]
    feature_names = list(metadata["feature_names"])
    if add_time:
        denom = max(float(np.nanmax(prefix_times)), 1.0)
        time_channel = (np.asarray(prefix_times, dtype=np.float32) / denom)[..., None]
        X = np.concatenate([np.asarray(features, dtype=np.float32), time_channel], axis=-1)
        feature_names.append("time_norm")
    else:
        X = features

    labels = pd.DataFrame(
        {
            "index": np.load(cache_dir / "label_index.npy"),
            "time_unbound": np.load(cache_dir / "label_time_unbound.npy"),
            "diagnosis": np.load(cache_dir / "label_diagnosis.npy"),
        }
    )
    data = {"times_complete": prefix_times}
    result_paths = [Path(path) for path in metadata["result_paths"]]
    return data, labels, result_paths, X, feature_names
