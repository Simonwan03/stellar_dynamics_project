from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    CACHE_FILES,
    CACHE_SCHEMA_VERSION,
    ForecastWindowDataset,
    PreparedForecastData,
    StandardScaler,
    build_source_lookup,
    inspect_sources,
    normalize_representation,
)


def _resolve_data_paths(data_glob: str) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in sorted(glob.glob(str(data_glob)))]
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched --data_glob={data_glob!r}")
    return paths


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_preparation_cache_signature(
    data_glob: str,
    representation: str,
    context_len: int,
    pred_len: int,
    stride: int,
    split_seed: int,
    train_frac: float,
    val_frac: float,
    group_size: int,
    completion_table: str | Path | None,
    require_complete_systems: bool,
    max_trajectories: int,
    max_variations_per_system: int,
    max_train_variations_per_system: int,
    max_windows_per_trajectory: int,
    window_sampling: str,
    drop_nan_windows: bool,
    max_scaler_trajectories: int,
) -> dict[str, Any]:
    """Build the exact input/config signature used to create a prepared cache."""
    paths = _resolve_data_paths(data_glob)
    completion_path = None
    completion_file = None
    if completion_table is not None:
        completion_path = Path(completion_table).expanduser().resolve()
        completion_file = _file_signature(completion_path)
    return {
        "source_paths": [str(path) for path in paths],
        "source_files": [_file_signature(path) for path in paths],
        "representation": normalize_representation(representation),
        "context_len": int(context_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "split_seed": int(split_seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "group_size": int(group_size),
        "completion_table": None if completion_path is None else str(completion_path),
        "completion_file": completion_file,
        "require_complete_systems": bool(require_complete_systems),
        "max_trajectories": int(max_trajectories),
        "max_variations_per_system": int(max_variations_per_system),
        "max_train_variations_per_system": int(max_train_variations_per_system),
        "max_windows_per_trajectory": int(max_windows_per_trajectory),
        "window_sampling": str(window_sampling),
        "drop_nan_windows": bool(drop_nan_windows),
        "max_scaler_trajectories": int(max_scaler_trajectories),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _cache_paths(cache_dir: str | Path) -> dict[str, Path]:
    directory = Path(cache_dir)
    return {name: directory / name for name in CACHE_FILES}


def save_prepared_cache(
    prepared: PreparedForecastData,
    cache_dir: str | Path,
    preparation: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Persist window records, scaler, metadata, and a source/config manifest."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and not cache_dir.is_dir():
        raise NotADirectoryError(f"Cache path is not a directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir)
    existing = [path.name for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Prepared cache already contains {existing}; pass --overwrite to replace it: {cache_dir}"
        )

    metadata = dict(prepared.feature_metadata)
    feature_names = [str(name) for name in metadata.get("channel_names", [])]
    if not feature_names:
        raise ValueError("Prepared feature metadata has no channel_names.")
    if len(feature_names) != prepared.scaler.mean.size:
        raise ValueError(
            "Feature/scaler size mismatch while saving cache: "
            f"{len(feature_names)} vs {prepared.scaler.mean.size}"
        )

    split_indices = {
        name: np.asarray(prepared.split_indices[name], dtype=np.int64).reshape(-1)
        for name in ("train", "val", "test")
    }
    valid_lengths = np.asarray(prepared.train_dataset.valid_lengths, dtype=np.int64).reshape(-1)
    for name, dataset in (
        ("train", prepared.train_dataset),
        ("val", prepared.val_dataset),
        ("test", prepared.test_dataset),
    ):
        if not np.array_equal(valid_lengths, dataset.valid_lengths):
            raise ValueError(f"Dataset valid_lengths differ while saving {name} cache records.")

    prepared_source_paths = [str(Path(path).expanduser().resolve()) for path in prepared.source_paths]
    if prepared_source_paths != list(preparation["source_paths"]):
        raise ValueError(
            "Prepared source paths do not match the cache signature. "
            f"prepared={prepared_source_paths[:2]} signature={preparation['source_paths'][:2]}"
        )

    arrays: dict[str, np.ndarray] = {"valid_lengths": valid_lengths}
    for name, dataset in (
        ("train", prepared.train_dataset),
        ("val", prepared.val_dataset),
        ("test", prepared.test_dataset),
    ):
        arrays[f"{name}_split_indices"] = split_indices[name]
        arrays[f"{name}_record_trajectories"] = np.asarray(dataset.record_trajectories, dtype=np.int64)
        arrays[f"{name}_record_starts"] = np.asarray(dataset.record_starts, dtype=np.int64)

    prepared.scaler.save(paths["scaler.npz"], feature_names)
    np.savez_compressed(paths["records.npz"], **arrays)
    with paths["feature_metadata.json"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=_json_default)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "preparation": preparation,
        "trajectory_count": int(prepared.trajectory_count),
        "time_steps": int(prepared.time_steps),
        "num_features": int(metadata["num_features"]),
        "feature_names": feature_names,
        "split_counts": {name: int(indices.size) for name, indices in split_indices.items()},
        "record_counts": {
            name: int(arrays[f"{name}_record_starts"].size) for name in ("train", "val", "test")
        },
    }
    with paths["manifest.json"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)


def _signature_mismatch(cached: dict[str, Any], requested: dict[str, Any]) -> str:
    lines = []
    for key in sorted(set(cached) | set(requested)):
        if cached.get(key) != requested.get(key):
            lines.append(f"  {key}: cached={cached.get(key)!r}, requested={requested.get(key)!r}")
    return "\n".join(lines)


def load_prepared_cache(
    cache_dir: str | Path,
    expected_preparation: dict[str, Any],
    verbose: bool = True,
) -> PreparedForecastData:
    """Load a prepared cache after validating its data files and preprocessing config."""
    paths = _cache_paths(cache_dir)
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared cache is missing {missing}: {cache_dir}")

    with paths["manifest.json"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prepared-cache schema {manifest.get('schema_version')}; "
            f"expected {CACHE_SCHEMA_VERSION}."
        )
    cached_preparation = manifest.get("preparation")
    if cached_preparation != expected_preparation:
        raise ValueError(
            "Prepared cache does not match the requested data/preprocessing settings:\n"
            + _signature_mismatch(cached_preparation or {}, expected_preparation)
        )

    representation = normalize_representation(cached_preparation["representation"])
    source_paths = [Path(path) for path in cached_preparation["source_paths"]]
    sources = inspect_sources(source_paths, representation)
    source_ids, local_ids = build_source_lookup(sources)
    with paths["feature_metadata.json"].open("r", encoding="utf-8") as handle:
        feature_metadata = json.load(handle)
    scaler, scaler_feature_names = StandardScaler.load(paths["scaler.npz"])
    feature_names = [str(name) for name in feature_metadata.get("channel_names", [])]
    if scaler_feature_names and scaler_feature_names != feature_names:
        raise ValueError("Prepared cache scaler channel names do not match feature metadata.")
    if int(feature_metadata.get("num_features", -1)) != scaler.mean.size:
        raise ValueError("Prepared cache feature metadata does not match scaler size.")

    with np.load(paths["records.npz"], allow_pickle=False) as payload:
        valid_lengths = np.asarray(payload["valid_lengths"], dtype=np.int64).reshape(-1)
        split_indices = {
            name: np.asarray(payload[f"{name}_split_indices"], dtype=np.int64).reshape(-1)
            for name in ("train", "val", "test")
        }
        records = {
            name: (
                np.asarray(payload[f"{name}_record_trajectories"], dtype=np.int64).reshape(-1),
                np.asarray(payload[f"{name}_record_starts"], dtype=np.int64).reshape(-1),
            )
            for name in ("train", "val", "test")
        }

    if valid_lengths.size != source_ids.size:
        raise ValueError(
            "Prepared cache valid_lengths size does not match source rows: "
            f"{valid_lengths.size} vs {source_ids.size}"
        )
    for name, indices in split_indices.items():
        if np.any(indices < 0) or np.any(indices >= valid_lengths.size):
            raise ValueError(f"Prepared cache {name} split contains invalid trajectory indices.")

    common = dict(
        sources=sources,
        source_ids=source_ids,
        local_ids=local_ids,
        representation=representation,
        feature_metadata=feature_metadata,
        scaler=scaler,
        context_len=int(cached_preparation["context_len"]),
        pred_len=int(cached_preparation["pred_len"]),
        stride=int(cached_preparation["stride"]),
        max_windows_per_trajectory=int(cached_preparation["max_windows_per_trajectory"]),
        window_sampling=str(cached_preparation["window_sampling"]),
        drop_nan_windows=bool(cached_preparation["drop_nan_windows"]),
        valid_lengths=valid_lengths,
    )
    datasets = {}
    for offset, name in enumerate(("train", "val", "test"), start=101):
        record_trajectories, record_starts = records[name]
        datasets[name] = ForecastWindowDataset(
            trajectory_indices=split_indices[name],
            seed=int(cached_preparation["split_seed"]) + offset,
            cached_record_trajectories=record_trajectories,
            cached_record_starts=record_starts,
            **common,
        )
    if verbose:
        print(
            f"[data] loaded prepared cache {Path(cache_dir)}: "
            f"train={len(datasets['train'])}, val={len(datasets['val'])}, "
            f"test={len(datasets['test'])}",
            flush=True,
        )
    return PreparedForecastData(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        scaler=scaler,
        feature_metadata=feature_metadata,
        split_indices=split_indices,
        source_paths=[str(source.path) for source in sources],
        trajectory_count=int(manifest["trajectory_count"]),
        time_steps=int(manifest["time_steps"]),
    )
