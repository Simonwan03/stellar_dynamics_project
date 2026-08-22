"""Reproducible system-level train/validation/test splits.

Task A, Task B, and Task C all observe the same hierarchical data: one
``raw_case_id`` contains a base system and its perturbation trajectories.  The
split must therefore be made once at that level, rather than independently at
the case or trajectory level for each task.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_SPLIT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_COLUMNS = (
    "raw_case_id",
    "stable_count",
    "stable_fraction",
    "stable_fraction_bin",
    "split",
)


def default_case_split_manifest_path(
    dataset_dir: str | Path,
    sampling_id: int,
    group_size: int,
    time_horizon: float,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Path:
    """Return the conventional path shared by all task entry points."""

    ratio_tag = "-".join(str(int(round(ratio * 100))) for ratio in (train_ratio, val_ratio, test_ratio))
    horizon_tag = f"{float(time_horizon):g}".replace(".", "p")
    return Path(dataset_dir) / (
        f"common_case_split_sampling_{int(sampling_id)}_g{int(group_size)}_"
        f"h{horizon_tag}_r{ratio_tag}_seed{int(seed)}.csv"
    )


def build_case_catalog(
    labels_path: str | Path,
    group_size: int = 32,
    time_horizon: float = 10000.0,
) -> pd.DataFrame:
    """Build the canonical complete-system catalog from a completion table.

    A case is eligible only when all ``group_size`` variation IDs are present
    among rows with ``time_unbound >= 0`` and ``diagnosis == 0``.  This is the
    common 4998-system universe for dataset_v3; incomplete systems are never
    silently assigned to one task's split.
    """

    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Completion table does not exist: {labels_path}")
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    return build_case_catalog_from_labels(labels, group_size=group_size, time_horizon=time_horizon)


def build_case_catalog_from_labels(
    labels: pd.DataFrame,
    group_size: int = 32,
    time_horizon: float = 10000.0,
) -> pd.DataFrame:
    """Build a canonical case catalog from an already loaded label frame."""

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if not np.isfinite(time_horizon) or time_horizon < 0.0:
        raise ValueError(f"time_horizon must be finite and non-negative, got {time_horizon}.")
    required = {"index", "time_unbound", "diagnosis"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels are missing required columns: {missing}")

    work = labels[["index", "time_unbound", "diagnosis"]].copy()
    work["index"] = pd.to_numeric(work["index"], errors="raise").astype(np.int64)
    work["time_unbound"] = pd.to_numeric(work["time_unbound"], errors="raise").astype(np.float64)
    work["diagnosis"] = pd.to_numeric(work["diagnosis"], errors="raise").astype(np.int64)
    work["raw_case_id"] = work["index"] // int(group_size)
    work["variation_id"] = work["index"] % int(group_size)
    valid = work[(work["time_unbound"] >= 0.0) & (work["diagnosis"] == 0)].copy()

    group_info = valid.groupby("raw_case_id", sort=True)["variation_id"].agg(
        count="size", unique="nunique", minimum="min", maximum="max"
    )
    complete_mask = (
        (group_info["count"] == group_size)
        & (group_info["unique"] == group_size)
        & (group_info["minimum"] == 0)
        & (group_info["maximum"] == group_size - 1)
    )
    complete_ids = group_info.index[complete_mask]
    if complete_ids.empty:
        raise ValueError("No complete valid systems were found in the completion table.")

    complete = valid[valid["raw_case_id"].isin(complete_ids)].copy()
    complete["stable_at_horizon"] = (complete["time_unbound"] >= float(time_horizon)).astype(np.int64)
    catalog = (
        complete.groupby("raw_case_id", sort=True)
        .agg(stable_count=("stable_at_horizon", "sum"), valid_variations=("stable_at_horizon", "size"))
        .reset_index()
    )
    if not np.all(catalog["valid_variations"].to_numpy(dtype=np.int64) == group_size):
        raise RuntimeError("The complete-system catalog contains a group with an unexpected variation count.")
    catalog["stable_count"] = catalog["stable_count"].astype(np.int64)
    catalog["stable_fraction"] = catalog["stable_count"].to_numpy(dtype=np.float64) / float(group_size)
    catalog["stable_fraction_bin"] = catalog["stable_fraction"].map(stable_fraction_bin).astype(np.int64)
    return catalog[["raw_case_id", "stable_count", "stable_fraction", "stable_fraction_bin"]]


def stable_fraction_bin(value: float) -> int:
    """Return the fixed coarse stratum used by the common split.

    The endpoint systems are kept in their own strata because they are the
    positive/negative tails of Task A's binary target.  Intermediate bins are
    broad enough to remain well populated while matching the final-horizon
    target used by Task C.
    """

    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"stable_fraction must lie in [0, 1], got {value}.")
    if value <= 0.0:
        return 0
    if value <= 0.25:
        return 1
    if value <= 0.50:
        return 2
    if value < 1.0:
        return 3
    return 4


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> np.ndarray:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError(f"Split ratios must be finite and positive, got {ratios.tolist()}.")
    if not math.isclose(float(ratios.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios.sum():g}.")
    return ratios


def _global_split_counts(total: int, ratios: np.ndarray) -> np.ndarray:
    if total < len(SPLIT_NAMES):
        raise ValueError(f"Need at least {len(SPLIT_NAMES)} systems, got {total}.")

    # Use largest remainder so the requested proportions are respected even
    # when the system count is not divisible by three.  The final correction
    # only matters for very small synthetic datasets and guarantees that each
    # split remains non-empty.
    quotas = total * ratios
    counts = np.floor(quotas).astype(np.int64)
    remainder = int(total - counts.sum())
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:remainder]] += 1
    for missing_split in np.flatnonzero(counts == 0):
        donors = np.argsort(-counts, kind="stable")
        donor = next((idx for idx in donors if counts[idx] > 1), None)
        if donor is None:
            raise RuntimeError(f"Could not make all splits non-empty: {counts.tolist()}")
        counts[donor] -= 1
        counts[missing_split] += 1
    if int(counts.sum()) != total:
        raise RuntimeError(f"Could not apportion {total} systems across splits: {counts.tolist()}")
    return counts


def _allocate_stratum_counts(stratum_sizes: np.ndarray, target: int) -> np.ndarray:
    """Apportion an exact split count across strata by largest remainder."""

    sizes = np.asarray(stratum_sizes, dtype=np.int64)
    if target < 0 or target > int(sizes.sum()):
        raise ValueError(f"Cannot allocate target={target} from {int(sizes.sum())} remaining systems.")
    if target == 0:
        return np.zeros_like(sizes)
    quotas = sizes.astype(np.float64) * float(target) / float(sizes.sum())
    counts = np.floor(quotas).astype(np.int64)
    remainder = int(target - counts.sum())
    order = np.argsort(-(quotas - counts), kind="stable")
    for stratum_idx in order:
        if remainder == 0:
            break
        if counts[stratum_idx] < sizes[stratum_idx]:
            counts[stratum_idx] += 1
            remainder -= 1
    if remainder:
        for stratum_idx in np.flatnonzero(counts < sizes):
            if remainder == 0:
                break
            take = min(remainder, int(sizes[stratum_idx] - counts[stratum_idx]))
            counts[stratum_idx] += take
            remainder -= take
    if remainder:
        raise RuntimeError("Could not apportion the requested count across strata.")
    return counts


def create_case_split_manifest(
    catalog: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign every catalog system to one deterministic split."""

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    required = {"raw_case_id", "stable_count", "stable_fraction", "stable_fraction_bin"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"Case catalog is missing required columns: {missing}")
    result = catalog[list(required)].copy()
    if result["raw_case_id"].duplicated().any():
        raise ValueError("Case catalog contains duplicate raw_case_id values.")
    result["raw_case_id"] = result["raw_case_id"].astype(np.int64)
    result["stable_fraction_bin"] = result["stable_fraction_bin"].astype(np.int64)
    result = result.sort_values("raw_case_id").reset_index(drop=True)

    global_counts = _global_split_counts(len(result), ratios)
    bin_values = np.sort(result["stable_fraction_bin"].unique())
    stratum_rows = [np.flatnonzero(result["stable_fraction_bin"].to_numpy() == value) for value in bin_values]
    stratum_sizes = np.asarray([rows.size for rows in stratum_rows], dtype=np.int64)

    test_counts = _allocate_stratum_counts(stratum_sizes, int(global_counts[2]))
    remaining_sizes = stratum_sizes - test_counts
    val_counts = _allocate_stratum_counts(remaining_sizes, int(global_counts[1]))

    rng = np.random.default_rng(seed)
    split = np.empty(len(result), dtype=object)
    for rows, n_val, n_test in zip(stratum_rows, val_counts, test_counts):
        shuffled = rows.copy()
        rng.shuffle(shuffled)
        split[shuffled[:n_test]] = "test"
        split[shuffled[n_test : n_test + n_val]] = "val"
        split[shuffled[n_test + n_val :]] = "train"
    result["split"] = split
    counts = result["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0).to_numpy(dtype=np.int64)
    if not np.array_equal(counts, global_counts):
        raise RuntimeError(f"Split counts {counts.tolist()} do not match target {global_counts.tolist()}.")
    return result[list(MANIFEST_COLUMNS)]


def ensure_case_split_manifest(
    labels_path: str | Path,
    manifest_path: str | Path,
    *,
    group_size: int = 32,
    time_horizon: float = 10000.0,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load a validated manifest or create it from the canonical label table."""

    manifest_path = Path(manifest_path)
    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    catalog = build_case_catalog(labels_path, group_size=group_size, time_horizon=time_horizon)
    if manifest_path.exists() and not overwrite:
        manifest = pd.read_csv(manifest_path)
        _validate_manifest(manifest, catalog, ratios)
        _validate_manifest_metadata(
            manifest_path,
            group_size=group_size,
            time_horizon=time_horizon,
            ratios=ratios,
        )
        return manifest[list(MANIFEST_COLUMNS)]

    manifest = create_case_split_manifest(
        catalog,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "manifest_version": CASE_SPLIT_VERSION,
        "group_size": int(group_size),
        "time_horizon": float(time_horizon),
        "seed": int(seed),
        "ratios": {name: float(value) for name, value in zip(SPLIT_NAMES, ratios)},
        "num_cases": int(len(manifest)),
        "split_counts": {
            name: int((manifest["split"] == name).sum()) for name in SPLIT_NAMES
        },
        "labels_path": str(Path(labels_path)),
    }
    _metadata_path(manifest_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_case_split_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Case split manifest does not exist: {path}")
    manifest = pd.read_csv(path)
    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Case split manifest is missing columns: {missing}")
    return manifest[list(MANIFEST_COLUMNS)]


def case_split_membership(sample_case_ids: np.ndarray, manifest: pd.DataFrame) -> np.ndarray:
    """Map arbitrary sample rows to their system-level split labels."""

    sample_case_ids = np.asarray(sample_case_ids).reshape(-1)
    if sample_case_ids.size == 0:
        return np.empty(0, dtype=object)
    try:
        sample_case_ids = sample_case_ids.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_case_ids must contain integer raw_case_id values.") from exc
    if manifest["raw_case_id"].duplicated().any():
        raise ValueError("Case split manifest contains duplicate raw_case_id values.")
    mapping = dict(
        zip(
            manifest["raw_case_id"].to_numpy(dtype=np.int64),
            manifest["split"].astype(str).to_numpy(),
        )
    )
    unknown = np.unique([case_id for case_id in sample_case_ids if int(case_id) not in mapping])
    if unknown.size:
        raise ValueError(
            f"Found {unknown.size} sample case IDs absent from the common manifest; "
            f"first IDs: {unknown[:10].tolist()}"
        )
    return np.asarray([mapping[int(case_id)] for case_id in sample_case_ids], dtype=object)


def case_split_indices(sample_case_ids: np.ndarray, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    membership = case_split_membership(sample_case_ids, manifest)
    return {
        name: np.flatnonzero(membership == name).astype(np.int64)
        for name in SPLIT_NAMES
    }


def missing_case_ids(sample_case_ids: np.ndarray, manifest: pd.DataFrame) -> np.ndarray:
    """Return canonical manifest systems absent from a task's loaded samples."""

    sample_case_ids = np.asarray(sample_case_ids, dtype=np.int64).reshape(-1)
    manifest_case_ids = manifest["raw_case_id"].to_numpy(dtype=np.int64)
    return np.setdiff1d(manifest_case_ids, np.unique(sample_case_ids), assume_unique=False)


def _metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".json")


def _validate_manifest(manifest: pd.DataFrame, catalog: pd.DataFrame, ratios: np.ndarray) -> None:
    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Case split manifest is missing columns: {missing}")
    actual = manifest[list(MANIFEST_COLUMNS)].copy()
    if actual["raw_case_id"].duplicated().any():
        raise ValueError("Case split manifest contains duplicate raw_case_id values.")
    expected = catalog.sort_values("raw_case_id").reset_index(drop=True)
    actual = actual.sort_values("raw_case_id").reset_index(drop=True)
    if not np.array_equal(
        actual["raw_case_id"].to_numpy(dtype=np.int64),
        expected["raw_case_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Case split manifest does not cover exactly the canonical complete-system universe.")
    for column in ("stable_count", "stable_fraction_bin"):
        if not np.array_equal(actual[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"Case split manifest column '{column}' disagrees with the completion table.")
    if not np.allclose(
        actual["stable_fraction"].to_numpy(dtype=np.float64),
        expected["stable_fraction"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Case split manifest stable_fraction values disagree with the completion table.")
    if not actual["split"].isin(SPLIT_NAMES).all():
        raise ValueError(f"Case split manifest contains unsupported split labels: {actual['split'].unique().tolist()}")
    expected_counts = _global_split_counts(len(expected), ratios)
    actual_counts = actual["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0).to_numpy(dtype=np.int64)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError(
            f"Case split manifest counts {actual_counts.tolist()} do not match requested ratios "
            f"{ratios.tolist()} (expected {expected_counts.tolist()})."
        )


def _validate_manifest_metadata(
    manifest_path: Path,
    *,
    group_size: int,
    time_horizon: float,
    ratios: np.ndarray,
) -> None:
    metadata_path = _metadata_path(manifest_path)
    if not metadata_path.exists():
        return
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read case split metadata: {metadata_path}") from exc
    if int(metadata.get("manifest_version", -1)) != CASE_SPLIT_VERSION:
        raise ValueError(f"Unsupported case split manifest version in {metadata_path}.")
    if int(metadata.get("group_size", -1)) != int(group_size):
        raise ValueError("Existing case split manifest uses a different group_size.")
    if not math.isclose(float(metadata.get("time_horizon", float("nan"))), float(time_horizon), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Existing case split manifest uses a different time_horizon.")
    stored_ratios = metadata.get("ratios", {})
    for name, expected in zip(SPLIT_NAMES, ratios):
        if not math.isclose(float(stored_ratios.get(name, float("nan"))), float(expected), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Existing case split manifest uses different split ratios.")
