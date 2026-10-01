#!/usr/bin/env python3
"""Run the sampling3 fixed-budget phase-ensemble experiment.

The experiment uses the version-1 system split, an orbital15 cache, a shared
TCN + DeepSets regressor, and a per-trajectory TCN probability-averaging
baseline. The default protocol is the first B=512 budget only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from tasks.common.case_split import create_case_split_manifest, stable_fraction_bin
from tasks.common.experiment_utils import ensure_dir, set_random_seed
from tasks.task_b.data import batch_id_from_path, discover_result_paths, load_aligned_labels
from tasks.task_b.phase_ensemble import (
    BudgetCondition,
    PairDeepSetsRegressor,
    PhaseTCNClassifier,
    budget_conditions,
    build_pair_partition,
    common_preinstability_mask,
    nested_phase_draws,
    pair_ids_to_variations,
    prefix_endpoints_in_label_time,
)


GROUP_SIZE = 32
PAIR_COUNT = GROUP_SIZE // 2
PAIR_MEMBERS = np.column_stack((np.arange(PAIR_COUNT), np.arange(PAIR_COUNT) + PAIR_COUNT)).astype(np.int16)
LABEL_HORIZON = 10000.0
PAIR_COUNTS = (1, 2, 4, 8)
PREFIX_CHOICES = (32, 64, 128, 256, 512)
FEATURE_CHANNELS = 15
DEFAULT_AUDIT = ROOT / "results/shared_analysis/sampling_3/system_level_audit_sampling_3/system_level_statistics.csv"
DEFAULT_LABEL_FALLBACK = ROOT / "results/task_b/sampling_3/task_b_sampling3_prefix512/labels_filtered.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--label-fallback", type=Path, default=DEFAULT_LABEL_FALLBACK)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/comparisons/sampling_3/fixed_budget_sampling3_B512_pair_aware")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--phase-draws", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-systems", type=int, default=8)
    parser.add_argument("--eval-batch-systems", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true", help="Write split/phase metadata and validate inputs without training.")
    return parser


def resolve_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = (
        Path(os.environ["STELLAR_DATA_DIR"]).expanduser()
        if "STELLAR_DATA_DIR" in os.environ else ROOT / "dataset_sampling_3_prefix1024",
    )
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("results_batch_*_sampling_3.hdf5")):
            return candidate.resolve()
    # Keep the repository's conventional path in the preflight report; the
    # subsequent missing-shard error names the required files explicitly.
    return candidates[0].resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_cache_path(explicit: Path | None, dataset_dir: Path, output_dir: Path) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    cache_name = ".fixed_budget_sampling3_orbital15_prefix1024.f32"
    minimum_free_bytes = 11 * 1024**3
    for base in (dataset_dir, Path("/tmp"), output_dir):
        if not base.is_dir() or not os.access(base, os.W_OK):
            continue
        if shutil.disk_usage(base).free >= minimum_free_bytes:
            return (base / cache_name).resolve()
    raise OSError(
        "No writable filesystem has the approximately 10 GiB free space needed for the sampling3 "
        "orbital15 feature cache. Pass --cache-path on a volume with sufficient space."
    )


def rebuild_split_manifest(
    audit: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> pd.DataFrame:
    required = {"system_id", "stable_count", "p_stable", "stable_fraction_bin", "ma_margin"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"Sampling3 system audit is missing columns: {missing}")
    if audit["system_id"].duplicated().any():
        raise ValueError("Sampling3 system audit contains duplicate system_id values.")
    catalog = audit.rename(
        columns={"system_id": "raw_case_id", "p_stable": "stable_fraction"}
    )[["raw_case_id", "stable_count", "stable_fraction"]].copy()
    # The audit uses six display bins (including distinct p=0 and p=1 tails);
    # version-1 case splitting uses its own five-bin stratification. Recompute
    # the latter with the shared helper instead of reusing the audit bins.
    catalog["stable_fraction_bin"] = catalog["stable_fraction"].map(stable_fraction_bin).astype(np.int64)
    manifest = create_case_split_manifest(
        catalog,
        seed=seed,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
    )
    manifest_path = output_dir / (
        f"common_case_split_sampling_3_g{GROUP_SIZE}_h{LABEL_HORIZON:g}_"
        f"r{int(ratios[0] * 100)}-{int(ratios[1] * 100)}-{int(ratios[2] * 100)}_seed{seed}.csv"
    )
    manifest.to_csv(manifest_path, index=False)
    split_counts = {name: int((manifest["split"] == name).sum()) for name in ("train", "val", "test")}
    save_json(
        manifest_path.with_suffix(".json"),
        {
            "manifest_version": 1,
            "sampling_id": 3,
            "group_size": GROUP_SIZE,
            "time_horizon": LABEL_HORIZON,
            "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "num_cases": len(manifest),
            "split_counts": split_counts,
            "reconstructed_from": "results/shared_analysis/sampling_3/system_level_audit_sampling_3/system_level_statistics.csv",
        },
    )
    expected_counts = {"train": 3498, "val": 750, "test": 750}
    if len(manifest) != 4998 or split_counts != expected_counts:
        raise ValueError(f"Unexpected sampling3 version-1 split: systems={len(manifest)}, counts={split_counts}")

    prior = ROOT / "results/task_b/sampling_3/task_b_sampling3_prefix512/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv"
    if prior.is_file() and seed == 42 and np.allclose(ratios, (0.70, 0.15, 0.15)):
        prior_manifest = pd.read_csv(prior).sort_values("raw_case_id").reset_index(drop=True)
        current = manifest.sort_values("raw_case_id").reset_index(drop=True)
        if not np.array_equal(prior_manifest["raw_case_id"], current["raw_case_id"]):
            raise ValueError("Rebuilt split system IDs disagree with the saved version-1 split.")
        if not np.array_equal(prior_manifest["split"].astype(str), current["split"].astype(str)):
            raise ValueError("Rebuilt split assignments disagree with the saved version-1 split.")
    return manifest


def save_pair_partition(system_ids: np.ndarray, output_dir: Path, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    order = build_pair_partition(system_ids, pair_count=PAIR_COUNT, heldout_pairs=8, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, system_id in enumerate(system_ids):
        for rank, pair_id in enumerate(order[index]):
            rows.append(
                {
                    "system_id": int(system_id),
                    "pair_id": int(pair_id),
                    "variation_i": int(PAIR_MEMBERS[pair_id, 0]),
                    "variation_j": int(PAIR_MEMBERS[pair_id, 1]),
                    "role": "candidate" if rank < 8 else "heldout",
                    "role_order": int(rank if rank < 8 else rank - 8),
                    "permutation_rank": int(rank),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "pair_partition.csv", index=False)
    return order, frame


def load_labels(
    dataset_dir: Path,
    fallback_path: Path,
    result_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, str]:
    labels_path = dataset_dir / "completed_systems_sampling_3.txt"
    if labels_path.is_file() and result_paths:
        batch_ids: list[int] = []
        row_counts: list[int] = []
        for path in result_paths:
            batch_ids.append(batch_id_from_path(path))
            with h5py.File(path, "r") as handle:
                row_counts.append(int(handle["times_complete"].shape[0]))
        return load_aligned_labels(labels_path, batch_ids, row_counts), str(labels_path)
    if fallback_path.is_file():
        labels = pd.read_csv(fallback_path)
        needed = {"index", "num_batch", "time_unbound", "diagnosis"}
        if not needed.issubset(labels.columns):
            raise ValueError(f"Saved sampling3 label fallback lacks columns: {sorted(needed - set(labels.columns))}")
        return labels, str(fallback_path)
    if labels_path.is_file():
        raw = pd.read_csv(
            labels_path,
            sep=r"\s+",
            header=None,
            names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
        )
        return raw, str(labels_path)
    raise FileNotFoundError(
        "Sampling3 labels are unavailable. Expected completed_systems_sampling_3.txt in "
        f"{dataset_dir} or the saved sampling3 label fallback {fallback_path}."
    )


def make_targets(
    labels: pd.DataFrame,
    audit: pd.DataFrame,
    manifest: pd.DataFrame,
    pair_order: np.ndarray,
    system_ids: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.copy()
    labels["index"] = pd.to_numeric(labels["index"], errors="raise").astype(np.int64)
    labels["time_unbound"] = pd.to_numeric(labels["time_unbound"], errors="raise").astype(np.float64)
    labels["diagnosis"] = pd.to_numeric(labels["diagnosis"], errors="raise").astype(np.int64)
    if labels["index"].duplicated().any():
        raise ValueError("Sampling3 completion labels contain duplicate trajectory IDs.")
    labels["system_id"] = labels["index"] // GROUP_SIZE
    labels["variation_id"] = labels["index"] % GROUP_SIZE
    labels["stable"] = labels["time_unbound"] >= LABEL_HORIZON
    labels["valid"] = (labels["time_unbound"] >= 0.0) & (labels["diagnosis"] == 0)
    retained = labels[labels["system_id"].isin(system_ids) & labels["valid"]]
    stable = np.zeros((len(system_ids), GROUP_SIZE), dtype=np.float32)
    event_times = np.full((len(system_ids), GROUP_SIZE), np.inf, dtype=np.float64)
    system_to_row = {int(system_id): index for index, system_id in enumerate(system_ids)}
    for row in retained.itertuples(index=False):
        system_row = system_to_row[int(row.system_id)]
        variation = int(row.variation_id)
        stable[system_row, variation] = float(row.stable)
        if float(row.time_unbound) < LABEL_HORIZON:
            event_times[system_row, variation] = float(row.time_unbound)
    counts = retained.groupby("system_id")["variation_id"].nunique().reindex(system_ids, fill_value=0).to_numpy()
    if not np.all(counts == GROUP_SIZE):
        bad = system_ids[counts != GROUP_SIZE][:10].tolist()
        raise ValueError(f"Completion labels do not contain all 32 valid variations for systems {bad}.")
    p_all = stable.mean(axis=1)
    audit_sorted = audit.set_index("system_id").reindex(system_ids)
    if audit_sorted["p_stable"].isna().any() or not np.allclose(
        p_all, audit_sorted["p_stable"].to_numpy(dtype=np.float64), atol=1e-8, rtol=0.0
    ):
        raise ValueError("Completion-table stable fractions disagree with the sampling3 audit table.")
    heldout_pairs = pair_order[:, 8:].astype(np.int64)
    heldout_variations = pair_ids_to_variations(heldout_pairs, PAIR_MEMBERS).reshape(len(system_ids), -1)
    heldout_target = np.take_along_axis(stable, heldout_variations, axis=1).mean(axis=1)
    membership = manifest.set_index("raw_case_id")["split"].reindex(system_ids).astype(str).to_numpy()
    targets = pd.DataFrame(
        {
            "system_id": system_ids,
            "split": membership,
            "p_stable_all32": p_all,
            "p_stable_heldout16": heldout_target,
            "stable_count_all32": stable.sum(axis=1).astype(np.int64),
            "stable_count_heldout16": np.take_along_axis(stable, heldout_variations, axis=1).sum(axis=1).astype(np.int64),
            "ma_margin": audit_sorted["ma_margin"].to_numpy(dtype=np.float64),
        }
    )
    targets.to_csv(output_dir / "system_targets.csv", index=False)
    return stable, p_all.astype(np.float32), heldout_target.astype(np.float32), event_times


def load_time_axis(paths: list[Path]) -> tuple[np.ndarray, int, float | None, dict[str, Any]]:
    if not paths:
        raise FileNotFoundError("No sampling3 HDF5 trajectory shards were found.")
    with h5py.File(paths[0], "r") as handle:
        if "times_complete" not in handle:
            raise KeyError(f"Missing times_complete in {paths[0]}")
        grid = np.asarray(handle["times_complete"][0], dtype=np.float64)
        source_steps = int(handle.attrs.get("source_time_steps", grid.size))
        attr = handle.attrs.get("source_time_end")
        source_end = None if attr is None else float(np.asarray(attr).reshape(-1)[0])
    deltas = np.diff(grid[np.isfinite(grid)])
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    interval = float(np.median(deltas)) if deltas.size else None
    inferred_end = float(grid[np.isfinite(grid)][0] + interval * (source_steps - 1)) if interval is not None else float(grid[-1])
    used_attr = bool(source_end is not None and np.isfinite(source_end) and source_end >= float(grid[-1]))
    metadata = {
        "source_file": str(paths[0]),
        "stored_time_steps": int(grid.size),
        "source_time_steps": int(source_steps),
        "stored_time_start": float(grid[0]),
        "stored_time_end": float(grid[-1]),
        "median_sampling_interval": interval,
        "source_time_end_attribute": source_end,
        "source_time_end_attribute_used": used_attr,
        "estimated_source_time_end": float(source_end if used_attr else inferred_end),
        "label_horizon": LABEL_HORIZON,
    }
    return grid, source_steps, source_end if used_attr else None, metadata


def resolve_system_rows(
    labels: pd.DataFrame,
    system_ids: np.ndarray,
    feature_finite: np.ndarray,
    total_feature_rows: int,
) -> np.ndarray:
    frame = labels[["index"]].copy()
    frame["index"] = pd.to_numeric(frame["index"], errors="raise").astype(np.int64)
    if len(frame) != total_feature_rows:
        raise ValueError(f"Feature rows ({total_feature_rows}) and aligned labels ({len(frame)}) differ.")
    if frame["index"].duplicated().any():
        raise ValueError("Aligned labels contain duplicate global trajectory IDs.")
    row_by_id = dict(zip(frame["index"].to_numpy(dtype=np.int64), np.arange(len(frame), dtype=np.int64)))
    rows = np.empty((len(system_ids), GROUP_SIZE), dtype=np.int64)
    for i, system_id in enumerate(system_ids):
        ids = int(system_id) * GROUP_SIZE + np.arange(GROUP_SIZE, dtype=np.int64)
        try:
            rows[i] = [row_by_id[int(index)] for index in ids]
        except KeyError as exc:
            raise ValueError(f"Missing feature/label row for trajectory {exc.args[0]}.") from exc
    if np.any(~np.asarray(feature_finite, dtype=bool)[rows]):
        bad_systems = system_ids[np.any(~np.asarray(feature_finite, dtype=bool)[rows], axis=1)][:10]
        raise ValueError(f"Non-finite orbital15 features found in complete systems {bad_systems.tolist()}.")
    return rows


def calculate_normalization(
    features: np.ndarray,
    train_rows: np.ndarray,
    batch_trajectories: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    total = np.zeros(FEATURE_CHANNELS, dtype=np.float64)
    total_sq = np.zeros(FEATURE_CHANNELS, dtype=np.float64)
    count = 0
    for start in range(0, selected.size, batch_trajectories):
        rows = selected[start : start + batch_trajectories]
        batch = np.asarray(features[rows, :, :], dtype=np.float32)
        total += batch.sum(axis=(0, 1), dtype=np.float64)
        total_sq += np.square(batch, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
        count += int(batch.shape[0] * batch.shape[1])
    if count == 0:
        raise ValueError("No training trajectories are available for normalization.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32).reshape(1, 1, -1), std.astype(np.float32).reshape(1, 1, -1)


def to_tensor_batch(
    features: np.ndarray,
    system_rows: np.ndarray,
    system_indices: np.ndarray,
    variations: np.ndarray,
    prefix: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    batch_size, phases = variations.shape
    chosen_rows = system_rows[system_indices[:, None], variations]
    values = np.asarray(features[chosen_rows.reshape(-1), :prefix, :], dtype=np.float32)
    values -= mean
    values /= std
    tensor = torch.from_numpy(np.ascontiguousarray(values.reshape(batch_size, phases, prefix, FEATURE_CHANNELS)))
    return tensor.to(device, non_blocking=device.type == "cuda")


def to_pair_tensor_batch(
    features: np.ndarray,
    system_rows: np.ndarray,
    system_indices: np.ndarray,
    pair_ids: np.ndarray,
    prefix: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    batch_size, pair_count = pair_ids.shape
    variation_ids = pair_ids_to_variations(pair_ids, PAIR_MEMBERS).reshape(batch_size, pair_count * 2)
    flat = to_tensor_batch(features, system_rows, system_indices, variation_ids, prefix, mean, std, device)
    return flat.reshape(batch_size, pair_count, 2, prefix, FEATURE_CHANNELS)


def singleton_member_mask(sides: np.ndarray, device: torch.device) -> torch.Tensor:
    """Build [systems, 1 pair, 2 members] masks for one observed member per pair."""
    chosen = np.asarray(sides, dtype=np.int64).reshape(-1)
    if np.any((chosen != 0) & (chosen != 1)):
        raise ValueError("Singleton member sides must be 0 or 1.")
    mask = np.zeros((chosen.size, 1, 2), dtype=np.float32)
    mask[np.arange(chosen.size), 0, chosen] = 1.0
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_regression_condition(
    model: PairDeepSetsRegressor,
    features: np.ndarray,
    system_rows: np.ndarray,
    system_indices: np.ndarray,
    selected_pairs: np.ndarray,
    prefix: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_systems: int,
    amp: bool,
    member_mask: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, system_indices.size, batch_systems):
        stop = min(start + batch_systems, system_indices.size)
        index_batch = system_indices[start:stop]
        pair_batch = selected_pairs[start:stop]
        x = to_pair_tensor_batch(features, system_rows, index_batch, pair_batch, prefix, mean, std, device)
        mask_batch = None if member_mask is None else member_mask[start:stop]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = model(x, mask_batch).squeeze(-1)
        predictions.append(pred.float().cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


@torch.no_grad()
def predict_baseline_condition(
    model: PhaseTCNClassifier,
    features: np.ndarray,
    system_rows: np.ndarray,
    system_indices: np.ndarray,
    selected_pairs: np.ndarray,
    prefix: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_systems: int,
    amp: bool,
    member_mask: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, system_indices.size, batch_systems):
        stop = min(start + batch_systems, system_indices.size)
        index_batch = system_indices[start:stop]
        pair_batch = selected_pairs[start:stop]
        x = to_pair_tensor_batch(features, system_rows, index_batch, pair_batch, prefix, mean, std, device)
        batch, pairs, members, steps, channels = x.shape
        if member_mask is None:
            mask = torch.ones((batch, pairs, members), dtype=x.dtype, device=device)
        else:
            mask = member_mask[start:stop].to(dtype=x.dtype, device=device)
            if mask.shape != (batch, pairs, members):
                raise ValueError("member_mask must have shape [systems, pairs, 2].")
        flat_mask = mask.reshape(-1).bool()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            selected_x = x.reshape(batch * pairs * members, steps, channels)[flat_mask]
            selected_prob = torch.sigmoid(model(selected_x)).reshape(-1)
            probabilities = x.new_zeros((batch * pairs * members,))
            probabilities[flat_mask] = selected_prob.to(dtype=probabilities.dtype)
            probabilities = probabilities.reshape(batch, pairs, members)
            probability = (probabilities * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        predictions.append(probability.float().cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def validation_mae_regression(
    model: PairDeepSetsRegressor,
    features: np.ndarray,
    system_rows: np.ndarray,
    val_indices: np.ndarray,
    phase_draws: np.ndarray,
    targets: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_systems: int,
    amp: bool,
    singleton_sides: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    values: list[float] = []
    for condition in conditions:
        is_singleton = np.isclose(condition.k, 0.5)
        selected_pairs = phase_draws[:, :1] if is_singleton else phase_draws[:, : int(condition.k)]
        mask = singleton_member_mask(singleton_sides, device) if is_singleton and singleton_sides is not None else None
        if is_singleton and mask is None:
            raise ValueError("K=0.5 validation requires one fixed member side per system.")
        pred = predict_regression_condition(
            model, features, system_rows, val_indices, selected_pairs, condition.t,
            mean, std, device, batch_systems, amp, member_mask=mask,
        )
        values.append(float(mean_absolute_error(targets[val_indices], pred)))
    return float(np.mean(values)), values


def train_regression_model(
    *,
    features: np.ndarray,
    system_rows: np.ndarray,
    stable_labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    phase_pool: np.ndarray,
    val_draws: np.ndarray,
    targets: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    model_name: str,
    val_singleton_sides: np.ndarray | None = None,
) -> tuple[PairDeepSetsRegressor, list[dict[str, float]]]:
    set_random_seed(seed)
    rng = np.random.default_rng(seed + 137)
    model = PairDeepSetsRegressor(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if args.amp and device.type == "cuda" else None
    amp = scaler is not None
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        loss_total = 0.0
        system_total = 0
        for start in range(0, order.size, args.batch_systems):
            batch_system_ids = order[start : start + args.batch_systems]
            batch_size = batch_system_ids.size
            condition = conditions[int(rng.integers(0, len(conditions)))]
            k = condition.k
            prefix = condition.t
            is_singleton = np.isclose(k, 0.5)
            sampled_pairs = 1 if is_singleton else int(k)
            pair_ranks = np.argsort(rng.random((batch_size, phase_pool.shape[1])), axis=1)[:, :sampled_pairs]
            pair_choices = np.take_along_axis(phase_pool[batch_system_ids], pair_ranks, axis=1)
            x = to_pair_tensor_batch(features, system_rows, batch_system_ids, pair_choices, prefix, mean, std, device)
            member_mask = None
            if is_singleton:
                member_sides = rng.integers(0, 2, size=batch_size, dtype=np.int64)
                member_mask = singleton_member_mask(member_sides, device)
            y = torch.as_tensor(targets[batch_system_ids, None], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(x, member_mask)
                loss = torch.nn.functional.mse_loss(prediction, y)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_total += float(loss.detach().item()) * batch_size
            system_total += batch_size
        val_mean_mae, val_condition_mae = validation_mae_regression(
            model,
            features,
            system_rows,
            val_indices,
            val_draws,
            targets,
            conditions,
            mean,
            std,
            device,
            args.eval_batch_systems,
            amp,
            singleton_sides=val_singleton_sides,
        )
        record = {
            "epoch": epoch,
            "train_mse": loss_total / max(system_total, 1),
            "val_mae_mean_conditions": val_mean_mae,
        }
        record.update({f"val_mae_k{condition.k}_t{condition.t}": value for condition, value in zip(conditions, val_condition_mae)})
        history.append(record)
        print(f"{model_name} seed={seed} epoch={epoch:02d} train_mse={record['train_mse']:.5f} val_mae={val_mean_mae:.5f}", flush=True)
        if val_mean_mae < best_mae:
            best_mae = val_mean_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def validation_mae_baseline(
    model: PhaseTCNClassifier,
    features: np.ndarray,
    system_rows: np.ndarray,
    val_indices: np.ndarray,
    phase_draws: np.ndarray,
    target: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_systems: int,
    amp: bool,
    singleton_sides: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    values: list[float] = []
    for condition in conditions:
        is_singleton = np.isclose(condition.k, 0.5)
        selected_pairs = phase_draws[:, :1] if is_singleton else phase_draws[:, : int(condition.k)]
        mask = singleton_member_mask(singleton_sides, device) if is_singleton and singleton_sides is not None else None
        if is_singleton and mask is None:
            raise ValueError("K=0.5 validation requires one fixed member side per system.")
        pred = predict_baseline_condition(
            model, features, system_rows, val_indices, selected_pairs, condition.t,
            mean, std, device, batch_systems, amp, member_mask=mask,
        )
        values.append(float(mean_absolute_error(target[val_indices], pred)))
    return float(np.mean(values)), values


def train_baseline_model(
    *,
    features: np.ndarray,
    system_rows: np.ndarray,
    stable_labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    val_draws_all: np.ndarray,
    p_all: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    val_singleton_sides: np.ndarray | None = None,
) -> tuple[PhaseTCNClassifier, list[dict[str, float]]]:
    set_random_seed(seed)
    rng = np.random.default_rng(seed + 911)
    model = PhaseTCNClassifier(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if args.amp and device.type == "cuda" else None
    amp = scaler is not None
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        loss_total = 0.0
        system_total = 0
        for start in range(0, order.size, args.batch_systems):
            batch_system_ids = order[start : start + args.batch_systems]
            batch_size = batch_system_ids.size
            variations = rng.integers(0, GROUP_SIZE, size=batch_size, dtype=np.int64)
            prefix = int(rng.choice(PREFIX_CHOICES))
            x4 = to_tensor_batch(
                features,
                system_rows,
                batch_system_ids,
                variations[:, None],
                prefix,
                mean,
                std,
                device,
            )
            x = x4[:, 0]
            y = torch.as_tensor(
                stable_labels[batch_system_ids, variations, None], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_total += float(loss.detach().item()) * batch_size
            system_total += batch_size
        val_mean_mae, val_condition_mae = validation_mae_baseline(
            model,
            features,
            system_rows,
            val_indices,
            val_draws_all,
            p_all,
            conditions,
            mean,
            std,
            device,
            args.eval_batch_systems,
            amp,
            singleton_sides=val_singleton_sides,
        )
        record = {
            "epoch": epoch,
            "train_bce": loss_total / max(system_total, 1),
            "val_mae_mean_conditions": val_mean_mae,
        }
        record.update({f"val_mae_k{condition.k}_t{condition.t}": value for condition, value in zip(conditions, val_condition_mae)})
        history.append(record)
        print(f"mean_probability seed={seed} epoch={epoch:02d} train_bce={record['train_bce']:.5f} val_mae={val_mean_mae:.5f}", flush=True)
        if val_mean_mae < best_mae:
            best_mae = val_mean_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_draws(
    model: torch.nn.Module,
    predictor: Any,
    features: np.ndarray,
    system_rows: np.ndarray,
    test_indices: np.ndarray,
    phase_draws: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[int, np.ndarray]:
    draw_count = phase_draws.shape[0]
    result = {condition.k: np.empty((draw_count, test_indices.size), dtype=np.float32) for condition in conditions}
    amp = bool(args.amp and device.type == "cuda")
    for condition in conditions:
        for draw in range(draw_count):
            result[condition.k][draw] = predictor(
                model,
                features,
                system_rows,
                test_indices,
                phase_draws[draw, :, : condition.k],
                condition.t,
                mean,
                std,
                device,
                args.eval_batch_systems,
                amp,
            )
        print(f"  predicted K={condition.k}, T={condition.t} over {draw_count} phase draws", flush=True)
    return result


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    y = np.asarray(target, dtype=np.float64)[valid]
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    if y.size == 0:
        return {"n_systems": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return {
        "n_systems": int(y.size),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
        "r2": float(r2_score(y, pred)) if y.size >= 2 else float("nan"),
    }


def bootstrap_metric_intervals(
    target: np.ndarray,
    prediction: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    y = np.asarray(target, dtype=np.float64)[valid]
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    if y.size < 2:
        return {name: (float("nan"), float("nan")) for name in ("mae", "rmse", "r2")}
    sample = rng.integers(0, y.size, size=(replicates, y.size))
    err = y - pred
    mae = np.abs(err[sample]).mean(axis=1)
    rmse = np.sqrt(np.square(err[sample]).mean(axis=1))
    r2 = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        ys = y[sample[i]]
        ps = pred[sample[i]]
        denominator = np.square(ys - ys.mean()).sum()
        r2[i] = 1.0 - np.square(ys - ps).sum() / denominator if denominator > 0 else np.nan
    return {
        "mae": tuple(float(x) for x in np.nanquantile(mae, [0.025, 0.975])),
        "rmse": tuple(float(x) for x in np.nanquantile(rmse, [0.025, 0.975])),
        "r2": tuple(float(x) for x in np.nanquantile(r2, [0.025, 0.975])),
    }


def bootstrap_delta_mae(
    target: np.ndarray,
    prediction_k1: np.ndarray,
    prediction_k: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    valid = np.isfinite(target) & np.isfinite(prediction_k1) & np.isfinite(prediction_k)
    y = np.asarray(target, dtype=np.float64)[valid]
    p1 = np.asarray(prediction_k1, dtype=np.float64)[valid]
    pk = np.asarray(prediction_k, dtype=np.float64)[valid]
    if y.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    paired_gain = np.abs(y - p1) - np.abs(y - pk)
    observed = float(paired_gain.mean())
    samples = rng.integers(0, y.size, size=(replicates, y.size))
    boot = paired_gain[samples].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    # Center the paired bootstrap distribution under H0: delta=0, then
    # compare the observed absolute gain with the two-sided null distribution.
    null_boot = boot - observed
    p_value = (np.count_nonzero(np.abs(null_boot) >= abs(observed)) + 1) / (replicates + 1)
    return observed, float(lower), float(upper), float(p_value)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return adjusted.tolist()
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = order.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def aggregate_predictions(
    seed_predictions: list[dict[int, np.ndarray]],
    conditions: tuple[BudgetCondition, ...],
    eligible_draws: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if not seed_predictions:
        raise ValueError("No seed predictions were provided.")
    counts = (
        np.full(seed_predictions[0][conditions[0].k].shape[1], seed_predictions[0][conditions[0].k].shape[0], dtype=np.int64)
        if eligible_draws is None
        else np.asarray(eligible_draws, dtype=bool).sum(axis=0).astype(np.int64)
    )
    aggregate: dict[int, np.ndarray] = {}
    for condition in conditions:
        stack = np.stack([item[condition.k] for item in seed_predictions], axis=0).astype(np.float64)
        if eligible_draws is None:
            aggregate[condition.k] = stack.mean(axis=(0, 1)).astype(np.float32)
        else:
            mask = np.asarray(eligible_draws, dtype=np.float64)[None, :, :]
            total = (stack * mask).sum(axis=(0, 1))
            denominator = counts * len(seed_predictions)
            aggregate[condition.k] = np.divide(
                total,
                denominator,
                out=np.full_like(total, np.nan, dtype=np.float64),
                where=denominator > 0,
            ).astype(np.float32)
    return aggregate, counts


def write_evaluation_outputs(
    *,
    output_dir: Path,
    system_ids: np.ndarray,
    split_names: np.ndarray,
    ma_margin: np.ndarray,
    p_all: np.ndarray,
    p_heldout: np.ndarray,
    test_indices: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    prediction_store: dict[tuple[str, str], list[dict[int, np.ndarray]]],
    candidate_pre: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_map = {"all_phase": p_all, "heldout_phase": p_heldout}
    system_rows_out: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    aggregate_store: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    count_store: dict[tuple[str, str, str], np.ndarray] = {}
    bootstrap_seed = int(args.seed + 424242)

    for (model_name, target_name), seed_predictions in prediction_store.items():
        pred, _ = aggregate_predictions(seed_predictions, conditions)
        aggregate_store[(model_name, target_name, "standard")] = pred
        for condition in conditions:
            values = pred[condition.k]
            target = target_map[target_name][test_indices]
            group = point_metrics(target, values)
            rng = np.random.default_rng(bootstrap_seed + condition.k + (0 if model_name == "deep_sets" else 100) + (0 if target_name == "all_phase" else 200))
            intervals = bootstrap_metric_intervals(target, values, args.bootstrap_replicates, rng)
            metric_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "control": "standard",
                    "subset": "all_test",
                    "K": condition.k, "M_pairs": condition.k,
                    "T": condition.t,
                    **group,
                    **{f"{name}_ci_low": bounds[0] for name, bounds in intervals.items()},
                    **{f"{name}_ci_high": bounds[1] for name, bounds in intervals.items()},
                }
            )
            for local_i, system_index in enumerate(test_indices):
                system_rows_out.append(
                    {
                        "system_id": int(system_ids[system_index]),
                        "split": str(split_names[system_index]),
                        "model": model_name,
                        "target": target_name,
                        "control": "standard",
                        "K": condition.k, "M_pairs": condition.k,
                        "T": condition.t,
                        "target_value": float(target[local_i]),
                        "prediction": float(values[local_i]),
                        "absolute_error": float(abs(target[local_i] - values[local_i])),
                        "ma_margin": float(ma_margin[system_index]),
                        "ma_group": "near" if abs(ma_margin[system_index]) < 0.1 else "far" if abs(ma_margin[system_index]) >= 0.3 else "middle",
                        "eligible_draws": int(args.phase_draws),
                    }
                )
        family_p: list[float] = []
        family_indices: list[int] = []
        target = target_map[target_name][test_indices]
        for condition in conditions:
            if condition.k == 1:
                continue
            rng = np.random.default_rng(bootstrap_seed + condition.k + (300 if model_name == "deep_sets" else 400) + (0 if target_name == "all_phase" else 500))
            observed, low, high, p_value = bootstrap_delta_mae(
                target,
                pred[1],
                pred[condition.k],
                args.bootstrap_replicates,
                rng,
            )
            family_indices.append(len(comparison_rows))
            family_p.append(p_value)
            comparison_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "control": "standard",
                    "subset": "all_test",
                    "K": condition.k, "M_pairs": condition.k,
                    "T": condition.t,
                    "n_systems": int(np.isfinite(target).sum()),
                    "delta_mae_k1_minus_k": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw_bootstrap": p_value,
                    "p_holm": np.nan,
                }
            )
        for idx, adjusted in zip(family_indices, holm_adjust(family_p)):
            comparison_rows[idx]["p_holm"] = adjusted

    # MA-boundary subgroup summaries use the same per-system predictions and
    # paired system bootstrap as the main comparisons.
    test_margin = ma_margin[test_indices]
    for (model_name, target_name), _seed_predictions in prediction_store.items():
        standard = aggregate_store[(model_name, target_name, "standard")]
        target = target_map[target_name][test_indices]
        subgroup_families = [
            ("ma_near", np.abs(test_margin) < 0.1),
            ("ma_far", np.abs(test_margin) >= 0.3),
        ]
        for subgroup_name, subgroup_mask in subgroup_families:
            family_p = []
            family_indices = []
            for condition in conditions:
                values = standard[condition.k][subgroup_mask]
                subgroup_target = target[subgroup_mask]
                group = point_metrics(subgroup_target, values)
                rng = np.random.default_rng(
                    bootstrap_seed + condition.k + (1100 if model_name == "deep_sets" else 1200)
                    + (0 if target_name == "all_phase" else 100) + (0 if subgroup_name == "ma_near" else 200)
                )
                intervals = bootstrap_metric_intervals(subgroup_target, values, args.bootstrap_replicates, rng)
                metric_rows.append(
                    {
                        "model": model_name,
                        "target": target_name,
                        "control": "standard",
                        "subset": subgroup_name,
                        "K": condition.k, "M_pairs": condition.k,
                        "T": condition.t,
                        **group,
                        **{f"{name}_ci_low": bounds[0] for name, bounds in intervals.items()},
                        **{f"{name}_ci_high": bounds[1] for name, bounds in intervals.items()},
                    }
                )
                if condition.k == 1:
                    continue
                rng = np.random.default_rng(
                    bootstrap_seed + condition.k + (1300 if model_name == "deep_sets" else 1400)
                    + (0 if target_name == "all_phase" else 100) + (0 if subgroup_name == "ma_near" else 200)
                )
                observed, low, high, p_value = bootstrap_delta_mae(
                    subgroup_target,
                    standard[1][subgroup_mask],
                    standard[condition.k][subgroup_mask],
                    args.bootstrap_replicates,
                    rng,
                )
                family_indices.append(len(comparison_rows))
                family_p.append(p_value)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "target": target_name,
                        "control": "standard",
                        "subset": subgroup_name,
                        "K": condition.k, "M_pairs": condition.k,
                        "T": condition.t,
                        "n_systems": int(subgroup_mask.sum()),
                        "delta_mae_k1_minus_k": observed,
                        "ci_low": low,
                        "ci_high": high,
                        "p_raw_bootstrap": p_value,
                        "p_holm": np.nan,
                    }
                )
            for idx, adjusted in zip(family_indices, holm_adjust(family_p)):
                comparison_rows[idx]["p_holm"] = adjusted

    # Strict control: the heldout-target estimate averages only common draw
    # indices where every phase in every B=512 allocation remained event-free.
    for (model_name, target_name), seed_predictions in prediction_store.items():
        if target_name != "heldout_phase":
            continue
        pre_pred, eligible_count = aggregate_predictions(seed_predictions, conditions, candidate_pre)
        aggregate_store[(model_name, target_name, "heldout_preinstability")] = pre_pred
        count_store[(model_name, target_name, "heldout_preinstability")] = eligible_count
        target = p_heldout[test_indices]
        family_p = []
        family_indices = []
        for condition in conditions:
            values = pre_pred[condition.k]
            mask = np.isfinite(values)
            group = point_metrics(target[mask], values[mask])
            rng = np.random.default_rng(bootstrap_seed + condition.k + (700 if model_name == "deep_sets" else 800))
            intervals = bootstrap_metric_intervals(target[mask], values[mask], args.bootstrap_replicates, rng)
            metric_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "control": "heldout_preinstability",
                    "subset": "common_eligible_systems",
                    "K": condition.k, "M_pairs": condition.k,
                    "T": condition.t,
                    **group,
                    **{f"{name}_ci_low": bounds[0] for name, bounds in intervals.items()},
                    **{f"{name}_ci_high": bounds[1] for name, bounds in intervals.items()},
                }
            )
            for local_i, system_index in enumerate(test_indices):
                if not np.isfinite(values[local_i]):
                    continue
                system_rows_out.append(
                    {
                        "system_id": int(system_ids[system_index]),
                        "split": str(split_names[system_index]),
                        "model": model_name,
                        "target": target_name,
                        "control": "heldout_preinstability",
                        "K": condition.k, "M_pairs": condition.k,
                        "T": condition.t,
                        "target_value": float(target[local_i]),
                        "prediction": float(values[local_i]),
                        "absolute_error": float(abs(target[local_i] - values[local_i])),
                        "ma_margin": float(ma_margin[system_index]),
                        "ma_group": "near" if abs(ma_margin[system_index]) < 0.1 else "far" if abs(ma_margin[system_index]) >= 0.3 else "middle",
                        "eligible_draws": int(eligible_count[local_i]),
                    }
                )
            if condition.k == 1:
                continue
            rng = np.random.default_rng(bootstrap_seed + condition.k + (900 if model_name == "deep_sets" else 1000))
            observed, low, high, p_value = bootstrap_delta_mae(
                target,
                pre_pred[1],
                pre_pred[condition.k],
                args.bootstrap_replicates,
                rng,
            )
            family_indices.append(len(comparison_rows))
            family_p.append(p_value)
            comparison_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "control": "heldout_preinstability",
                    "subset": "common_eligible_systems",
                    "K": condition.k, "M_pairs": condition.k,
                    "T": condition.t,
                    "n_systems": int(np.isfinite(pre_pred[condition.k]).sum()),
                    "delta_mae_k1_minus_k": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw_bootstrap": p_value,
                    "p_holm": np.nan,
                }
            )
        for idx, adjusted in zip(family_indices, holm_adjust(family_p)):
            comparison_rows[idx]["p_holm"] = adjusted

    predictions_frame = pd.DataFrame(system_rows_out)
    metrics_frame = pd.DataFrame(metric_rows)
    comparisons_frame = pd.DataFrame(comparison_rows)
    predictions_frame.to_csv(output_dir / "per_system_predictions.csv", index=False)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    comparisons_frame.to_csv(output_dir / "paired_bootstrap_comparisons.csv", index=False)
    save_json(
        output_dir / "metrics.json",
        {
            "primary_metric": "system-level MAE",
            "bootstrap_unit": "base system",
            "bootstrap_replicates": args.bootstrap_replicates,
            "pair_draws": args.phase_draws,
            "preinstability_eligibility": "same pair draw must be eligible across all four 2*M_pairs*T=512 allocations",
            "models": sorted(predictions_frame["model"].unique().tolist()) if not predictions_frame.empty else [],
        },
    )
    return predictions_frame, metrics_frame, comparisons_frame


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(
    output_dir: Path,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    system_predictions: pd.DataFrame,
    p_all_test: np.ndarray,
    ma_margin_test: np.ndarray,
    condition_map: dict[int, int],
    n_bootstrap: int,
    seed: int,
) -> None:
    # Figure 1: primary fixed-budget MAE curve, plus the averaging baseline.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    primary = metrics[(metrics["control"] == "standard") & (metrics["target"] == "all_phase")]
    for model_name, label in (("deep_sets", "TCN + DeepSets"), ("mean_probability", "TCN mean probability")):
        frame = primary[primary["model"] == model_name].sort_values("K")
        if frame.empty:
            continue
        err_low = frame["mae"] - frame["mae_ci_low"]
        err_high = frame["mae_ci_high"] - frame["mae"]
        ax.errorbar(frame["K"], frame["mae"], yerr=np.vstack([err_low, err_high]), marker="o", capsize=3, label=label)
    ax.set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"], xlabel="Observed matched pairs M", ylabel="System-level MAE", title="Fixed budget B = 512 stored points")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure1_fixed_budget_tradeoff")

    # Figure 2: all phases, heldout target, and heldout + pre-instability.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    panels = (
        ("standard", "all_phase", "All-phase target"),
        ("standard", "heldout_phase", "Held-out phases"),
        ("heldout_preinstability", "heldout_phase", "Held-out + pre-instability"),
    )
    for ax, (control, target, title) in zip(axes, panels):
        frame = metrics[(metrics["control"] == control) & (metrics["target"] == target)]
        for model_name, label in (("deep_sets", "TCN + DeepSets"), ("mean_probability", "TCN mean probability")):
            curve = frame[frame["model"] == model_name].sort_values("K")
            if curve.empty:
                continue
            ax.plot(curve["K"], curve["mae"], marker="o", label=label)
        ax.set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"], xlabel="Observed matched pairs M", title=title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("System-level MAE")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed-budget hard controls")
    _save_figure(fig, output_dir, "figure2_hard_controls")

    # Figure 3: paired breadth gain near and far from the MA boundary.
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    test_predictions = system_predictions[
        (system_predictions["model"] == "deep_sets")
        & (system_predictions["target"] == "all_phase")
        & (system_predictions["control"] == "standard")
    ]
    near_ids = np.flatnonzero(np.abs(ma_margin_test) < 0.1)
    far_ids = np.flatnonzero(np.abs(ma_margin_test) >= 0.3)
    y_all = p_all_test
    groups = (("near", near_ids, r"Near boundary ($|m|<0.1$)"), ("far", far_ids, r"Far ($|m|\geq0.3$)"))
    offsets = {"near": -0.09, "far": 0.09}
    rng = np.random.default_rng(seed + 65537)
    for group_name, ids, label in groups:
        group_frame = test_predictions[test_predictions["ma_group"] == group_name]
        for condition in sorted(condition_map):
            if condition == 1:
                continue
            subset1 = group_frame[group_frame["K"] == 1].set_index("system_id")
            subsetk = group_frame[group_frame["K"] == condition].set_index("system_id")
            shared = subset1.index.intersection(subsetk.index)
            if shared.empty:
                continue
            a = subset1.loc[shared, "absolute_error"].to_numpy(dtype=np.float64)
            b = subsetk.loc[shared, "absolute_error"].to_numpy(dtype=np.float64)
            gain = a - b
            boots = gain[rng.integers(0, gain.size, size=(n_bootstrap, gain.size))].mean(axis=1)
            low, high = np.quantile(boots, [0.025, 0.975])
            xpos = condition + offsets[group_name]
            ax.errorbar(xpos, gain.mean(), yerr=[[gain.mean() - low], [high - gain.mean()]], marker="o", capsize=3, label=label if condition == 2 else None, color=("#2474B5" if group_name == "near" else "#D55E00"))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set(xscale="log", xticks=[2, 4, 8], xticklabels=["2", "4", "8"], xlabel="Matched pairs M compared with M=1 at equal B=512", ylabel=r"Breadth gain $\Delta$MAE = MAE(1) - MAE(K)", title="Breadth gain by MA-boundary distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure3_ma_boundary_breadth_gain")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.group_size != GROUP_SIZE:
        raise ValueError("This runner is defined for the 32-variation sampling3 systems.")
    if args.train_seeds <= 0 or args.phase_draws <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("train-seeds, phase-draws, and bootstrap-replicates must be positive.")
    if args.batch_systems <= 0 or args.eval_batch_systems <= 0:
        raise ValueError("Batch sizes must be positive.")
    conditions = budget_conditions(args.budget, PAIR_COUNTS, max_prefix=1024, phases_per_unit=2)
    if args.budget == 512 and tuple(condition.t for condition in conditions) != (256, 128, 64, 32):
        raise ValueError("The pair-aware B=512 allocations must be (M,T)=(1,256),(2,128),(4,64),(8,32).")

    args.output_dir = ensure_dir(args.output_dir.resolve())
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    args.cache_path = resolve_cache_path(args.cache_path, dataset_dir, args.output_dir)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    audit = pd.read_csv(args.audit_csv)
    audit = audit.sort_values("system_id").reset_index(drop=True)
    system_ids = audit["system_id"].to_numpy(dtype=np.int64)
    manifest = rebuild_split_manifest(audit, args.output_dir, seed=args.seed, ratios=ratios)
    pair_order, _ = save_pair_partition(system_ids, args.output_dir, args.seed)
    manifest_sorted = manifest.sort_values("raw_case_id").reset_index(drop=True)
    split_names = manifest_sorted["split"].astype(str).to_numpy()
    ma_margin = audit.set_index("system_id").reindex(system_ids)["ma_margin"].to_numpy(dtype=np.float64)
    test_indices = np.flatnonzero(split_names == "test")
    near_count = int(np.count_nonzero(np.abs(ma_margin[test_indices]) < 0.1))
    far_count = int(np.count_nonzero(np.abs(ma_margin[test_indices]) >= 0.3))

    result_paths = discover_result_paths(dataset_dir, 3) if dataset_dir.is_dir() and list(dataset_dir.glob("results_batch_*_sampling_3.hdf5")) else []
    labels, label_source = load_labels(dataset_dir, args.label_fallback, result_paths)
    stable, p_all, p_heldout, failure_times = make_targets(
        labels, audit, manifest_sorted, pair_order, system_ids, args.output_dir
    )
    phase_seeds = [int(args.seed + 1009 * index) for index in range(args.train_seeds)]
    config: dict[str, Any] = {
        "experiment": "sampling3_fixed_budget_matched_pair_ensemble",
        "pairing": "variation i paired with variation i+16 for i=0,...,15; each sampled unit contains both trajectories",
        "sampling_id": 3,
        "dataset_dir": dataset_dir,
        "label_source": label_source,
        "audit_csv": args.audit_csv,
        "output_dir": args.output_dir,
        "cache_path": args.cache_path,
        "budget_definition": "stored orbital15 input points, 2 * M_pairs * T",
        "budget": int(args.budget),
        "conditions": [{"M_pairs": condition.k, "trajectories": 2 * condition.k, "T": condition.t, "stored_points": condition.budget} for condition in conditions],
        "max_stored_prefix": 1024,
        "feature_channels": FEATURE_CHANNELS,
        "feature_representation": "orbital15; angle channels encoded as sin/cos; float32 cache",
        "group_size": GROUP_SIZE,
        "split_version": 1,
        "split_seed": int(args.seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_counts": manifest_sorted["split"].value_counts().to_dict(),
        "heldout_partition": "fixed per system, 8 candidate matched pairs + 8 heldout matched pairs",
        "pair_partition_seed": int(args.seed),
        "pair_subset_draws": int(args.phase_draws),
        "random_streams": {"base_train_seed": int(args.seed), "per_run_seed_step": 1009, "deep_sets_all_model_seeds": phase_seeds, "deep_sets_heldout_model_seeds": [seed + 200000 for seed in phase_seeds], "mean_probability_model_seeds": [seed + 400000 for seed in phase_seeds], "train_rng_offsets": {"deep_sets": 137, "mean_probability": 911}, "evaluation_rng_seeds": {"test_all_phase": int(args.seed + 31031), "test_candidate": int(args.seed + 41041), "validation_all_phase": int(args.seed + 51051), "validation_candidate": int(args.seed + 61061)}},
        "target_all_phases": "stable fraction among all 32 variations; stable iff time_unbound >= 10000",
        "target_heldout": "stable fraction among the fixed 8 heldout pairs (16 variations)",
        "model": {"encoder": "shared TCN per trajectory", "pair_embedding": "concatenate member mean and absolute difference", "set_pooling": "DeepSets over matched pairs", "hidden": 160, "levels": 5, "temporal_pooling": "mean", "deep_sets_phi_rho_hidden": 128, "dropout": args.dropout, "output": "sigmoid [0,1]"},
        "baseline": "binary TCN per-trajectory stable probability, averaged over both members of each observed pair",
        "training": {"loss": "MSE for pair regressor; BCE for trajectory baseline", "checkpoint": "mean validation MAE across four B=512 pair allocations", "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs, "patience": args.patience, "train_seeds": phase_seeds, "batch_systems": args.batch_systems, "training_condition_sampling": "uniform over the four fixed-budget allocations (M_pairs,T)=(1,256),(2,128),(4,64),(8,32); include both pair members", "stored_points_per_training_example": int(args.budget)},
        "evaluation": {"primary_metric": "system-level MAE", "auxiliary_metrics": ["RMSE", "R2"], "bootstrap_unit": "base system", "bootstrap_replicates": args.bootstrap_replicates, "holm_family": "three M>1 comparisons within each model/target/control/subgroup"},
        "preinstability": {"condition": "both members of every observed pair have time_unbound > mapped prefix endpoint", "common_eligibility": "same pair draw eligible for all four B=512 allocations", "right_censored_value": 10000, "right_censor_treatment": "no event observed; represented by +infinity for event-time eligibility"},
        "ma_groups": {"near": "abs(ma_margin) < 0.1", "middle_excluded": "0.1 <= abs(ma_margin) < 0.3", "far": "abs(ma_margin) >= 0.3", "test_counts": {"near": near_count, "far": far_count}},
        "run_scope": "single-seed preliminary run when --train-seeds=1; full protocol defaults to five seeds",
        "status": "prepared; feature cache and training not yet started",
    }
    save_json(args.output_dir / "config.json", config)
    print(f"Sampling3 systems: {len(system_ids)}; split={manifest_sorted['split'].value_counts().to_dict()}; MA test groups near={near_count}, far={far_count}", flush=True)

    if args.prepare_only:
        print(f"Prepared experiment files in {args.output_dir}", flush=True)
        return

    if not result_paths:
        config["status"] = "blocked_missing_sampling3_feature_shards"
        config["required_data"] = f"{dataset_dir}/results_batch_00_sampling_3.hdf5 through results_batch_39_sampling_3.hdf5"
        save_json(args.output_dir / "config.json", config)
        raise FileNotFoundError(
            "Sampling3 orbital15 trajectory shards are not mounted. Expected files like "
            f"{dataset_dir}/results_batch_00_sampling_3.hdf5. The runner has written the rebuilt split, "
            "fixed phase partition, targets, and config; mount/provide the sampling3 prefix1024 shards and rerun."
        )
    if len(result_paths) != 40:
        raise ValueError(f"Expected the full 40 sampling3 shards, found {len(result_paths)} in {dataset_dir}.")

    from tasks.task_b.highres import make_cache

    shape, finite = make_cache(
        result_paths,
        args.cache_path,
        prefix_len=1024,
        time_stride=1,
        rebuild=False,
    )
    if shape[1:] != (1024, FEATURE_CHANNELS):
        raise ValueError(f"Unexpected sampling3 orbital15 cache shape: {shape}")
    feature_array = np.memmap(args.cache_path, mode="r", dtype=np.float32, shape=shape)
    batch_ids = [batch_id_from_path(path) for path in result_paths]
    row_counts = []
    for path in result_paths:
        with h5py.File(path, "r") as handle:
            row_counts.append(int(handle["times_complete"].shape[0]))
    aligned_labels = load_aligned_labels(dataset_dir / "completed_systems_sampling_3.txt", batch_ids, row_counts)
    system_rows = resolve_system_rows(aligned_labels, system_ids, finite, shape[0])

    train_indices = np.flatnonzero(split_names == "train")
    val_indices = np.flatnonzero(split_names == "val")
    test_indices = np.flatnonzero(split_names == "test")
    train_rows = system_rows[train_indices].reshape(-1)
    mean, std = calculate_normalization(feature_array, train_rows)
    np.savez(args.output_dir / "orbital15_normalization.npz", mean=mean, std=std)

    grid, source_steps, source_end, time_metadata = load_time_axis(result_paths)
    endpoints = prefix_endpoints_in_label_time(
        grid,
        [condition.t for condition in conditions],
        source_time_steps=source_steps,
        label_horizon=LABEL_HORIZON,
        source_time_end=source_end,
    )
    config["time_axis"] = time_metadata
    config["prefix_endpoint_label_time"] = {str(condition.t): float(value) for condition, value in zip(conditions, endpoints)}
    config["feature_cache"] = {"path": args.cache_path, "shape": shape, "dtype": "float32"}
    config["status"] = "features cached; training started"
    save_json(args.output_dir / "config.json", config)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print("CUDA requested but unavailable; using CPU.", flush=True)
    amp = bool(args.amp and device.type == "cuda")
    args.amp = amp
    all_pool = np.tile(np.arange(PAIR_COUNT, dtype=np.int16), (len(system_ids), 1))
    candidate_pool = pair_order[:, :8].copy()
    seed_values = phase_seeds
    rng_eval_all = np.random.default_rng(args.seed + 31031)
    rng_eval_candidate = np.random.default_rng(args.seed + 41041)
    test_draws_all = nested_phase_draws(all_pool[test_indices], args.phase_draws, rng_eval_all)
    test_draws_candidate = nested_phase_draws(candidate_pool[test_indices], args.phase_draws, rng_eval_candidate)
    val_draws_all = nested_phase_draws(all_pool[val_indices], 1, np.random.default_rng(args.seed + 51051))[0]
    val_draws_candidate = nested_phase_draws(candidate_pool[val_indices], 1, np.random.default_rng(args.seed + 61061))[0]
    _ = val_draws_candidate  # Heldout-target models use the same five-condition validation protocol below.
    failure_test = failure_times[test_indices]
    all_pre = common_preinstability_mask(failure_test, test_draws_all, conditions, endpoints, pair_members=PAIR_MEMBERS)
    candidate_pre = common_preinstability_mask(failure_test, test_draws_candidate, conditions, endpoints, pair_members=PAIR_MEMBERS)
    pd.DataFrame(
        {
            "K": [condition.k for condition in conditions],
            "M_pairs": [condition.k for condition in conditions],
            "trajectories": [2 * condition.k for condition in conditions],
            "T": [condition.t for condition in conditions],
            "prefix_endpoint_label_time": endpoints,
            "candidate_common_eligible_system_draws": [int(candidate_pre.sum())] * len(conditions),
            "allphase_common_eligible_system_draws": [int(all_pre.sum())] * len(conditions),
        }
    ).to_csv(args.output_dir / "preinstability_eligibility.csv", index=False)

    train_pool_all = all_pool
    train_pool_candidate = candidate_pool
    prediction_store: dict[tuple[str, str], list[dict[int, np.ndarray]]] = {
        (model, target): []
        for model in ("deep_sets", "mean_probability")
        for target in ("all_phase", "heldout_phase")
    }
    for seed_index, seed in enumerate(seed_values):
        print(f"\n=== training seed {seed_index + 1}/{len(seed_values)} (seed={seed}) ===", flush=True)
        for mode, pool, target_values, val_draws in (
            ("all_phase", train_pool_all, p_all, val_draws_all),
            ("heldout_phase", train_pool_candidate, p_heldout, val_draws_candidate),
        ):
            model, history = train_regression_model(
                features=feature_array,
                system_rows=system_rows,
                stable_labels=stable,
                train_indices=train_indices,
                val_indices=val_indices,
                phase_pool=pool,
                val_draws=val_draws,
                targets=target_values,
                conditions=conditions,
                mean=mean,
                std=std,
                device=device,
                args=args,
                seed=seed + (0 if mode == "all_phase" else 200000),
                model_name=f"deep_sets_{mode}",
            )
            pd.DataFrame(history).to_csv(args.output_dir / f"train_log_deep_sets_{mode}_seed{seed}.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "seed": seed, "mode": mode}, args.output_dir / f"deep_sets_{mode}_seed{seed}.pt")
            draw_pool = test_draws_all if mode == "all_phase" else test_draws_candidate
            predictions = predict_draws(
                model, predict_regression_condition, feature_array, system_rows, test_indices,
                draw_pool, conditions, mean, std, device, args
            )
            prediction_store[("deep_sets", mode)].append(predictions)

        baseline, baseline_history = train_baseline_model(
            features=feature_array,
            system_rows=system_rows,
            stable_labels=stable,
            train_indices=train_indices,
            val_indices=val_indices,
            val_draws_all=val_draws_all,
            p_all=p_all,
            conditions=conditions,
            mean=mean,
            std=std,
            device=device,
            args=args,
            seed=seed + 400000,
        )
        pd.DataFrame(baseline_history).to_csv(args.output_dir / f"train_log_mean_probability_seed{seed}.csv", index=False)
        torch.save({"state_dict": baseline.state_dict(), "seed": seed}, args.output_dir / f"mean_probability_seed{seed}.pt")
        for mode, draw_pool in (("all_phase", test_draws_all), ("heldout_phase", test_draws_candidate)):
            predictions = predict_draws(
                baseline, predict_baseline_condition, feature_array, system_rows, test_indices,
                draw_pool, conditions, mean, std, device, args
            )
            prediction_store[("mean_probability", mode)].append(predictions)

    prediction_frame, metric_frame, comparison_frame = write_evaluation_outputs(
        output_dir=args.output_dir,
        system_ids=system_ids,
        split_names=split_names,
        ma_margin=ma_margin,
        p_all=p_all,
        p_heldout=p_heldout,
        test_indices=test_indices,
        conditions=conditions,
        prediction_store=prediction_store,
        candidate_pre=candidate_pre,
        args=args,
    )
    make_figures(
        args.output_dir,
        metric_frame,
        comparison_frame,
        prediction_frame,
        p_all[test_indices],
        ma_margin[test_indices],
        {condition.k: condition.t for condition in conditions},
        args.bootstrap_replicates,
        args.seed,
    )
    config["status"] = "complete_preliminary_single_seed" if args.train_seeds == 1 else "complete"
    config["models_saved"] = True
    save_json(args.output_dir / "config.json", config)
    print(f"Experiment complete. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
