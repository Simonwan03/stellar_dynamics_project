#!/usr/bin/env python3
"""Train an XGBoost Task B classifier on prefix orbital15 statistics.

This is a strict prefix-only, SPoCK-inspired feature baseline.  It does not
run an additional N-body integration and therefore cannot produce the original
SPoCK MEGNO feature.  Instead it summarizes each of the 15 orbital15 channels
over the observed prefix with eight statistics:

``initial, final, mean, std, slope, min, max, delta``

The split and label ordering are inherited from the existing Task B
``prefix_512`` directory so that this run is directly comparable to the Panda
and Chronos-2 probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset_v3"
DEFAULT_PREFIX_DIR = REPO_ROOT / "results/task_b/sampling_2/task_b_common/prefix_512"
DEFAULT_OUTPUT_DIR = (
    EXPERIMENT_ROOT
    / "outputs/panda_unbound_dataset/spock_inspired_task_b_split_prefix96_xgb"
)

STAT_NAMES = ("initial", "final", "mean", "std", "slope", "min", "max", "delta")
ORBITAL15_CHANNEL_NAMES = (
    "aratio_0",
    "mratio_0",
    "mratio_1",
    "ecc_0",
    "ecc_1",
    "inc_0",
    "inc_1",
    "AP_sin_0",
    "AP_sin_1",
    "AP_cos_0",
    "AP_cos_1",
    "LAN_sin_0",
    "LAN_sin_1",
    "LAN_cos_0",
    "LAN_cos_1",
)
ORBITAL_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def labels_fingerprint(labels: pd.DataFrame) -> str:
    columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        columns.append("row_in_batch")
    identity = labels[columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def channel_names() -> list[str]:
    return [f"{channel}__{stat}" for channel in ORBITAL15_CHANNEL_NAMES for stat in STAT_NAMES]


def build_orbital15(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build the exact 15-channel representation used by Panda/Chronos."""
    aratio = np.asarray(data["aratios_complete"], dtype=np.float32)
    mratio = np.asarray(data["mratios_complete"], dtype=np.float32)
    ecc = np.asarray(data["eccs_complete"], dtype=np.float32)
    inc = np.asarray(data["incs_complete"], dtype=np.float32)
    ap = np.asarray(data["APs_complete"], dtype=np.float32)
    lan = np.asarray(data["LANs_complete"], dtype=np.float32)
    x = np.concatenate(
        [aratio, mratio, ecc, inc, np.sin(ap), np.cos(ap), np.sin(lan), np.cos(lan)],
        axis=-1,
    )
    if x.ndim != 3 or x.shape[-1] != 15:
        raise ValueError(f"Expected orbital15 shape (N,T,15), got {x.shape}")
    return x


def summarize_prefix(x: np.ndarray) -> np.ndarray:
    """Return eight statistics per channel, with slope per normalized prefix."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 15:
        raise ValueError(f"Expected (N,T,15), got {values.shape}")
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values).all(axis=(1, 2))).sum())
        raise ValueError(f"Found {bad} non-finite orbital15 prefix rows.")

    n_rows, time_steps, _ = values.shape
    normalized_time = np.linspace(0.0, 1.0, time_steps, dtype=np.float32)
    centered_time = normalized_time - normalized_time.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        raise ValueError("The prefix must contain at least two time steps.")

    initial = values[:, 0, :]
    final = values[:, -1, :]
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    slope = np.einsum("ntc,t->nc", values, centered_time) / denominator
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    delta = final - initial
    features = np.stack([initial, final, mean, std, slope, minimum, maximum, delta], axis=2)
    features = features.reshape(n_rows, 15 * len(STAT_NAMES))
    if not np.isfinite(features).all():
        raise ValueError("Prefix summary generated non-finite values.")
    return features.astype(np.float32, copy=False)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def load_task_labels(prefix_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = pd.read_csv(prefix_dir / "labels_filtered.csv")
    split_file = np.load(prefix_dir / "split_indices.npz")
    split_rows = {
        name: np.asarray(split_file[f"{name}_idx"], dtype=np.int64)
        for name in ("train", "val", "test")
    }
    all_rows = np.concatenate(list(split_rows.values()))
    if all_rows.size != len(labels) or np.unique(all_rows).size != len(labels):
        raise ValueError("Task B split does not cover every label exactly once.")
    if "row_in_batch" not in labels.columns:
        raise ValueError("labels_filtered.csv must contain row_in_batch for exact shard access.")
    return labels, split_rows


def extract_feature_cache(
    labels: pd.DataFrame,
    dataset_dir: Path,
    sampling_id: int,
    prefix_steps: int,
    cache_path: Path,
    batch_size: int,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if prefix_steps <= 1:
        raise ValueError("--prefix-steps must be greater than one.")
    if batch_size <= 0:
        raise ValueError("--read-batch-size must be positive.")
    metadata_path = cache_path.with_suffix(".json")
    expected_shape = (len(labels), 15 * len(STAT_NAMES))
    expected_metadata = {
        "rows": len(labels),
        "feature_dim": expected_shape[1],
        "prefix_steps": prefix_steps,
        "sampling_id": sampling_id,
        "representation": "orbital15",
        "statistics": list(STAT_NAMES),
        "slope_time": "normalized_prefix_0_to_1",
        "label_fingerprint": labels_fingerprint(labels),
        "dtype": "float32",
    }
    if cache_path.exists() and metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text())
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected_shape and all(
            metadata.get(key) == value for key, value in expected_metadata.items()
        ):
            print(f"Reusing feature cache: {cache_path}", flush=True)
            return cached, metadata
        raise ValueError(f"Existing cache is incompatible: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npy")
    features = np.lib.format.open_memmap(
        temp_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    labels_by_batch = labels.groupby("batch_id", sort=True)
    processed = 0
    started = time.perf_counter()
    try:
        for batch_id, group in labels_by_batch:
            batch_id = int(batch_id)
            task_positions = group.index.to_numpy(dtype=np.int64)
            local_rows = group["row_in_batch"].to_numpy(dtype=np.int64)
            read_order = np.argsort(local_rows)
            restore_order = np.argsort(read_order)
            sorted_rows = local_rows[read_order]
            shard_path = dataset_dir / f"results_batch_{batch_id:02d}_sampling_{sampling_id}.hdf5"
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            with h5py.File(shard_path, "r") as handle:
                for start in range(0, len(sorted_rows), batch_size):
                    stop = min(start + batch_size, len(sorted_rows))
                    selected = sorted_rows[start:stop]
                    data = {
                        key: np.asarray(handle[key][selected, :prefix_steps, ...], dtype=np.float32)
                        for key in ORBITAL_KEYS
                    }
                    orbital15 = build_orbital15(data)
                    summary = summarize_prefix(orbital15)
                    # Rows within each group are sorted for HDF5 access, then
                    # restored to the original labels_filtered order.
                    original_positions = task_positions[read_order[start:stop]]
                    features[original_positions] = summary
                    processed += len(summary)
            features.flush()
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"feature_cache batch={batch_id:02d} rows={processed}/{len(labels)} "
                f"rate={processed/elapsed:.1f}/s",
                flush=True,
            )
    except BaseException:
        features.flush()
        raise
    features.flush()
    del features
    os.replace(temp_path, cache_path)
    metadata = {
        **expected_metadata,
        "channels": list(ORBITAL15_CHANNEL_NAMES),
        "feature_names": channel_names(),
        "read_batch_size": batch_size,
        "created_seconds": time.time(),
    }
    write_json(metadata_path, metadata)
    return np.load(cache_path, mmap_mode="r"), metadata


def train_and_evaluate(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_rows: dict[str, np.ndarray],
    output_dir: Path,
    prefix_steps: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    reg_lambda: float,
    n_jobs: int,
    early_stopping_rounds: int,
    time_horizon: float,
) -> dict[str, Any]:
    target_unbound = (
        labels["time_unbound"].to_numpy(dtype=np.float32) < time_horizon
    ).astype(np.int8)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbosity=1,
    )
    print(
        f"Training XGBoost feature_dim={features.shape[1]} n_estimators={n_estimators} "
        f"max_depth={max_depth} n_jobs={n_jobs}",
        flush=True,
    )
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(np.asarray(features[split_rows["val"]]), target_unbound[split_rows["val"]])],
        "verbose": True,
    }
    model.fit(np.asarray(features[split_rows["train"]]), target_unbound[split_rows["train"]], **fit_kwargs)

    metrics: dict[str, Any] = {
        "model": "xgboost",
        "representation": "orbital15_summary",
        "prefix_steps": int(prefix_steps),
        "feature_dim": int(features.shape[1]),
        "target": f"time_unbound < {time_horizon:g}",
        "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
        "splits": {split: int(rows.size) for split, rows in split_rows.items()},
    }
    prediction_parts: list[pd.DataFrame] = []
    for split, rows in split_rows.items():
        probability_unbound = model.predict_proba(np.asarray(features[rows]))[:, 1]
        y_unbound = target_unbound[rows]
        probability_stable = 1.0 - probability_unbound
        y_stable = 1 - y_unbound
        metrics[split] = {
            "unbound_positive": probability_metrics(y_unbound, probability_unbound),
            "stable_positive": probability_metrics(y_stable, probability_stable),
        }
        frame = labels.iloc[rows][
            ["index", "num_batch", "row_in_batch", "time_unbound", "diagnosis"]
        ].copy()
        frame["split"] = split
        frame["target_unbound"] = y_unbound
        frame["prob_unbound"] = probability_unbound
        frame["prob_stable"] = probability_stable
        prediction_parts.append(frame)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        output_dir / "predictions_all.csv", index=False
    )
    # XGBoost 1.7.x and the current scikit-learn 1.8 do not agree on the
    # estimator mixin metadata when calling ``XGBClassifier.save_model``.
    # Persist the native booster directly; predictions remain identical and
    # the complete sklearn hyperparameters are recorded in run_config.json.
    model.get_booster().save_model(str(output_dir / "xgboost_model.json"))
    gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame(
        {
            "feature": channel_names(),
            "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(features.shape[1])],
        }
    )
    total_gain = float(importance["gain"].sum())
    importance["gain_fraction"] = (
        importance["gain"] / total_gain if total_gain > 0.0 else importance["gain"]
    )
    importance.sort_values("gain", ascending=False).to_csv(
        output_dir / "feature_importance_gain.csv", index=False
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix-steps", type=int, default=96)
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--read-batch-size", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    if args.prefix_steps <= 1:
        raise ValueError("--prefix-steps must be greater than one.")
    args.dataset_dir = args.dataset_dir.resolve()
    args.prefix_dir = args.prefix_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output:
        for filename in ("metrics.json", "predictions_all.csv", "xgboost_model.json", "run_config.json"):
            path = args.output_dir / filename
            if path.exists():
                path.unlink()

    labels, split_rows = load_task_labels(args.prefix_dir)
    cache_path = args.output_dir / f"orbital15_prefix{args.prefix_steps}_summary.float32.npy"
    features, metadata = extract_feature_cache(
        labels=labels,
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        prefix_steps=args.prefix_steps,
        cache_path=cache_path,
        batch_size=args.read_batch_size,
        overwrite=args.overwrite_cache,
    )
    metrics = train_and_evaluate(
        features=features,
        labels=labels,
        split_rows=split_rows,
        output_dir=args.output_dir,
        prefix_steps=args.prefix_steps,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,
        time_horizon=args.time_horizon,
    )
    write_json(
        args.output_dir / "run_config.json",
        {
            "dataset_dir": str(args.dataset_dir),
            "prefix_dir": str(args.prefix_dir),
            "output_dir": str(args.output_dir),
            "prefix_steps": args.prefix_steps,
            "sampling_id": args.sampling_id,
            "time_horizon": args.time_horizon,
            "statistics": list(STAT_NAMES),
            "feature_dim": int(features.shape[1]),
            "xgboost": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
                "n_jobs": args.n_jobs,
                "early_stopping_rounds": args.early_stopping_rounds,
                "seed": args.seed,
            },
            "cache_metadata": metadata,
            "metrics_path": str(args.output_dir / "metrics.json"),
            "feature_cache": str(cache_path),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
