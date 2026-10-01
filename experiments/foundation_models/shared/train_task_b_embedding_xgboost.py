#!/usr/bin/env python3
"""Train an XGBoost head on a frozen Panda or Chronos-2 Task B embedding cache.

The cache must contain one row per sample in the exact Task B label order.  The
script keeps the existing prefix=96 split and target, but replaces the linear
probe with an XGBoost classifier.  It is intentionally separate from the
physics-summary script so the comparison is easy to audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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
DEFAULT_PREFIX_DIR = REPO_ROOT / "results/task_b/sampling_2/task_b_common/prefix_512"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def labels_fingerprint(labels: pd.DataFrame) -> str:
    columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        columns.append("row_in_batch")
    identity = labels[columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


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


def load_task(prefix_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = pd.read_csv(prefix_dir / "labels_filtered.csv")
    split_file = np.load(prefix_dir / "split_indices.npz")
    split_rows = {
        split: np.asarray(split_file[f"{split}_idx"], dtype=np.int64)
        for split in ("train", "val", "test")
    }
    all_rows = np.concatenate(list(split_rows.values()))
    if all_rows.size != len(labels) or np.unique(all_rows).size != len(labels):
        raise ValueError("The Task B split does not cover each label exactly once.")
    return labels, split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--model-name", required=True, choices=("panda96", "chronos2_96_reg"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.2)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n_estimators <= 0 or args.max_depth <= 0 or args.n_jobs == 0:
        raise ValueError("n-estimators, max-depth and n-jobs must be valid positive values.")
    args.feature_cache = args.feature_cache.resolve()
    args.prefix_dir = args.prefix_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in (
            "metrics.json",
            "predictions_all.csv",
            "xgboost_model.json",
            "feature_importance_gain.csv",
            "run_config.json",
        ):
            path = args.output_dir / name
            if path.exists():
                path.unlink()

    labels, split_rows = load_task(args.prefix_dir)
    metadata_path = args.feature_cache.with_suffix(".json")
    if not args.feature_cache.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Feature cache and metadata are required: {args.feature_cache}")
    metadata = json.loads(metadata_path.read_text())
    expected_fingerprint = labels_fingerprint(labels)
    if metadata.get("label_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Embedding cache label fingerprint does not match the aligned Task B labels. "
            f"cache={metadata.get('label_fingerprint')} expected={expected_fingerprint}"
        )
    if int(metadata.get("rows", -1)) != len(labels):
        raise ValueError("Embedding cache row count does not match Task B labels.")
    if int(metadata.get("prefix_steps", -1)) != 96:
        raise ValueError(f"Expected prefix_steps=96, got {metadata.get('prefix_steps')!r}")

    features = np.load(args.feature_cache, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError(f"Unexpected embedding shape: {features.shape}")
    if not np.isfinite(np.asarray(features[: min(32, len(features))], dtype=np.float32)).all():
        raise ValueError("Embedding cache contains non-finite values.")

    target_unbound = (labels["time_unbound"].to_numpy(dtype=np.float32) < 10000.0).astype(np.int8)
    train_rows = split_rows["train"]
    val_rows = split_rows["val"]
    print(
        f"model={args.model_name} cache={args.feature_cache} shape={features.shape} "
        f"train={len(train_rows)} val={len(val_rows)} test={len(split_rows['test'])}",
        flush=True,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        max_bin=args.max_bin,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=1,
    )
    x_train = np.asarray(features[train_rows], dtype=np.float32)
    x_val = np.asarray(features[val_rows], dtype=np.float32)
    model.fit(
        x_train,
        target_unbound[train_rows],
        eval_set=[(x_val, target_unbound[val_rows])],
        verbose=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    del x_train, x_val

    metrics: dict[str, Any] = {
        "model": "xgboost",
        "embedding_model": args.model_name,
        "feature_cache": str(args.feature_cache),
        "feature_dim": int(features.shape[1]),
        "prefix_steps": 96,
        "target": "time_unbound < 10000",
        "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
        "splits": {split: int(rows.size) for split, rows in split_rows.items()},
    }
    prediction_parts: list[pd.DataFrame] = []
    for split, rows in split_rows.items():
        x_split = np.asarray(features[rows], dtype=np.float32)
        probability_unbound = model.predict_proba(x_split)[:, 1]
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
        del x_split

    pd.concat(prediction_parts, ignore_index=True).to_csv(
        args.output_dir / "predictions_all.csv", index=False
    )
    model.get_booster().save_model(str(args.output_dir / "xgboost_model.json"))
    gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame(
        {
            "feature_index": np.arange(features.shape[1], dtype=np.int64),
            "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(features.shape[1])],
        }
    ).sort_values("gain", ascending=False)
    total_gain = float(importance["gain"].sum())
    importance["gain_fraction"] = importance["gain"] / total_gain if total_gain > 0 else 0.0
    importance.to_csv(args.output_dir / "feature_importance_gain.csv", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model_name": args.model_name,
            "feature_cache": str(args.feature_cache),
            "prefix_dir": str(args.prefix_dir),
            "feature_dim": int(features.shape[1]),
            "prefix_steps": 96,
            "xgboost": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
                "max_bin": args.max_bin,
                "n_jobs": args.n_jobs,
                "early_stopping_rounds": args.early_stopping_rounds,
                "seed": args.seed,
            },
            "cache_metadata": metadata,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
