#!/usr/bin/env python3
"""Fuse a frozen Task B logit with prefix-only physics summary features.

This is the quick, reproducible stacking variant: the foundation-model
probabilities come from an already-trained Task B probe and are converted to a
single logit.  That logit is concatenated with the 120-dimensional
SPoCK-inspired prefix summary cache and a logistic-regression meta-head is
selected on the existing validation split.

For a strict stacking paper result, replace the in-sample train logits with
out-of-fold logits; the test split and all physics features remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_PREFIX_DIR = REPO_ROOT / "results/task_b_prefix_ablation/prefix_512"
DEFAULT_PHYSICS_CACHE = (
    EXPERIMENT_ROOT
    / "outputs/panda_unbound_dataset/spock_inspired_task_b_split_prefix96_xgb/"
    / "orbital15_prefix96_summary.float32.npy"
)
DEFAULT_BASE_ROOT = EXPERIMENT_ROOT / "outputs/panda_unbound_dataset"


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
        raise ValueError("The Task B split does not cover every label exactly once.")
    return labels, split_rows


def align_base_logit(labels: pd.DataFrame, prediction_path: Path) -> np.ndarray:
    predictions = pd.read_csv(prediction_path)
    if "index" not in predictions.columns or "prob_unbound" not in predictions.columns:
        raise ValueError(f"Prediction file lacks index/prob_unbound: {prediction_path}")
    if predictions["index"].duplicated().any():
        raise ValueError(f"Duplicate indices in prediction file: {prediction_path}")
    indexed = predictions.set_index("index")
    missing = labels.loc[~labels["index"].isin(indexed.index), "index"]
    if len(missing):
        raise ValueError(f"{len(missing)} Task B rows are missing from {prediction_path}")
    probability = indexed.loc[labels["index"], "prob_unbound"].to_numpy(dtype=np.float64)
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability)).astype(np.float32)


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    labels: pd.DataFrame,
    split_rows: dict[str, np.ndarray],
    output_dir: Path,
    base_model: str,
    base_prediction_path: Path,
    c_values: list[float],
) -> dict[str, Any]:
    best_model: Pipeline | None = None
    best_c = None
    best_val_auc = -np.inf
    val_results: list[dict[str, float]] = []
    for c_value in c_values:
        candidate = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c_value),
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        candidate.fit(X[split_rows["train"]], y[split_rows["train"]])
        val_probability = candidate.predict_proba(X[split_rows["val"]])[:, 1]
        val_auc = float(roc_auc_score(y[split_rows["val"]], val_probability))
        val_results.append({"C": float(c_value), "val_auroc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = float(c_value)
            best_model = candidate
    if best_model is None or best_c is None:
        raise RuntimeError("No fusion model was selected.")

    metrics: dict[str, Any] = {
        "model": "logit_plus_physics_logistic_regression",
        "base_model": base_model,
        "base_prediction_path": str(base_prediction_path),
        "physics_feature_dim": int(X.shape[1] - 1),
        "fusion_feature_dim": int(X.shape[1]),
        "stacking_mode": "in_sample_train_base_logit_quick_probe",
        "target": "time_unbound < 10000",
        "selected_C": best_c,
        "validation_grid": val_results,
        "splits": {split: int(rows.size) for split, rows in split_rows.items()},
    }
    prediction_parts: list[pd.DataFrame] = []
    for split, rows in split_rows.items():
        probability_unbound = best_model.predict_proba(X[rows])[:, 1]
        y_unbound = y[rows]
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
    joblib.dump(best_model, output_dir / "fusion_logistic.joblib")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", choices=("panda96", "chronos2_96_reg"), required=True)
    parser.add_argument("--base-predictions", type=Path, default=None)
    parser.add_argument("--physics-cache", type=Path, default=DEFAULT_PHYSICS_CACHE)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", dest="c_values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.physics_cache = args.physics_cache.resolve()
    args.prefix_dir = args.prefix_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.base_predictions is None:
        if args.base_model == "panda96":
            args.base_predictions = DEFAULT_BASE_ROOT / "panda_task_b_split_prefix_96_aligned/predictions_all.csv"
        else:
            args.base_predictions = DEFAULT_BASE_ROOT / "chronos2_task_b_split_prefix96_reg/predictions_all.csv"
    args.base_predictions = args.base_predictions.resolve()
    if args.overwrite:
        for name in ("metrics.json", "predictions_all.csv", "fusion_logistic.joblib", "run_config.json"):
            path = args.output_dir / name
            if path.exists():
                path.unlink()

    labels, split_rows = load_task(args.prefix_dir)
    physics_metadata_path = args.physics_cache.with_suffix(".json")
    if not args.physics_cache.exists() or not physics_metadata_path.exists():
        raise FileNotFoundError(f"Physics cache and metadata are required: {args.physics_cache}")
    physics_metadata = json.loads(physics_metadata_path.read_text())
    if physics_metadata.get("label_fingerprint") != labels_fingerprint(labels):
        raise ValueError("Physics cache labels do not match Task B labels.")
    physics = np.load(args.physics_cache, mmap_mode="r")
    if physics.shape != (len(labels), 120):
        raise ValueError(f"Expected physics cache shape {(len(labels), 120)}, got {physics.shape}")
    physics = np.asarray(physics, dtype=np.float32)
    if not np.isfinite(physics).all():
        raise ValueError("Physics cache contains non-finite values.")

    foundation_logit = align_base_logit(labels, args.base_predictions)
    if not np.isfinite(foundation_logit).all():
        raise ValueError("Foundation logit contains non-finite values.")
    X = np.concatenate([foundation_logit[:, None], physics], axis=1)
    y = (labels["time_unbound"].to_numpy(dtype=np.float32) < 10000.0).astype(np.int8)
    print(
        f"base={args.base_model} X={X.shape} train={len(split_rows['train'])} "
        f"val={len(split_rows['val'])} test={len(split_rows['test'])}",
        flush=True,
    )
    metrics = fit_and_evaluate(
        X=X,
        y=y,
        labels=labels,
        split_rows=split_rows,
        output_dir=args.output_dir,
        base_model=args.base_model,
        base_prediction_path=args.base_predictions,
        c_values=args.c_values,
    )
    write_json(
        args.output_dir / "run_config.json",
        {
            "base_model": args.base_model,
            "base_predictions": str(args.base_predictions),
            "physics_cache": str(args.physics_cache),
            "prefix_dir": str(args.prefix_dir),
            "fusion_feature_dim": int(X.shape[1]),
            "physics_feature_dim": int(physics.shape[1]),
            "C_values": [float(value) for value in args.c_values],
            "stacking_mode": "in_sample_train_base_logit_quick_probe",
            "physics_metadata": physics_metadata,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
