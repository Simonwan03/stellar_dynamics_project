#!/usr/bin/env python3
"""Run Task-B dynamical summaries + XGBoost on the Task-F masked protocol.

For each requested mask ratio this script uses the same Task-F data filter,
system-level split, labels, and deterministic validation/test masks.  It
computes the Task-B eight statistics (120 features for orbital15) only over
observed time positions, then trains one XGBoost classifier per mask ratio.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task_b_highres import (  # noqa: E402
    FEATURE_NAMES,
    discover_result_paths,
    load_aligned_labels,
    make_cache,
)
from tasks.common.case_split import case_split_indices, ensure_case_split_manifest  # noqa: E402
from tasks.common.experiment_utils import (  # noqa: E402
    binary_metrics_from_probs,
    ensure_dir,
    save_binary_curves,
    save_json,
    set_random_seed,
)
from tasks.task_f.main import best_f1_threshold, default_split_path, ratio_tag  # noqa: E402
from tasks.task_f.masking import random_observation_mask, validate_mask_ratio  # noqa: E402
from tasks.task_f.summaries import (  # noqa: E402
    TASK_B_STAT_NAMES,
    summarize_observed_prefix,
    task_b_summary_feature_names,
)


def parse_mask_ratios(value: str) -> tuple[float, ...]:
    raw = [float(item.strip()) for item in value.split(",") if item.strip()]
    ratios = tuple(validate_mask_ratio(item / 100.0 if item >= 1.0 else item) for item in raw)
    if not ratios:
        raise ValueError("At least one mask ratio is required.")
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"Mask ratios must be unique, got {ratios}.")
    return ratios


def summary_batch(
    features: np.ndarray,
    sample_ids: np.ndarray,
    mask_ratio: float,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Summarize one batch with the Task-F deterministic mask policy."""
    values = np.asarray(features, dtype=np.float32)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    observed = np.empty(values.shape[:2], dtype=bool)
    for row, sample_id in enumerate(sample_ids):
        observed[row] = random_observation_mask(
            values.shape[1],
            mask_ratio,
            seed=seed,
            sample_id=int(sample_id),
            epoch=epoch,
        )
    return summarize_observed_prefix(values, observed)


def make_summary_features(
    features: np.memmap,
    global_indices: np.ndarray,
    sample_ids: np.ndarray,
    rows: np.ndarray,
    mask_ratio: float,
    *,
    seed: int,
    epoch: int,
    batch_size: int,
    output_path: Path | None = None,
) -> np.ndarray:
    """Materialize summary features in filtered-label order."""
    rows = np.asarray(rows, dtype=np.int64)
    selected_global = global_indices[rows]
    selected_ids = sample_ids[rows]
    shape = (len(rows), features.shape[2] * len(TASK_B_STAT_NAMES))
    if output_path is None:
        output = np.empty(shape, dtype=np.float32)
    else:
        output_path = output_path.resolve()
        ensure_dir(output_path.parent)
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_values = np.asarray(features[selected_global[start:stop]], dtype=np.float32)
        output[start:stop] = summary_batch(
            batch_values,
            selected_ids[start:stop],
            mask_ratio,
            seed=seed,
            epoch=epoch,
        )
    output.flush()
    return output


def load_task_f_data(args: argparse.Namespace) -> tuple[np.memmap, dict[str, Any]]:
    """Load the Task-F cache and reproduce its exact filtered split."""
    paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    batch_ids = [int(Path(path).name.split("_")[2]) for path in paths]
    row_counts: list[int] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            row_counts.append(int(handle["times_complete"].shape[0]))
    cache_path = (
        args.cache_path
        or Path("/tmp") / f"task_f_sampling_{args.sampling_id}_prefix_{args.prefix_len}_orbital15.f32"
    ).resolve()
    shape, finite = make_cache(paths, cache_path, args.prefix_len, args.time_stride, args.rebuild_cache)
    features = np.memmap(cache_path, mode="r", dtype=np.float32, shape=shape)

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    labels = load_aligned_labels(labels_path, batch_ids, row_counts)
    all_case_ids = labels["index"].to_numpy(dtype=np.int64) // args.group_size
    split_manifest = args.split_manifest or default_split_path(
        args.dataset_dir, args.sampling_id, args.group_size, args.time_horizon, args.seed
    )
    manifest = ensure_case_split_manifest(
        labels_path,
        split_manifest,
        group_size=args.group_size,
        time_horizon=args.time_horizon,
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    valid = (
        np.isin(all_case_ids, manifest["raw_case_id"].to_numpy(dtype=np.int64))
        & (labels["time_unbound"].to_numpy(dtype=np.float64) >= 0.0)
        & (labels["diagnosis"].to_numpy(dtype=np.int64) == 0)
        & finite
    )
    filtered = labels.loc[valid].reset_index(drop=True)
    global_indices = np.flatnonzero(valid).astype(np.int64)
    sample_ids = filtered["index"].to_numpy(dtype=np.int64)
    sample_case_ids = sample_ids // args.group_size
    targets = (filtered["time_unbound"].to_numpy(dtype=np.float64) >= args.time_horizon).astype(np.float32)
    splits = case_split_indices(sample_case_ids, manifest)
    split_membership = np.empty(len(filtered), dtype=object)
    for name, rows in splits.items():
        split_membership[rows] = name
    filtered["raw_case_id"] = sample_case_ids
    filtered["split"] = split_membership
    return features, {
        "labels": filtered,
        "global_indices": global_indices,
        "sample_ids": sample_ids,
        "targets": targets,
        "splits": splits,
        "split_manifest": split_manifest,
        "cache_path": cache_path,
        "shape": shape,
    }


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install the project's xgboost dependency in the stellar environment."
        ) from exc
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=1,
    )
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_val, y_val.astype(np.int32))],
        "verbose": True,
    }
    if args.early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
    model.fit(x_train, y_train.astype(np.int32), **fit_kwargs)
    return model


def comparison_row(mask_ratio: float, model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    at_half = metrics["test_at_0.5"]
    at_val = metrics["test_val_f1_threshold"]
    return {
        "mask_ratio": mask_ratio,
        "model": model_name,
        "accuracy@0.5": at_half["accuracy"],
        "balanced_accuracy@0.5": at_half["balanced_accuracy"],
        "f1@0.5": at_half["f1"],
        "f1@val_threshold": at_val["f1"],
        "accuracy@val_threshold": at_val["accuracy"],
        "auc": at_half["auc"],
        "average_precision": at_half["average_precision"],
        "brier": at_half["brier"],
    }


def load_existing_comparison(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, path, key in (
        ("TCN", ROOT / "results/task_f_masked_prefix/metrics.json", "tcn"),
        ("Attention", ROOT / "results/task_f_masked_prefix_attention/metrics.json", "attention"),
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for tag, ratio_payload in payload.items():
            if key not in ratio_payload:
                continue
            ratio = float(ratio_payload[key]["mask_ratio"])
            rows.append(comparison_row(ratio, model_name, ratio_payload[key]))
    return rows


def write_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["mask_ratio", "model"])
    frame.to_csv(output_dir / "comparison.csv", index=False)
    columns = [
        "mask_ratio",
        "model",
        "accuracy@0.5",
        "balanced_accuracy@0.5",
        "f1@0.5",
        "f1@val_threshold",
        "auc",
        "average_precision",
        "brier",
    ]
    lines = [
        "# Task F: Task-B dynamical summaries + XGBoost comparison",
        "",
        "Metrics use the same Task-F test split and deterministic test mask for each ratio.",
        "The summary features use only observed positions; initial/final are the first/last observed values.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = [
            f"{row['mask_ratio']:.0%}",
            str(row["model"]),
            f"{row['accuracy@0.5']:.4f}",
            f"{row['balanced_accuracy@0.5']:.4f}",
            f"{row['f1@0.5']:.4f}",
            f"{row['f1@val_threshold']:.4f}",
            f"{row['auc']:.4f}",
            f"{row['average_precision']:.4f}",
            f"{row['brier']:.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sampling_3_prefix1024"))
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_f_summary_xgboost"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--mask-ratios", type=str, default="0,25,50,75")
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--save-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = ensure_dir(args.output_dir.resolve())
    ratios = parse_mask_ratios(args.mask_ratios)
    if args.read_batch_size <= 0:
        raise ValueError("--read-batch-size must be positive.")
    if args.overwrite_output:
        for path in args.output_dir.glob("mask*_xgboost_*"):
            if path.is_file():
                path.unlink()
        for name in ("metrics.json", "comparison.csv", "comparison.md", "config.json"):
            path = args.output_dir / name
            if path.exists():
                path.unlink()

    set_random_seed(args.seed)
    features, data = load_task_f_data(args)
    labels: pd.DataFrame = data["labels"]
    global_indices = data["global_indices"]
    sample_ids = data["sample_ids"]
    targets = data["targets"]
    splits: dict[str, np.ndarray] = data["splits"]
    feature_names = task_b_summary_feature_names(FEATURE_NAMES)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "dataset_dir": args.dataset_dir,
            "output_dir": args.output_dir,
            "mask_ratios": ratios,
            "prefix_shape": data["shape"],
            "summary_statistics": TASK_B_STAT_NAMES,
            "summary_feature_dim": len(feature_names),
            "summary_feature_names": feature_names,
            "summary_mask_policy": (
                "only observed time positions; initial/final are first/last observed; "
                "slope uses original normalized time coordinates"
            ),
            "split_manifest": data["split_manifest"],
            "split_counts": {name: int(rows.size) for name, rows in splits.items()},
            "stable_rate": float(targets.mean()),
            "cache_path": data["cache_path"],
        },
    )

    all_metrics: dict[str, Any] = {}
    comparison_rows = load_existing_comparison(args.output_dir)
    y_test = targets[splits["test"]]
    for ratio in ratios:
        tag = ratio_tag(ratio)
        print(f"Preparing Task-B summaries for mask={ratio:.0%}", flush=True)
        feature_path = (
            args.output_dir / f"{tag}_summary_features.float32.npy" if args.save_features else None
        )
        summary_features = make_summary_features(
            features,
            global_indices,
            sample_ids,
            np.arange(len(targets), dtype=np.int64),
            ratio,
            seed=args.seed,
            epoch=0,
            batch_size=args.read_batch_size,
            output_path=feature_path,
        )
        x_train = np.asarray(summary_features[splits["train"]])
        x_val = np.asarray(summary_features[splits["val"]])
        x_test = np.asarray(summary_features[splits["test"]])
        y_train = targets[splits["train"]]
        y_val = targets[splits["val"]]
        model = train_xgboost(x_train, y_train, x_val, y_val, args=args)
        p_train = model.predict_proba(x_train)[:, 1]
        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]
        threshold = best_f1_threshold(y_val, p_val)
        model_metrics = {
            "parameters": int(len(feature_names)),
            "mask_ratio": ratio,
            "summary_statistics": TASK_B_STAT_NAMES,
            "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
            "validation_threshold": threshold,
            "train": binary_metrics_from_probs(targets[splits["train"]], p_train),
            "validation": binary_metrics_from_probs(y_val, p_val),
            "test_at_0.5": binary_metrics_from_probs(y_test, p_test),
            "test_val_f1_threshold": binary_metrics_from_probs(y_test, p_test, threshold),
        }
        all_metrics[tag] = {"xgboost": model_metrics}
        model.get_booster().save_model(str(args.output_dir / f"{tag}_xgboost_model.json"))
        gain = model.get_booster().get_score(importance_type="gain")
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(len(feature_names))],
            }
        )
        total_gain = float(importance["gain"].sum())
        importance["gain_fraction"] = importance["gain"] / total_gain if total_gain > 0.0 else 0.0
        importance.sort_values("gain", ascending=False).to_csv(
            args.output_dir / f"{tag}_xgboost_feature_importance.csv", index=False
        )
        prediction_frame = labels.iloc[splits["test"]][
            ["index", "raw_case_id", "time_unbound", "diagnosis", "split"]
        ].copy()
        prediction_frame["target_stable"] = y_test.astype(np.int64)
        prediction_frame["prob_stable"] = p_test
        prediction_frame["mask_ratio"] = ratio
        prediction_frame.to_csv(args.output_dir / f"{tag}_xgboost_test_predictions.csv", index=False)
        save_binary_curves(
            args.output_dir / f"{tag}_xgboost_test_curves.png",
            y_test,
            p_test,
            f"Task F Task-B summaries + XGBoost mask={ratio:.0%}",
        )
        comparison_rows.append(comparison_row(ratio, "Summary + XGBoost", model_metrics))
        save_json(args.output_dir / "metrics.json", all_metrics)
        print(
            f"mask={ratio:.0%} test_auc={model_metrics['test_at_0.5']['auc']:.4f} "
            f"test_f1@val={model_metrics['test_val_f1_threshold']['f1']:.4f}",
            flush=True,
        )
    save_json(args.output_dir / "metrics.json", all_metrics)
    write_comparison(args.output_dir, comparison_rows)
    print(f"Task F summary XGBoost complete. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    from scripts.run_task_f_summary_xgboost_fair import main as fair_main
    fair_main()
