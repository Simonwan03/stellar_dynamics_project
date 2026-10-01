#!/usr/bin/env python3
"""Fair-seed runner for Task-B summaries + XGBoost on Task F.

The implementation helpers live in ``run_task_f_summary_xgboost.py``.  This
entry point materializes train, validation, and test summaries separately so
each split receives the same mask seed as the native Task-F experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_f.summary_xgboost import (
    TASK_B_STAT_NAMES,
    build_parser,
    comparison_row,
    ensure_dir,
    load_existing_comparison,
    load_task_f_data,
    make_summary_features,
    parse_mask_ratios,
    save_binary_curves,
    save_json,
    set_random_seed,
    task_b_summary_feature_names,
    train_xgboost,
    write_comparison,
)
from tasks.task_b.highres import FEATURE_NAMES
from tasks.task_f.main import ratio_tag
from tasks.common.experiment_utils import binary_metrics_from_probs


def materialize_split_summaries(
    features: np.memmap,
    data: dict[str, Any],
    ratio: float,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    feature_names: list[str],
) -> np.ndarray:
    targets = data["targets"]
    global_indices = data["global_indices"]
    sample_ids = data["sample_ids"]
    splits = data["splits"]
    tag = ratio_tag(ratio)
    feature_path = output_dir / f"{tag}_summary_features.float32.npy" if args.save_features else None
    shape = (len(targets), len(feature_names))
    if feature_path is None:
        summary_features = np.empty(shape, dtype=np.float32)
    else:
        summary_features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=shape)

    # These are the exact seeds used by Task F's train/validation/test loaders.
    split_mask_settings = {
        "train": (args.seed, 1),
        "val": (args.seed + 10_000, 0),
        "test": (args.seed + 20_000, 0),
    }
    for split_name, split_rows in splits.items():
        split_seed, split_epoch = split_mask_settings[split_name]
        temp_path = output_dir / f".{tag}_{split_name}_summary.tmp.npy"
        split_features = make_summary_features(
            features,
            global_indices,
            sample_ids,
            split_rows,
            ratio,
            seed=split_seed,
            epoch=split_epoch,
            batch_size=args.read_batch_size,
            output_path=temp_path,
        )
        summary_features[split_rows] = np.asarray(split_features)
        del split_features
        temp_path.unlink(missing_ok=True)
    if hasattr(summary_features, "flush"):
        summary_features.flush()
    return summary_features


def main() -> None:
    args = build_parser().parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = ensure_dir(args.output_dir.resolve())
    ratios = parse_mask_ratios(args.mask_ratios)
    if args.read_batch_size <= 0:
        raise ValueError("--read-batch-size must be positive.")
    if args.overwrite_output:
        for path in args.output_dir.glob("mask*"):
            if path.is_file():
                path.unlink()
        for name in ("metrics.json", "comparison.csv", "comparison.md", "config.json"):
            (args.output_dir / name).unlink(missing_ok=True)

    set_random_seed(args.seed)
    features, data = load_task_f_data(args)
    labels: pd.DataFrame = data["labels"]
    targets = data["targets"]
    splits = data["splits"]
    feature_names = task_b_summary_feature_names(FEATURE_NAMES)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "mask_ratios": ratios,
            "prefix_shape": data["shape"],
            "summary_statistics": TASK_B_STAT_NAMES,
            "summary_feature_dim": len(feature_names),
            "summary_feature_names": feature_names,
            "summary_mask_policy": (
                "only observed time positions; initial/final are first/last observed; "
                "slope uses original normalized time coordinates"
            ),
            "train_mask_seed_epoch": {"seed": args.seed, "epoch": 1},
            "validation_mask_seed_epoch": {"seed": args.seed + 10_000, "epoch": 0},
            "test_mask_seed_epoch": {"seed": args.seed + 20_000, "epoch": 0},
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
        summary_features = materialize_split_summaries(
            features,
            data,
            ratio,
            args=args,
            output_dir=args.output_dir,
            feature_names=feature_names,
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
        from tasks.task_f.main import best_f1_threshold

        threshold = best_f1_threshold(y_val, p_val)
        model_metrics = {
            "feature_dim": len(feature_names),
            "mask_ratio": ratio,
            "summary_statistics": TASK_B_STAT_NAMES,
            "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
            "validation_threshold": threshold,
            "train": binary_metrics_from_probs(y_train, p_train),
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
    main()
