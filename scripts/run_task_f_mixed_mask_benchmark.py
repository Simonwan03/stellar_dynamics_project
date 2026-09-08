#!/usr/bin/env python3
"""Run the prefix=256 mixed-mask Task-F robustness benchmark.

Each training regime fits one model per architecture and evaluates that model
on a fixed random/single-block/multi-block corruption grid.  The default run
contains clean-only, random-only, and the proposed mixed-mask regime, with TCN,
masked Attention, and Task-B dynamical summaries + XGBoost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task_b_highres import FEATURE_NAMES, calculate_normalization  # noqa: E402
from scripts.run_task_f_summary_xgboost import (  # noqa: E402
    build_parser as build_summary_parser,
    load_task_f_data,
    parse_mask_ratios,
)
from tasks.common.experiment_utils import (  # noqa: E402
    binary_metrics_from_probs,
    ensure_dir,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.task_f.main import best_f1_threshold, predict, train_model  # noqa: E402
from tasks.task_f.masking import apply_observation_mask  # noqa: E402
from tasks.task_f.mixed_masking import (  # noqa: E402
    MASK_KINDS,
    MASK_REGIMES,
    fixed_mask,
    sample_training_mask,
)
from tasks.task_f.models import build_model  # noqa: E402
from tasks.task_f.summaries import (  # noqa: E402
    TASK_B_STAT_NAMES,
    summarize_observed_prefix,
    task_b_summary_feature_names,
)


MODEL_NAMES = ("tcn", "attention", "xgboost")
TEST_KINDS = ("random", "single_block", "multi_block")


def parse_regimes(value: str) -> tuple[str, ...]:
    regimes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not regimes or any(item not in MASK_REGIMES for item in regimes):
        raise ValueError(f"Choose training regimes from {MASK_REGIMES}, got {regimes}.")
    if len(set(regimes)) != len(regimes):
        raise ValueError(f"Training regimes must be unique, got {regimes}.")
    return regimes


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not models or any(item not in MODEL_NAMES for item in models):
        raise ValueError(f"Choose models from {MODEL_NAMES}, got {models}.")
    if len(set(models)) != len(models):
        raise ValueError(f"Models must be unique, got {models}.")
    return models


def ratio_tag(ratio: float) -> str:
    return f"{int(round(100.0 * ratio)):02d}"


def cell_tag(kind: str, ratio: float, location: str | None = None) -> str:
    base = f"{kind}_{ratio_tag(ratio)}"
    return f"{base}_{location}" if location else base


def normalize_percent(value: float) -> float:
    value = float(value)
    return value / 100.0 if value >= 1.0 else value


class MixedMaskDataset(Dataset):
    """Memmap-backed sequence dataset for dynamic or fixed masks."""

    def __init__(
        self,
        features: np.memmap,
        rows: np.ndarray,
        global_indices: np.ndarray,
        targets: np.ndarray,
        sample_ids: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        seed: int,
        regime: str | None = None,
        fixed_kind: str | None = None,
        fixed_ratio: float = 0.0,
        fixed_start: int | None = None,
        dynamic: bool = False,
        view: int = 0,
    ) -> None:
        self.features = features
        self.rows = np.asarray(rows, dtype=np.int64)
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.seed = int(seed)
        self.regime = regime
        self.fixed_kind = fixed_kind
        self.fixed_ratio = float(fixed_ratio)
        self.fixed_start = fixed_start
        self.dynamic = bool(dynamic)
        self.view = int(view)
        self.epoch = 0
        if (regime is None) == (fixed_kind is None):
            raise ValueError("Specify exactly one of regime or fixed_kind.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, time_steps: int, sample_id: int) -> np.ndarray:
        if self.regime is not None:
            observed, _, _ = sample_training_mask(
                time_steps,
                seed=self.seed,
                sample_id=sample_id,
                epoch=self.epoch if self.dynamic else 0,
                regime=self.regime,
                view=self.view,
            )
            return observed
        return fixed_mask(
            time_steps,
            self.fixed_ratio,
            kind=self.fixed_kind or "random",
            seed=self.seed,
            sample_id=sample_id,
            epoch=0,
            view=self.view,
            fixed_start=self.fixed_start,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[index])
        values = np.asarray(self.features[self.global_indices[row]], dtype=np.float32)
        values = np.array((values - self.mean[0]) / self.std[0], dtype=np.float32, copy=True)
        observed = self._mask(values.shape[0], int(self.sample_ids[row]))
        masked = apply_observation_mask(values, observed, append_mask_channel=True)
        target = np.asarray([self.targets[row]], dtype=np.float32)
        return torch.from_numpy(masked), torch.from_numpy(target)


def make_loader(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    num_workers: int,
    train: bool = False,
    regime: str | None = None,
    fixed_kind: str | None = None,
    fixed_ratio: float = 0.0,
    fixed_start: int | None = None,
    dynamic: bool = False,
    view: int = 0,
) -> tuple[MixedMaskDataset, DataLoader]:
    dataset = MixedMaskDataset(
        features,
        rows,
        data["global_indices"],
        data["targets"],
        data["sample_ids"],
        mean,
        std,
        seed=seed,
        regime=regime,
        fixed_kind=fixed_kind,
        fixed_ratio=fixed_ratio,
        fixed_start=fixed_start,
        dynamic=dynamic,
        view=view,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


def observed_for_training_batch(
    time_steps: int,
    sample_ids: np.ndarray,
    *,
    seed: int,
    epoch: int,
    regime: str,
    view: int,
) -> np.ndarray:
    masks = np.empty((len(sample_ids), time_steps), dtype=bool)
    for row, sample_id in enumerate(np.asarray(sample_ids, dtype=np.int64)):
        masks[row], _, _ = sample_training_mask(
            time_steps,
            seed=seed,
            sample_id=int(sample_id),
            epoch=epoch,
            regime=regime,
            view=view,
        )
    return masks


def observed_for_fixed_batch(
    time_steps: int,
    sample_ids: np.ndarray,
    *,
    seed: int,
    kind: str,
    ratio: float,
    view: int = 0,
    fixed_start: int | None = None,
) -> np.ndarray:
    masks = np.empty((len(sample_ids), time_steps), dtype=bool)
    for row, sample_id in enumerate(np.asarray(sample_ids, dtype=np.int64)):
        masks[row] = fixed_mask(
            time_steps,
            ratio,
            kind=kind,
            seed=seed,
            sample_id=int(sample_id),
            view=view,
            fixed_start=fixed_start,
        )
    return masks


def summary_features_for_rows(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    *,
    mask_policy: str,
    seed: int,
    batch_size: int,
    regime: str | None = None,
    kind: str | None = None,
    ratio: float = 0.0,
    epoch: int = 0,
    view: int = 0,
    fixed_start: int | None = None,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    global_indices = data["global_indices"]
    sample_ids = data["sample_ids"]
    shape = (len(rows), features.shape[2] * len(TASK_B_STAT_NAMES))
    output = np.empty(shape, dtype=np.float32)
    time_steps = int(features.shape[1])
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        values = np.asarray(features[global_indices[batch_rows]], dtype=np.float32)
        ids = sample_ids[batch_rows]
        if mask_policy == "training":
            observed = observed_for_training_batch(
                time_steps,
                ids,
                seed=seed,
                epoch=epoch,
                regime=regime or "mixed",
                view=view,
            )
        elif mask_policy == "fixed":
            observed = observed_for_fixed_batch(
                time_steps,
                ids,
                seed=seed,
                kind=kind or "random",
                ratio=ratio,
                view=view,
                fixed_start=fixed_start,
            )
        else:
            raise ValueError(f"Unknown summary mask policy {mask_policy!r}.")
        output[start:stop] = summarize_observed_prefix(values, observed)
    return output


def make_training_summaries(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    *,
    regime: str,
    seed: int,
    batch_size: int,
    views: int,
) -> np.ndarray:
    actual_views = 1 if regime == "clean_only" else int(views)
    chunks = [
        summary_features_for_rows(
            features,
            data,
            rows,
            mask_policy="training",
            seed=seed,
            batch_size=batch_size,
            regime=regime,
            epoch=view + 1,
            view=view,
        )
        for view in range(actual_views)
    ]
    return np.concatenate(chunks, axis=0)


def make_validation_summaries(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    *,
    regime: str,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    return summary_features_for_rows(
        features,
        data,
        rows,
        mask_policy="training",
        seed=seed,
        batch_size=batch_size,
        regime=regime,
        epoch=0,
        view=0,
    )


def make_test_summaries(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    *,
    kind: str,
    ratio: float,
    seed: int,
    batch_size: int,
    view: int = 0,
    fixed_start: int | None = None,
) -> np.ndarray:
    return summary_features_for_rows(
        features,
        data,
        rows,
        mask_policy="fixed",
        seed=seed,
        batch_size=batch_size,
        kind=kind,
        ratio=ratio,
        view=view,
        fixed_start=fixed_start,
    )


def build_test_cells(test_ratios: tuple[float, ...], location_ratio: float) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for ratio in test_ratios:
        if ratio == 0.0:
            cells.append({"tag": cell_tag("clean", ratio), "kind": "random", "ratio": 0.0})
            continue
        for kind in TEST_KINDS:
            cells.append({"tag": cell_tag(kind, ratio), "kind": kind, "ratio": ratio})
    length = int(round(256 * location_ratio))
    start_map = {
        "early": 0,
        "middle": (256 - length) // 2,
        "late": 256 - length,
    }
    for location, start in start_map.items():
        cells.append(
            {
                "tag": cell_tag("single_block", location_ratio, location),
                "kind": "single_block",
                "ratio": location_ratio,
                "location": location,
                "fixed_start": start,
            }
        )
    return cells


def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
    random_state: int,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("XGBoost is required in the stellar environment.") from exc
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
        random_state=random_state,
        verbosity=0,
    )
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_val, y_val.astype(np.int32))],
        "verbose": False,
    }
    if args.early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
    model.fit(x_train, y_train.astype(np.int32), **fit_kwargs)
    return model


def model_metric_record(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "test_at_0.5": binary_metrics_from_probs(y_true, prob, 0.5),
        "test_val_f1_threshold": binary_metrics_from_probs(y_true, prob, threshold),
    }


def comparison_row(
    regime: str,
    model: str,
    cell: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    at_half = metrics["test_at_0.5"]
    at_val = metrics["test_val_f1_threshold"]
    return {
        "training_regime": regime,
        "model": model,
        "cell": cell["tag"],
        "mask_kind": cell["kind"],
        "mask_ratio": float(cell["ratio"]),
        "location": cell.get("location", ""),
        "accuracy@0.5": at_half["accuracy"],
        "balanced_accuracy@0.5": at_half["balanced_accuracy"],
        "f1@0.5": at_half["f1"],
        "f1@val_threshold": at_val["f1"],
        "auc": at_half["auc"],
        "average_precision": at_half["average_precision"],
        "brier": at_half["brier"],
    }


def write_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["training_regime", "mask_ratio", "cell", "model"])
    frame.to_csv(output_dir / "comparison.csv", index=False)
    columns = [
        "training_regime",
        "model",
        "cell",
        "accuracy@0.5",
        "balanced_accuracy@0.5",
        "f1@val_threshold",
        "auc",
        "average_precision",
        "brier",
    ]
    lines = [
        "# Task F prefix=256 mixed-mask benchmark",
        "",
        "One model is trained per architecture and training regime, then evaluated on fixed corruption cells.",
        "The Task-B summaries use only observed timestamps; slope uses original time coordinates.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = [
            str(row["training_regime"]),
            str(row["model"]),
            str(row["cell"]),
            f"{row['accuracy@0.5']:.4f}",
            f"{row['balanced_accuracy@0.5']:.4f}",
            f"{row['f1@val_threshold']:.4f}",
            f"{row['auc']:.4f}",
            f"{row['average_precision']:.4f}",
            f"{row['brier']:.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = build_summary_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("results/task_f_mixed_mask_benchmark_seed42"),
        prefix_len=256,
        models="tcn,attention,xgboost",
    )
    parser.add_argument("--models", type=str, default="tcn,attention,xgboost")
    parser.add_argument(
        "--training-regimes",
        type=str,
        default="clean_only,random_only,mixed",
        help="Comma-separated regimes: clean_only, random_only, mixed.",
    )
    parser.add_argument(
        "--test-ratios",
        type=str,
        default="0,10,20,30,40,50,60,70",
        help="Test ratios, in percent for values >=1.",
    )
    parser.add_argument("--location-ratio", type=float, default=25.0)
    parser.add_argument("--train-summary-views", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tcn-hidden", type=int, default=160)
    parser.add_argument("--tcn-levels", type=int, default=5)
    parser.add_argument("--attention-hidden", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prefix_len != 256:
        raise ValueError("This benchmark is deliberately fixed to --prefix-len 256.")
    if args.num_workers < 0 or args.batch_size <= 0:
        raise ValueError("num-workers must be non-negative and batch-size must be positive.")
    if args.train_summary_views <= 0:
        raise ValueError("--train-summary-views must be positive.")
    regimes = parse_regimes(args.training_regimes)
    models = parse_models(args.models)
    test_ratios = parse_mask_ratios(args.test_ratios)
    location_ratio = normalize_percent(args.location_ratio)
    if not 0.0 < location_ratio < 1.0:
        raise ValueError("location ratio must be in (0, 1).")

    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = ensure_dir(args.output_dir.resolve())
    if args.overwrite_output:
        for path in args.output_dir.iterdir():
            if path.is_file():
                path.unlink()
    set_random_seed(args.seed)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    features, data = load_task_f_data(args)
    labels: pd.DataFrame = data["labels"]
    targets = data["targets"]
    splits: dict[str, np.ndarray] = data["splits"]
    mean, std = calculate_normalization(
        features,
        data["global_indices"],
        splits["train"],
        args.read_batch_size,
        data["shape"][1],
        data["shape"][2],
    )
    np.savez(args.output_dir / "sequence_normalization.npz", mean=mean, std=std)
    labels.to_csv(args.output_dir / "labels_filtered.csv", index=False)
    np.savez(
        args.output_dir / "split_indices.npz",
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
        case_ids=labels["raw_case_id"].to_numpy(dtype=np.int64),
        cache_indices=data["global_indices"],
    )
    summary_names = task_b_summary_feature_names(FEATURE_NAMES)
    cells = build_test_cells(test_ratios, location_ratio)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "training_regimes": regimes,
            "models": models,
            "test_ratios": test_ratios,
            "location_ratio": location_ratio,
            "location_intervals_prefix256": {
                "early": [0, int(round(256 * location_ratio))],
                "middle": [(256 - int(round(256 * location_ratio))) // 2,
                           (256 + int(round(256 * location_ratio))) // 2],
                "late": [256 - int(round(256 * location_ratio)), 256],
            },
            "device": str(device),
            "use_amp": use_amp,
            "summary_statistics": TASK_B_STAT_NAMES,
            "summary_feature_dim": len(summary_names),
            "summary_feature_names": summary_names,
            "summary_mask_policy": (
                "Task-B 120 summaries over observed timestamps; initial/final are first/last observed; "
                "slope uses original normalized time coordinates"
            ),
            "training_mask_policy": {
                "clean_probability": 0.10,
                "ratio_range": [0.0, 0.50],
                "conditional_type_probabilities": {
                    "single_block": 0.50,
                    "multi_block": 0.30,
                    "random": 0.20,
                },
                "block_min_length": 4,
                "block_min_gap": 4,
            },
            "test_cells": cells,
            "split_counts": {name: int(rows.size) for name, rows in splits.items()},
            "stable_rate": float(targets.mean()),
            "cache_path": data["cache_path"],
        },
    )

    y_train = targets[splits["train"]]
    y_val = targets[splits["val"]]
    y_test = targets[splits["test"]]
    metrics_payload: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []

    for regime_index, regime in enumerate(regimes):
        print(f"\\n=== training regime: {regime} ===", flush=True)
        metrics_payload[regime] = {}
        val_seq_seed = args.seed + 10_000
        test_seq_seed = args.seed + 20_000
        train_seq_seed = args.seed
        val_dataset, val_loader = make_loader(
            features,
            data,
            splits["val"],
            mean,
            std,
            seed=val_seq_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            regime=regime,
            dynamic=False,
        )

        trained_models: dict[str, Any] = {}
        thresholds: dict[str, float] = {}

        for model_index, model_name in enumerate(models):
            model_seed = args.seed + regime_index * 100 + model_index
            set_random_seed(model_seed)
            print(f"[{regime}] training {model_name} (seed={model_seed})", flush=True)
            if model_name == "xgboost":
                train_summary = make_training_summaries(
                    features,
                    data,
                    splits["train"],
                    regime=regime,
                    seed=train_seq_seed,
                    batch_size=args.read_batch_size,
                    views=args.train_summary_views,
                )
                val_summary = make_validation_summaries(
                    features,
                    data,
                    splits["val"],
                    regime=regime,
                    seed=val_seq_seed,
                    batch_size=args.read_batch_size,
                )
                print(
                    f"[{regime}] XGBoost features train={train_summary.shape} val={val_summary.shape}",
                    flush=True,
                )
                model = train_xgb(
                    train_summary,
                    np.tile(y_train, 1 if regime == "clean_only" else args.train_summary_views),
                    val_summary,
                    y_val,
                    args=args,
                    random_state=model_seed,
                )
                p_val = model.predict_proba(val_summary)[:, 1]
                thresholds[model_name] = best_f1_threshold(y_val, p_val)
                trained_models[model_name] = model
                model_payload = {
                    "feature_dim": len(summary_names),
                    "train_views": 1 if regime == "clean_only" else args.train_summary_views,
                    "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
                    "validation_threshold": thresholds[model_name],
                }
                model.get_booster().save_model(
                    str(args.output_dir / f"{regime}_xgboost_model.json")
                )
                gain = model.get_booster().get_score(importance_type="gain")
                importance = pd.DataFrame(
                    {
                        "feature": summary_names,
                        "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(len(summary_names))],
                    }
                )
                total_gain = float(importance["gain"].sum())
                importance["gain_fraction"] = importance["gain"] / total_gain if total_gain else 0.0
                importance.sort_values("gain", ascending=False).to_csv(
                    args.output_dir / f"{regime}_xgboost_feature_importance.csv", index=False
                )
                metrics_payload[regime][model_name] = {"validation": model_payload, "cells": {}}
            else:
                train_dataset, train_loader = make_loader(
                    features,
                    data,
                    splits["train"],
                    mean,
                    std,
                    seed=train_seq_seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train=True,
                    regime=regime,
                    dynamic=True,
                )
                model = build_model(
                    model_name,
                    data["shape"][2],
                    dropout=args.dropout,
                    tcn_hidden=args.tcn_hidden,
                    tcn_levels=args.tcn_levels,
                    attention_hidden=args.attention_hidden,
                    attention_heads=args.attention_heads,
                    attention_layers=args.attention_layers,
                    max_len=args.prefix_len,
                    append_mask_channel=True,
                )
                model, history = train_model(
                    model,
                    train_dataset,
                    train_loader,
                    val_loader,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.early_stopping_patience,
                    device=device,
                    use_amp=use_amp,
                )
                y_val_model, p_val = predict(model, val_loader, device, use_amp)
                thresholds[model_name] = best_f1_threshold(y_val_model, p_val)
                trained_models[model_name] = model
                model_payload = {
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "epochs_completed": len(history),
                    "validation_threshold": thresholds[model_name],
                }
                save_csv(args.output_dir / f"{regime}_{model_name}_train_log.csv", history)
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    args.output_dir / f"{regime}_{model_name}_model.pt",
                )
                metrics_payload[regime][model_name] = {"validation": model_payload, "cells": {}}
            save_json(args.output_dir / "metrics.json", metrics_payload)
            print(
                f"[{regime}] {model_name} trained; validation threshold={thresholds[model_name]:.5f}",
                flush=True,
            )

        for cell_index, cell in enumerate(cells, start=1):
            tag = cell["tag"]
            print(
                f"[{regime}] evaluating cell {cell_index}/{len(cells)}: {tag}",
                flush=True,
            )
            xgb_test_summary: np.ndarray | None = None
            for model_name in models:
                model = trained_models[model_name]
                if model_name == "xgboost":
                    xgb_test_summary = make_test_summaries(
                        features,
                        data,
                        splits["test"],
                        kind=cell["kind"],
                        ratio=float(cell["ratio"]),
                        seed=test_seq_seed,
                        batch_size=args.read_batch_size,
                        fixed_start=cell.get("fixed_start"),
                    )
                    probability = model.predict_proba(xgb_test_summary)[:, 1]
                else:
                    _, test_loader = make_loader(
                        features,
                        data,
                        splits["test"],
                        mean,
                        std,
                        seed=test_seq_seed,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        fixed_kind=cell["kind"],
                        fixed_ratio=float(cell["ratio"]),
                        fixed_start=cell.get("fixed_start"),
                    )
                    _, probability = predict(model, test_loader, device, use_amp)
                cell_metrics = model_metric_record(y_test, probability, thresholds[model_name])
                cell_metrics["mask_ratio"] = float(cell["ratio"])
                cell_metrics["mask_kind"] = cell["kind"]
                if cell.get("location"):
                    cell_metrics["location"] = cell["location"]
                metrics_payload[regime][model_name]["cells"][tag] = cell_metrics
                comparison_rows.append(comparison_row(regime, model_name, cell, cell_metrics))
                np.savez_compressed(
                    args.output_dir / f"{regime}_{model_name}_{tag}_predictions.npz",
                    y=y_test.astype(np.int8),
                    probability=probability.astype(np.float32),
                )
                auc = cell_metrics["test_at_0.5"]["auc"]
                print(f"[{regime}] {model_name} {tag}: AUROC={auc:.4f}", flush=True)
            save_json(args.output_dir / "metrics.json", metrics_payload)
            write_comparison(args.output_dir, comparison_rows)

    save_json(args.output_dir / "metrics.json", metrics_payload)
    write_comparison(args.output_dir, comparison_rows)
    print(f"Task F mixed-mask benchmark complete. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

