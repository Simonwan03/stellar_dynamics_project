#!/usr/bin/env python3
"""Compare Task-F models under one- and two-block contiguous masks."""
from __future__ import annotations

import argparse
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
    comparison_row,
    load_existing_comparison,
    load_task_f_data,
    parse_mask_ratios,
    train_xgboost,
)
from tasks.common.experiment_utils import (  # noqa: E402
    binary_metrics_from_probs,
    ensure_dir,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.task_f.block_masking import contiguous_mask_mode_observation  # noqa: E402
from tasks.task_f.main import best_f1_threshold, predict, train_model  # noqa: E402
from tasks.task_f.masking import apply_observation_mask  # noqa: E402
from tasks.task_f.models import build_model  # noqa: E402
from tasks.task_f.summaries import (  # noqa: E402
    TASK_B_STAT_NAMES,
    summarize_observed_prefix,
    task_b_summary_feature_names,
)


class ContiguousMaskedMemmapDataset(Dataset):
    """Task-F dataset using deterministic one- or two-block masks."""

    def __init__(
        self,
        features: np.memmap,
        rows: np.ndarray,
        global_indices: np.ndarray,
        targets: np.ndarray,
        sample_ids: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        mask_ratio: float,
        seed: int,
        mode: str,
        dynamic: bool,
    ) -> None:
        self.features = features
        self.rows = np.asarray(rows, dtype=np.int64)
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)
        self.mode = mode
        self.dynamic = bool(dynamic)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[index])
        values = np.asarray(self.features[self.global_indices[row]], dtype=np.float32)
        values = np.array((values - self.mean[0]) / self.std[0], dtype=np.float32, copy=True)
        observed = contiguous_mask_mode_observation(
            values.shape[0],
            self.mask_ratio,
            seed=self.seed,
            sample_id=int(self.sample_ids[row]),
            epoch=self.epoch if self.dynamic else 0,
            mode=self.mode,
        )
        masked = apply_observation_mask(values, observed, append_mask_channel=True)
        target = np.asarray([self.targets[row]], dtype=np.float32)
        return torch.from_numpy(masked), torch.from_numpy(target)


def make_block_loader(
    features: np.memmap,
    data: dict[str, Any],
    rows: np.ndarray,
    ratio: float,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
    mode: str,
    train: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[ContiguousMaskedMemmapDataset, DataLoader]:
    dataset = ContiguousMaskedMemmapDataset(
        features,
        rows,
        data["global_indices"],
        data["targets"],
        data["sample_ids"],
        mean,
        std,
        ratio,
        seed,
        mode,
        dynamic=train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
    )
    return dataset, loader


def block_summary_batch(
    values: np.ndarray,
    sample_ids: np.ndarray,
    ratio: float,
    *,
    seed: int,
    epoch: int,
    mode: str,
) -> np.ndarray:
    observed = np.empty(values.shape[:2], dtype=bool)
    for row, sample_id in enumerate(np.asarray(sample_ids, dtype=np.int64)):
        observed[row] = contiguous_mask_mode_observation(
            values.shape[1],
            ratio,
            seed=seed,
            sample_id=int(sample_id),
            epoch=epoch,
            mode=mode,
        )
    return summarize_observed_prefix(values, observed)


def make_block_summary_features(
    features: np.memmap,
    data: dict[str, Any],
    ratio: float,
    *,
    mode: str,
    seed: int,
    batch_size: int,
    save_path: Path | None,
) -> np.ndarray:
    targets = data["targets"]
    global_indices = data["global_indices"]
    sample_ids = data["sample_ids"]
    splits = data["splits"]
    shape = (len(targets), features.shape[2] * len(TASK_B_STAT_NAMES))
    if save_path is None:
        output = np.empty(shape, dtype=np.float32)
    else:
        output = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32, shape=shape)
    split_mask_settings = {
        "train": (seed, 1),
        "val": (seed + 10_000, 0),
        "test": (seed + 20_000, 0),
    }
    for split_name, rows in splits.items():
        split_seed, split_epoch = split_mask_settings[split_name]
        selected_global = global_indices[rows]
        selected_ids = sample_ids[rows]
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            values = np.asarray(features[selected_global[start:stop]], dtype=np.float32)
            output[rows[start:stop]] = block_summary_batch(
                values,
                selected_ids[start:stop],
                ratio,
                seed=split_seed,
                epoch=split_epoch,
                mode=mode,
            )
    if hasattr(output, "flush"):
        output.flush()
    return output


def load_random_summary_rows() -> list[dict[str, Any]]:
    path = ROOT / "results/task_f_summary_xgboost/metrics.json"
    if not path.exists():
        return []
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for ratio_payload in payload.values():
        if "xgboost" not in ratio_payload:
            continue
        row = comparison_row(
            float(ratio_payload["xgboost"]["mask_ratio"]),
            "Summary + XGBoost (random)",
            ratio_payload["xgboost"],
        )
        rows.append(row)
    return rows


def write_block_comparison(output_dir: Path, rows: list[dict[str, Any]], mode: str) -> None:
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
        f"# Task F contiguous mask comparison ({mode})",
        "",
        "Block masks use the same Task-F split, labels, and seed offsets as the random-mask runs.",
        "Dynamical summaries use only observed positions; initial/final are first/last observed.",
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
    parser = build_summary_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("results/task_f_contiguous"))
    parser.add_argument("--mask-mode", choices=("contiguous1", "contiguous2"), required=True)
    parser.add_argument("--models", type=str, default="tcn,attention,xgboost")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = ensure_dir(args.output_dir.resolve())
    ratios = parse_mask_ratios(args.mask_ratios)
    models = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    allowed = {"tcn", "attention", "xgboost"}
    if not models or set(models) - allowed:
        raise ValueError(f"Choose models from {sorted(allowed)}, got {models}.")
    if args.batch_size <= 0 or args.read_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.overwrite_output:
        for path in args.output_dir.glob("mask*"):
            if path.is_file():
                path.unlink()
        for name in ("metrics.json", "comparison.csv", "comparison.md", "config.json"):
            (args.output_dir / name).unlink(missing_ok=True)

    set_random_seed(args.seed)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    features, data = load_task_f_data(args)
    labels: pd.DataFrame = data["labels"]
    targets = data["targets"]
    splits = data["splits"]
    mean, std = calculate_normalization(
        features,
        data["global_indices"],
        splits["train"],
        args.batch_size,
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
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "mask_mode": args.mask_mode,
            "mask_ratios": ratios,
            "device": str(device),
            "use_amp": use_amp,
            "summary_statistics": TASK_B_STAT_NAMES,
            "summary_feature_dim": len(summary_names),
            "summary_feature_names": summary_names,
            "summary_mask_policy": (
                "only observed positions; first/last observed for initial/final; "
                "slope uses original normalized time coordinates"
            ),
            "split_counts": {name: int(rows.size) for name, rows in splits.items()},
            "stable_rate": float(targets.mean()),
            "cache_path": data["cache_path"],
        },
    )

    all_metrics: dict[str, Any] = {}
    comparison_rows = load_existing_comparison(args.output_dir) + load_random_summary_rows()
    y_test = targets[splits["test"]]
    for ratio in ratios:
        tag = f"{args.mask_mode}_{int(round(ratio * 100)):02d}"
        all_metrics[tag] = {}
        print(f"Preparing contiguous summaries and loaders: mode={args.mask_mode} mask={ratio:.0%}", flush=True)
        summary_path = args.output_dir / f"{tag}_summary_features.float32.npy" if args.save_features else None
        summary_features = make_block_summary_features(
            features,
            data,
            ratio,
            mode=args.mask_mode,
            seed=args.seed,
            batch_size=args.read_batch_size,
            save_path=summary_path,
        )
        x_train = np.asarray(summary_features[splits["train"]])
        x_val = np.asarray(summary_features[splits["val"]])
        x_test = np.asarray(summary_features[splits["test"]])
        y_train = targets[splits["train"]]
        y_val = targets[splits["val"]]
        train_ds, train_loader = make_block_loader(
            features, data, splits["train"], ratio, mean=mean, std=std,
            seed=args.seed, mode=args.mask_mode, train=True, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        _, val_loader = make_block_loader(
            features, data, splits["val"], ratio, mean=mean, std=std,
            seed=args.seed + 10_000, mode=args.mask_mode, train=False, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        _, test_loader = make_block_loader(
            features, data, splits["test"], ratio, mean=mean, std=std,
            seed=args.seed + 20_000, mode=args.mask_mode, train=False, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        for model_name in models:
            set_random_seed(args.seed)
            if model_name == "xgboost":
                model = train_xgboost(x_train, y_train, x_val, y_val, args=args)
                p_train = model.predict_proba(x_train)[:, 1]
                p_val = model.predict_proba(x_val)[:, 1]
                p_test = model.predict_proba(x_test)[:, 1]
                model.get_booster().save_model(str(args.output_dir / f"{tag}_xgboost_model.json"))
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
                    args.output_dir / f"{tag}_xgboost_feature_importance.csv", index=False
                )
                model_metrics = {
                    "feature_dim": len(summary_names),
                    "mask_ratio": ratio,
                    "mask_mode": args.mask_mode,
                    "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
                    "validation_threshold": best_f1_threshold(y_val, p_val),
                    "test_at_0.5": binary_metrics_from_probs(y_test, p_test),
                    "test_val_f1_threshold": binary_metrics_from_probs(
                        y_test, p_test, best_f1_threshold(y_val, p_val)
                    ),
                }
                model_label = "Summary + XGBoost (block)"
                prediction_frame = labels.iloc[splits["test"]][
                    ["index", "raw_case_id", "time_unbound", "diagnosis", "split"]
                ].copy()
                prediction_frame["target_stable"] = y_test.astype(np.int64)
                prediction_frame["prob_stable"] = p_test
                prediction_frame["mask_ratio"] = ratio
                prediction_frame.to_csv(args.output_dir / f"{tag}_xgboost_test_predictions.csv", index=False)
                save_binary_curves(
                    args.output_dir / f"{tag}_xgboost_test_curves.png", y_test, p_test,
                    f"Task F {args.mask_mode} summary + XGBoost mask={ratio:.0%}",
                )
            else:
                model = build_model(
                    model_name,
                    data["shape"][2],
                    dropout=args.dropout,
                    tcn_hidden=args.tcn_hidden,
                    tcn_levels=args.tcn_levels,
                    attention_hidden=args.attention_hidden,
                    attention_heads=args.attention_heads,
                    attention_layers=args.attention_layers,
                    max_len=data["shape"][1],
                    append_mask_channel=True,
                )
                model, history = train_model(
                    model,
                    train_ds,
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
                p_test = predict(model, test_loader, device, use_amp)[1]
                threshold = best_f1_threshold(y_val_model, p_val)
                model_metrics = {
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "mask_ratio": ratio,
                    "mask_mode": args.mask_mode,
                    "validation_threshold": threshold,
                    "test_at_0.5": binary_metrics_from_probs(y_test, p_test),
                    "test_val_f1_threshold": binary_metrics_from_probs(y_test, p_test, threshold),
                }
                model_label = model_name.capitalize()
                save_csv(args.output_dir / f"{tag}_{model_name}_train_log.csv", history)
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    args.output_dir / f"{tag}_{model_name}_model.pt",
                )
                save_binary_curves(
                    args.output_dir / f"{tag}_{model_name}_test_curves.png", y_test, p_test,
                    f"Task F {args.mask_mode} {model_name} mask={ratio:.0%}",
                )
            all_metrics[tag][model_name] = model_metrics
            comparison_rows.append(comparison_row(ratio, model_label, model_metrics))
            save_json(args.output_dir / "metrics.json", all_metrics)
            print(
                f"mode={args.mask_mode} mask={ratio:.0%} model={model_name} "
                f"auc={model_metrics['test_at_0.5']['auc']:.4f} "
                f"f1@val={model_metrics['test_val_f1_threshold']['f1']:.4f}",
                flush=True,
            )
    save_json(args.output_dir / "metrics.json", all_metrics)
    write_block_comparison(args.output_dir, comparison_rows, args.mask_mode)
    print(f"Task F contiguous mask comparison complete. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
