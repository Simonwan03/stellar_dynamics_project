#!/usr/bin/env python3
"""Train a linear Task B probe on a frozen Chronos-2 feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

DEFAULT_FEATURE_CACHE = (
    EXPERIMENT_ROOT / "outputs/panda_unbound_dataset/chronos2_task_b_prefix96_reg_features.float16.npy"
)
DEFAULT_PREFIX_DIR = REPO_ROOT / "results/task_b_prefix_ablation/prefix_512"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_reg"


class CachedFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, rows: np.ndarray):
        self.features = features
        self.targets = np.asarray(targets, dtype=np.float32)
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[item])
        # The cache is float16 on disk; train the linear head in float32.
        feature = np.asarray(self.features[row], dtype=np.float32)
        return torch.from_numpy(feature), torch.tensor(self.targets[row], dtype=torch.float32)


class CachedLinearProbe(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.classifier = nn.Linear(int(feature_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(-1)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=True))
        targets.append(target.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    labels = pd.read_csv(args.prefix_dir / "labels_filtered.csv")
    split_file = np.load(args.prefix_dir / "split_indices.npz")
    split_rows = {name: split_file[f"{name}_idx"].astype(np.int64) for name in ("train", "val", "test")}
    all_split_rows = np.concatenate(list(split_rows.values()))
    if np.unique(all_split_rows).size != len(labels) or all_split_rows.size != len(labels):
        raise ValueError("The train/val/test split does not cover each Task B label exactly once.")

    metadata_path = args.feature_cache.with_suffix(".json")
    if not args.feature_cache.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Feature cache and metadata are both required: {args.feature_cache}")
    cache_metadata = read_json(metadata_path)
    expected_fingerprint = labels_fingerprint(labels)
    if cache_metadata.get("label_fingerprint") != expected_fingerprint:
        raise ValueError("Feature cache labels do not match the requested Task B label ordering.")
    if cache_metadata.get("pooling") != "reg":
        raise ValueError(f"Expected a Chronos-2 REG cache, got pooling={cache_metadata.get('pooling')!r}")
    if int(cache_metadata.get("prefix_steps", -1)) != 96:
        raise ValueError(f"Expected prefix_steps=96, got {cache_metadata.get('prefix_steps')!r}")

    features = np.load(args.feature_cache, mmap_mode="r")
    expected_shape = (len(labels), 15 * 768)
    if features.shape != expected_shape:
        raise ValueError(f"Expected feature cache shape {expected_shape}, got {features.shape}")
    if not np.isfinite(np.asarray(features[: min(32, len(features))], dtype=np.float32)).all():
        raise ValueError("Feature cache contains non-finite values in its first rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for filename in (
            "checkpoint_best.pt",
            "history.csv",
            "metrics.json",
            "predictions_all.csv",
            "run_config.json",
            "segmented_test_metrics.csv",
        ):
            path = args.output_dir / filename
            if path.exists():
                path.unlink()

    set_seed(args.seed)
    target_unbound = (labels["time_unbound"].to_numpy(dtype=np.float32) < 10000.0).astype(np.float32)
    loaders = {
        split: DataLoader(
            CachedFeatureDataset(features, target_unbound, rows),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        for split, rows in split_rows.items()
    }
    # Keep a deterministic, non-shuffled view for saved predictions.  The
    # training loader intentionally shuffles only the training split.
    eval_loaders = {
        split: DataLoader(
            CachedFeatureDataset(features, target_unbound, rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        for split, rows in split_rows.items()
    }

    model = CachedLinearProbe(features.shape[1]).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_target = target_unbound[split_rows["train"]]
    positive = float(train_target.sum())
    negative = float(len(train_target) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"Chronos-2 REG cache={args.feature_cache}; feature_dim={features.shape[1]}; "
        f"device={device}; trainable_parameters={trainable_parameters:,}",
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_features, target in loaders["train"]:
            batch_features = batch_features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            count += len(target)
        val_target, val_probability = predict(model, eval_loaders["val"], device)
        val_auc = float(roc_auc_score(val_target, val_probability))
        record = {"epoch": epoch, "train_loss": loss_sum / count, "val_auroc": val_auc}
        history.append(record)
        print(f"epoch={epoch:03d} loss={record['train_loss']:.6f} val_auroc={val_auc:.6f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "feature_dim": int(features.shape[1]),
            "feature_cache": str(args.feature_cache),
            "pooling": "reg",
            "head": "linear",
            "trainable_parameters": trainable_parameters,
            "best_val_auroc": best_auc,
        },
        args.output_dir / "checkpoint_best.pt",
    )

    all_metrics: dict[str, Any] = {}
    prediction_parts: list[pd.DataFrame] = []
    for split, rows in split_rows.items():
        y_unbound, prob_unbound = predict(model, eval_loaders[split], device)
        y_stable = 1.0 - y_unbound
        prob_stable = 1.0 - prob_unbound
        all_metrics[split] = {
            "unbound_positive": probability_metrics(y_unbound, prob_unbound),
            "stable_positive": probability_metrics(y_stable, prob_stable),
        }
        prediction = labels.iloc[rows][["index", "num_batch", "time_unbound", "diagnosis"]].copy()
        prediction["task_b_row"] = rows
        prediction["split"] = split
        prediction["target_unbound"] = y_unbound.astype(np.int64)
        prediction["prob_unbound"] = prob_unbound
        prediction["prob_stable"] = prob_stable
        prediction_parts.append(prediction)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(args.output_dir / "predictions_all.csv", index=False)
    test_predictions = predictions.loc[predictions["split"] == "test"]
    time_unbound = test_predictions["time_unbound"]
    prefix_end_time = 96.0 * 5.0
    segment_definitions = (
        ("prefix 96内已 unbound", time_unbound < prefix_end_time, True),
        ("prefix 96后才 unbound", (time_unbound >= prefix_end_time) & (time_unbound < 10000.0), True),
        ("一直到结束仍稳定", time_unbound >= 10000.0, False),
    )
    segmented_rows = []
    for segment, mask, expected_unbound in segment_definitions:
        probability = test_predictions.loc[mask, "prob_unbound"]
        segmented_rows.append(
            {
                "segment": segment,
                "count": int(mask.sum()),
                "correct_or_recall": float(((probability >= 0.5) == expected_unbound).mean()),
                "mean_prob_unbound": float(probability.mean()),
            }
        )
    pd.DataFrame(segmented_rows).to_csv(args.output_dir / "segmented_test_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    write_json(args.output_dir / "metrics.json", all_metrics)
    write_json(
        args.output_dir / "run_config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "feature_cache_metadata": cache_metadata,
            "feature_shape": list(features.shape),
            "trainable_parameters": trainable_parameters,
            "split_counts": {split: int(len(rows)) for split, rows in split_rows.items()},
        },
    )
    print(json.dumps(all_metrics["test"], ensure_ascii=False, indent=2), flush=True)
    print(f"Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
