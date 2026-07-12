from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_name_list(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def stratified_split_indices(
    y: np.ndarray,
    test_size: int | float = 0.2,
    val_size: int | float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    n = y.size
    if n < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {n}.")

    def resolve_size(size: int | float, base: int) -> int:
        if isinstance(size, float) and 0.0 < size < 1.0:
            return int(round(base * size))
        return int(size)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    def sample_stratified(pool: np.ndarray, target_size: int) -> np.ndarray:
        target_size = max(0, min(int(target_size), pool.size - 1))
        if target_size == 0:
            return np.array([], dtype=np.int64)
        classes = np.unique(y[pool])
        selected: list[np.ndarray] = []
        remaining = target_size
        for cls in classes[:-1]:
            cls_idx = pool[y[pool] == cls]
            take = int(round(target_size * cls_idx.size / pool.size))
            take = min(take, cls_idx.size)
            selected.append(rng.choice(cls_idx, size=take, replace=False))
            remaining -= take
        cls_idx = pool[y[pool] == classes[-1]]
        selected.append(rng.choice(cls_idx, size=min(remaining, cls_idx.size), replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
        if out.size < target_size:
            extras = np.setdiff1d(pool, out, assume_unique=False)
            out = np.concatenate([out, rng.choice(extras, size=target_size - out.size, replace=False)])
        rng.shuffle(out)
        return out.astype(np.int64)

    test_count = resolve_size(test_size, n)
    test_idx = sample_stratified(all_idx, test_count)
    train_val_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
    val_count = resolve_size(val_size, train_val_idx.size)
    val_idx = sample_stratified(train_val_idx, val_count)
    train_idx = np.setdiff1d(train_val_idx, val_idx, assume_unique=False)
    rng.shuffle(train_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def standardize_from_train(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    out_train = ((X_train - mu) / std).astype(np.float32)
    out_arrays = [((arr - mu) / std).astype(np.float32) for arr in arrays]
    return out_train, out_arrays, {"mean": mu.astype(np.float32), "std": std.astype(np.float32)}


def binary_metrics_from_probs(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    prob = np.asarray(prob).reshape(-1).astype(np.float64)
    pred = (prob >= threshold).astype(np.int64)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "positive_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "predicted_positive_rate": float(pred.mean()) if pred.size else float("nan"),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, prob))
        metrics["average_precision"] = float(average_precision_score(y_true, prob))
    else:
        metrics["auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def save_binary_curves(path: str | Path, y_true: np.ndarray, prob: np.ndarray, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    prob = np.asarray(prob).reshape(-1).astype(np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, prob)
        precision, recall, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(fpr, tpr)
        axes[1].plot(recall, precision)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--")
    axes[0].set_title("ROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def select_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def close_h5_handles(objects: list[Any]) -> None:
    for obj in objects:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
