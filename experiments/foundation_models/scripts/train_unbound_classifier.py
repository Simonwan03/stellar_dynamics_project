from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.project_data import (  # noqa: E402
    build_representation,
    case_summary,
    choose_default_dataset,
    clean_and_group_trajectories,
    ensure_dir,
    load_labels,
    load_result_arrays,
    load_yaml,
    required_keys_for_representation,
    stratified_case_split,
    summarize_audit,
    write_json,
)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TemporalConvClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


@dataclass
class SplitMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    positive_rate: float
    n: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict[str, Any], allow_cpu_flag: bool) -> torch.device:
    requested = str(cfg.get("device", "cuda")).lower()
    allow_cpu = bool(cfg.get("allow_cpu_fallback", False)) or allow_cpu_flag
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if allow_cpu:
            print("WARNING: CUDA is not available; using CPU fallback because allow_cpu_fallback is enabled.")
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is not available in this environment. Re-run on a GPU-visible shell, "
            "or pass --allow-cpu for a CPU fallback sanity run."
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def metric_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float, threshold: float) -> SplitMetrics:
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true_i = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_true_i == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_i == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_i == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_i == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return SplitMetrics(
        loss=float(loss),
        accuracy=float((tp + tn) / max(1, len(y_true_i))),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=metric_auroc(y_true_i, y_prob),
        positive_rate=float(y_true_i.mean()) if len(y_true_i) else float("nan"),
        n=int(len(y_true_i)),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[SplitMetrics, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()) * len(yb))
            ys.append(yb.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.float32)
    y_prob = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)
    mean_loss = sum(losses) / max(1, len(y_true))
    return compute_metrics(y_true, y_prob, mean_loss, threshold), y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unbound_dataset.yaml")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(exp_root / args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_cases is not None:
        cfg["max_cases"] = args.max_cases
    set_seed(int(cfg["seed"]))

    device = resolve_device(cfg, allow_cpu_flag=args.allow_cpu)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    choice = choose_default_dataset(cfg.get("sampling_id_preference", [0]))
    representation = str(cfg.get("representation", "orbital15"))
    max_cases = cfg.get("max_cases")
    max_rows = None if max_cases is None else int(max_cases) * 8
    prefix_steps = int(cfg["prefix_steps"])
    keys = required_keys_for_representation(representation)

    print("Dataset choice:", summarize_audit(choice))
    print("Representation:", representation)
    print("Prefix steps:", prefix_steps)

    labels = load_labels(choice, max_rows=max_rows)
    data = load_result_arrays(choice, keys=keys, time_steps=prefix_steps, max_rows=max_rows)
    x = build_representation(data, representation=representation)
    x, labels = clean_and_group_trajectories(x, labels)

    final_time = float(labels["time_unbound"].max())
    labels = labels.copy()
    labels["final_unbound"] = (labels["time_unbound"] < final_time).astype(int)
    case_df = case_summary(labels)
    case_df = stratified_case_split(
        case_df,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    labels = labels.merge(case_df[["case_id", "split", "tail_label", "stable_count"]], on="case_id", how="left")

    split_masks = {split: labels["split"].to_numpy() == split for split in ["train", "val", "test"]}
    train_mask = split_masks["train"]
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = x[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    y = labels["final_unbound"].to_numpy(dtype=np.float32)

    output_root = exp_root / str(cfg["output_root"]) / str(cfg["run_name"])
    ensure_dir(output_root)

    write_json(
        output_root / "dataset_summary.json",
        {
            "source_dataset": summarize_audit(choice),
            "representation": representation,
            "prefix_steps": prefix_steps,
            "feature_shape": list(x.shape),
            "final_time": final_time,
            "label_definition": "final_unbound = time_unbound < max(time_unbound among retained valid dataset rows)",
            "split_counts": {split: int(mask.sum()) for split, mask in split_masks.items()},
            "positive_counts": {split: int(y[mask].sum()) for split, mask in split_masks.items()},
            "positive_rates": {split: float(y[mask].mean()) for split, mask in split_masks.items()},
        },
    )
    labels.to_csv(output_root / "manifest_with_labels.csv", index=False)
    np.savez_compressed(output_root / "normalization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    loaders = {}
    for split, mask in split_masks.items():
        loaders[split] = DataLoader(
            SequenceDataset(x[mask], y[mask]),
            batch_size=int(cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
        )

    model = TemporalConvClassifier(
        in_channels=int(x.shape[-1]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    pos = float(y[train_mask].sum())
    neg = float(train_mask.sum() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    best_val = -float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_n += len(yb)
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device, float(cfg["threshold"]))
        record = {"epoch": epoch, "train_loss": train_loss_sum / max(1, train_n), "val": asdict(val_metrics)}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"val_auc={val_metrics.auroc:.4f} val_f1={val_metrics.f1:.4f} val_acc={val_metrics.accuracy:.4f}"
        )
        score = val_metrics.auroc if np.isfinite(val_metrics.auroc) else val_metrics.f1
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "feature_mean": mean.astype(np.float32),
                    "feature_std": std.astype(np.float32),
                    "epoch": epoch,
                    "best_val_score": best_val,
                },
                output_root / "checkpoint_best.pt",
            )

    checkpoint = torch.load(output_root / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    for split in ["train", "val", "test"]:
        split_metrics, y_true, y_prob = evaluate(model, loaders[split], criterion, device, float(cfg["threshold"]))
        metrics[split] = asdict(split_metrics)
        pred_df = labels.loc[split_masks[split], ["case_id", "perturb_id", "num_batch", "index", "time_unbound", "final_unbound", "tail_label"]].copy()
        pred_df["prob_unbound"] = y_prob
        pred_df["pred_unbound"] = (y_prob >= float(cfg["threshold"])).astype(int)
        pred_df.to_csv(output_root / f"predictions_{split}.csv", index=False)

    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "history.json", history)
    write_json(output_root / "run_config.json", cfg)
    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {output_root}")


if __name__ == "__main__":
    main()
