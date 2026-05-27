from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiment_utils import ensure_dir, parse_name_list, save_csv, save_json, select_device, set_random_seed
from task_b_prefix_classifiers import (
    build_prefix_features,
    discover_result_paths,
    filter_valid_samples,
    load_aligned_labels,
    load_result_prefixes,
)


class TrajectorySurvivalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.q = self.y.mean(dim=1)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.q[index]


class DirectCurveDataset(Dataset):
    def __init__(self, X_case: np.ndarray, q: np.ndarray):
        self.X_case = torch.from_numpy(X_case).float()
        self.q = torch.from_numpy(q).float()

    def __len__(self) -> int:
        return self.X_case.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_case[index], self.q[index]


def parse_future_times(value: str) -> np.ndarray:
    arr = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Pass at least one future time.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"Future times must be strictly increasing, got {arr}.")
    return arr


def build_case_tensors(
    X_traj: np.ndarray,
    labels,
    group_size: int,
    future_times: np.ndarray,
    max_cases: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.copy()
    labels["case_id"] = labels["index"] // group_size
    labels["variation_id"] = labels["index"] % group_size
    labels["_row"] = np.arange(len(labels), dtype=np.int64)
    complete_ids = []
    for case_id, group in labels.groupby("case_id", sort=True):
        if len(group) != group_size:
            continue
        if set(group["variation_id"].to_numpy(dtype=int)) != set(range(group_size)):
            continue
        complete_ids.append(int(case_id))
        if max_cases is not None and len(complete_ids) >= max_cases:
            break
    if not complete_ids:
        raise ValueError("No complete case groups survived filtering.")

    X_cases: list[np.ndarray] = []
    y_cases: list[np.ndarray] = []
    for case_id in complete_ids:
        group = labels[labels["case_id"] == case_id].sort_values("variation_id")
        rows = group["_row"].to_numpy(dtype=np.int64)
        times = group["time_unbound"].to_numpy(dtype=np.float32)
        X_cases.append(X_traj[rows])
        y_cases.append((times[:, None] >= future_times[None, :]).astype(np.float32))
    return (
        np.stack(X_cases).astype(np.float32),
        np.stack(y_cases).astype(np.float32),
        np.asarray(complete_ids, dtype=np.int64),
    )


def stratified_case_split(q: np.ndarray, test_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = q.shape[0]
    anchor = q[:, -1]
    bins = np.digitize(anchor, bins=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]), right=True)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for cls in np.unique(bins):
        idx = np.where(bins == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        n_val = int(round(len(idx) * val_ratio))
        if len(idx) > 2:
            n_test = min(max(n_test, 1), len(idx) - 2)
            n_val = min(max(n_val, 1), len(idx) - n_test - 1)
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test : n_test + n_val].tolist())
        train.extend(idx[n_test + n_val :].tolist())
    for arr in (train, val, test):
        rng.shuffle(arr)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64)


def standardize_case_train(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mu = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((X_train - mu) / std).astype(np.float32),
        [((arr - mu) / std).astype(np.float32) for arr in arrays],
        {"mean": mu.astype(np.float32), "std": std.astype(np.float32)},
    )


class ConvTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


class GRUTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.dropout(torch.nn.functional.gelu(self.input(x))))
        return h[:, -1]


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        return self.encoder(h)[:, 0]


class MonotoneHead(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, dropout: float, monotone: bool):
        super().__init__()
        self.horizon = horizon
        self.monotone = monotone
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if not self.monotone or self.horizon == 1:
            return raw
        z0 = raw[..., :1]
        drops = torch.nn.functional.softplus(raw[..., 1:])
        return torch.cat([z0, z0 - torch.cumsum(drops, dim=-1)], dim=-1)


class TrajectoryContextSurvivalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        encoder_type: str,
        dropout: float,
        monotone: bool,
    ):
        super().__init__()
        if encoder_type == "gru":
            self.encoder = GRUTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "attention":
            self.encoder = TemporalAttentionEncoder(input_dim, hidden_dim, heads=4, layers=1, dropout=dropout)
        else:
            self.encoder = ConvTrajectoryEncoder(input_dim, hidden_dim, dropout)
        self.head = MonotoneHead(hidden_dim * 2, horizon, hidden_dim, dropout, monotone=monotone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, group, time, dim = x.shape
        emb = self.encoder(x.reshape(bsz * group, time, dim)).reshape(bsz, group, -1)
        context = emb.mean(dim=1, keepdim=True).expand(-1, group, -1)
        return self.head(torch.cat([emb, context], dim=-1))


class DirectCurveModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float):
        super().__init__()
        self.encoder = ConvTrajectoryEncoder(input_dim, hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class DualAttentionSurvivalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float, monotone: bool):
        super().__init__()
        self.temporal = TemporalAttentionEncoder(input_dim, hidden_dim, heads=4, layers=1, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.ensemble = nn.TransformerEncoder(layer, num_layers=1)
        self.head = MonotoneHead(hidden_dim * 2, horizon, hidden_dim, dropout, monotone=monotone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, group, time, dim = x.shape
        emb = self.temporal(x.reshape(bsz * group, time, dim)).reshape(bsz, group, -1)
        contextual = self.ensemble(emb)
        context = contextual.mean(dim=1, keepdim=True).expand(-1, group, -1)
        return self.head(torch.cat([contextual, context], dim=-1))


class TrajectorySurvivalLoss(nn.Module):
    def __init__(self, lambda_mono: float = 0.05, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.lambda_mono = float(lambda_mono)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        mono = torch.relu(probs[..., 1:] - probs[..., :-1]).mean() if probs.size(-1) > 1 else torch.tensor(0.0, device=logits.device)
        return bce + self.lambda_mono * mono


class DirectCurveLoss(nn.Module):
    def __init__(self, lambda_mono: float = 0.05):
        super().__init__()
        self.lambda_mono = float(lambda_mono)

    def forward(self, logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        mse = torch.mean((probs - q) ** 2)
        mono = torch.relu(probs[:, 1:] - probs[:, :-1]).mean() if probs.size(-1) > 1 else torch.tensor(0.0, device=logits.device)
        return mse + self.lambda_mono * mono


def compute_pos_weight(y: np.ndarray, min_value: float = 0.25, max_value: float = 4.0) -> torch.Tensor:
    positives = y.sum(axis=(0, 1)).astype(np.float32)
    total = float(y.shape[0] * y.shape[1])
    negatives = total - positives
    weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(np.clip(weight, min_value, max_value).reshape(1, 1, -1), dtype=torch.float32)


@torch.no_grad()
def evaluate_trajectory_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(float(loss.item()) * xb.size(0))
        preds.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        targets.append(yb.mean(dim=1).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    abs_err = np.abs(pred - target)
    return {
        "loss": float(np.sum(losses) / len(loader.dataset)),
        "overall_mae": float(abs_err.mean()),
        "final_mae": float(abs_err[:, -1].mean()),
        "horizon_mae": abs_err.mean(axis=0),
        "preds": pred,
        "targets": target,
    }


@torch.no_grad()
def evaluate_direct_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for xb, qb in loader:
        xb = xb.to(device)
        qb = qb.to(device)
        logits = model(xb)
        loss = criterion(logits, qb)
        losses.append(float(loss.item()) * xb.size(0))
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(qb.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    abs_err = np.abs(pred - target)
    return {
        "loss": float(np.sum(losses) / len(loader.dataset)),
        "overall_mae": float(abs_err.mean()),
        "final_mae": float(abs_err[:, -1].mean()),
        "horizon_mae": abs_err.mean(axis=0),
        "preds": pred,
        "targets": target,
    }


def train_trajectory_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
        val = evaluate_trajectory_model(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset), "val_loss": val["loss"], "val_mae": val["overall_mae"]}
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} val_mae={row['val_mae']:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_direct_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, qb in train_loader:
            xb = xb.to(device)
            qb = qb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), qb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
        val = evaluate_direct_model(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset), "val_loss": val["loss"], "val_mae": val["overall_mae"]}
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} val_mae={row['val_mae']:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def mean_baseline_metrics(q_train: np.ndarray, q_eval: np.ndarray) -> dict[str, Any]:
    pred = np.broadcast_to(q_train.mean(axis=0, keepdims=True), q_eval.shape)
    abs_err = np.abs(pred - q_eval)
    return {
        "overall_mae": float(abs_err.mean()),
        "final_mae": float(abs_err[:, -1].mean()),
        "horizon_mae": abs_err.mean(axis=0),
        "preds": pred,
        "targets": q_eval,
    }


def save_survival_plots(output_dir: Path, future_times: np.ndarray, metrics: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, payload in metrics.items():
        if "horizon_mae" in payload:
            ax.plot(future_times, payload["horizon_mae"], marker="o", label=name)
    ax.set_xlabel("Future time")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Survival-fraction MAE by horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_mae.png", dpi=160)
    plt.close(fig)

    best_name = min((k for k, v in metrics.items() if "preds" in v), key=lambda k: metrics[k]["overall_mae"])
    best = metrics[best_name]
    count = min(6, best["preds"].shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.2 * count), sharex=True)
    if count == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(future_times, best["targets"][i], marker="o", label="target")
        ax.plot(future_times, best["preds"][i], marker="x", label=best_name)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Future time")
    fig.tight_layout()
    fig.savefig(output_dir / "example_survival_curves.png", dpi=160)
    plt.close(fig)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k not in {"preds", "targets"}}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task C: trajectory-level multi-horizon survival prediction.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_c"))
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=200)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--future-times", type=str, default="500,1000,2000,4000,6000,8000,10000")
    parser.add_argument("--models", type=str, default="baseline,direct,traj,dual")
    parser.add_argument("--encoder-type", choices=["cnn", "gru", "attention"], default="cnn")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda-mono", type=float, default=0.05)
    parser.add_argument("--no-monotone-head", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--no-time", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    future_times = parse_future_times(args.future_times)
    requested_models = set(parse_name_list(args.models))

    result_paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    data, batch_ids, rows_loaded = load_result_prefixes(
        result_paths,
        prefix_len=args.prefix_len,
        time_stride=args.time_stride,
        rows_per_shard=args.rows_per_shard,
        include_geom=not args.no_geom,
        add_time=not args.no_time,
    )
    labels = load_aligned_labels(args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt", batch_ids, rows_loaded)
    X_traj, feature_names = build_prefix_features(data, add_time=not args.no_time, include_geom=not args.no_geom)
    X_traj, _, labels, _ = filter_valid_samples(X_traj, labels, data, time_horizon=None)
    X_cases, y_cases, case_ids = build_case_tensors(X_traj, labels, args.group_size, future_times, max_cases=args.max_cases)
    q_cases = y_cases.mean(axis=1)

    train_idx, val_idx, test_idx = stratified_case_split(q_cases, args.test_ratio, args.val_ratio, args.seed)
    X_train, X_val, X_test = X_cases[train_idx], X_cases[val_idx], X_cases[test_idx]
    y_train, y_val, y_test = y_cases[train_idx], y_cases[val_idx], y_cases[test_idx]
    q_train, q_val, q_test = q_cases[train_idx], q_cases[val_idx], q_cases[test_idx]
    X_train, [X_val, X_test], norm = standardize_case_train(X_train, X_val, X_test)

    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, case_ids=case_ids)
    np.savez(output_dir / "normalization_stats.npz", mean=norm["mean"], std=norm["std"])
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "future_times": future_times,
            "result_paths": result_paths,
            "num_cases": int(X_cases.shape[0]),
            "input_shape": tuple(X_cases.shape[1:]),
            "feature_names": feature_names,
            "train_cases": int(train_idx.size),
            "val_cases": int(val_idx.size),
            "test_cases": int(test_idx.size),
        },
    )

    train_loader = DataLoader(TrajectorySurvivalDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(TrajectorySurvivalDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(TrajectorySurvivalDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics: dict[str, dict[str, Any]] = {}
    if "baseline" in requested_models:
        metrics["baseline"] = mean_baseline_metrics(q_train, q_test)

    if "direct" in requested_models:
        X_case_train = X_train.mean(axis=1)
        X_case_val = X_val.mean(axis=1)
        X_case_test = X_test.mean(axis=1)
        direct_train_loader = DataLoader(DirectCurveDataset(X_case_train, q_train), batch_size=args.batch_size, shuffle=True)
        direct_val_loader = DataLoader(DirectCurveDataset(X_case_val, q_val), batch_size=args.batch_size, shuffle=False)
        direct_test_loader = DataLoader(DirectCurveDataset(X_case_test, q_test), batch_size=args.batch_size, shuffle=False)
        model = DirectCurveModel(X_train.shape[-1], args.hidden_dim, len(future_times), args.dropout)
        criterion = DirectCurveLoss(lambda_mono=args.lambda_mono)
        model, history = train_direct_model(model, direct_train_loader, direct_val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["direct"] = evaluate_direct_model(model, direct_test_loader, criterion, device)
        save_csv(output_dir / "direct_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "direct_model.pt")

    pos_weight = None if args.no_pos_weight else compute_pos_weight(y_train)
    criterion = TrajectorySurvivalLoss(lambda_mono=args.lambda_mono, pos_weight=pos_weight)

    if "traj" in requested_models:
        model = TrajectoryContextSurvivalModel(
            X_train.shape[-1],
            args.hidden_dim,
            len(future_times),
            encoder_type=args.encoder_type,
            dropout=args.dropout,
            monotone=not args.no_monotone_head,
        )
        model, history = train_trajectory_model(model, train_loader, val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["traj"] = evaluate_trajectory_model(model, test_loader, criterion, device)
        save_csv(output_dir / "traj_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "traj_model.pt")

    if "dual" in requested_models:
        model = DualAttentionSurvivalModel(
            X_train.shape[-1],
            args.hidden_dim,
            len(future_times),
            dropout=args.dropout,
            monotone=not args.no_monotone_head,
        )
        model, history = train_trajectory_model(model, train_loader, val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["dual"] = evaluate_trajectory_model(model, test_loader, criterion, device)
        save_csv(output_dir / "dual_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "dual_model.pt")

    save_survival_plots(output_dir, future_times, metrics)
    np.savez(
        output_dir / "test_predictions.npz",
        **{f"{name}_preds": payload["preds"] for name, payload in metrics.items() if "preds" in payload},
        targets=q_test,
        future_times=future_times,
    )
    save_json(output_dir / "metrics.json", {name: compact_metrics(payload) for name, payload in metrics.items()})
    print(f"Task C complete. Outputs: {output_dir}")
    for name, payload in metrics.items():
        print(name, compact_metrics(payload))


if __name__ == "__main__":
    main()
