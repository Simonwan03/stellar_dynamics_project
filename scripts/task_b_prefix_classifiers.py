from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiment_utils import (
    binary_metrics_from_probs,
    ensure_dir,
    parse_name_list,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
    stratified_split_indices,
)


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
RUNS_PER_BATCH = 8000
ORBITAL_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "masses_complete",
    "eccs_complete",
    "smaxes_complete",
    "energy_complete",
    "angMomentum_complete",
    "APs_complete",
    "LANs_complete",
    "incs_complete",
)
INITIAL_KEYS = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
ANGLE_RESULT_KEYS = {"APs_complete", "LANs_complete", "incs_complete"}
ANGLE_INITIAL_KEYS = {"APs", "LANs", "incs"}


def batch_id_from_path(path: str | Path) -> int:
    match = RESULT_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse result batch id from {path}")
    return int(match.group(1))


def discover_result_paths(dataset_dir: str | Path, sampling_id: int, max_shards: int | None = None) -> list[Path]:
    paths = sorted(Path(dataset_dir).glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No result shards found in {dataset_dir} for sampling {sampling_id}.")
    return paths


def sincos(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


def pairwise_geom(pos: np.ndarray, vel: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pairs = ((0, 1), (0, 2), (1, 2))
    channels: list[np.ndarray] = []
    for i, j in pairs:
        dr = pos[:, :, i, :] - pos[:, :, j, :]
        dv = vel[:, :, i, :] - vel[:, :, j, :]
        dist = np.linalg.norm(dr, axis=-1, keepdims=True)
        speed = np.linalg.norm(dv, axis=-1, keepdims=True)
        radial = (dr * dv).sum(axis=-1, keepdims=True) / np.maximum(dist, eps)
        channels.extend([dist, speed, radial])
    return np.concatenate(channels, axis=-1).astype(np.float32)


def load_result_prefixes(
    result_paths: list[Path],
    prefix_len: int,
    time_stride: int,
    rows_per_shard: int | None,
    include_geom: bool,
    add_time: bool,
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    data_parts: dict[str, list[np.ndarray]] = {key: [] for key in ORBITAL_KEYS}
    if include_geom:
        data_parts["positions_complete"] = []
        data_parts["velocities_complete"] = []
    if add_time:
        data_parts["times_complete"] = []
    batch_ids: list[int] = []
    rows_loaded: list[int] = []
    time_sel = slice(0, prefix_len, time_stride)

    for path in result_paths:
        batch_id = batch_id_from_path(path)
        with h5py.File(path, "r") as handle:
            n = handle["times_complete"].shape[0]
            if rows_per_shard is not None:
                n = min(n, rows_per_shard)
            row_sel = slice(0, n)
            for key in data_parts:
                if key == "times_complete":
                    arr = np.asarray(handle[key][row_sel, time_sel], dtype=np.float32)
                else:
                    arr = np.asarray(handle[key][row_sel, time_sel, ...], dtype=np.float32)
                data_parts[key].append(arr)
        batch_ids.append(batch_id)
        rows_loaded.append(n)

    data = {key: np.concatenate(parts, axis=0) for key, parts in data_parts.items()}
    return data, batch_ids, rows_loaded


def load_aligned_labels(
    labels_path: str | Path,
    batch_ids: list[int],
    rows_loaded: list[int],
    runs_per_batch: int = RUNS_PER_BATCH,
) -> pd.DataFrame:
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    labels = labels.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float32",
            "wall_time": "float32",
            "diagnosis": "int64",
        }
    )
    labels["batch_id"] = labels["index"] // runs_per_batch
    labels["row_in_batch"] = labels["index"] % runs_per_batch
    frames = []
    for batch_id, row_count in zip(batch_ids, rows_loaded):
        frame = labels[(labels["batch_id"] == batch_id) & (labels["row_in_batch"] < row_count)].copy()
        frame = frame.sort_values("row_in_batch")
        if len(frame) != row_count:
            raise ValueError(f"Expected {row_count} labels for batch {batch_id}, got {len(frame)}.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_prefix_features(
    data: dict[str, np.ndarray],
    add_time: bool = True,
    include_geom: bool = True,
) -> tuple[np.ndarray, list[str]]:
    channels: list[np.ndarray] = []
    names: list[str] = []
    for key in ORBITAL_KEYS:
        arr = data[key]
        if key in ANGLE_RESULT_KEYS:
            arr = sincos(arr)
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1).astype(np.float32)
        channels.append(flat)
        names.extend([f"{key}_{i}" for i in range(flat.shape[-1])])
    if include_geom:
        geom = pairwise_geom(data["positions_complete"], data["velocities_complete"])
        channels.append(geom)
        names.extend([f"geom_{i}" for i in range(geom.shape[-1])])
    if add_time:
        time = data["times_complete"].astype(np.float32)
        denom = max(float(np.nanmax(time)), 1.0)
        time = (time / denom)[..., None]
        channels.append(time)
        names.append("time_norm")
    X = np.concatenate(channels, axis=-1).astype(np.float32)
    return X, names


def load_initial_features(ics_path: str | Path, global_indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    with h5py.File(ics_path, "r") as handle:
        for key in INITIAL_KEYS:
            arr = np.asarray(handle[key][global_indices], dtype=np.float32)
            if key in ANGLE_INITIAL_KEYS:
                arr = sincos(arr)
            flat = arr.reshape(arr.shape[0], -1)
            arrays.append(flat)
            names.extend([f"{key}_{i}" for i in range(flat.shape[1])])
    return np.concatenate(arrays, axis=1).astype(np.float32), names


def filter_valid_samples(
    X: np.ndarray,
    labels: pd.DataFrame,
    data: dict[str, np.ndarray],
    time_horizon: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    valid_label = (labels["time_unbound"] >= 0.0) & (labels["diagnosis"] == 0)
    finite_prefix = np.isfinite(X).all(axis=(1, 2))
    mask = valid_label.to_numpy() & finite_prefix
    X = X[mask]
    labels = labels.loc[mask].reset_index(drop=True)
    if time_horizon is None:
        time_horizon = float(labels["time_unbound"].max())
    y = (labels["time_unbound"].to_numpy(dtype=np.float32) >= float(time_horizon)).astype(np.float32)
    return X, y, labels, mask


def standardize_sequence_train(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((X_train - mu) / std).astype(np.float32),
        [((arr - mu) / std).astype(np.float32) for arr in arrays],
        {"mean": mu.astype(np.float32), "std": std.astype(np.float32)},
    )


def standardize_static_train(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((X_train - mu) / std).astype(np.float32),
        [((arr - mu) / std).astype(np.float32) for arr in arrays],
        {"mean": mu.astype(np.float32), "std": std.astype(np.float32)},
    )


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_static: np.ndarray | None = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.reshape(-1, 1)).float()
        self.X_static = None if X_static is None else torch.from_numpy(X_static).float()

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.X_static is None:
            return self.X[index], self.y[index]
        return self.X[index], self.X_static[index], self.y[index]


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            trim = y.size(-1) - x.size(-1)
            y = y[..., trim // 2 : trim // 2 + x.size(-1)]
        return y + self.skip(x)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, levels: int = 4, dropout: float = 0.2):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = input_dim
        for level in range(levels):
            blocks.append(TCNBlock(in_ch, hidden, dilation=2**level, dropout=dropout))
            in_ch = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x.transpose(1, 2)).mean(dim=-1)
        return self.head(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, heads: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(self.pos(h))
        return self.head(h[:, 0])


class ICPrefixClassifier(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.prefix = CNN1DClassifier(input_dim=input_dim, hidden=hidden, dropout=dropout)
        self.prefix.head = nn.Identity()
        self.static = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        hp = self.prefix.net(x.transpose(1, 2)).squeeze(-1)
        hs = self.static(x_static)
        return self.head(torch.cat([hp, hs], dim=-1))


class LinearSVM(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[LinearSVM, list[dict[str, float]]]:
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    mu = X_train_2d.mean(axis=0, keepdims=True)
    std = X_train_2d.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_2d = ((X_train_2d - mu) / std).astype(np.float32)
    X_val_2d = ((X_val_2d - mu) / std).astype(np.float32)
    y_svm = (y_train * 2.0 - 1.0).astype(np.float32)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_2d).float(), torch.from_numpy(y_svm).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    model = LinearSVM(X_train_2d.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            score = model(xb)
            loss = torch.clamp(1.0 - yb * score, min=0.0).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        prob = predict_svm(model, X_val_2d, device)
        metrics = binary_metrics_from_probs(y_val, prob)
        history.append({"epoch": epoch, "loss": loss_sum / max(count, 1), "val_auc": metrics["auc"]})
    model.norm_mean = torch.from_numpy(mu.astype(np.float32))
    model.norm_std = torch.from_numpy(std.astype(np.float32))
    return model, history


@torch.no_grad()
def predict_svm(model: LinearSVM, X_2d: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(X_2d).float(), batch_size=4096, shuffle=False)
    probs: list[np.ndarray] = []
    for xb in loader:
        score = model(xb.to(device))
        probs.append(torch.sigmoid(score).cpu().numpy())
    return np.concatenate(probs)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_auc = -float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                logits = model(xb.to(device))
            else:
                xb, xs, yb = batch
                logits = model(xb.to(device), xs.to(device))
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * yb.size(0)
            count += yb.size(0)
        y_val, prob = predict_classifier(model, val_loader, device)
        metrics = binary_metrics_from_probs(y_val, prob)
        history.append({"epoch": epoch, "loss": loss_sum / max(count, 1), "val_auc": metrics["auc"], "val_f1": metrics["f1"]})
        if metrics["auc"] > best_auc:
            best_auc = float(metrics["auc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val_auc={history[-1]['val_auc']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 2:
            xb, yb = batch
            logits = model(xb.to(device))
        else:
            xb, xs, yb = batch
            logits = model(xb.to(device), xs.to(device))
        y_all.append(yb.numpy().reshape(-1))
        p_all.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(y_all), np.concatenate(p_all)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task B: hard-label prefix trajectory classifiers.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_b"))
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=100)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--time-horizon", type=float, default=None)
    parser.add_argument("--models", type=str, default="svm,cnn,tcn,attention,ic_prefix")
    parser.add_argument("--test-size", type=float, default=0.125)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    models = set(parse_name_list(args.models))

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
    X, feature_names = build_prefix_features(data, add_time=not args.no_time, include_geom=not args.no_geom)
    X, y, labels, valid_mask = filter_valid_samples(X, labels, data, args.time_horizon)
    X_static = None
    static_names: list[str] = []
    if "ic_prefix" in models:
        global_indices = labels["index"].to_numpy(dtype=np.int64)
        X_static, static_names = load_initial_features(args.dataset_dir / f"ICs_sampling_{args.sampling_id}.hdf5", global_indices)

    train_idx, val_idx, test_idx = stratified_split_indices(y, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train, [X_val, X_test], seq_norm = standardize_sequence_train(X_train, X_val, X_test)
    static_norm = None
    Xs_train = Xs_val = Xs_test = None
    if X_static is not None:
        Xs_train, Xs_val, Xs_test = X_static[train_idx], X_static[val_idx], X_static[test_idx]
        Xs_train, [Xs_val, Xs_test], static_norm = standardize_static_train(Xs_train, Xs_val, Xs_test)

    labels.to_csv(output_dir / "labels_filtered.csv", index=False)
    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    np.savez(output_dir / "sequence_normalization.npz", mean=seq_norm["mean"], std=seq_norm["std"])
    if static_norm is not None:
        np.savez(output_dir / "static_normalization.npz", mean=static_norm["mean"], std=static_norm["std"])
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "result_paths": result_paths,
            "num_samples": int(len(y)),
            "positive_rate": float(y.mean()),
            "input_shape": tuple(X.shape[1:]),
            "feature_names": feature_names,
            "static_feature_names": static_names,
        },
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics: dict[str, Any] = {}

    if "svm" in models:
        svm, history = train_linear_svm(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        X_test_2d = ((X_test_2d - svm.norm_mean.numpy()) / svm.norm_std.numpy()).astype(np.float32)
        prob = predict_svm(svm, X_test_2d, device)
        metrics["svm"] = {"test": binary_metrics_from_probs(y_test, prob)}
        save_csv(output_dir / "svm_train_log.csv", history)
        torch.save({"model_state_dict": svm.state_dict()}, output_dir / "svm_model.pt")
        save_binary_curves(output_dir / "svm_test_curves.png", y_test, prob, "Task B SVM")

    model_factories = {
        "cnn": lambda: CNN1DClassifier(X_train.shape[-1], hidden=args.hidden_dim, dropout=args.dropout),
        "tcn": lambda: TCNClassifier(X_train.shape[-1], hidden=args.hidden_dim, dropout=args.dropout),
        "attention": lambda: AttentionClassifier(X_train.shape[-1], hidden=args.hidden_dim, dropout=args.dropout),
    }
    for name, factory in model_factories.items():
        if name not in models:
            continue
        model, history = train_classifier(factory(), train_loader, val_loader, args.epochs, args.lr, args.weight_decay, device)
        y_eval, prob = predict_classifier(model, test_loader, device)
        metrics[name] = {"test": binary_metrics_from_probs(y_eval, prob)}
        save_csv(output_dir / f"{name}_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / f"{name}_model.pt")
        save_binary_curves(output_dir / f"{name}_test_curves.png", y_eval, prob, f"Task B {name}")

    if "ic_prefix" in models:
        if Xs_train is None or Xs_val is None or Xs_test is None:
            raise RuntimeError("Static IC features were not prepared.")
        ic_train_loader = DataLoader(
            SequenceDataset(X_train, y_train, Xs_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        ic_val_loader = DataLoader(
            SequenceDataset(X_val, y_val, Xs_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        ic_test_loader = DataLoader(
            SequenceDataset(X_test, y_test, Xs_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = ICPrefixClassifier(X_train.shape[-1], Xs_train.shape[-1], hidden=args.hidden_dim, dropout=args.dropout)
        model, history = train_classifier(model, ic_train_loader, ic_val_loader, args.epochs, args.lr, args.weight_decay, device)
        y_eval, prob = predict_classifier(model, ic_test_loader, device)
        metrics["ic_prefix"] = {"test": binary_metrics_from_probs(y_eval, prob)}
        save_csv(output_dir / "ic_prefix_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "ic_prefix_model.pt")
        save_binary_curves(output_dir / "ic_prefix_test_curves.png", y_eval, prob, "Task B IC+Prefix")

    save_json(output_dir / "metrics.json", metrics)
    print(f"Task B complete. Outputs: {output_dir}")
    for name, payload in metrics.items():
        print(name, payload["test"])


if __name__ == "__main__":
    main()
