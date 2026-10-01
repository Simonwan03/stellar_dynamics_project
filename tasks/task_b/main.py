from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.experiment_utils import (
    binary_metrics_from_probs,
    ensure_dir,
    parse_name_list,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.common.case_split import (
    case_split_indices,
    default_case_split_manifest_path,
    ensure_case_split_manifest,
    missing_case_ids,
)
from tasks.common.orbital_schema import ANGLE_RESULT_KEYS, FULL_RESULT_KEYS, ORBITAL15_RESULT_KEYS
from tasks.common.sequence_layers import PositionalEncoding, TCNBlock
from tasks.task_b.data import (
    GROUP_SIZE,
    RESULT_RE,
    RUNS_PER_BATCH,
    batch_id_from_path,
    build_group_fraction_samples,
    build_prefix_features,
    discover_result_paths,
    filter_valid_samples,
    grouped_case_split_indices,
    load_aligned_labels,
    load_result_prefixes,
    pairwise_geom,
    sincos,
    standardize_sequence_train,
)


DEFAULT_TIME_HORIZON = 10000.0


def regression_metrics_from_preds(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    pred = np.clip(np.asarray(pred).reshape(-1).astype(np.float64), 0.0, 1.0)
    return {
        "mse": float(mean_squared_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)) if y_true.size > 1 else float("nan"),
    }


def infer_temporal_model_scale(num_steps: int) -> dict[str, dict[str, int]]:
    if num_steps <= 100:
        return {
            "cnn": {"hidden": 128},
            "tcn": {"hidden": 128, "levels": 4},
            "gru": {"hidden": 128, "layers": 2},
            "attention": {"hidden": 128, "layers": 2, "heads": 4},
        }
    if num_steps <= 200:
        return {
            "cnn": {"hidden": 160},
            "tcn": {"hidden": 128, "levels": 5},
            "gru": {"hidden": 128, "layers": 2},
            "attention": {"hidden": 128, "layers": 3, "heads": 4},
        }
    return {
        "cnn": {"hidden": 192},
        "tcn": {"hidden": 160, "levels": 5},
        "gru": {"hidden": 160, "layers": 2},
        "attention": {"hidden": 160, "layers": 3, "heads": 4},
    }


def is_improved(metric: float, best_metric: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "max":
        return metric > (best_metric + min_delta)
    if mode == "min":
        return metric < (best_metric - min_delta)
    raise ValueError(f"Unsupported early-stopping mode: {mode}")


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.reshape(-1, 1)).float()

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self.X[index], self.y[index]


class MLPClassifier(nn.Module):
    """Flattened-prefix MLP baseline for trajectory binary classification."""
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout)])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))

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


class GRUClassifier(nn.Module):
    """Temporal GRU baseline for trajectory prefixes.

    A two-layer unidirectional GRU is used by default.  The hidden width is
    selected from the same prefix-length scale table as the other temporal
    models, keeping the parameter count between the CNN and TCN baselines for
    the default 15-channel representation.
    """

    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if layers < 1:
            raise ValueError(f"GRU layers must be positive, got {layers}")
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h = self.norm(h[-1])
        return self.head(h)


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


class GroupVariationAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        group_size: int = GROUP_SIZE,
        hidden: int = 96,
        heads: int = 4,
        variation_layers: int = 1,
        temporal_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.group_size = group_size
        self.input_proj = nn.Linear(input_dim, hidden)
        self.variation_embedding = nn.Parameter(torch.zeros(1, 1, group_size, hidden))
        self.variation_cls = nn.Parameter(torch.zeros(1, 1, 1, hidden))
        variation_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.variation_encoder = nn.TransformerEncoder(variation_layer, num_layers=variation_layers)
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = PositionalEncoding(hidden)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"GroupVariationAttentionRegressor expects [B, T, group, C], got {tuple(x.shape)}")
        bsz, steps, group, channels = x.shape
        if group != self.group_size:
            raise ValueError(f"Expected group size {self.group_size}, got {group}")
        h = self.input_proj(x.reshape(bsz * steps, group, channels)).reshape(bsz, steps, group, -1)
        h = h + self.variation_embedding[:, :, :group, :]
        cls = self.variation_cls.expand(bsz, steps, -1, -1)
        h = torch.cat([cls, h], dim=2).reshape(bsz * steps, group + 1, -1)
        group_h = self.variation_encoder(h)[:, 0].reshape(bsz, steps, -1)
        temporal_cls = self.temporal_cls.expand(bsz, -1, -1)
        temporal = torch.cat([temporal_cls, group_h], dim=1)
        temporal = self.temporal_encoder(self.pos(temporal))
        return self.head(temporal[:, 0])


class LinearSVM(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class LinearRegressor(nn.Module):
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
    early_stopping_patience: int = 0,
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
    best_state = None
    best_auc = -float("inf")
    epochs_without_improvement = 0
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
        if is_improved(metrics["auc"], best_auc, mode="max"):
            best_auc = float(metrics["auc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"early_stopping epoch={epoch:03d} val_auc={metrics['auc']:.4f} best_auc={best_auc:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.norm_mean = torch.from_numpy(mu.astype(np.float32))
    model.norm_std = torch.from_numpy(std.astype(np.float32))
    return model, history


def train_linear_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    early_stopping_patience: int = 0,
) -> tuple[LinearRegressor, list[dict[str, float]]]:
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    mu = X_train_2d.mean(axis=0, keepdims=True)
    std = X_train_2d.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_2d = ((X_train_2d - mu) / std).astype(np.float32)
    X_val_2d = ((X_val_2d - mu) / std).astype(np.float32)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_2d).float(), torch.from_numpy(y_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    model = LinearRegressor(X_train_2d.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_mae = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(xb))
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        val_pred = predict_linear_regressor(model, X_val_2d, device)
        metrics = regression_metrics_from_preds(y_val, val_pred)
        history.append({"epoch": epoch, "loss": loss_sum / max(count, 1), "val_mse": metrics["mse"], "val_mae": metrics["mae"]})
        if is_improved(metrics["mae"], best_mae, mode="min"):
            best_mae = float(metrics["mae"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"early_stopping epoch={epoch:03d} val_mae={metrics['mae']:.5f} best_mae={best_mae:.5f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
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


@torch.no_grad()
def predict_linear_regressor(model: LinearRegressor, X_2d: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(X_2d).float(), batch_size=4096, shuffle=False)
    preds: list[np.ndarray] = []
    for xb in loader:
        pred = torch.sigmoid(model(xb.to(device)))
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    early_stopping_patience: int = 0,
    amp: bool = False,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    history: list[dict[str, float]] = []
    best_state = None
    best_auc = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            xb = xb.to(device, non_blocking=use_amp)
            yb = yb.to(device, non_blocking=use_amp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.item()) * yb.size(0)
            count += yb.size(0)
        y_val, prob = predict_classifier(model, val_loader, device, amp=use_amp)
        metrics = binary_metrics_from_probs(y_val, prob)
        history.append({"epoch": epoch, "loss": loss_sum / max(count, 1), "val_auc": metrics["auc"], "val_f1": metrics["f1"]})
        if is_improved(metrics["auc"], best_auc, mode="max"):
            best_auc = float(metrics["auc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val_auc={history[-1]['val_auc']:.4f}")
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"early_stopping epoch={epoch:03d} val_auc={metrics['auc']:.4f} best_auc={best_auc:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_sequence_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    early_stopping_patience: int = 0,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_mae = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(xb))
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * yb.size(0)
            count += yb.size(0)
        y_val, pred = predict_sequence_regressor(model, val_loader, device)
        metrics = regression_metrics_from_preds(y_val, pred)
        history.append({"epoch": epoch, "loss": loss_sum / max(count, 1), "val_mse": metrics["mse"], "val_mae": metrics["mae"]})
        if is_improved(metrics["mae"], best_mae, mode="min"):
            best_mae = float(metrics["mae"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val_mae={history[-1]['val_mae']:.5f}")
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"early_stopping epoch={epoch:03d} val_mae={metrics['mae']:.5f} best_mae={best_mae:.5f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for xb, yb in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(xb.to(device, non_blocking=use_amp))
        y_all.append(yb.numpy().reshape(-1))
        p_all.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(y_all), np.concatenate(p_all)


@torch.no_grad()
def predict_sequence_regressor(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for xb, yb in loader:
        pred = torch.sigmoid(model(xb.to(device)))
        y_all.append(yb.numpy().reshape(-1))
        p_all.append(pred.cpu().numpy().reshape(-1))
    return np.concatenate(y_all), np.concatenate(p_all)


def save_test_predictions(
    path: Path,
    labels: pd.DataFrame,
    test_idx: np.ndarray,
    target_stable: np.ndarray,
    probability_stable: np.ndarray,
) -> None:
    """Persist test probabilities for overall and future-only Task B analysis."""
    target_stable = np.asarray(target_stable, dtype=np.int64).reshape(-1)
    probability_stable = np.asarray(probability_stable, dtype=np.float64).reshape(-1)
    if len(test_idx) != target_stable.size or target_stable.size != probability_stable.size:
        raise ValueError("Test prediction arrays do not match test row count.")
    frame = labels.iloc[np.asarray(test_idx, dtype=np.int64)][
        ["index", "num_batch", "time_unbound", "diagnosis", "raw_case_id", "split"]
    ].copy()
    frame["task_b_row"] = np.asarray(test_idx, dtype=np.int64)
    frame["target_stable"] = target_stable
    frame["prob_stable"] = probability_stable
    frame["target_unbound"] = 1 - target_stable
    frame["prob_unbound"] = 1.0 - probability_stable
    frame.to_csv(path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task B: prefix trajectory models.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_b/sampling_2/task_b_main"))
    parser.add_argument(
        "--representation",
        choices=["orbital15", "full"],
        default="orbital15",
        help="orbital15 keeps the 15-channel Panda-style orbital sequence; full restores the wider orbital+geometry/time stack.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["trajectory_binary", "group_fraction"],
        default="trajectory_binary",
        help="trajectory_binary predicts one trajectory label; group_fraction predicts stable fraction across each 32-trajectory system group.",
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument(
        "--group-aggregation",
        choices=["concat", "set_attention"],
        default="concat",
        help="How group_fraction combines the 32 prefixes: concat flattens variations into channels; set_attention keeps a variation axis for group_attention.",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=100)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--time-horizon", type=float, default=DEFAULT_TIME_HORIZON)
    parser.add_argument("--models", type=str, default="svm,cnn,tcn,attention")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument(
        "--val-ratio",
        "--val-size",
        dest="val_ratio",
        type=float,
        default=0.15,
        help="Validation fraction of total systems. --val-size is a legacy alias.",
    )
    parser.add_argument(
        "--test-ratio",
        "--test-size",
        dest="test_ratio",
        type=float,
        default=0.15,
        help="Test fraction of total systems. --test-size is a legacy alias.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Shared raw_case_id split CSV; created from the completion table when absent.",
    )
    parser.add_argument("--overwrite-split-manifest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=None)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--tcn-levels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-geom", action="store_true", help="Only applies to --representation full.")
    parser.add_argument("--no-time", action="store_true", help="Only applies to --representation full.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    models = set(parse_name_list(args.models))
    if args.representation == "orbital15":
        result_keys = ORBITAL15_RESULT_KEYS
        include_geom = False
        add_time = False
    else:
        result_keys = FULL_RESULT_KEYS
        include_geom = not args.no_geom
        add_time = not args.no_time

    result_paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    data, batch_ids, rows_loaded = load_result_prefixes(
        result_paths,
        result_keys=result_keys,
        prefix_len=args.prefix_len,
        time_stride=args.time_stride,
        rows_per_shard=args.rows_per_shard,
        include_geom=include_geom,
        add_time=add_time,
    )
    labels = load_aligned_labels(args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt", batch_ids, rows_loaded)
    X, feature_names = build_prefix_features(data, result_keys=result_keys, add_time=add_time, include_geom=include_geom)
    X, y, labels, valid_mask = filter_valid_samples(X, labels, data, args.time_horizon)
    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    split_manifest = args.split_manifest or default_case_split_manifest_path(
        args.dataset_dir,
        args.sampling_id,
        args.group_size,
        args.time_horizon,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    split_manifest_df = ensure_case_split_manifest(
        labels_path,
        split_manifest,
        group_size=args.group_size,
        time_horizon=args.time_horizon,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite_split_manifest,
    )
    loaded_case_ids = (labels["index"].to_numpy(dtype=np.int64) // args.group_size).astype(np.int64)
    manifest_case_ids = split_manifest_df["raw_case_id"].to_numpy(dtype=np.int64)
    known_case_mask = np.isin(loaded_case_ids, manifest_case_ids)
    if not np.all(known_case_mask):
        dropped = int((~known_case_mask).sum())
        print(f"Dropping {dropped} valid trajectory rows outside the common complete-system universe.")
        X = X[known_case_mask]
        y = y[known_case_mask]
        labels = labels.loc[known_case_mask].reset_index(drop=True)
    if args.max_shards is None and args.rows_per_shard is None:
        remaining_case_ids = (labels["index"].to_numpy(dtype=np.int64) // args.group_size).astype(np.int64)
        missing = missing_case_ids(remaining_case_ids, split_manifest_df)
        counts = pd.Series(remaining_case_ids).value_counts()
        expected_case_ids = split_manifest_df["raw_case_id"].to_numpy(dtype=np.int64)
        incomplete = expected_case_ids[counts.reindex(expected_case_ids, fill_value=0).to_numpy() != args.group_size]
        missing = np.union1d(missing, incomplete)
        if missing.size:
            raise ValueError(
                "Task B does not contain the full common system universe after input filtering; "
                f"missing {missing.size} raw_case_id values (first: {missing[:10].tolist()})."
            )
    if args.target_mode == "group_fraction":
        X, y, labels = build_group_fraction_samples(
            X,
            y,
            labels,
            group_size=args.group_size,
            aggregation=args.group_aggregation,
        )
    if args.target_mode == "trajectory_binary":
        sample_case_ids = (labels["index"].to_numpy(dtype=np.int64) // args.group_size).astype(np.int64)
    else:
        sample_case_ids = labels["raw_case_id"].to_numpy(dtype=np.int64)
    split_indices = case_split_indices(sample_case_ids, split_manifest_df)
    train_idx, val_idx, test_idx = (
        split_indices["train"],
        split_indices["val"],
        split_indices["test"],
    )
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train, [X_val, X_test], seq_norm = standardize_sequence_train(X_train, X_val, X_test)
    temporal_scale = infer_temporal_model_scale(X_train.shape[1])
    cnn_hidden = args.hidden_dim if args.hidden_dim is not None else temporal_scale["cnn"]["hidden"]
    tcn_hidden = args.hidden_dim if args.hidden_dim is not None else temporal_scale["tcn"]["hidden"]
    tcn_levels = args.tcn_levels if args.tcn_levels is not None else temporal_scale["tcn"]["levels"]
    attention_hidden = args.hidden_dim if args.hidden_dim is not None else temporal_scale["attention"]["hidden"]
    attention_layers = args.attention_layers if args.attention_layers is not None else temporal_scale["attention"]["layers"]
    attention_heads = args.attention_heads
    gru_hidden = args.hidden_dim if args.hidden_dim is not None else temporal_scale["gru"]["hidden"]
    gru_layers = args.gru_layers if args.gru_layers is not None else temporal_scale["gru"]["layers"]
    group_attention_hidden = args.hidden_dim if args.hidden_dim is not None else attention_hidden

    split_membership = np.empty(len(sample_case_ids), dtype=object)
    for split_name, split_rows in split_indices.items():
        split_membership[split_rows] = split_name
    labels = labels.copy()
    labels["raw_case_id"] = sample_case_ids
    labels["split"] = split_membership
    labels.to_csv(output_dir / "labels_filtered.csv", index=False)
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        case_ids=sample_case_ids,
    )
    np.savez(output_dir / "sequence_normalization.npz", mean=seq_norm["mean"], std=seq_norm["std"])
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "result_paths": result_paths,
            "num_samples": int(len(y)),
            "target_mean": float(y.mean()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "input_shape": tuple(X.shape[1:]),
            "feature_names": feature_names,
            "split_manifest": split_manifest,
            "split_unit": "complete raw_case_id system",
            "split_horizon": float(args.time_horizon),
            "split_ratios": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "split_counts": {name: int(rows.size) for name, rows in split_indices.items()},
            "split_case_counts": {
                name: int(np.unique(sample_case_ids[rows]).size)
                for name, rows in split_indices.items()
            },
            "effective_model_scale": {
                "cnn": {"hidden": cnn_hidden},
                "tcn": {"hidden": tcn_hidden, "levels": tcn_levels},
                "gru": {"hidden": gru_hidden, "layers": gru_layers},
                "attention": {"hidden": attention_hidden, "layers": attention_layers, "heads": attention_heads},
                "group_attention": {"hidden": group_attention_hidden, "heads": attention_heads},
            },
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

    if args.target_mode == "group_fraction":
        if "svm" in models:
            reg, history = train_linear_regressor(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                early_stopping_patience=args.early_stopping_patience,
            )
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
            X_test_2d = ((X_test_2d - reg.norm_mean.numpy()) / reg.norm_std.numpy()).astype(np.float32)
            pred = predict_linear_regressor(reg, X_test_2d, device)
            metrics["linear"] = {"test": regression_metrics_from_preds(y_test, pred)}
            save_csv(output_dir / "linear_train_log.csv", history)
            torch.save(
                {
                    "model_state_dict": reg.state_dict(),
                    "norm_mean": reg.norm_mean.clone(),
                    "norm_std": reg.norm_std.clone(),
                },
                output_dir / "linear_model.pt",
            )

        if args.group_aggregation == "concat":
            model_factories = {
                "cnn": lambda: CNN1DClassifier(X_train.shape[-1], hidden=cnn_hidden, dropout=args.dropout),
                "tcn": lambda: TCNClassifier(X_train.shape[-1], hidden=tcn_hidden, levels=tcn_levels, dropout=args.dropout),
                "gru": lambda: GRUClassifier(
                    X_train.shape[-1], hidden=gru_hidden, layers=gru_layers, dropout=args.dropout
                ),
                "attention": lambda: AttentionClassifier(
                    X_train.shape[-1],
                    hidden=attention_hidden,
                    heads=attention_heads,
                    layers=attention_layers,
                    dropout=args.dropout,
                ),
            }
        else:
            unsupported = sorted((models & {"cnn", "tcn", "gru", "attention"}) - {"group_attention"})
            if unsupported:
                print(f"Skipping {unsupported} in set_attention mode; use --models group_attention or svm.")
            model_factories = {
                "group_attention": lambda: GroupVariationAttentionRegressor(
                    X_train.shape[-1],
                    group_size=args.group_size,
                    hidden=group_attention_hidden,
                    heads=attention_heads,
                    dropout=args.dropout,
                ),
            }
        for name, factory in model_factories.items():
            if name not in models:
                continue
            model, history = train_sequence_regressor(
                factory(),
                train_loader,
                val_loader,
                args.epochs,
                args.lr,
                args.weight_decay,
                device,
                early_stopping_patience=args.early_stopping_patience,
            )
            y_eval, pred = predict_sequence_regressor(model, test_loader, device)
            metrics[name] = {"test": regression_metrics_from_preds(y_eval, pred)}
            save_csv(output_dir / f"{name}_train_log.csv", history)
            torch.save({"model_state_dict": model.state_dict()}, output_dir / f"{name}_model.pt")

        save_json(output_dir / "metrics.json", metrics)
        print(f"Task B complete. Outputs: {output_dir}")
        for name, payload in metrics.items():
            print(name, payload["test"])
        return

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
            early_stopping_patience=args.early_stopping_patience,
        )
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        X_test_2d = ((X_test_2d - svm.norm_mean.numpy()) / svm.norm_std.numpy()).astype(np.float32)
        prob = predict_svm(svm, X_test_2d, device)
        metrics["svm"] = {"test": binary_metrics_from_probs(y_test, prob)}
        save_csv(output_dir / "svm_train_log.csv", history)
        save_test_predictions(output_dir / "svm_test_predictions.csv", labels, test_idx, y_test, prob)
        torch.save(
            {
                "model_state_dict": svm.state_dict(),
                "norm_mean": svm.norm_mean.clone(),
                "norm_std": svm.norm_std.clone(),
            },
            output_dir / "svm_model.pt",
        )
        save_binary_curves(output_dir / "svm_test_curves.png", y_test, prob, "Task B SVM")

    model_factories = {
        "mlp": lambda: MLPClassifier(
            X_train.shape[1] * X_train.shape[2],
            hidden=(256, 128, 64),
            dropout=args.dropout,
        ),
        "cnn": lambda: CNN1DClassifier(X_train.shape[-1], hidden=cnn_hidden, dropout=args.dropout),
        "tcn": lambda: TCNClassifier(X_train.shape[-1], hidden=tcn_hidden, levels=tcn_levels, dropout=args.dropout),
        "gru": lambda: GRUClassifier(X_train.shape[-1], hidden=gru_hidden, layers=gru_layers, dropout=args.dropout),
        "attention": lambda: AttentionClassifier(
            X_train.shape[-1],
            hidden=attention_hidden,
            heads=attention_heads,
            layers=attention_layers,
            dropout=args.dropout,
        ),
    }
    for name, factory in model_factories.items():
        if name not in models:
            continue
        model, history = train_classifier(
            factory(),
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            device,
            early_stopping_patience=args.early_stopping_patience,
        )
        y_eval, prob = predict_classifier(model, test_loader, device)
        metrics[name] = {"test": binary_metrics_from_probs(y_eval, prob)}
        save_csv(output_dir / f"{name}_train_log.csv", history)
        save_test_predictions(output_dir / f"{name}_test_predictions.csv", labels, test_idx, y_eval, prob)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / f"{name}_model.pt")
        save_binary_curves(output_dir / f"{name}_test_curves.png", y_eval, prob, f"Task B {name}")

    save_json(output_dir / "metrics.json", metrics)
    print(f"Task B complete. Outputs: {output_dir}")
    for name, payload in metrics.items():
        print(name, payload["test"])


if __name__ == "__main__":
    main()
