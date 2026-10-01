from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.experiment_utils import ensure_dir, parse_name_list, save_csv, save_json, select_device, set_random_seed
from tasks.common.case_split import (
    case_split_indices,
    default_case_split_manifest_path,
    ensure_case_split_manifest,
    missing_case_ids,
)
from tasks.common.orbital_schema import FULL_RESULT_KEYS, ORBITAL15_RESULT_KEYS
from tasks.task_b.data import (
    build_prefix_features,
    discover_result_paths,
    filter_valid_samples,
    load_aligned_labels,
    load_result_prefixes,
)
from tasks.task_c.cache import load_cached_prefixes


class TrajectorySurvivalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, q: np.ndarray | None = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.q = torch.from_numpy(q).float() if q is not None else self.y.mean(dim=1)

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


class InitialConditionCurveDataset(Dataset):
    """Static initial-condition inputs paired with system survival curves."""

    def __init__(self, X_ic: np.ndarray, q: np.ndarray):
        self.X_ic = torch.from_numpy(X_ic).float()
        self.q = torch.from_numpy(q).float()

    def __len__(self) -> int:
        return self.X_ic.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_ic[index], self.q[index]


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
    complete_ids: list[int] = []
    expected_variations = np.arange(group_size, dtype=np.int64)
    for case_id, group in labels.groupby("case_id", sort=True):
        variations = np.sort(group["variation_id"].to_numpy(dtype=np.int64))
        if len(group) == group_size and np.array_equal(variations, expected_variations):
            complete_ids.append(int(case_id))
    if max_cases is not None:
        complete_ids = complete_ids[:max_cases]
    if not complete_ids:
        raise ValueError("No complete case groups survived filtering.")

    selected = labels[labels["case_id"].isin(complete_ids)].sort_values(["case_id", "variation_id"])
    rows = selected["_row"].to_numpy(dtype=np.int64).reshape(len(complete_ids), group_size)
    times = selected["time_unbound"].to_numpy(dtype=np.float32).reshape(len(complete_ids), group_size)
    return (
        X_traj[rows].astype(np.float32),
        (times[:, :, None] >= future_times[None, None, :]).astype(np.float32),
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
    if X_train.ndim == 4:
        reduce_axes = (0, 1, 2)
    elif X_train.ndim == 3:
        reduce_axes = (0, 1)
    elif X_train.ndim == 2:
        reduce_axes = (0,)
    else:
        raise ValueError(f"Expected case tensors with 2, 3, or 4 dimensions, got shape {X_train.shape}.")
    mu = X_train.mean(axis=reduce_axes, keepdims=True)
    std = X_train.std(axis=reduce_axes, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((X_train - mu) / std).astype(np.float32),
        [((arr - mu) / std).astype(np.float32) for arr in arrays],
        {"mean": mu.astype(np.float32), "std": std.astype(np.float32)},
    )


def aggregate_case_trajectories(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return X.mean(axis=1).astype(np.float32)
    if mode == "mean_std":
        stats = [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
        ]
        return np.concatenate(stats, axis=-1).astype(np.float32)
    raise ValueError(f"Unknown case aggregation mode: {mode}")


def build_variation_subset_indices(
    group_size: int,
    variation_count: int,
    mode: str = "balanced",
    seed: int = 42,
) -> np.ndarray:
    """Return deterministic variation indices for a reduced perturbation set.

    For the dataset's 32-variation layout, the balanced schedule samples both
    16-variation halves and is nested across 1, 2, 4, 8, 16, and 32 members:
    [0] -> [0,16] -> [0,8,16,24] -> ... -> [0..15,16..31].
    """
    if variation_count < 1 or variation_count > group_size:
        raise ValueError(f"variation_count must be in [1, {group_size}], got {variation_count}.")
    if mode not in {"balanced", "random"}:
        raise ValueError(f"Unknown variation subset mode: {mode}")
    if variation_count == group_size:
        return np.arange(group_size, dtype=np.int64)

    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(group_size, size=variation_count, replace=False).astype(np.int64))

    if group_size != 32 or variation_count not in {1, 2, 4, 8, 16}:
        raise ValueError(
            "balanced variation subsets currently support group_size=32 and "
            "variation_count in {1,2,4,8,16,32}."
        )
    if variation_count == 1:
        return np.asarray([0], dtype=np.int64)
    local_schedules = {
        1: [0],
        2: [0, 8],
        4: [0, 4, 8, 12],
        8: list(range(0, 16, 2)),
        16: list(range(16)),
    }
    per_half = variation_count // 2
    local = np.asarray(local_schedules[per_half], dtype=np.int64)
    return np.concatenate([local, local + group_size // 2])


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


class HazardHead(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hazard_logits_to_survival(logits: torch.Tensor) -> torch.Tensor:
    hazard = torch.sigmoid(logits)
    return torch.cumprod(1.0 - hazard, dim=-1)


class TrajectoryContextSurvivalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        encoder_type: str,
        dropout: float,
    ):
        super().__init__()
        if encoder_type == "gru":
            self.encoder = GRUTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "attention":
            self.encoder = TemporalAttentionEncoder(input_dim, hidden_dim, heads=4, layers=1, dropout=dropout)
        else:
            self.encoder = ConvTrajectoryEncoder(input_dim, hidden_dim, dropout)
        self.variation_head = HazardHead(hidden_dim * 2, horizon, hidden_dim, dropout)
        self.system_head = HazardHead(hidden_dim, horizon, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, group, time, dim = x.shape
        emb = self.encoder(x.reshape(bsz * group, time, dim)).reshape(bsz, group, -1)
        context = emb.mean(dim=1, keepdim=True).expand(-1, group, -1)
        variation_logits = self.variation_head(torch.cat([emb, context], dim=-1))
        system_logits = self.system_head(emb.mean(dim=1))
        return variation_logits, system_logits


class DirectCurveModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float):
        super().__init__()
        self.encoder = ConvTrajectoryEncoder(input_dim, hidden_dim, dropout)
        self.head = HazardHead(hidden_dim, horizon, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class InitialConditionMLP(nn.Module):
    """MLP baseline using one 15-dimensional orbital IC per system."""

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        bottleneck_dim = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualAttentionSurvivalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float):
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
        self.variation_head = HazardHead(hidden_dim * 2, horizon, hidden_dim, dropout)
        self.system_head = HazardHead(hidden_dim, horizon, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, group, time, dim = x.shape
        emb = self.temporal(x.reshape(bsz * group, time, dim)).reshape(bsz, group, -1)
        contextual = self.ensemble(emb)
        context = contextual.mean(dim=1, keepdim=True).expand(-1, group, -1)
        variation_logits = self.variation_head(torch.cat([contextual, context], dim=-1))
        system_logits = self.system_head(contextual.mean(dim=1))
        return variation_logits, system_logits


class VariationHazardNLL(nn.Module):
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        previous_survival = torch.cat([torch.ones_like(y[..., :1]), y[..., :-1]], dim=-1)
        failures = (previous_survival - y).clamp(0.0, 1.0)
        interval_nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, failures, reduction="none")
        return (interval_nll * previous_survival).sum() / previous_survival.sum().clamp_min(1.0)


class BinomialHazardNLL(nn.Module):
    def forward(self, logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        previous_survival = torch.cat([torch.ones_like(q[:, :1]), q[:, :-1]], dim=-1)
        failures = (previous_survival - q).clamp_min(0.0)
        hazard_target = failures / previous_survival.clamp_min(1e-8)
        interval_nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, hazard_target, reduction="none")
        return (interval_nll * previous_survival).sum() / previous_survival.sum().clamp_min(1e-8)


class DualHeadSurvivalLoss(nn.Module):
    """Joint variation-level and system-level survival objective.

    The system head is trained directly against the observed 32-variation
    survival fraction.  A small consistency term keeps it close to the mean
    of the variation-head survival curves without forcing the two heads to be
    identical.
    """

    def __init__(self, system_weight: float = 1.0, consistency_weight: float = 0.1):
        super().__init__()
        if system_weight < 0.0 or consistency_weight < 0.0:
            raise ValueError("Dual-head loss weights must be non-negative.")
        self.system_weight = float(system_weight)
        self.consistency_weight = float(consistency_weight)
        self.variation_criterion = VariationHazardNLL()
        self.system_criterion = BinomialHazardNLL()

    def components(
        self,
        variation_logits: torch.Tensor,
        system_logits: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        variation_loss = self.variation_criterion(variation_logits, y)
        system_loss = self.system_criterion(system_logits, q)
        variation_survival = hazard_logits_to_survival(variation_logits)
        system_survival = hazard_logits_to_survival(system_logits)
        mean_variation_survival = variation_survival.mean(dim=1)
        consistency_loss = torch.nn.functional.smooth_l1_loss(system_survival, mean_variation_survival)
        total = variation_loss + self.system_weight * system_loss + self.consistency_weight * consistency_loss
        return {
            "loss": total,
            "variation_loss": variation_loss,
            "system_loss": system_loss,
            "consistency_loss": consistency_loss,
        }

    def forward(
        self,
        variation_logits: torch.Tensor,
        system_logits: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(variation_logits, system_logits, y, q)["loss"]


@torch.no_grad()
def evaluate_trajectory_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: DualHeadSurvivalLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_sums = {"loss": 0.0, "variation_loss": 0.0, "system_loss": 0.0, "consistency_loss": 0.0}
    system_preds: list[np.ndarray] = []
    mean_variation_preds: list[np.ndarray] = []
    system_targets: list[np.ndarray] = []
    variation_preds: list[np.ndarray] = []
    variation_targets: list[np.ndarray] = []
    for xb, yb, qb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        qb = qb.to(device)
        variation_logits, system_logits = model(xb)
        components = criterion.components(variation_logits, system_logits, yb, qb)
        for name in loss_sums:
            loss_sums[name] += float(components[name].item()) * xb.size(0)
        variation_survival = hazard_logits_to_survival(variation_logits)
        system_survival = hazard_logits_to_survival(system_logits)
        variation_preds.append(variation_survival.cpu().numpy())
        mean_variation_preds.append(variation_survival.mean(dim=1).cpu().numpy())
        system_preds.append(system_survival.cpu().numpy())
        variation_targets.append(yb.cpu().numpy())
        system_targets.append(qb.cpu().numpy())
    system_pred = np.concatenate(system_preds, axis=0)
    mean_variation_pred = np.concatenate(mean_variation_preds, axis=0)
    system_target = np.concatenate(system_targets, axis=0)
    variation_pred = np.concatenate(variation_preds, axis=0)
    variation_target = np.concatenate(variation_targets, axis=0)
    system_abs_err = np.abs(system_pred - system_target)
    mean_variation_abs_err = np.abs(mean_variation_pred - system_target)
    variation_abs_err = np.abs(variation_pred - variation_target)
    dataset_size = len(loader.dataset)
    return {
        **{name: float(value / dataset_size) for name, value in loss_sums.items()},
        "overall_mae": float(system_abs_err.mean()),
        "final_mae": float(system_abs_err[:, -1].mean()),
        "horizon_mae": system_abs_err.mean(axis=0),
        "variation_overall_mae": float(variation_abs_err.mean()),
        "variation_final_mae": float(variation_abs_err[:, :, -1].mean()),
        "mean_variation_overall_mae": float(mean_variation_abs_err.mean()),
        "mean_variation_final_mae": float(mean_variation_abs_err[:, -1].mean()),
        "preds": system_pred,
        "targets": system_target,
        "variation_preds": variation_pred,
        "variation_targets": variation_target,
        "mean_variation_preds": mean_variation_pred,
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
        preds.append(hazard_logits_to_survival(logits).cpu().numpy())
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
    criterion: DualHeadSurvivalLoss,
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
        train_loss_sums = {"variation_loss": 0.0, "system_loss": 0.0, "consistency_loss": 0.0}
        for xb, yb, qb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            qb = qb.to(device)
            optimizer.zero_grad(set_to_none=True)
            variation_logits, system_logits = model(xb)
            components = criterion.components(variation_logits, system_logits, yb, qb)
            loss = components["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            for name in train_loss_sums:
                train_loss_sums[name] += float(components[name].item()) * xb.size(0)
        val = evaluate_trajectory_model(model, val_loader, criterion, device)
        train_size = len(train_loader.dataset)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / train_size,
            "train_variation_loss": train_loss_sums["variation_loss"] / train_size,
            "train_system_loss": train_loss_sums["system_loss"] / train_size,
            "train_consistency_loss": train_loss_sums["consistency_loss"] / train_size,
            "val_loss": val["loss"],
            "val_variation_loss": val["variation_loss"],
            "val_system_loss": val["system_loss"],
            "val_consistency_loss": val["consistency_loss"],
            "val_mae": val["overall_mae"],
            "val_mean_variation_mae": val["mean_variation_overall_mae"],
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_system_mae={row['val_mae']:.5f} "
            f"val_mean_variation_mae={row['val_mean_variation_mae']:.5f}"
        )
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


def train_initial_condition_model(
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
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader.dataset),
            "val_loss": val["loss"],
            "val_mae": val["overall_mae"],
        }
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
    preferred_models = ["direct", "direct_mean_std", "traj", "dual", "ic_mlp"]
    model_names = [name for name in preferred_models if name in metrics and "preds" in metrics[name]]
    if not model_names:
        raise ValueError("No neural-model predictions available for plotting.")

    display_names = {
        "direct": "direct",
        "direct_mean_std": "direct_stats",
        "traj": "traj",
        "dual": "dual",
        "ic_mlp": "IC-MLP",
    }
    colors = {
        "direct": "#D1495B",
        "direct_mean_std": "#EDAE49",
        "traj": "#00798C",
        "dual": "#30638E",
        "ic_mlp": "#6A4C93",
    }
    targets = metrics[model_names[0]]["targets"]

    # Figure 1: mean survival curve over all test cases.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(future_times, targets.mean(axis=0), color="black", linewidth=2.8, marker="o", label="true")
    for name in model_names:
        ax.plot(
            future_times,
            metrics[name]["preds"].mean(axis=0),
            color=colors[name],
            linewidth=2,
            marker="o",
            label=display_names[name],
        )
    ax.set_xlabel("Future time")
    ax.set_ylabel("Mean survival fraction")
    ax.set_title("Mean survival curve")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_mean_survival_curve.png", dpi=180)
    plt.close(fig)

    # Figure 2: representative stable, middle, and unstable systems.
    final_target = targets[:, -1]
    case_specs = [("stable", 1.0), ("middle", 0.5), ("unstable", 0.0)]
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, target_value in case_specs:
        order = np.argsort(np.abs(final_target - target_value))
        index = next((int(i) for i in order if int(i) not in used), int(order[0]))
        selected.append((label, index))
        used.add(index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    for ax, (label, index) in zip(axes, selected):
        ax.plot(future_times, targets[index], color="black", linewidth=2.8, marker="o", label="true")
        for name in model_names:
            ax.plot(
                future_times,
                metrics[name]["preds"][index],
                color=colors[name],
                linewidth=1.7,
                marker="o",
                label=display_names[name],
            )
        ax.set_title(f"{label}: case {index}, q(10000)={final_target[index]:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Future time")
    axes[0].set_ylabel("Survival fraction")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_representative_cases.png", dpi=180)
    plt.close(fig)

    # Figure 3: horizon-wise MAE comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in model_names:
        ax.plot(
            future_times,
            metrics[name]["horizon_mae"],
            color=colors[name],
            linewidth=2,
            marker="o",
            label=display_names[name],
        )
    ax.set_xlabel("Future time")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Per-horizon survival-fraction MAE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_per_horizon_mae.png", dpi=180)
    plt.close(fig)

    # Figure 4: predicted versus true survival fraction at the final horizon.
    scatter_horizon = int(np.argmin(np.abs(future_times - 10000.0)))
    columns = 2
    rows = int(np.ceil(len(model_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 4.5 * rows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, model_names):
        pred = metrics[name]["preds"][:, scatter_horizon]
        ax.scatter(targets[:, scatter_horizon], pred, s=18, alpha=0.45, color=colors[name], edgecolors="none")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"{display_names[name]} | MAE={np.mean(np.abs(pred - targets[:, scatter_horizon])):.3f}")
        ax.set_xlabel("True q")
        ax.set_ylabel("Predicted q")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    for ax in axes_arr[len(model_names) :]:
        ax.axis("off")
    fig.suptitle(f"Predicted vs true survival fraction at t={future_times[scatter_horizon]:g}", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_q_pred_vs_true.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: heatmaps for the best neural model, sorted by final true survival.
    best_name = min(model_names, key=lambda name: metrics[name]["overall_mae"])
    order = np.argsort(targets[:, -1])[::-1]
    true_sorted = targets[order]
    pred_sorted = metrics[best_name]["preds"][order]
    error_sorted = np.abs(pred_sorted - true_sorted)
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    panels = [
        ("True q", true_sorted, "viridis", 0.0, 1.0),
        (f"Predicted q: {display_names[best_name]}", pred_sorted, "viridis", 0.0, 1.0),
        ("Absolute error", error_sorted, "magma", 0.0, max(0.1, float(error_sorted.max()))),
    ]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Future horizon")
        ax.set_xticks(np.arange(len(future_times)), labels=[f"{time:g}" for time in future_times], rotation=45)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Test cases sorted by true q(10000)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_5_survival_heatmaps.png", dpi=180)
    plt.close(fig)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in metrics.items()
        if k not in {"preds", "targets", "variation_preds", "variation_targets", "mean_variation_preds"}
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task C: trajectory-level multi-horizon survival prediction.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_c/sampling_2/task_c_main"))
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--variation-count",
        type=int,
        default=None,
        help="Use a subset of each complete variation group; defaults to all group_size variations.",
    )
    parser.add_argument(
        "--variation-target",
        choices=["subset", "full32"],
        default="subset",
        help="Target the selected variation fraction or the full 32-variation system fraction.",
    )
    parser.add_argument(
        "--variation-subset-mode",
        choices=["balanced", "random"],
        default="balanced",
        help="How to select a reduced variation set.",
    )
    parser.add_argument("--variation-subset-seed", type=int, default=42)
    parser.add_argument(
        "--input-mode",
        choices=["prefix", "initial"],
        default="prefix",
        help="Use trajectory prefixes (prefix) or one t=0 orbital15 IC per system (initial).",
    )
    parser.add_argument(
        "--prefix-representation",
        choices=["full", "orbital15"],
        default="full",
        help="Prefix feature representation; orbital15 avoids loading geometry and auxiliary full-stack channels.",
    )
    parser.add_argument(
        "--prefix-cache",
        type=Path,
        default=None,
        help="Load prefix features and labels from a shared cache instead of reading HDF5 shards.",
    )
    parser.add_argument("--prefix-len", type=int, default=200)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--future-times", type=str, default="500,800,1500,2500,4000,6500,10000")
    parser.add_argument(
        "--future-time-mode",
        choices=["absolute"],
        default="absolute",
        help="Interpret future-times as absolute time_unbound thresholds.",
    )
    parser.add_argument("--models", type=str, default="baseline,direct,direct_mean_std,traj,dual")
    parser.add_argument("--encoder-type", choices=["cnn", "gru", "attention"], default="cnn")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-horizon",
        type=float,
        default=10000.0,
        help="Fixed horizon used only to stratify the shared A/B/C system split.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Shared raw_case_id split CSV; created from the completion table when absent.",
    )
    parser.add_argument("--overwrite-split-manifest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--system-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the direct system-level survival head in traj/dual models.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.1,
        help="Weight for matching the system head to the mean variation-head survival curve.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    requested_future_times = parse_future_times(args.future_times)
    requested_models = set(parse_name_list(args.models))

    if args.input_mode == "initial":
        if args.variation_count is not None or args.variation_target != "subset" or args.future_time_mode != "absolute":
            raise ValueError("Variation-count ablations are only supported in prefix mode.")
        allowed_models = {"baseline", "ic_mlp"}
        unsupported = sorted(requested_models - allowed_models)
        if unsupported:
            raise ValueError(
                f"Initial-condition mode supports baseline,ic_mlp; unsupported models: {unsupported}"
            )
        if "ic_mlp" not in requested_models:
            raise ValueError("Initial-condition mode requires --models ic_mlp or baseline,ic_mlp.")
        result_keys = ORBITAL15_RESULT_KEYS
        effective_prefix_len = 1
        include_geom = False
        add_time = False
    else:
        if args.prefix_representation == "orbital15":
            result_keys = ORBITAL15_RESULT_KEYS
            include_geom = False
            add_time = False
        else:
            result_keys = FULL_RESULT_KEYS
            include_geom = not args.no_geom
            add_time = not args.no_time
        effective_prefix_len = args.prefix_len

    if args.prefix_cache is not None:
        data, labels, result_paths, X_traj, feature_names = load_cached_prefixes(
            args.prefix_cache,
            prefix_len=effective_prefix_len,
            time_stride=args.time_stride,
            add_time=add_time,
            result_keys=result_keys,
            include_geom=include_geom,
        )
    else:
        load_time_metadata = add_time
        result_paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
        data, batch_ids, rows_loaded = load_result_prefixes(
            result_paths,
            result_keys=result_keys,
            prefix_len=effective_prefix_len,
            time_stride=args.time_stride,
            rows_per_shard=args.rows_per_shard,
            include_geom=include_geom,
            add_time=load_time_metadata,
        )
        labels = load_aligned_labels(
            args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt",
            batch_ids,
            rows_loaded,
        )
        X_traj, feature_names = build_prefix_features(
            data,
            result_keys=result_keys,
            add_time=add_time,
            include_geom=include_geom,
        )

    observed_time_end = None
    future_times = requested_future_times
    X_traj, _, labels, _ = filter_valid_samples(X_traj, labels, data, time_horizon=None)
    X_cases, y_cases, case_ids = build_case_tensors(X_traj, labels, args.group_size, future_times, max_cases=args.max_cases)
    initial_condition_max_abs_diff = None
    if args.input_mode == "initial":
        if X_cases.ndim != 4 or X_cases.shape[2] != 1:
            raise ValueError(f"Expected one-step IC case tensor, got shape {X_cases.shape}.")
        initial_condition_max_abs_diff = float(np.max(np.abs(X_cases - X_cases[:, :1, :, :])))
        X_cases = X_cases[:, 0, 0, :]
    full_y_cases = y_cases
    full_q_cases = full_y_cases.mean(axis=1)
    variation_indices = np.arange(args.group_size, dtype=np.int64)
    variation_count = args.variation_count if args.variation_count is not None else args.group_size
    if args.input_mode == "prefix" and variation_count != args.group_size:
        variation_indices = build_variation_subset_indices(
            group_size=args.group_size,
            variation_count=variation_count,
            mode=args.variation_subset_mode,
            seed=args.variation_subset_seed,
        )
        X_cases = X_cases[:, variation_indices]
        y_cases = y_cases[:, variation_indices]
    if args.variation_target == "full32":
        q_cases = full_q_cases
    else:
        q_cases = y_cases.mean(axis=1)

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    split_manifest = args.split_manifest or default_case_split_manifest_path(
        args.dataset_dir,
        args.sampling_id,
        args.group_size,
        args.split_horizon,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    split_manifest_df = ensure_case_split_manifest(
        labels_path,
        split_manifest,
        group_size=args.group_size,
        time_horizon=args.split_horizon,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite_split_manifest,
    )
    if args.max_shards is None and args.rows_per_shard is None and args.max_cases is None:
        missing = missing_case_ids(case_ids, split_manifest_df)
        if missing.size:
            raise ValueError(
                "Task C does not contain the full common system universe after input filtering; "
                f"missing {missing.size} raw_case_id values (first: {missing[:10].tolist()})."
            )
    split_indices = case_split_indices(case_ids, split_manifest_df)
    train_idx, val_idx, test_idx = (
        split_indices["train"],
        split_indices["val"],
        split_indices["test"],
    )
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
            "effective_prefix_len": effective_prefix_len,
            "future_times": future_times,
            "future_time_mode": args.future_time_mode,
            "future_time_offsets": requested_future_times if args.future_time_mode == "after_prefix" else None,
            "prefix_representation": args.prefix_representation,
            "result_paths": result_paths,
            "num_cases": int(X_cases.shape[0]),
            "input_shape": tuple(X_cases.shape[1:]),
            "feature_names": feature_names,
            "train_cases": int(train_idx.size),
            "val_cases": int(val_idx.size),
            "test_cases": int(test_idx.size),
            "initial_condition_max_abs_diff": initial_condition_max_abs_diff,
            "variation_count": int(variation_count),
            "variation_target": args.variation_target,
            "variation_subset_mode": args.variation_subset_mode,
            "variation_subset_seed": int(args.variation_subset_seed),
            "variation_indices": variation_indices,
            "observed_time_end": observed_time_end,
            "split_manifest": split_manifest,
            "split_unit": "complete raw_case_id system",
            "split_horizon": float(args.split_horizon),
            "split_ratios": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "split_counts": {name: int(rows.size) for name, rows in split_indices.items()},
        },
    )

    if args.input_mode == "initial":
        train_loader = DataLoader(InitialConditionCurveDataset(X_train, q_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(InitialConditionCurveDataset(X_val, q_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(InitialConditionCurveDataset(X_test, q_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(TrajectorySurvivalDataset(X_train, y_train, q_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(TrajectorySurvivalDataset(X_val, y_val, q_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(TrajectorySurvivalDataset(X_test, y_test, q_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics: dict[str, dict[str, Any]] = {}
    if "baseline" in requested_models:
        metrics["baseline"] = mean_baseline_metrics(q_train, q_test)

    if args.input_mode == "initial":
        criterion = BinomialHazardNLL()
        model = InitialConditionMLP(
            input_dim=X_train.shape[-1],
            horizon=len(future_times),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        model, history = train_initial_condition_model(
            model,
            train_loader,
            val_loader,
            criterion,
            args.epochs,
            args.lr,
            args.weight_decay,
            device,
        )
        metrics["ic_mlp"] = evaluate_direct_model(model, test_loader, criterion, device)
        save_csv(output_dir / "ic_mlp_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "ic_mlp_model.pt")
        save_survival_plots(output_dir, future_times, metrics)
        np.savez(
            output_dir / "test_predictions.npz",
            **{f"{name}_preds": payload["preds"] for name, payload in metrics.items() if "preds" in payload},
            targets=q_test,
            future_times=future_times,
        )
        save_json(output_dir / "metrics.json", {name: compact_metrics(payload) for name, payload in metrics.items()})
        print(f"Task C initial-condition complete. Outputs: {output_dir}")
        for name, payload in metrics.items():
            print(name, compact_metrics(payload))
        return

    if "direct" in requested_models:
        X_case_train = aggregate_case_trajectories(X_train, mode="mean")
        X_case_val = aggregate_case_trajectories(X_val, mode="mean")
        X_case_test = aggregate_case_trajectories(X_test, mode="mean")
        direct_train_loader = DataLoader(DirectCurveDataset(X_case_train, q_train), batch_size=args.batch_size, shuffle=True)
        direct_val_loader = DataLoader(DirectCurveDataset(X_case_val, q_val), batch_size=args.batch_size, shuffle=False)
        direct_test_loader = DataLoader(DirectCurveDataset(X_case_test, q_test), batch_size=args.batch_size, shuffle=False)
        model = DirectCurveModel(X_case_train.shape[-1], args.hidden_dim, len(future_times), args.dropout)
        criterion = BinomialHazardNLL()
        model, history = train_direct_model(model, direct_train_loader, direct_val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["direct"] = evaluate_direct_model(model, direct_test_loader, criterion, device)
        save_csv(output_dir / "direct_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "direct_model.pt")

    if "direct_mean_std" in requested_models:
        X_case_train = aggregate_case_trajectories(X_train, mode="mean_std")
        X_case_val = aggregate_case_trajectories(X_val, mode="mean_std")
        X_case_test = aggregate_case_trajectories(X_test, mode="mean_std")
        direct_train_loader = DataLoader(DirectCurveDataset(X_case_train, q_train), batch_size=args.batch_size, shuffle=True)
        direct_val_loader = DataLoader(DirectCurveDataset(X_case_val, q_val), batch_size=args.batch_size, shuffle=False)
        direct_test_loader = DataLoader(DirectCurveDataset(X_case_test, q_test), batch_size=args.batch_size, shuffle=False)
        model = DirectCurveModel(X_case_train.shape[-1], args.hidden_dim, len(future_times), args.dropout)
        criterion = BinomialHazardNLL()
        model, history = train_direct_model(model, direct_train_loader, direct_val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["direct_mean_std"] = evaluate_direct_model(model, direct_test_loader, criterion, device)
        save_csv(output_dir / "direct_mean_std_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "direct_mean_std_model.pt")

    criterion = DualHeadSurvivalLoss(
        system_weight=args.system_loss_weight,
        consistency_weight=args.consistency_weight,
    )

    if "traj" in requested_models:
        model = TrajectoryContextSurvivalModel(
            X_train.shape[-1],
            args.hidden_dim,
            len(future_times),
            encoder_type=args.encoder_type,
            dropout=args.dropout,
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
        )
        model, history = train_trajectory_model(model, train_loader, val_loader, criterion, args.epochs, args.lr, args.weight_decay, device)
        metrics["dual"] = evaluate_trajectory_model(model, test_loader, criterion, device)
        save_csv(output_dir / "dual_train_log.csv", history)
        torch.save({"model_state_dict": model.state_dict()}, output_dir / "dual_model.pt")

    save_survival_plots(output_dir, future_times, metrics)
    prediction_arrays: dict[str, np.ndarray] = {}
    for name, payload in metrics.items():
        if "preds" in payload:
            prediction_arrays[f"{name}_preds"] = payload["preds"]
        if "variation_preds" in payload:
            prediction_arrays[f"{name}_variation_preds"] = payload["variation_preds"]
        if "mean_variation_preds" in payload:
            prediction_arrays[f"{name}_mean_variation_preds"] = payload["mean_variation_preds"]
    np.savez(
        output_dir / "test_predictions.npz",
        **prediction_arrays,
        targets=q_test,
        variation_targets=y_test,
        future_times=future_times,
    )
    save_json(output_dir / "metrics.json", {name: compact_metrics(payload) for name, payload in metrics.items()})
    print(f"Task C complete. Outputs: {output_dir}")
    for name, payload in metrics.items():
        print(name, compact_metrics(payload))


if __name__ == "__main__":
    main()
