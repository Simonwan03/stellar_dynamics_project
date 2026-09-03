from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import h5py
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


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
RUNS_PER_BATCH = 8000
GROUP_SIZE = 32
DEFAULT_TIME_HORIZON = 10000.0
FULL_RESULT_KEYS = (
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
ORBITAL15_RESULT_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
)
ANGLE_RESULT_KEYS = {"APs_complete", "LANs_complete"}


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
    result_keys: tuple[str, ...],
    prefix_len: int,
    time_stride: int,
    rows_per_shard: int | None,
    include_geom: bool,
    add_time: bool,
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    data_parts: dict[str, list[np.ndarray]] = {key: [] for key in result_keys}
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
    # The completion table already records the output batch.  Derive the row
    # order within each batch from the global index so shards with a different
    # row count (for example sampling_3: 4000 instead of 8000) align correctly.
    labels["batch_id"] = labels["num_batch"]
    labels = labels.sort_values(["batch_id", "index"]).reset_index(drop=True)
    labels["row_in_batch"] = labels.groupby("batch_id", sort=False).cumcount()
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
    result_keys: tuple[str, ...],
    add_time: bool = True,
    include_geom: bool = True,
) -> tuple[np.ndarray, list[str]]:
    channels: list[np.ndarray] = []
    names: list[str] = []
    for key in result_keys:
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


def build_group_fraction_samples(
    X: np.ndarray,
    y: np.ndarray,
    labels: pd.DataFrame,
    group_size: int = GROUP_SIZE,
    aggregation: str = "concat",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels = labels.copy()
    labels["raw_case_id"] = labels["index"] // group_size
    labels["variation_id"] = labels["index"] % group_size
    labels["_row"] = np.arange(len(labels), dtype=np.int64)
    group_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for case_id, frame in labels.groupby("raw_case_id", sort=True):
        if len(frame) != group_size:
            continue
        frame = frame.sort_values("variation_id")
        if not np.array_equal(frame["variation_id"].to_numpy(dtype=np.int64), np.arange(group_size)):
            continue
        rows = frame["_row"].to_numpy(dtype=np.int64)
        if not np.isfinite(X[rows]).all():
            continue
        group_rows.append(rows)
        stable = y[rows].astype(np.float32)
        meta_rows.append(
            {
                "raw_case_id": int(case_id),
                "stable_count": int(stable.sum()),
                "stable_fraction": float(stable.mean()),
                "valid_variations": int(group_size),
                "min_time_unbound": float(frame["time_unbound"].min()),
                "median_time_unbound": float(frame["time_unbound"].median()),
                "max_time_unbound": float(frame["time_unbound"].max()),
            }
        )
    if not group_rows:
        raise ValueError("No complete finite trajectory groups were found for group_fraction mode.")
    rows_arr = np.stack(group_rows, axis=0)
    X_group = X[rows_arr]  # [G, group, T, C]
    X_group = X_group.transpose(0, 2, 1, 3)  # [G, T, group, C]
    if aggregation == "concat":
        X_group = X_group.reshape(X_group.shape[0], X_group.shape[1], -1)
    elif aggregation == "set_attention":
        pass
    else:
        raise ValueError(f"Unsupported group aggregation: {aggregation}")
    y_group = np.asarray([row["stable_fraction"] for row in meta_rows], dtype=np.float32)
    return X_group.astype(np.float32), y_group, pd.DataFrame(meta_rows)


def grouped_case_split_indices(
    y: np.ndarray,
    sample_case_ids: np.ndarray,
    test_size: int | float = 0.2,
    val_size: int | float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1).astype(np.float32)
    sample_case_ids = np.asarray(sample_case_ids).reshape(-1).astype(np.int64)
    if y.size != sample_case_ids.size:
        raise ValueError(f"y and sample_case_ids must have the same length, got {y.size} and {sample_case_ids.size}.")

    unique_case_ids, inverse = np.unique(sample_case_ids, return_inverse=True)
    n_cases = unique_case_ids.size
    if n_cases < 3:
        raise ValueError(f"Need at least 3 cases for train/val/test split, got {n_cases}.")

    def resolve_size(size: int | float, base: int) -> int:
        if isinstance(size, float) and 0.0 < size < 1.0:
            return int(round(base * size))
        return int(size)

    case_scores = np.zeros(n_cases, dtype=np.float32)
    for case_idx in range(n_cases):
        rows = inverse == case_idx
        case_scores[case_idx] = float(y[rows].mean())

    # Coarse buckets keep split balance while avoiding sparse 33-class stratification on stable_count.
    case_bins = np.zeros(n_cases, dtype=np.int64)
    case_bins[(case_scores > 0.0) & (case_scores < 0.5)] = 1
    case_bins[(case_scores >= 0.5) & (case_scores < 1.0)] = 2
    case_bins[case_scores >= 1.0] = 3

    rng = np.random.default_rng(seed)
    all_case_idx = np.arange(n_cases)

    def sample_stratified_cases(pool: np.ndarray, target_size: int) -> np.ndarray:
        target_size = max(0, min(int(target_size), pool.size - 1))
        if target_size == 0:
            return np.array([], dtype=np.int64)
        classes = np.unique(case_bins[pool])
        selected: list[np.ndarray] = []
        remaining = target_size
        for cls in classes[:-1]:
            cls_idx = pool[case_bins[pool] == cls]
            take = int(round(target_size * cls_idx.size / pool.size))
            take = min(take, cls_idx.size)
            if take > 0:
                selected.append(rng.choice(cls_idx, size=take, replace=False))
                remaining -= take
        cls_idx = pool[case_bins[pool] == classes[-1]]
        take = min(remaining, cls_idx.size)
        if take > 0:
            selected.append(rng.choice(cls_idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
        if out.size < target_size:
            extras = np.setdiff1d(pool, out, assume_unique=False)
            out = np.concatenate([out, rng.choice(extras, size=target_size - out.size, replace=False)])
        rng.shuffle(out)
        return out.astype(np.int64)

    test_case_count = resolve_size(test_size, n_cases)
    test_case_idx = sample_stratified_cases(all_case_idx, test_case_count)
    train_val_case_idx = np.setdiff1d(all_case_idx, test_case_idx, assume_unique=False)
    val_case_count = resolve_size(val_size, train_val_case_idx.size)
    val_case_idx = sample_stratified_cases(train_val_case_idx, val_case_count)
    train_case_idx = np.setdiff1d(train_val_case_idx, val_case_idx, assume_unique=False)
    rng.shuffle(train_case_idx)

    def expand(case_idx: np.ndarray) -> np.ndarray:
        mask = np.isin(inverse, case_idx)
        rows = np.flatnonzero(mask).astype(np.int64)
        rng.shuffle(rows)
        return rows

    return expand(train_case_idx), expand(val_case_idx), expand(test_case_idx)


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_b"))
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
