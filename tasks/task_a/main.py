from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4
import copy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.experiment_utils import (
    ensure_dir,
    parse_int_list,
    parse_name_list,
    save_csv,
    save_json,
    save_binary_curves,
    select_device,
    set_random_seed,
    standardize_from_train,
    binary_metrics_from_probs,
    select_binary_threshold_by_f1,
)
from tasks.common.case_split import (
    case_split_indices,
    default_case_split_manifest_path,
    ensure_case_split_manifest,
    missing_case_ids,
)
from tasks.common.traditional_stability import (
    load_initial_conditions,
    load_vynatheya2022_initial_conditions,
)
from tasks.common.pretrained_triple_mlp import (
    DEFAULT_PRETRAINED_GHOST_MLP_PATH,
    DEFAULT_PRETRAINED_MLP_PATH,
    PRETRAINED_GHOST_MLP_SHA256,
    PRETRAINED_GHOST_MLP_SOURCE_URL,
    PRETRAINED_MLP_SHA256,
    PRETRAINED_MLP_SOURCE_URL,
    load_pretrained_mlp_initial_conditions,
)


DEFAULT_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
ANGLE_FEATURES = {"APs", "LANs"}
MLP_PRESETS: dict[str, dict[str, Any]] = {
    "small": {"hidden": (64, 64, 32), "dropout": 0.1},
    "base": {"hidden": (128, 128, 64), "dropout": 0.2},
    "large": {"hidden": (256, 256, 128), "dropout": 0.2},
}
XGB_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "n_jobs": -1,
    },
    "base": {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "n_jobs": -1,
    },
    "strong": {
        "objective": "reg:squarederror",
        "n_estimators": 2000,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "n_jobs": -1,
    },
}


def sincos(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


def inspect_hdf5(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append({"name": name, "shape": tuple(obj.shape), "dtype": str(obj.dtype)})

        handle.visititems(collect)
    return rows


def regroup_initial_conditions(
    input_path: str | Path,
    output_path: str | Path,
    group_size: int = 32,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    angle_sincos: bool = True,
    overwrite: bool = False,
) -> dict[str, tuple[int, ...]]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return inspect_tokenized_shapes(output_path)

    tokenized: dict[str, np.ndarray] = {}
    with h5py.File(input_path, "r") as fin:
        for name in features:
            if name not in fin:
                raise KeyError(f"Missing feature {name!r} in {input_path}")
            x = fin[name][:]
            n = (x.shape[0] // group_size) * group_size
            x = x[:n].reshape(-1, group_size, *x.shape[1:])
            if angle_sincos and name in ANGLE_FEATURES:
                x = sincos(x)
            tokenized[name] = x

    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}-{uuid4().hex}")
    try:
        with h5py.File(tmp_path, "w") as fout:
            for name, arr in tokenized.items():
                fout.create_dataset(name, data=arr, compression="gzip")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {name: tuple(arr.shape) for name, arr in tokenized.items()}


def inspect_tokenized_shapes(path: str | Path) -> dict[str, tuple[int, ...]]:
    with h5py.File(path, "r") as handle:
        return {name: tuple(handle[name].shape) for name in sorted(handle.keys())}


def expected_tokenized_shapes(
    input_path: str | Path,
    group_size: int = 32,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    angle_sincos: bool = True,
) -> dict[str, tuple[int, ...]]:
    input_path = Path(input_path)
    expected: dict[str, tuple[int, ...]] = {}
    with h5py.File(input_path, "r") as handle:
        for name in features:
            if name not in handle:
                raise KeyError(f"Missing feature {name!r} in {input_path}")
            shape = list(handle[name].shape[1:])
            if angle_sincos and name in ANGLE_FEATURES:
                shape[-1] *= 2
            expected[name] = (handle[name].shape[0] // group_size, group_size, *shape)
    return expected


def build_group_stable_labels(
    labels_path: str | Path,
    group_size: int = 32,
    threshold_time: float = 10000.0,
    require_complete: bool = True,
    target_mode: str = "stable_fraction",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if target_mode not in {"stable_fraction", "binary_any"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    df = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    df = df.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float32",
            "wall_time": "float32",
            "diagnosis": "int64",
        }
    )
    df = df.sort_values("index").reset_index(drop=True)
    df["raw_case_id"] = df["index"] // group_size
    df["variation_id"] = df["index"] % group_size
    df["valid"] = (df["time_unbound"] >= 0.0) & (df["diagnosis"] == 0)
    valid = df[df["valid"]].copy()
    if require_complete:
        counts = valid["raw_case_id"].value_counts().sort_index()
        complete_case_ids = counts[counts == group_size].index.to_numpy()
        valid = valid[valid["raw_case_id"].isin(complete_case_ids)].copy()

    valid["survived_horizon"] = (valid["time_unbound"] >= threshold_time).astype(np.int32)
    grouped = (
        valid.groupby("raw_case_id")
        .agg(
            stable_count=("survived_horizon", "sum"),
            stable_fraction=("survived_horizon", "mean"),
            valid_variations=("survived_horizon", "size"),
            min_time_unbound=("time_unbound", "min"),
            median_time_unbound=("time_unbound", "median"),
            max_time_unbound=("time_unbound", "max"),
        )
        .reset_index()
    )
    grouped["stable_binary"] = (grouped["stable_count"] == group_size).astype(np.int32)
    grouped["any_unbound"] = (grouped["stable_count"] < group_size).astype(np.int32)
    case_ids = grouped["raw_case_id"].to_numpy(dtype=np.int64)
    if target_mode == "stable_fraction":
        y = grouped["stable_fraction"].to_numpy(dtype=np.float32)
    else:
        # A system is positive/stable only when all 32 variations survive.
        # Thus one variation unbound before threshold_time gives label 0.
        y = grouped["stable_binary"].to_numpy(dtype=np.float32)
    return case_ids, y, grouped


def load_tokenized_features(
    tokenized_path: str | Path,
    case_ids: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    feature_names: list[str] = []
    with h5py.File(tokenized_path, "r") as handle:
        n_cases = min(handle[name].shape[0] for name in features)
        valid_case_ids = np.asarray(case_ids, dtype=np.int64)
        if valid_case_ids.max(initial=-1) >= n_cases:
            raise IndexError(
                f"Label case id exceeds tokenized IC rows: max case={valid_case_ids.max()}, rows={n_cases}."
            )
        for name in features:
            arr = np.asarray(handle[name][valid_case_ids], dtype=np.float32)
            flat = arr[:, 0, ...].reshape(arr.shape[0], -1)
            names = [f"{name}_first_{i}" for i in range(flat.shape[1])]
            arrays.append(flat)
            feature_names.extend(names)
    X = np.concatenate(arrays, axis=1).astype(np.float32)
    return X, feature_names


def filter_finite_cases(
    X: np.ndarray,
    y: np.ndarray,
    label_df: pd.DataFrame,
    case_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    finite_mask = np.isfinite(X).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        print(f"Dropping {dropped} cases with non-finite initial-condition features.")
    return X[finite_mask], y[finite_mask], label_df.loc[finite_mask].reset_index(drop=True), case_ids[finite_mask]


def equation2_group_predictions(
    input_path: str | Path,
    case_ids: np.ndarray,
    group_size: int,
) -> pd.DataFrame:
    """Return Equation (2) fraction and all-stable predictions for systems."""

    predictions = load_initial_conditions(Path(input_path), group_size=group_size)
    groups = aggregate_variation_stability(predictions, group_size=group_size)
    selected = groups.reindex(np.asarray(case_ids, dtype=np.int64))
    if selected.isna().any(axis=None):
        missing = selected[selected.isna().any(axis=1)].index.to_numpy(dtype=np.int64)
        raise ValueError(
            "Equation (2) did not produce predictions for all selected systems; "
            f"missing raw_case_id values: {missing[:10].tolist()}"
        )
    return selected


def aggregate_variation_stability(
    predictions: pd.DataFrame,
    group_size: int,
) -> pd.DataFrame:
    """Aggregate hard per-variation criterion outputs to Task A systems."""

    required = {"group_id", "variation_id", "stable", "stability_margin"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Criterion predictions are missing columns: {missing}")
    groups = (
        predictions.groupby("group_id", sort=True)
        .agg(
            variations=("variation_id", "size"),
            unique_variations=("variation_id", "nunique"),
            predicted_stable_fraction=("stable", "mean"),
            stability_margin=("stability_margin", "min"),
        )
    )
    incomplete = groups[
        (groups["variations"] != group_size) | (groups["unique_variations"] != group_size)
    ]
    if not incomplete.empty:
        raise ValueError(
            f"Expected exactly {group_size} unique variations per system; "
            f"found {len(incomplete)} incomplete systems."
        )
    groups["prediction_stable"] = (groups["stability_margin"] > 0.0).astype(np.int8)
    return groups


def equation4_group_predictions(
    input_path: str | Path,
    case_ids: np.ndarray,
    group_size: int,
) -> pd.DataFrame:
    """Return Equation (4) fraction and all-stable predictions for systems."""

    predictions = load_vynatheya2022_initial_conditions(Path(input_path), group_size=group_size)
    groups = aggregate_variation_stability(predictions, group_size=group_size)
    selected = groups.reindex(np.asarray(case_ids, dtype=np.int64))
    if selected.isna().any(axis=None):
        missing = selected[selected.isna().any(axis=1)].index.to_numpy(dtype=np.int64)
        raise ValueError(
            "Equation (4) did not produce predictions for all selected systems; "
            f"missing raw_case_id values: {missing[:10].tolist()}"
        )
    return selected


def aggregate_pretrained_mlp_predictions(
    predictions: pd.DataFrame,
    group_size: int,
) -> pd.DataFrame:
    """Aggregate official per-variation MLP probabilities to Task A systems."""

    required = {
        "group_id",
        "variation_id",
        "stable",
        "probability_stable",
        "probability_unstable",
        "in_training_domain",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Pretrained MLP predictions are missing columns: {missing}")
    groups = predictions.groupby("group_id", sort=True).agg(
        variations=("variation_id", "size"),
        unique_variations=("variation_id", "nunique"),
        predicted_stable_fraction=("probability_stable", "mean"),
        hard_stable_fraction=("stable", "mean"),
        probability_stable=("probability_stable", "min"),
        probability_unstable=("probability_unstable", "max"),
        variations_inside_training_domain=("in_training_domain", "sum"),
    )
    incomplete = groups[
        (groups["variations"] != group_size) | (groups["unique_variations"] != group_size)
    ]
    if not incomplete.empty:
        raise ValueError(
            f"Expected exactly {group_size} unique variations per system; "
            f"found {len(incomplete)} incomplete systems."
        )
    groups["prediction_stable"] = (groups["probability_unstable"] < 0.5).astype(np.int8)
    return groups


def pretrained_mlp_group_predictions(
    input_path: str | Path,
    model_path: str | Path,
    case_ids: np.ndarray,
    group_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return official pretrained MLP predictions for selected Task A systems."""

    predictions, metadata = load_pretrained_mlp_initial_conditions(
        Path(input_path), Path(model_path), group_size=group_size
    )
    groups = aggregate_pretrained_mlp_predictions(predictions, group_size=group_size)
    selected = groups.reindex(np.asarray(case_ids, dtype=np.int64))
    if selected.isna().any(axis=None):
        missing = selected[selected.isna().any(axis=1)].index.to_numpy(dtype=np.int64)
        raise ValueError(
            "Pretrained MLP did not produce predictions for all selected systems; "
            f"missing raw_case_id values: {missing[:10].tolist()}"
        )
    return selected, metadata


def resolve_binary_threshold(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    requested_threshold: float | None,
) -> tuple[float, str]:
    if requested_threshold is not None:
        threshold = float(requested_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"--binary-threshold must lie in [0, 1], got {requested_threshold}.")
        return threshold, "fixed_cli_value"
    return select_binary_threshold_by_f1(y_val, prob_val), "validation_max_f1"


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    *arrays: tuple[np.ndarray, np.ndarray],
) -> tuple[DataLoader, ...]:
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.reshape(-1, 1)).float())
    loaders: list[DataLoader] = [DataLoader(train_ds, batch_size=batch_size, shuffle=True)]
    for X, y in arrays:
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.reshape(-1, 1)).float())
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False))
    return tuple(loaders)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.2,
        output_mode: str = "mse",
    ):
        super().__init__()
        if output_mode not in {"mse", "bce"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        if output_mode == "mse":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.output_mode = output_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics_from_preds(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    pred = np.asarray(pred).reshape(-1).astype(np.float64)
    pred = np.clip(pred, 0.0, 1.0)
    error = pred - y_true
    mse = float(mean_squared_error(y_true, pred))
    binomial_noise = float(np.mean(y_true * (1.0 - y_true) / 32.0)) if y_true.size else float("nan")
    target_ranks = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
    pred_ranks = pd.Series(pred).rank(method="average").to_numpy(dtype=np.float64)
    target_bins: dict[str, dict[str, float | int]] = {}
    for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        if upper == 1.0:
            mask = (y_true >= lower) & (y_true <= upper)
        else:
            mask = (y_true >= lower) & (y_true < upper)
        label = f"{lower:g}-{upper:g}"
        if np.any(mask):
            target_bins[label] = {
                "count": int(mask.sum()),
                "mae": float(np.mean(np.abs(error[mask]))),
                "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
                "bias": float(np.mean(error[mask])),
            }
        else:
            target_bins[label] = {"count": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "median_absolute_error": float(median_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)) if y_true.size > 1 else float("nan"),
        "explained_variance": float(explained_variance_score(y_true, pred)) if y_true.size > 1 else float("nan"),
        "pearson": _safe_correlation(y_true, pred),
        "spearman": _safe_correlation(target_ranks, pred_ranks),
        "mean_bias_error": float(np.mean(error)) if y_true.size else float("nan"),
        "target_mean": float(np.mean(y_true)) if y_true.size else float("nan"),
        "prediction_mean": float(np.mean(pred)) if pred.size else float("nan"),
        "within_abs_error_0.05": float(np.mean(np.abs(error) <= 0.05)) if y_true.size else float("nan"),
        "within_abs_error_0.10": float(np.mean(np.abs(error) <= 0.10)) if y_true.size else float("nan"),
        "within_abs_error_0.20": float(np.mean(np.abs(error) <= 0.20)) if y_true.size else float("nan"),
        "binomial_sampling_mse_estimate": binomial_noise,
        "mse_over_binomial_noise": float(mse / binomial_noise) if binomial_noise > 1e-12 else float("nan"),
        "by_target_bin": target_bins,
    }


def save_binary_predictions(
    path: str | Path,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> None:
    rows = [
        {"y_true": int(label), "probability": float(score), "prediction": int(score >= threshold)}
        for label, score in zip(np.asarray(y_true).reshape(-1), np.asarray(prob).reshape(-1))
    ]
    save_csv(path, rows)


def save_regression_scatter(path: str | Path, y_true: np.ndarray, pred: np.ndarray, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    y_true = np.asarray(y_true).reshape(-1)
    pred = np.clip(np.asarray(pred).reshape(-1), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, pred, s=12, alpha=0.45)
    ax.plot([0, 1], [0, 1], color="0.3", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True stable fraction")
    ax.set_ylabel("Predicted stable fraction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def predict_regressor(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for xb, yb in loader:
        output = model(xb.to(device))
        if getattr(model, "output_mode", "mse") == "bce":
            output = torch.sigmoid(output)
        preds.append(output.cpu().numpy().reshape(-1))
        labels.append(yb.numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(preds)


def train_regressor(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    in_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: tuple[int, ...],
    loss_mode: str,
    target_mode: str,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
) -> tuple[MLPRegressor, list[dict[str, float]], dict[str, Any]]:
    model = MLPRegressor(in_dim=in_dim, hidden=hidden, dropout=dropout, output_mode=loss_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_mode == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_mode == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    history: list[dict[str, float]] = []
    classification = target_mode == "binary_any"
    best_eval_metric = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        y_eval, eval_pred = predict_regressor(model, eval_loader, device)
        if classification:
            eval_metrics = binary_metrics_from_probs(y_eval, eval_pred)
            # Brier score is the probability MSE and is a smooth, threshold-free
            # validation criterion for selecting the best binary checkpoint.
            eval_metric = float(np.mean((np.asarray(y_eval) - np.asarray(eval_pred)) ** 2))
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / max(count, 1),
                    "eval_brier": eval_metric,
                    "eval_accuracy": eval_metrics["accuracy"],
                    "eval_f1": eval_metrics["f1"],
                    "eval_auc": eval_metrics["auc"],
                }
            )
        else:
            eval_metrics = regression_metrics_from_preds(y_eval, eval_pred)
            eval_metric = eval_metrics["mse"]
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / max(count, 1),
                    "eval_mse": eval_metrics["mse"],
                    "eval_mae": eval_metrics["mae"],
                    "eval_r2": eval_metrics["r2"],
                }
            )
        improved = eval_metric < (best_eval_metric - early_stopping_min_delta)
        if improved:
            best_eval_metric = float(eval_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if classification:
            print(
                f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} "
                f"eval_brier={history[-1]['eval_brier']:.5f} "
                f"eval_f1={history[-1]['eval_f1']:.5f}"
            )
        else:
            print(
                f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} "
                f"eval_mse={history[-1]['eval_mse']:.5f} eval_mae={history[-1]['eval_mae']:.5f}"
            )
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_eval_metric={best_eval_metric:.5f}"
            )
            break
    model.load_state_dict(best_state)
    fit_info = {
        "best_epoch": int(best_epoch),
        "best_eval_metric": float(best_eval_metric),
        "monitor": "brier" if classification else "mse",
        "epochs_ran": int(len(history)),
        "stopped_early": bool(len(history) < epochs),
        "early_stopping_patience": None if early_stopping_patience is None else int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
    }
    return model, history, fit_info


def run_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    random_state: int,
    preset: dict[str, Any],
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install xgboost or omit --models xgboost.") from exc

    model = xgb.XGBRegressor(
        **preset,
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    train_pred = np.clip(model.predict(X_train), 0.0, 1.0)
    test_pred = np.clip(model.predict(X_test), 0.0, 1.0)
    model.get_booster().save_model(output_dir / "xgboost_fraction_model.json")
    return {
        "train": regression_metrics_from_preds(y_train, train_pred),
        "test": regression_metrics_from_preds(y_test, test_pred),
    }


def run_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    random_state: int,
    preset: dict[str, Any],
    requested_threshold: float | None = None,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install xgboost or omit --models xgboost.") from exc

    classifier_preset = dict(preset)
    classifier_preset["objective"] = "binary:logistic"
    model = xgb.XGBClassifier(
        **classifier_preset,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
    )
    model.fit(X_train, y_train.astype(np.int32))
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    threshold, threshold_source = resolve_binary_threshold(y_val, val_prob, requested_threshold)
    model.get_booster().save_model(output_dir / "xgboost_binary_model.json")
    return {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "train": binary_metrics_from_probs(y_train, train_prob, threshold=threshold),
        "val": binary_metrics_from_probs(y_val, val_prob, threshold=threshold),
        "test": binary_metrics_from_probs(y_test, test_prob, threshold=threshold),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task A: group-level stable-fraction regression from tokenized ICs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument(
        "--target-mode",
        choices=["stable_fraction", "binary_any"],
        default="stable_fraction",
        help="Predict the 32-variation stable fraction, or binary all-stable (any unbound => 0).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_a/sampling_2/task_a_main"))
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--regroup", action="store_true", help="Create/update ICs_sampling_*_groupof*.hdf5.")
    parser.add_argument("--overwrite-tokenized", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated methods. Defaults to equation2,equation4,mlp,xgboost "
            "for both target modes."
        ),
    )
    parser.add_argument(
        "--pretrained-mlp-path",
        type=Path,
        default=DEFAULT_PRETRAINED_MLP_PATH,
        help=(
            "Official Vynatheya et al. mlp_model_trip_v1.2.2.pkl artifact. "
            "Required only when --models includes pretrained_mlp."
        ),
    )
    parser.add_argument(
        "--pretrained-ghost-mlp-path",
        type=Path,
        default=DEFAULT_PRETRAINED_GHOST_MLP_PATH,
        help=(
            "Official Vynatheya et al. mlp_model_trip_ghost_v1.2.2.pkl artifact. "
            "Required only when --models includes pretrained_ghost_mlp."
        ),
    )
    parser.add_argument(
        "--mlp-size",
        choices=sorted(MLP_PRESETS),
        default="base",
        help="Preset MLP profile. Can still be overridden with --hidden and/or --dropout.",
    )
    parser.add_argument(
        "--xgb-size",
        choices=sorted(XGB_PRESETS),
        default="base",
        help="Preset XGBoost profile.",
    )
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
    parser.add_argument("--max-cases", type=int, default=None, help="Optional first-N case limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden", type=str, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Stop MLP training if validation MSE does not improve for this many epochs. Use 0 or a negative value to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-MSE improvement required to reset the early-stopping patience counter.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=None,
        help="Binary decision threshold. Defaults to the validation-set F1 optimum.",
    )
    parser.add_argument(
        "--mlp-loss",
        choices=["mse", "bce"],
        default=None,
        help="MLP loss. Defaults to mse for stable_fraction and bce for binary_any.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    effective_mlp_loss = args.mlp_loss or ("bce" if args.target_mode == "binary_any" else "mse")

    ics_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}.hdf5"
    tokenized_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}_groupof{args.group_size}_taska_orbital15.hdf5"
    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    selected_features = parse_name_list(args.features) or list(DEFAULT_FEATURES)
    mlp_preset = MLP_PRESETS[args.mlp_size]
    xgb_preset = XGB_PRESETS[args.xgb_size]
    hidden = tuple(parse_int_list(args.hidden) or mlp_preset["hidden"])
    dropout = float(args.dropout if args.dropout is not None else mlp_preset["dropout"])
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    if args.models is None:
        models = {"equation2", "equation4", "mlp", "xgboost"}
    else:
        models = set(parse_name_list(args.models))
    if "equation_2" in models:
        models.remove("equation_2")
        models.add("equation2")
    for alias in ("equation_4", "vynatheya2022"):
        if alias in models:
            models.remove(alias)
            models.add("equation4")
    for alias in ("paper_mlp", "vynatheya_mlp", "pretrained_paper_mlp"):
        if alias in models:
            models.remove(alias)
            models.add("pretrained_mlp")
    for alias in ("ghost_mlp", "paper_ghost_mlp", "vynatheya_ghost_mlp"):
        if alias in models:
            models.remove(alias)
            models.add("pretrained_ghost_mlp")
    allowed_models = {
        "equation2",
        "equation4",
        "pretrained_mlp",
        "pretrained_ghost_mlp",
        "mlp",
        "xgboost",
    }
    unsupported = sorted(models - allowed_models)
    if unsupported:
        raise ValueError(
            f"Unsupported Task A methods for target_mode={args.target_mode}: {unsupported}. "
            f"Allowed methods: {sorted(allowed_models)}"
        )

    if args.regroup or not tokenized_path.exists():
        tokenized_shapes = regroup_initial_conditions(
            ics_path,
            tokenized_path,
            group_size=args.group_size,
            features=tuple(selected_features),
            overwrite=args.overwrite_tokenized,
        )
    else:
        tokenized_shapes = inspect_tokenized_shapes(tokenized_path)
        expected_shapes = expected_tokenized_shapes(
            ics_path,
            group_size=args.group_size,
            features=tuple(selected_features),
        )
        shape_mismatches = {
            name: {"found": tokenized_shapes.get(name), "expected": expected_shapes[name]}
            for name in selected_features
            if tokenized_shapes.get(name) != expected_shapes[name]
        }
        if shape_mismatches:
            raise ValueError(
                f"Tokenized IC file {tokenized_path} has unexpected layout: {shape_mismatches}. "
                "Re-run with --regroup --overwrite-tokenized."
            )

    case_ids, y, label_df = build_group_stable_labels(
        labels_path,
        group_size=args.group_size,
        threshold_time=args.time_horizon,
        target_mode=args.target_mode,
    )
    max_observed_horizon = float(label_df["max_time_unbound"].max())
    if args.time_horizon > max_observed_horizon + 1e-6:
        raise ValueError(
            f"Requested --time-horizon={args.time_horizon:g} exceeds the maximum "
            f"time_unbound={max_observed_horizon:g} in {labels_path}. "
            "Use a horizon within the recorded completion-time range."
        )
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
    if args.max_cases is not None:
        max_cases = max(1, int(args.max_cases))
        case_ids = case_ids[:max_cases]
        y = y[:max_cases]
        label_df = label_df.iloc[:max_cases].reset_index(drop=True)
    X, feature_names = load_tokenized_features(tokenized_path, case_ids, selected_features)
    X, y, label_df, case_ids = filter_finite_cases(X, y, label_df, case_ids)
    if args.max_cases is None:
        missing = missing_case_ids(case_ids, split_manifest_df)
        if missing.size:
            raise ValueError(
                "Task A does not contain the full common system universe after finite-value filtering; "
                f"missing {missing.size} raw_case_id values (first: {missing[:10].tolist()})."
            )
    if args.target_mode == "binary_any" and np.unique(y).size < 2:
        raise ValueError("binary_any requires both all-stable and any-unbound systems after filtering.")
    split_indices = case_split_indices(case_ids, split_manifest_df)
    train_idx, val_idx, test_idx = (
        split_indices["train"],
        split_indices["val"],
        split_indices["test"],
    )
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train, [X_val, X_test], norm = standardize_from_train(X_train, X_val, X_test)

    split_membership = np.empty(len(case_ids), dtype=object)
    for split_name, split_rows in split_indices.items():
        split_membership[split_rows] = split_name
    label_df = label_df.copy()
    label_df["split"] = split_membership
    label_df.to_csv(output_dir / "group_labels.csv", index=False)
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        case_ids=case_ids,
    )
    np.savez(output_dir / "normalization_stats.npz", mean=norm["mean"], std=norm["std"])
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "models": ",".join(sorted(models)),
            "device": str(device),
            "tokenized_path": tokenized_path,
            "tokenized_shapes": tokenized_shapes,
            "num_cases": int(len(y)),
            "num_features": int(X.shape[1]),
            "feature_names": feature_names,
            "target_mode": args.target_mode,
            "mlp_hidden": hidden,
            "mlp_dropout": dropout,
            "mlp_loss": effective_mlp_loss,
            "xgb_preset": xgb_preset,
            "label_mean": float(y.mean()),
            "label_min": float(y.min()),
            "label_max": float(y.max()),
            "split_manifest": split_manifest,
            "split_unit": "complete raw_case_id system",
            "split_horizon": float(args.time_horizon),
            "split_ratios": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "split_counts": {name: int(rows.size) for name, rows in split_indices.items()},
            "binary_threshold_policy": (
                {
                    "mlp_xgboost": (
                        "validation_max_f1"
                        if args.binary_threshold is None
                        else "fixed_cli_value"
                    )
                    if models & {"mlp", "xgboost"}
                    else None,
                    "equation2_equation4": (
                        "fixed_physics_boundary" if models & {"equation2", "equation4"} else None
                    ),
                    "pretrained_mlp": (
                        "fixed_upstream_0.5_plus_validation_max_f1"
                        if "pretrained_mlp" in models
                        else None
                    ),
                    "pretrained_ghost_mlp": (
                        "fixed_upstream_0.5_plus_validation_max_f1"
                        if "pretrained_ghost_mlp" in models
                        else None
                    ),
                }
            )
            if args.target_mode == "binary_any"
            else None,
            "binary_threshold_override": args.binary_threshold,
            "equation2_group_rule": (
                "binary_any uses the minimum Equation (2) margin; stable_fraction uses the "
                "mean hard stability decision across variations"
                if "equation2" in models
                else None
            ),
            "equation4_group_rule": (
                "binary_any uses the minimum Equation (4) margin; stable_fraction uses the "
                "mean hard stability decision across variations"
                if "equation4" in models
                else None
            ),
            "equation4_reference": (
                "Vynatheya et al. 2022, arXiv:2207.03151, Equations (3)-(4)"
                if "equation4" in models
                else None
            ),
            "equation4_lk_radicand_policy": (
                "clip to zero outside the classical LK inclination window"
                if "equation4" in models
                else None
            ),
            "pretrained_mlp_source": (
                PRETRAINED_MLP_SOURCE_URL if "pretrained_mlp" in models else None
            ),
            "pretrained_mlp_expected_sha256": (
                PRETRAINED_MLP_SHA256 if "pretrained_mlp" in models else None
            ),
            "pretrained_mlp_group_rule": (
                "binary_any uses the minimum variation stable probability at fixed 0.5; "
                "a secondary operating point selects the validation-F1 threshold; "
                "stable_fraction uses the mean variation stable probability"
                if "pretrained_mlp" in models
                else None
            ),
            "pretrained_ghost_mlp_source": (
                PRETRAINED_GHOST_MLP_SOURCE_URL
                if "pretrained_ghost_mlp" in models
                else None
            ),
            "pretrained_ghost_mlp_expected_sha256": (
                PRETRAINED_GHOST_MLP_SHA256
                if "pretrained_ghost_mlp" in models
                else None
            ),
            "pretrained_ghost_mlp_group_rule": (
                "binary_any uses the minimum variation stable probability at fixed 0.5; "
                "a secondary operating point selects the validation-F1 threshold; "
                "stable_fraction uses the mean variation stable probability"
                if "pretrained_ghost_mlp" in models
                else None
            ),
        },
    )

    all_metrics: dict[str, Any] = {}
    if "equation2" in models:
        equation2_groups = equation2_group_predictions(
            ics_path,
            case_ids,
            group_size=args.group_size,
        )
        equation2_fraction = equation2_groups["predicted_stable_fraction"].to_numpy(
            dtype=np.float64
        )
        equation2_binary = equation2_groups["prediction_stable"].to_numpy(dtype=np.float64)
        equation2_score = (
            equation2_binary if args.target_mode == "binary_any" else equation2_fraction
        )
        if args.target_mode == "binary_any":
            equation2_metrics = {
                split_name: binary_metrics_from_probs(y[split_rows], equation2_score[split_rows])
                for split_name, split_rows in split_indices.items()
            }
        else:
            equation2_metrics = {
                split_name: regression_metrics_from_preds(y[split_rows], equation2_score[split_rows])
                for split_name, split_rows in split_indices.items()
            }
        all_metrics["equation2"] = equation2_metrics
        equation2_rows = []
        for row, case_id in enumerate(case_ids):
            equation2_rows.append(
                {
                    "raw_case_id": int(case_id),
                    "split": str(split_membership[row]),
                    "y_true": float(y[row]),
                    "predicted_stable_fraction": float(equation2_fraction[row]),
                    "minimum_stability_margin": float(
                        equation2_groups.iloc[row]["stability_margin"]
                    ),
                    "prediction_stable": int(equation2_binary[row]),
                }
            )
        save_csv(output_dir / "equation2_predictions.csv", equation2_rows)

    if "equation4" in models:
        equation4_groups = equation4_group_predictions(
            ics_path,
            case_ids,
            group_size=args.group_size,
        )
        equation4_fraction = equation4_groups["predicted_stable_fraction"].to_numpy(
            dtype=np.float64
        )
        equation4_binary = equation4_groups["prediction_stable"].to_numpy(dtype=np.float64)
        equation4_score = (
            equation4_binary if args.target_mode == "binary_any" else equation4_fraction
        )
        if args.target_mode == "binary_any":
            equation4_metrics = {
                split_name: binary_metrics_from_probs(y[split_rows], equation4_score[split_rows])
                for split_name, split_rows in split_indices.items()
            }
        else:
            equation4_metrics = {
                split_name: regression_metrics_from_preds(y[split_rows], equation4_score[split_rows])
                for split_name, split_rows in split_indices.items()
            }
        all_metrics["equation4"] = equation4_metrics
        equation4_rows = []
        for row, case_id in enumerate(case_ids):
            equation4_rows.append(
                {
                    "raw_case_id": int(case_id),
                    "split": str(split_membership[row]),
                    "y_true": float(y[row]),
                    "predicted_stable_fraction": float(equation4_fraction[row]),
                    "minimum_stability_margin": float(
                        equation4_groups.iloc[row]["stability_margin"]
                    ),
                    "prediction_stable": int(equation4_binary[row]),
                }
            )
        save_csv(output_dir / "equation4_predictions.csv", equation4_rows)

    pretrained_model_paths = {
        "pretrained_mlp": args.pretrained_mlp_path,
        "pretrained_ghost_mlp": args.pretrained_ghost_mlp_path,
    }
    for pretrained_name, pretrained_path in pretrained_model_paths.items():
        if pretrained_name not in models:
            continue
        pretrained_groups, pretrained_metadata = pretrained_mlp_group_predictions(
            ics_path,
            pretrained_path,
            case_ids,
            group_size=args.group_size,
        )
        pretrained_fraction = pretrained_groups["predicted_stable_fraction"].to_numpy(
            dtype=np.float64
        )
        pretrained_binary_probability = pretrained_groups["probability_stable"].to_numpy(
            dtype=np.float64
        )
        pretrained_score = (
            pretrained_binary_probability
            if args.target_mode == "binary_any"
            else pretrained_fraction
        )
        if args.target_mode == "binary_any":
            pretrained_metrics = {
                split_name: binary_metrics_from_probs(
                    y[split_rows], pretrained_score[split_rows], threshold=0.5
                )
                for split_name, split_rows in split_indices.items()
            }
            pretrained_metrics["threshold"] = 0.5
            pretrained_metrics["threshold_source"] = "fixed_upstream_value"
            tuned_threshold = select_binary_threshold_by_f1(
                y[val_idx], pretrained_score[val_idx]
            )
            pretrained_metrics["validation_tuned"] = {
                "threshold": tuned_threshold,
                "threshold_source": "validation_max_f1",
                **{
                    split_name: binary_metrics_from_probs(
                        y[split_rows], pretrained_score[split_rows], threshold=tuned_threshold
                    )
                    for split_name, split_rows in split_indices.items()
                },
            }
        else:
            pretrained_metrics = {
                split_name: regression_metrics_from_preds(
                    y[split_rows], pretrained_score[split_rows]
                )
                for split_name, split_rows in split_indices.items()
            }
        pretrained_metadata = {
            **pretrained_metadata,
            "binary_group_score": "minimum variation probability_stable",
            "stable_fraction_group_score": "mean variation probability_stable",
        }
        pretrained_metrics["metadata"] = pretrained_metadata
        all_metrics[pretrained_name] = pretrained_metrics
        pretrained_rows = []
        for row, case_id in enumerate(case_ids):
            pretrained_rows.append(
                {
                    "raw_case_id": int(case_id),
                    "split": str(split_membership[row]),
                    "y_true": float(y[row]),
                    "predicted_stable_fraction": float(pretrained_fraction[row]),
                    "hard_stable_fraction": float(
                        pretrained_groups.iloc[row]["hard_stable_fraction"]
                    ),
                    "probability_stable": float(pretrained_binary_probability[row]),
                    "probability_unstable": float(
                        pretrained_groups.iloc[row]["probability_unstable"]
                    ),
                    "prediction_stable": int(
                        pretrained_groups.iloc[row]["prediction_stable"]
                    ),
                    "prediction_stable_val_tuned": (
                        int(pretrained_binary_probability[row] >= tuned_threshold)
                        if args.target_mode == "binary_any"
                        else None
                    ),
                    "variations_inside_training_domain": int(
                        pretrained_groups.iloc[row]["variations_inside_training_domain"]
                    ),
                }
            )
        save_csv(output_dir / f"{pretrained_name}_predictions.csv", pretrained_rows)
        save_json(output_dir / f"{pretrained_name}_metadata.json", pretrained_metadata)

    if "mlp" in models:
        train_loader, test_loader = make_loaders(X_train, y_train, args.batch_size, (X_test, y_test))
        val_loader = make_loaders(X_train, y_train, args.batch_size, (X_val, y_val))[1]
        mlp, history, fit_info = train_regressor(
            train_loader,
            val_loader,
            in_dim=X_train.shape[1],
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=dropout,
            hidden=hidden,
            loss_mode=effective_mlp_loss,
            target_mode=args.target_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        y_train_eval, train_pred = predict_regressor(mlp, train_loader, device)
        y_val_eval, val_pred = predict_regressor(mlp, val_loader, device)
        y_test_eval, test_pred = predict_regressor(mlp, test_loader, device)
        binary_threshold = None
        binary_threshold_source = None
        if args.target_mode == "binary_any":
            binary_threshold, binary_threshold_source = resolve_binary_threshold(
                y_val_eval,
                val_pred,
                args.binary_threshold,
            )
        model_stem = "binary" if args.target_mode == "binary_any" else "fraction"
        torch.save(
            {"model_state_dict": mlp.state_dict(), "feature_names": feature_names},
            output_dir / f"mlp_{model_stem}_model.pt",
        )
        save_csv(output_dir / f"mlp_{model_stem}_train_log.csv", history)
        if args.target_mode == "binary_any":
            assert binary_threshold is not None and binary_threshold_source is not None
            save_binary_predictions(
                output_dir / "mlp_binary_test_predictions.csv",
                y_test_eval,
                test_pred,
                threshold=binary_threshold,
            )
            save_binary_curves(output_dir / "mlp_binary_test_curves.png", y_test_eval, test_pred, "Task A binary MLP")
            all_metrics["mlp"] = {
                "fit": fit_info,
                "threshold": binary_threshold,
                "threshold_source": binary_threshold_source,
                "train": binary_metrics_from_probs(y_train_eval, train_pred, threshold=binary_threshold),
                "val": binary_metrics_from_probs(y_val_eval, val_pred, threshold=binary_threshold),
                "test": binary_metrics_from_probs(y_test_eval, test_pred, threshold=binary_threshold),
            }
        else:
            save_regression_scatter(output_dir / "mlp_fraction_test_scatter.png", y_test_eval, test_pred, "Task A MLP")
            all_metrics["mlp"] = {
                "fit": fit_info,
                "train": regression_metrics_from_preds(y_train_eval, train_pred),
                "val": regression_metrics_from_preds(y_val_eval, val_pred),
                "test": regression_metrics_from_preds(y_test_eval, test_pred),
            }

    if "xgboost" in models:
        if args.target_mode == "binary_any":
            all_metrics["xgboost"] = run_xgboost_classifier(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                output_dir,
                random_state=args.seed,
                preset=xgb_preset,
                requested_threshold=args.binary_threshold,
            )
        else:
            all_metrics["xgboost"] = run_xgboost_regressor(
                X_train,
                y_train,
                X_test,
                y_test,
                output_dir,
                random_state=args.seed,
                preset=xgb_preset,
            )

    save_json(output_dir / "metrics.json", all_metrics)
    print(f"Task A complete. Outputs: {output_dir}")
    for model_name, payload in all_metrics.items():
        if isinstance(payload, dict) and "test" in payload:
            print(model_name, payload["test"])


if __name__ == "__main__":
    main()
