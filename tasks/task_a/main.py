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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    select_device,
    set_random_seed,
    standardize_from_train,
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
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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
    case_ids = grouped["raw_case_id"].to_numpy(dtype=np.int64)
    y = grouped["stable_fraction"].to_numpy(dtype=np.float32)
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


def regression_metrics_from_preds(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    pred = np.asarray(pred).reshape(-1).astype(np.float64)
    pred = np.clip(pred, 0.0, 1.0)
    return {
        "mse": float(mean_squared_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)) if y_true.size > 1 else float("nan"),
    }


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
    best_eval_mse = float("inf")
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
        eval_metrics = regression_metrics_from_preds(y_eval, eval_pred)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(count, 1),
                "eval_mse": eval_metrics["mse"],
                "eval_mae": eval_metrics["mae"],
                "eval_r2": eval_metrics["r2"],
            }
        )
        improved = eval_metrics["mse"] < (best_eval_mse - early_stopping_min_delta)
        if improved:
            best_eval_mse = float(eval_metrics["mse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} "
            f"eval_mse={history[-1]['eval_mse']:.5f} eval_mae={history[-1]['eval_mae']:.5f}"
        )
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_eval_mse={best_eval_mse:.5f}"
            )
            break
    model.load_state_dict(best_state)
    fit_info = {
        "best_epoch": int(best_epoch),
        "best_eval_mse": float(best_eval_mse),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task A: group-level stable-fraction regression from tokenized ICs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_a"))
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--regroup", action="store_true", help="Create/update ICs_sampling_*_groupof*.hdf5.")
    parser.add_argument("--overwrite-tokenized", action="store_true")
    parser.add_argument("--models", type=str, default="mlp,xgboost")
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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction of total cases for MLP monitoring.")
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
        "--mlp-loss",
        choices=["mse", "bce"],
        default="mse",
        help="MLP regression loss. 'mse' uses Sigmoid+MSE; 'bce' uses logits+BCEWithLogitsLoss with soft fraction labels.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)

    ics_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}.hdf5"
    tokenized_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}_groupof{args.group_size}_taska_orbital15.hdf5"
    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    selected_features = parse_name_list(args.features) or list(DEFAULT_FEATURES)
    mlp_preset = MLP_PRESETS[args.mlp_size]
    xgb_preset = XGB_PRESETS[args.xgb_size]
    hidden = tuple(parse_int_list(args.hidden) or mlp_preset["hidden"])
    dropout = float(args.dropout if args.dropout is not None else mlp_preset["dropout"])
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    models = set(parse_name_list(args.models))

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
    )
    if args.max_cases is not None:
        max_cases = max(1, int(args.max_cases))
        case_ids = case_ids[:max_cases]
        y = y[:max_cases]
        label_df = label_df.iloc[:max_cases].reset_index(drop=True)
    X, feature_names = load_tokenized_features(tokenized_path, case_ids, selected_features)
    X, y, label_df, case_ids = filter_finite_cases(X, y, label_df, case_ids)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.seed,
    )
    val_fraction_of_train_val = float(args.val_size) / max(1e-8, 1.0 - float(args.test_size))
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=args.seed,
    )
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train, [X_val, X_test], norm = standardize_from_train(X_train, X_val, X_test)

    label_df.to_csv(output_dir / "group_labels.csv", index=False)
    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    np.savez(output_dir / "normalization_stats.npz", mean=norm["mean"], std=norm["std"])
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "tokenized_path": tokenized_path,
            "tokenized_shapes": tokenized_shapes,
            "num_cases": int(len(y)),
            "num_features": int(X.shape[1]),
            "feature_names": feature_names,
            "mlp_hidden": hidden,
            "mlp_dropout": dropout,
            "mlp_loss": args.mlp_loss,
            "xgb_preset": xgb_preset,
            "label_mean": float(y.mean()),
            "label_min": float(y.min()),
            "label_max": float(y.max()),
        },
    )

    all_metrics: dict[str, Any] = {}
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
            loss_mode=args.mlp_loss,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        y_train_eval, train_pred = predict_regressor(mlp, train_loader, device)
        y_val_eval, val_pred = predict_regressor(mlp, val_loader, device)
        y_test_eval, test_pred = predict_regressor(mlp, test_loader, device)
        torch.save(
            {"model_state_dict": mlp.state_dict(), "feature_names": feature_names},
            output_dir / "mlp_fraction_model.pt",
        )
        save_csv(output_dir / "mlp_fraction_train_log.csv", history)
        save_regression_scatter(output_dir / "mlp_fraction_test_scatter.png", y_test_eval, test_pred, "Task A MLP")
        all_metrics["mlp"] = {
            "fit": fit_info,
            "train": regression_metrics_from_preds(y_train_eval, train_pred),
            "val": regression_metrics_from_preds(y_val_eval, val_pred),
            "test": regression_metrics_from_preds(y_test_eval, test_pred),
        }

    if "xgboost" in models:
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
