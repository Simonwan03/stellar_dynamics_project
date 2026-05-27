from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_utils import (
    binary_metrics_from_probs,
    ensure_dir,
    parse_int_list,
    parse_name_list,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
    standardize_from_train,
)


DEFAULT_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
ANGLE_FEATURES = {"incs", "APs", "LANs"}


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
    with h5py.File(output_path, "w") as fout:
        for name, arr in tokenized.items():
            fout.create_dataset(name, data=arr, compression="gzip")
    return {name: tuple(arr.shape) for name, arr in tokenized.items()}


def inspect_tokenized_shapes(path: str | Path) -> dict[str, tuple[int, ...]]:
    with h5py.File(path, "r") as handle:
        return {name: tuple(handle[name].shape) for name in sorted(handle.keys())}


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
            is_stable=("survived_horizon", "max"),
            valid_variations=("survived_horizon", "size"),
            min_time_unbound=("time_unbound", "min"),
            median_time_unbound=("time_unbound", "median"),
            max_time_unbound=("time_unbound", "max"),
        )
        .reset_index()
    )
    case_ids = grouped["raw_case_id"].to_numpy(dtype=np.int64)
    y = grouped["is_stable"].to_numpy(dtype=np.float32)
    return case_ids, y, grouped


def load_tokenized_features(
    tokenized_path: str | Path,
    case_ids: np.ndarray,
    features: list[str],
    reduction: str = "flatten",
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
            if reduction == "flatten":
                flat = arr.reshape(arr.shape[0], -1)
                names = [f"{name}_{i}" for i in range(flat.shape[1])]
            elif reduction == "mean_std":
                mean = arr.mean(axis=1)
                std = arr.std(axis=1)
                flat = np.concatenate([mean.reshape(arr.shape[0], -1), std.reshape(arr.shape[0], -1)], axis=1)
                names = [f"{name}_mean_{i}" for i in range(mean.reshape(arr.shape[0], -1).shape[1])]
                names += [f"{name}_std_{i}" for i in range(std.reshape(arr.shape[0], -1).shape[1])]
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
            arrays.append(flat)
            feature_names.extend(names)
    X = np.concatenate(arrays, axis=1).astype(np.float32)
    return X, feature_names


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.reshape(-1, 1)).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test.reshape(-1, 1)).float())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_mlp(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        labels.append(yb.numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(probs)


def train_mlp(
    train_loader: DataLoader,
    test_loader: DataLoader,
    in_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: tuple[int, ...],
) -> tuple[MLPBinaryClassifier, list[dict[str, float]]]:
    model = MLPBinaryClassifier(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        _, train_prob = predict_mlp(model, train_loader, device)
        y_train = train_loader.dataset.tensors[1].numpy().reshape(-1)
        y_test, test_prob = predict_mlp(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(count, 1),
                "train_auc": binary_metrics_from_probs(y_train, train_prob)["auc"],
                "test_auc": binary_metrics_from_probs(y_test, test_prob)["auc"],
            }
        )
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} test_auc={history[-1]['test_auc']:.4f}")
    return model, history


def run_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install xgboost or omit --models xgboost.") from exc

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    model.save_model(output_dir / "xgboost_model.json")
    save_binary_curves(output_dir / "xgboost_test_curves.png", y_test, test_prob, "Task A XGBoost")
    return {
        "train": binary_metrics_from_probs(y_train, train_prob),
        "test": binary_metrics_from_probs(y_test, test_prob),
    }


def run_seed_sweep(
    X: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    test_size: int | float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: tuple[int, ...],
    device: torch.device,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        set_random_seed(seed)
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=seed,
            stratify=y if len(np.unique(y)) == 2 else None,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, [X_test], _ = standardize_from_train(X_train, X_test)
        train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size)
        model, _ = train_mlp(
            train_loader,
            test_loader,
            in_dim=X_train.shape[1],
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            hidden=hidden,
        )
        y_eval, prob = predict_mlp(model, test_loader, device)
        metrics = binary_metrics_from_probs(y_eval, prob)
        rows.append(
            {
                "seed": float(seed),
                "auc": float(metrics["auc"]),
                "average_precision": float(metrics["average_precision"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task A: group-level stable-at-horizon classifiers from tokenized ICs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_a"))
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--reduction", choices=["flatten", "mean_std"], default="flatten")
    parser.add_argument("--regroup", action="store_true", help="Create/update ICs_sampling_*_groupof*.hdf5.")
    parser.add_argument("--overwrite-tokenized", action="store_true")
    parser.add_argument("--models", type=str, default="mlp,xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=str, default="128,128,64")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed-sweep", type=str, default=None, help="Comma-separated seeds for an MLP split-sensitivity sweep.")
    parser.add_argument("--sweep-epochs", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)

    ics_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}.hdf5"
    tokenized_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}_groupof{args.group_size}.hdf5"
    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    selected_features = parse_name_list(args.features) or list(DEFAULT_FEATURES)
    hidden = tuple(parse_int_list(args.hidden) or [128, 128, 64])
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

    case_ids, y, label_df = build_group_stable_labels(
        labels_path,
        group_size=args.group_size,
        threshold_time=args.time_horizon,
    )
    X, feature_names = load_tokenized_features(tokenized_path, case_ids, selected_features, reduction=args.reduction)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) == 2 else None,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, [X_test], norm = standardize_from_train(X_train, X_test)

    label_df.to_csv(output_dir / "group_labels.csv", index=False)
    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)
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
            "label_positive_rate": float(y.mean()),
        },
    )

    all_metrics: dict[str, Any] = {}
    if "mlp" in models:
        train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, args.batch_size)
        mlp, history = train_mlp(
            train_loader,
            test_loader,
            in_dim=X_train.shape[1],
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden=hidden,
        )
        y_train_eval, train_prob = predict_mlp(mlp, train_loader, device)
        y_test_eval, test_prob = predict_mlp(mlp, test_loader, device)
        torch.save({"model_state_dict": mlp.state_dict(), "feature_names": feature_names}, output_dir / "mlp_model.pt")
        save_csv(output_dir / "mlp_train_log.csv", history)
        save_binary_curves(output_dir / "mlp_test_curves.png", y_test_eval, test_prob, "Task A MLP")
        all_metrics["mlp"] = {
            "train": binary_metrics_from_probs(y_train_eval, train_prob),
            "test": binary_metrics_from_probs(y_test_eval, test_prob),
        }

    if "xgboost" in models:
        all_metrics["xgboost"] = run_xgboost(X_train, y_train, X_test, y_test, output_dir)

    sweep_seeds = parse_int_list(args.seed_sweep)
    if sweep_seeds:
        rows = run_seed_sweep(
            X,
            y,
            seeds=sweep_seeds,
            test_size=args.test_size,
            batch_size=args.batch_size,
            epochs=args.sweep_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden=hidden,
            device=device,
        )
        save_csv(output_dir / "mlp_seed_sweep.csv", rows)
        all_metrics["mlp_seed_sweep"] = {
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "std_auc": float(np.std([row["auc"] for row in rows])),
            "rows": rows,
        }

    save_json(output_dir / "metrics.json", all_metrics)
    print(f"Task A complete. Outputs: {output_dir}")
    for model_name, payload in all_metrics.items():
        if isinstance(payload, dict) and "test" in payload:
            print(model_name, payload["test"])


if __name__ == "__main__":
    main()
