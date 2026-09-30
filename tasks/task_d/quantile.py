"""Task D: predict a system's median orbital-decorrelation time."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
PAIR_COUNT = 16
GROUP_SIZE = 32
ORBITAL_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
)
ANGLE_KEYS = frozenset({"APs_complete", "LANs_complete"})


@dataclass
class RegressionConfig:
    dataset_dir: str | Path
    output_dir: str | Path
    sampling_id: int = 3
    completion_table_name: str = "completed_systems_sampling_3.txt"
    threshold_power: float = 5.0
    min_valid_pairs: int = 8
    interpolate_crossings: bool = True
    models: tuple[str, ...] = ("constant", "xgboost", "mlp")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.15
    patience: int = 15
    device: str | None = None
    max_shards: int | None = None
    max_systems: int | None = None
    bootstrap_samples: int = 2000


# Compatibility for existing imports. This is no longer a quantile task.
QuantileConfig = RegressionConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def _discover_paths(dataset_dir: Path, sampling_id: int) -> list[Path]:
    paths = sorted(dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if not paths:
        raise FileNotFoundError(
            f"No results_batch_*_sampling_{sampling_id}.hdf5 under {dataset_dir}"
        )
    return paths


def _batch_id(path: Path) -> int:
    match = RESULT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot parse batch id from {path}")
    return int(match.group(1))


def _load_completion(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "batch", "time_unbound", "aux", "diagnosis"],
    )
    return table.astype(
        {
            "index": "int64",
            "batch": "int64",
            "time_unbound": "float64",
            "diagnosis": "int64",
        }
    )


def interpolate_first_crossing(
    growth: np.ndarray,
    times: np.ndarray,
    threshold: float,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return crossing time, event indicator, and sampled crossing index."""
    growth = np.asarray(growth, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    hits = np.isfinite(growth) & (growth >= threshold) & np.isfinite(times)[None, :]
    observed = hits.any(axis=1)
    indices = np.argmax(hits, axis=1).astype(np.int64)
    crossing = np.full(growth.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(observed)
    crossing[rows] = times[indices[rows]]
    if not interpolate or not rows.size:
        return crossing, observed, indices

    previous = indices[rows] - 1
    can_interpolate = (
        (previous >= 0)
        & np.isfinite(growth[rows, previous])
        & np.isfinite(growth[rows, indices[rows]])
        & (growth[rows, previous] > 0.0)
        & (growth[rows, indices[rows]] > growth[rows, previous])
        & (growth[rows, previous] < threshold)
    )
    selected = rows[can_interpolate]
    if selected.size:
        left_index = indices[selected] - 1
        right_index = indices[selected]
        left_log = np.log(growth[selected, left_index])
        right_log = np.log(growth[selected, right_index])
        fraction = (math.log(threshold) - left_log) / (right_log - left_log)
        fraction = np.clip(fraction, 0.0, 1.0)
        crossing[selected] = times[left_index] + fraction * (
            times[right_index] - times[left_index]
        )
    return crossing, observed, indices


def aggregate_pair_times(
    crossing_periods: Sequence[float],
    min_valid_pairs: int,
    threshold_power: float | None = None,
) -> dict[str, Any]:
    """Take the median observed crossing time without dividing by k."""
    del threshold_power  # accepted only for compatibility with older callers
    crossing = np.asarray(crossing_periods, dtype=np.float64)
    pair_times = crossing[np.isfinite(crossing) & (crossing > 0.0)]
    count = int(pair_times.size)
    system_time = float(np.median(pair_times)) if count >= min_valid_pairs else math.nan
    return {
        "num_valid_pairs": count,
        "median_decorrelation_time_periods": system_time,
        "pair_crossing_p25_periods": (
            float(np.quantile(pair_times, 0.25)) if count else math.nan
        ),
        "pair_crossing_p50_periods": (
            float(np.quantile(pair_times, 0.50)) if count else math.nan
        ),
        "pair_crossing_p75_periods": (
            float(np.quantile(pair_times, 0.75)) if count else math.nan
        ),
    }


def _flatten_initial_features(handle: h5py.File) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for key in ORBITAL_KEYS:
        values = np.asarray(handle[key][:, 0, ...], dtype=np.float64)
        values = values.reshape(values.shape[0], -1)
        base_names = [f"{key}_{index}" for index in range(values.shape[1])]
        if key in ANGLE_KEYS:
            blocks.extend((np.sin(values), np.cos(values)))
            names.extend([f"sin_{name}" for name in base_names])
            names.extend([f"cos_{name}" for name in base_names])
        else:
            blocks.append(values)
            names.extend(base_names)
    return np.concatenate(blocks, axis=1), names


def build_dataset(
    config: RegressionConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    """Build one base-system feature vector and one e^k median crossing label."""
    dataset_dir = Path(config.dataset_dir)
    paths = _discover_paths(dataset_dir, config.sampling_id)
    if config.max_shards is not None:
        paths = paths[: config.max_shards]
    completion = _load_completion(dataset_dir / config.completion_table_name)
    threshold = math.exp(config.threshold_power)
    feature_rows: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    time_grid: np.ndarray | None = None
    initial_outer_periods: list[float] = []

    for path in paths:
        batch = _batch_id(path)
        batch_completion = completion.loc[completion["batch"] == batch].sort_values("index")
        with h5py.File(path, "r") as handle:
            ratios = np.asarray(handle["aratios_complete"][:, :, 0])
            times = np.asarray(handle["times_complete"][0], dtype=np.float64)
            periods = np.asarray(handle["periods_complete"][:, 0, 1], dtype=np.float64)
            initial_features, current_names = _flatten_initial_features(handle)
        if ratios.dtype != np.float64:
            raise ValueError(f"Task D requires float64 aratios_complete: {path}")
        if ratios.shape[0] != len(batch_completion) or ratios.shape[0] % GROUP_SIZE:
            raise ValueError(f"Unexpected group/completion alignment in {path}")
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError(f"Initial feature layout differs in {path}")
        if time_grid is None:
            time_grid = times
        elif not np.allclose(times, time_grid, rtol=1e-12, atol=1e-9):
            raise ValueError(f"Time grid differs in {path}")

        groups = ratios.shape[0] // GROUP_SIZE
        indices = batch_completion["index"].to_numpy(dtype=np.int64).reshape(groups, GROUP_SIZE)
        system_ids = indices[:, 0] // GROUP_SIZE
        expected = system_ids[:, None] * GROUP_SIZE + np.arange(GROUP_SIZE)
        if not np.array_equal(indices, expected):
            raise ValueError(f"System rows are not contiguous groups of 32 in {path}")
        valid_rows = (
            (batch_completion["time_unbound"].to_numpy(dtype=np.float64) >= 0.0)
            & (batch_completion["diagnosis"].to_numpy(dtype=np.int64) == 0)
        ).reshape(groups, GROUP_SIZE)
        complete = valid_rows.all(axis=1)

        ratios = ratios.reshape(groups, GROUP_SIZE, ratios.shape[1])
        periods = periods.reshape(groups, GROUP_SIZE)
        initial_features = initial_features.reshape(groups, GROUP_SIZE, -1)
        separation = np.abs(ratios[:, :PAIR_COUNT] - ratios[:, PAIR_COUNT:])
        initial_separation = separation[:, :, 0]
        growth = np.full_like(separation, np.nan)
        denominator_ok = np.isfinite(initial_separation) & (initial_separation > 0.0)
        np.divide(
            separation,
            initial_separation[:, :, None],
            out=growth,
            where=denominator_ok[:, :, None],
        )
        crossing, observed, crossing_index = interpolate_first_crossing(
            growth.reshape(-1, growth.shape[-1]),
            times,
            threshold,
            interpolate=config.interpolate_crossings,
        )
        # Normalize each pair by the nominal member's initial outer period.
        nominal_period = periods[:, :PAIR_COUNT].reshape(-1)
        observed &= (
            np.isfinite(nominal_period)
            & (nominal_period > 0.0)
            & denominator_ok.reshape(-1)
        )
        crossing_periods = np.full_like(crossing, np.nan)
        crossing_periods[observed] = crossing[observed] / nominal_period[observed]
        crossing_periods = crossing_periods.reshape(groups, PAIR_COUNT)
        crossing_index = crossing_index.reshape(groups, PAIR_COUNT)

        for local_group in np.flatnonzero(complete):
            aggregate = aggregate_pair_times(
                crossing_periods[local_group],
                min_valid_pairs=config.min_valid_pairs,
            )
            system_time = float(aggregate["median_decorrelation_time_periods"])
            if not np.isfinite(system_time):
                continue
            row = {
                "system_id": int(system_ids[local_group]),
                "batch_id": int(batch),
                "local_group_id": int(local_group),
                "threshold_power": float(config.threshold_power),
                "threshold": float(threshold),
                "log_median_decorrelation_time": float(np.log(system_time)),
                **aggregate,
                "pair_crossing_index_min": int(np.min(crossing_index[local_group])),
                "pair_crossing_index_median": float(np.median(crossing_index[local_group])),
            }
            label_rows.append(row)
            feature_rows.append(initial_features[local_group, 0].copy())
            initial_outer_periods.append(float(periods[local_group, 0]))
            if config.max_systems is not None and len(label_rows) >= config.max_systems:
                break
        if config.max_systems is not None and len(label_rows) >= config.max_systems:
            break

    if not label_rows or feature_names is None or time_grid is None:
        raise ValueError("No valid Task D systems were built")
    order = np.argsort(np.asarray([row["system_id"] for row in label_rows], dtype=np.int64))
    labels = pd.DataFrame(label_rows).iloc[order].reset_index(drop=True)
    X = np.stack(feature_rows, axis=0)[order]
    y = labels["log_median_decorrelation_time"].to_numpy(dtype=np.float64)
    metadata = {
        "num_systems": int(len(labels)),
        "num_valid_pairs_required": int(config.min_valid_pairs),
        "threshold_power": float(config.threshold_power),
        "threshold": float(threshold),
        "label": "log(median_r(T_cross,r(e^k)/P_out,0,r))",
        "name": "median orbital-decorrelation time",
        "divided_by_threshold_power": False,
        "interpolate_crossings": bool(config.interpolate_crossings),
        "time_steps": int(len(time_grid)),
        "sample_spacing_periods": float(
            np.median(np.diff(time_grid))
            / np.nanmedian(np.asarray(initial_outer_periods, dtype=np.float64))
        ),
    }
    return X, y, labels, feature_names, metadata


def summarize_set_features(
    X_set: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Legacy helper retained for old analysis imports; not used by Task D."""
    summaries = (
        np.nanmean(X_set, axis=1),
        np.nanstd(X_set, axis=1),
        np.nanmin(X_set, axis=1),
        np.nanmax(X_set, axis=1),
    )
    names = [
        f"{stat}__{name}"
        for stat in ("mean", "std", "min", "max")
        for name in feature_names
    ]
    return np.concatenate(summaries, axis=1), names


def select_tabular_features(
    X_set: np.ndarray, feature_names: Sequence[str], mode: str
) -> tuple[np.ndarray, list[str]]:
    """Legacy feature selector; the production pipeline always uses row zero."""
    if mode == "summary":
        return summarize_set_features(X_set, feature_names)
    if mode == "first":
        return X_set[:, 0, :].copy(), list(feature_names)
    raise ValueError(f"Unknown Task D tabular feature mode: {mode}")


def make_split(y: np.ndarray, config: RegressionConfig) -> dict[str, np.ndarray]:
    ratios = np.asarray([config.train_ratio, config.val_ratio, config.test_ratio])
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to one")
    indices = np.arange(len(y), dtype=np.int64)
    strata = pd.qcut(y, q=min(10, len(y)), labels=False, duplicates="drop")
    try:
        train, remainder = train_test_split(
            indices,
            test_size=float(config.val_ratio + config.test_ratio),
            random_state=config.seed,
            stratify=strata,
        )
        remainder_strata = np.asarray(strata)[remainder]
        val_fraction = config.val_ratio / (config.val_ratio + config.test_ratio)
        test, val = train_test_split(
            remainder,
            test_size=val_fraction,
            random_state=config.seed,
            stratify=remainder_strata,
        )
    except ValueError:
        train, remainder = train_test_split(
            indices,
            test_size=float(config.val_ratio + config.test_ratio),
            random_state=config.seed,
        )
        val_fraction = config.val_ratio / (config.val_ratio + config.test_ratio)
        test, val = train_test_split(
            remainder, test_size=val_fraction, random_state=config.seed
        )
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def _metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    statistic = (
        spearmanr(y, prediction).statistic
        if np.unique(y).size > 1 and np.unique(prediction).size > 1
        else math.nan
    )
    return {
        "mae_log": float(mean_absolute_error(y, prediction)),
        "rmse_log": float(mean_squared_error(y, prediction) ** 0.5),
        "spearman": float(statistic) if np.isfinite(statistic) else math.nan,
        "factor_2_accuracy": float(np.mean(np.abs(y - prediction) <= math.log(2.0))),
    }


def evaluate_regression(
    y: np.ndarray,
    prediction: np.ndarray,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return point metrics and paired bootstrap 95% confidence intervals."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    point = _metric_values(y, prediction)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(bootstrap_samples):
        sample = rng.integers(0, len(y), size=len(y))
        values = _metric_values(y[sample], prediction[sample])
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(value)
    confidence_intervals: dict[str, dict[str, float | None]] = {}
    for name, values in draws.items():
        if values:
            low, high = np.quantile(values, (0.025, 0.975))
            confidence_intervals[name] = {
                "low": float(low),
                "high": float(high),
            }
        else:
            confidence_intervals[name] = {"low": None, "high": None}
    return {
        "count": int(len(y)),
        **{name: (value if np.isfinite(value) else None) for name, value in point.items()},
        "bootstrap_95_ci": confidence_intervals,
        "bootstrap_samples": int(bootstrap_samples),
    }


if nn is not None:

    class MLPRegressor(nn.Module):
        def __init__(self, input_dim: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)


def _fit_torch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: RegressionConfig,
) -> nn.Module:
    if torch is None:
        raise ImportError("PyTorch is required for the mlp model")
    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    wait = 0
    for _ in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb.to(device)) - yb.to(device)) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(
                torch.mean((model(X_val_tensor) - y_val_tensor) ** 2).item()
            )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def _predict_torch(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb).numpy())
    return np.concatenate(outputs).astype(np.float64)


def _fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("XGBoost is required for --models xgboost.") from exc
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return np.asarray(model.predict(X_test), dtype=np.float64), model


def _plot_predictions(
    path: Path, y: np.ndarray, predictions: dict[str, np.ndarray]
) -> None:
    columns = len(predictions)
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4.5), squeeze=False)
    for axis, (name, pred) in zip(axes[0], predictions.items(), strict=True):
        axis.scatter(np.exp(y), np.exp(pred), s=8, alpha=0.35)
        low = min(float(np.exp(y).min()), float(np.exp(pred).min()))
        high = max(float(np.exp(y).max()), float(np.exp(pred).max()))
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(name)
        axis.set_xlabel("true median decorrelation time / P_out,0")
        axis.set_ylabel("predicted median decorrelation time / P_out,0")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_regression_pipeline(config: RegressionConfig) -> dict[str, Any]:
    unknown = sorted(set(config.models) - {"constant", "xgboost", "mlp"})
    if unknown:
        raise ValueError(f"Unknown Task D models: {unknown}")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, labels, feature_names, label_metadata = build_dataset(config)
    split = make_split(y, config)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X[train_idx])
    X_val = imputer.transform(X[val_idx])
    X_test = imputer.transform(X[test_idx])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}
    if "constant" in config.models:
        predictions["constant"] = np.full(
            len(test_idx), np.median(y[train_idx]), dtype=np.float64
        )
    if "xgboost" in config.models:
        predictions["xgboost"], fitted_models["xgboost"] = _fit_xgboost(
            X_train, y[train_idx], X_test, config.seed
        )
    if "mlp" in config.models:
        model = MLPRegressor(X_train_scaled.shape[1], config.dropout)
        model = _fit_torch(
            model,
            X_train_scaled,
            y[train_idx],
            X_val_scaled,
            y[val_idx],
            config,
        )
        predictions["mlp"] = _predict_torch(model, X_test_scaled, config.batch_size)
        fitted_models["mlp"] = model

    metrics = {
        name: evaluate_regression(
            y[test_idx],
            pred,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        )
        for name, pred in predictions.items()
    }
    prediction_frame = labels.iloc[test_idx][
        [
            "system_id",
            "median_decorrelation_time_periods",
            "log_median_decorrelation_time",
        ]
    ].reset_index(drop=True)
    for name, pred in predictions.items():
        prediction_frame[f"{name}_log"] = pred
        prediction_frame[f"{name}_periods"] = np.exp(pred)

    labels.to_csv(output_dir / "task_d_labels.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    split_names = np.full(len(labels), "test", dtype=object)
    split_names[train_idx] = "train"
    split_names[val_idx] = "val"
    pd.DataFrame({"system_id": labels["system_id"], "split": split_names}).to_csv(
        output_dir / "split.csv", index=False
    )
    if "xgboost" in fitted_models:
        model = fitted_models["xgboost"]
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            output_dir / "xgboost_feature_importance.csv", index=False
        )
        model.get_booster().save_model(output_dir / "xgboost.json")
    if "mlp" in fitted_models:
        torch.save(fitted_models["mlp"].state_dict(), output_dir / "mlp.pt")
    _plot_predictions(output_dir / "predictions.png", y[test_idx], predictions)

    result = {
        "task": "initial conditions to median orbital-decorrelation time",
        "config": asdict(config),
        "label_metadata": label_metadata,
        "feature_metadata": {
            "representation": "first nominal variation only",
            "features": feature_names,
            "angles": "sin/cos encoded",
            "feature_count": len(feature_names),
        },
        "split": {name: int(len(indices)) for name, indices in split.items()},
        "target_summary": {
            "count": int(len(y)),
            "p25_periods": float(np.exp(np.quantile(y, 0.25))),
            "median_periods": float(np.exp(np.median(y))),
            "p75_periods": float(np.exp(np.quantile(y, 0.75))),
        },
        "model_metrics": metrics,
    }
    _write_json(output_dir / "metrics.json", result)
    return result


# Compatibility for existing callers; behavior is the single-output pipeline.
run_quantile_pipeline = run_regression_pipeline
