"""Task D: initial conditions to system Lyapunov-time conditional quantiles."""

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
PAIR_COUNT = 16
GROUP_SIZE = 32
QUANTILES = (0.25, 0.50, 0.75)
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
class QuantileConfig:
    dataset_dir: str | Path
    output_dir: str | Path
    sampling_id: int = 3
    completion_table_name: str = "completed_systems_sampling_3.txt"
    threshold_power: float = 5.0
    min_valid_pairs: int = 8
    interpolate_crossings: bool = True
    models: tuple[str, ...] = ("constant", "gbr", "mlp", "deepsets")
    tabular_features: str = "summary"
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
    table = table.astype(
        {
            "index": "int64",
            "batch": "int64",
            "time_unbound": "float64",
            "diagnosis": "int64",
        }
    )
    return table


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
    threshold_power: float,
    min_valid_pairs: int,
) -> dict[str, Any]:
    """Aggregate pair proxy times through the median observable growth rate."""
    crossing = np.asarray(crossing_periods, dtype=np.float64)
    valid = np.isfinite(crossing) & (crossing > 0.0)
    pair_tl = crossing[valid] / float(threshold_power)
    pair_lambda = 1.0 / pair_tl
    count = int(valid.sum())
    if count < min_valid_pairs:
        system_lambda = math.nan
        system_tl = math.nan
    else:
        system_lambda = float(np.median(pair_lambda))
        system_tl = float(1.0 / system_lambda)
    return {
        "num_valid_pairs": count,
        "lambda_system_per_period": system_lambda,
        "lyapunov_time_system_periods": system_tl,
        "pair_tl_p25_periods": float(np.quantile(pair_tl, 0.25)) if count else math.nan,
        "pair_tl_p50_periods": float(np.quantile(pair_tl, 0.50)) if count else math.nan,
        "pair_tl_p75_periods": float(np.quantile(pair_tl, 0.75)) if count else math.nan,
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


def build_dataset(config: QuantileConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    """Build [system, nominal variation, feature] inputs and e^k system labels."""
    dataset_dir = Path(config.dataset_dir)
    paths = _discover_paths(dataset_dir, config.sampling_id)
    if config.max_shards is not None:
        paths = paths[: config.max_shards]
    completion = _load_completion(dataset_dir / config.completion_table_name)
    threshold = math.exp(config.threshold_power)
    set_blocks: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    time_grid: np.ndarray | None = None
    initial_outer_periods: list[float] = []

    for path in paths:
        batch = _batch_id(path)
        batch_completion = completion.loc[completion["batch"] == batch].sort_values("index")
        with h5py.File(path, "r") as handle:
            ratios = np.asarray(handle["aratios_complete"][:, :, 0], dtype=np.float64)
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
        pair_period = np.nanmedian(
            np.stack((periods[:, :PAIR_COUNT], periods[:, PAIR_COUNT:]), axis=-1),
            axis=-1,
        ).reshape(-1)
        observed &= np.isfinite(pair_period) & (pair_period > 0.0) & denominator_ok.reshape(-1)
        crossing_periods = np.full_like(crossing, np.nan)
        crossing_periods[observed] = crossing[observed] / pair_period[observed]
        crossing_periods = crossing_periods.reshape(groups, PAIR_COUNT)
        crossing_index = crossing_index.reshape(groups, PAIR_COUNT)

        for local_group in np.flatnonzero(complete):
            aggregate = aggregate_pair_times(
                crossing_periods[local_group],
                config.threshold_power,
                config.min_valid_pairs,
            )
            if not np.isfinite(aggregate["lyapunov_time_system_periods"]):
                continue
            system_tl = float(aggregate["lyapunov_time_system_periods"])
            row = {
                "system_id": int(system_ids[local_group]),
                "batch_id": int(batch),
                "local_group_id": int(local_group),
                "threshold_power": float(config.threshold_power),
                "threshold": float(threshold),
                "log_lyapunov_time_system": float(np.log(system_tl)),
                **aggregate,
                "pair_crossing_index_min": int(np.min(crossing_index[local_group])),
                "pair_crossing_index_median": float(np.median(crossing_index[local_group])),
            }
            label_rows.append(row)
            set_blocks.append(initial_features[local_group, :PAIR_COUNT])
            initial_outer_periods.append(
                float(np.nanmedian(pair_period.reshape(groups, PAIR_COUNT)[local_group]))
            )
            if config.max_systems is not None and len(label_rows) >= config.max_systems:
                break
        if config.max_systems is not None and len(label_rows) >= config.max_systems:
            break

    if not label_rows or feature_names is None or time_grid is None:
        raise ValueError("No valid Task D systems were built")
    labels = pd.DataFrame(label_rows).sort_values("system_id").reset_index(drop=True)
    order = np.argsort(np.asarray([row["system_id"] for row in label_rows], dtype=np.int64))
    X_set = np.stack(set_blocks, axis=0)[order]
    y = labels["log_lyapunov_time_system"].to_numpy(dtype=np.float64)
    metadata = {
        "num_systems": int(len(labels)),
        "num_nominal_variations": PAIR_COUNT,
        "num_features_per_variation": int(X_set.shape[-1]),
        "threshold_power": float(config.threshold_power),
        "threshold": float(threshold),
        "label": "log(T_L,Q/P_out,0), T_L,Q=1/median_pair(k/t_cross,e^k)",
        "interpolate_crossings": bool(config.interpolate_crossings),
        "time_steps": int(len(time_grid)),
        "sample_spacing_periods": float(
            np.median(np.diff(time_grid))
            / np.nanmedian(np.asarray(initial_outer_periods, dtype=np.float64))
        ),
    }
    return X_set, y, labels, feature_names, metadata


def summarize_set_features(X_set: np.ndarray, feature_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    summaries = (
        np.nanmean(X_set, axis=1),
        np.nanstd(X_set, axis=1),
        np.nanmin(X_set, axis=1),
        np.nanmax(X_set, axis=1),
    )
    names = [f"{stat}__{name}" for stat in ("mean", "std", "min", "max") for name in feature_names]
    return np.concatenate(summaries, axis=1), names


def select_tabular_features(
    X_set: np.ndarray,
    feature_names: Sequence[str],
    mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Select the system-level input used by tabular models and the MLP."""
    if mode == "summary":
        return summarize_set_features(X_set, feature_names)
    if mode == "first":
        return X_set[:, 0, :].copy(), list(feature_names)
    raise ValueError(f"Unknown Task D tabular feature mode: {mode}")


def make_split(y: np.ndarray, config: QuantileConfig) -> dict[str, np.ndarray]:
    ratios = np.asarray([config.train_ratio, config.val_ratio, config.test_ratio])
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to one")
    indices = np.arange(len(y), dtype=np.int64)
    strata = pd.qcut(y, q=min(10, len(y)), labels=False, duplicates="drop")
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
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def pinball_numpy(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    residual = y - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def evaluate_quantiles(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"count": int(len(y)), "quantiles": {}}
    spearman = (
        float(spearmanr(y, prediction[:, 1]).statistic)
        if np.unique(prediction[:, 1]).size > 1
        else math.nan
    )
    for column, quantile in enumerate(QUANTILES):
        coverage = float(np.mean(y <= prediction[:, column]))
        result["quantiles"][f"q{int(100 * quantile)}"] = {
            "pinball_loss": pinball_numpy(y, prediction[:, column], quantile),
            "coverage": coverage,
            "coverage_error": coverage - quantile,
        }
    result.update(
        {
            "mean_pinball_loss": float(
                np.mean(
                    [pinball_numpy(y, prediction[:, i], q) for i, q in enumerate(QUANTILES)]
                )
            ),
            "central_50_coverage": float(
                np.mean((y >= prediction[:, 0]) & (y <= prediction[:, 2]))
            ),
            "mean_interval_width_log": float(np.mean(prediction[:, 2] - prediction[:, 0])),
            "quantile_crossing_rate": float(
                np.mean((prediction[:, 0] > prediction[:, 1]) | (prediction[:, 1] > prediction[:, 2]))
            ),
            "median_mae_log": float(mean_absolute_error(y, prediction[:, 1])),
            "median_rmse_log": float(mean_squared_error(y, prediction[:, 1]) ** 0.5),
            "median_spearman": spearman if np.isfinite(spearman) else None,
            "median_factor_2_accuracy": float(np.mean(np.abs(y - prediction[:, 1]) <= math.log(2.0))),
        }
    )
    return result


if nn is not None:

    class OrderedQuantileHead(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.output = nn.Linear(input_dim, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raw = self.output(x)
            median = raw[:, 0]
            lower = median - F.softplus(raw[:, 1])
            upper = median + F.softplus(raw[:, 2])
            return torch.stack((lower, median, upper), dim=1)


    class QuantileMLP(nn.Module):
        def __init__(self, input_dim: int, dropout: float) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = OrderedQuantileHead(128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(x))


    class DeepSetsQuantileRegressor(nn.Module):
        def __init__(self, input_dim: int, dropout: float) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.rho = nn.Sequential(
                nn.Linear(64 * 3, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.head = OrderedQuantileHead(64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.phi(x)
            pooled = torch.cat(
                (
                    encoded.mean(dim=1),
                    encoded.std(dim=1, unbiased=False),
                    encoded.max(dim=1).values,
                ),
                dim=1,
            )
            return self.head(self.rho(pooled))


def _pinball_torch(y: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    quantiles = prediction.new_tensor(QUANTILES).view(1, -1)
    residual = y[:, None] - prediction
    return torch.maximum(quantiles * residual, (quantiles - 1.0) * residual).mean()


def _fit_torch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: QuantileConfig,
) -> nn.Module:
    if torch is None:
        raise ImportError("PyTorch is required for mlp/deepsets models")
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
            loss = _pinball_torch(yb.to(device), model(xb.to(device)))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(_pinball_torch(y_val_tensor, model(X_val_tensor)).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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


def _fit_gbr(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int
) -> tuple[np.ndarray, list[GradientBoostingRegressor]]:
    predictions: list[np.ndarray] = []
    models: list[GradientBoostingRegressor] = []
    for quantile in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=10,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
        models.append(model)
    prediction = np.column_stack(predictions)
    prediction.sort(axis=1)
    return prediction, models


def _fit_xgboost_quantiles(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int
) -> tuple[np.ndarray, list[Any]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost >= 2.0 is required for --models xgboost."
        ) from exc

    major_version = int(xgb.__version__.split(".", maxsplit=1)[0])
    if major_version < 2:
        raise ImportError(
            f"XGBoost >= 2.0 is required for native quantile regression; found {xgb.__version__}."
        )

    predictions: list[np.ndarray] = []
    models: list[Any] = []
    for quantile in QUANTILES:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
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
        predictions.append(model.predict(X_test))
        models.append(model)
    prediction = np.column_stack(predictions)
    prediction.sort(axis=1)
    return prediction, models


def _plot_predictions(path: Path, y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    columns = len(predictions)
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4.5), squeeze=False)
    for axis, (name, pred) in zip(axes[0], predictions.items(), strict=True):
        axis.scatter(np.exp(y), np.exp(pred[:, 1]), s=8, alpha=0.35)
        low = min(float(np.exp(y).min()), float(np.exp(pred[:, 1]).min()))
        high = max(float(np.exp(y).max()), float(np.exp(pred[:, 1]).max()))
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(name)
        axis.set_xlabel("true system $T_{L,Q}/P_{out,0}$")
        axis.set_ylabel("predicted median")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_quantile_pipeline(config: QuantileConfig) -> dict[str, Any]:
    unknown = sorted(
        set(config.models) - {"constant", "gbr", "xgboost", "mlp", "deepsets"}
    )
    if unknown:
        raise ValueError(f"Unknown Task D models: {unknown}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_set, y, labels, feature_names, label_metadata = build_dataset(config)
    X_tabular, tabular_names = select_tabular_features(
        X_set, feature_names, config.tabular_features
    )
    split = make_split(y, config)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

    summary_imputer = SimpleImputer(strategy="median")
    X_summary_train = summary_imputer.fit_transform(X_tabular[train_idx])
    X_summary_val = summary_imputer.transform(X_tabular[val_idx])
    X_summary_test = summary_imputer.transform(X_tabular[test_idx])
    summary_scaler = StandardScaler()
    X_summary_train_scaled = summary_scaler.fit_transform(X_summary_train)
    X_summary_val_scaled = summary_scaler.transform(X_summary_val)
    X_summary_test_scaled = summary_scaler.transform(X_summary_test)

    channel_median = np.nanmedian(X_set[train_idx], axis=(0, 1))
    channel_median = np.where(np.isfinite(channel_median), channel_median, 0.0)
    X_set_clean = np.where(np.isfinite(X_set), X_set, channel_median[None, None, :])
    channel_mean = X_set_clean[train_idx].mean(axis=(0, 1))
    channel_std = X_set_clean[train_idx].std(axis=(0, 1))
    channel_std = np.where(channel_std < 1e-8, 1.0, channel_std)
    X_set_scaled = (X_set_clean - channel_mean[None, None, :]) / channel_std[None, None, :]

    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}
    if "constant" in config.models:
        values = np.quantile(y[train_idx], QUANTILES)
        predictions["constant"] = np.broadcast_to(values, (len(test_idx), 3)).copy()
    if "gbr" in config.models:
        predictions["gbr"], gbr_models = _fit_gbr(
            X_summary_train, y[train_idx], X_summary_test, config.seed
        )
        fitted_models["gbr"] = gbr_models
    if "xgboost" in config.models:
        predictions["xgboost"], xgboost_models = _fit_xgboost_quantiles(
            X_summary_train, y[train_idx], X_summary_test, config.seed
        )
        fitted_models["xgboost"] = xgboost_models
    if "mlp" in config.models:
        model = QuantileMLP(X_summary_train_scaled.shape[1], config.dropout)
        model = _fit_torch(
            model,
            X_summary_train_scaled,
            y[train_idx],
            X_summary_val_scaled,
            y[val_idx],
            config,
        )
        predictions["mlp"] = _predict_torch(model, X_summary_test_scaled, config.batch_size)
        fitted_models["mlp"] = model
    if "deepsets" in config.models:
        model = DeepSetsQuantileRegressor(X_set_scaled.shape[-1], config.dropout)
        model = _fit_torch(
            model,
            X_set_scaled[train_idx],
            y[train_idx],
            X_set_scaled[val_idx],
            y[val_idx],
            config,
        )
        predictions["deepsets"] = _predict_torch(
            model, X_set_scaled[test_idx], config.batch_size
        )
        fitted_models["deepsets"] = model

    metrics = {name: evaluate_quantiles(y[test_idx], pred) for name, pred in predictions.items()}
    prediction_frame = labels.iloc[test_idx][
        ["system_id", "lyapunov_time_system_periods", "log_lyapunov_time_system"]
    ].reset_index(drop=True)
    for name, pred in predictions.items():
        for column, quantile in enumerate((25, 50, 75)):
            prediction_frame[f"{name}_q{quantile}_log"] = pred[:, column]
            prediction_frame[f"{name}_q{quantile}_periods"] = np.exp(pred[:, column])

    labels.to_csv(output_dir / "task_d_labels.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    split_names = np.full(len(labels), "test", dtype=object)
    split_names[train_idx] = "train"
    split_names[val_idx] = "val"
    pd.DataFrame({"system_id": labels["system_id"], "split": split_names}).to_csv(
        output_dir / "split.csv", index=False
    )
    if "gbr" in fitted_models:
        importance = fitted_models["gbr"][1].feature_importances_
        pd.DataFrame({"feature": tabular_names, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(output_dir / "gbr_median_feature_importance.csv", index=False)
    if "xgboost" in fitted_models:
        importance = fitted_models["xgboost"][1].feature_importances_
        pd.DataFrame({"feature": tabular_names, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(output_dir / "xgboost_median_feature_importance.csv", index=False)
        for quantile, model in zip((25, 50, 75), fitted_models["xgboost"], strict=True):
            model.get_booster().save_model(output_dir / f"xgboost_q{quantile}.json")
    for name in ("mlp", "deepsets"):
        if name in fitted_models:
            torch.save(fitted_models[name].state_dict(), output_dir / f"{name}.pt")
    _plot_predictions(output_dir / "median_predictions.png", y[test_idx], predictions)

    result = {
        "task": "initial conditions to system Lyapunov-time conditional quantiles",
        "config": asdict(config),
        "label_metadata": label_metadata,
        "feature_metadata": {
            "representation": "16 nominal variations as an unordered set",
            "per_variation_features": feature_names,
            "tabular_features": config.tabular_features,
            "tabular_summary": (
                "mean/std/min/max" if config.tabular_features == "summary" else None
            ),
            "tabular_feature_count": len(tabular_names),
        },
        "split": {name: int(len(indices)) for name, indices in split.items()},
        "target_summary": {
            "count": int(len(y)),
            "p25_periods": float(np.exp(np.quantile(y, 0.25))),
            "p50_periods": float(np.exp(np.quantile(y, 0.50))),
            "p75_periods": float(np.exp(np.quantile(y, 0.75))),
        },
        "model_metrics": metrics,
    }
    _write_json(output_dir / "metrics.json", result)
    return result
