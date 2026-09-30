"""
Task G: Stability Boundary Sensitivity Prediction.

Task G predicts boundary sensitivity: for a configured perturbation pair
``(i, i + 16)`` from the same 32-variation system, the target is whether the
two trajectories have different stable/unstable outcomes at each horizon.
Unlike Task C, which predicts whether a trajectory or system is stable until a
horizon, Task G predicts whether a tiny configured perturbation changes the
stable/unstable outcome. This probes systems near the stability boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.orbital_schema import DEFAULT_VALID_PAIRS, INITIAL_FEATURE_KEYS  # noqa: E402

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover
    pearsonr = None
    spearmanr = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DEFAULT_HORIZONS = (500.0, 800.0, 1500.0, 2500.0, 4000.0, 6500.0, 10000.0)
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")


@dataclass
class TaskGConfig:
    data_glob: str = "dataset_v3/results_batch_*_sampling_2.hdf5"
    completed_systems: Path = ROOT / "dataset_v3" / "completed_systems_sampling_2.txt"
    output_dir: Path = ROOT / "results" / "task_g_boundary_sensitivity"
    representation: str = "pair_ic54"
    models: tuple[str, ...] = ("constant", "logistic", "random_forest", "xgboost", "mlp")
    horizons: tuple[float, ...] = DEFAULT_HORIZONS
    group_size: int = 32
    seed: int = 42
    max_systems: int | None = None
    rf_estimators: int = 500
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 2
    xgb_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 4
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    patience: int = 8
    device: str | None = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def parse_batch_id(path: str | Path) -> int:
    match = RESULT_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def discover_result_paths(data_glob: str) -> list[Path]:
    pattern = str((ROOT / data_glob) if not Path(data_glob).is_absolute() else data_glob)
    paths = sorted(Path().glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(pattern[1:]))
    if not paths:
        raise FileNotFoundError(f"No HDF5 result shards matched {data_glob!r}")
    return paths


def load_complete_systems(path: Path, group_size: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    raw = raw.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float64",
            "wall_time": "float64",
            "diagnosis": "int64",
        }
    )
    raw["system_id"] = raw["index"] // group_size
    raw["variation_id"] = raw["index"] % group_size
    raw["valid"] = (raw["time_unbound"] >= 0.0) & (raw["diagnosis"] == 0)

    dropped = {"invalid_rows": int((~raw["valid"]).sum()), "incomplete_or_duplicate_systems": 0}
    complete_frames: list[pd.DataFrame] = []
    for _, group in raw[raw["valid"]].groupby("system_id", sort=True):
        vids = np.sort(group["variation_id"].to_numpy(dtype=np.int64))
        if len(group) == group_size and np.array_equal(vids, np.arange(group_size)):
            complete_frames.append(group.sort_values("variation_id"))
        else:
            dropped["incomplete_or_duplicate_systems"] += 1
    if not complete_frames:
        raise ValueError("No complete valid systems found.")
    complete = pd.concat(complete_frames, ignore_index=True)
    return complete.sort_values(["system_id", "variation_id"]).reset_index(drop=True), dropped


def build_system_arrays(
    complete: pd.DataFrame,
    horizons: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    system_ids = np.asarray(sorted(complete["system_id"].unique()), dtype=np.int64)
    times = np.empty((len(system_ids), group_size), dtype=np.float64)
    for row_idx, system_id in enumerate(system_ids):
        group = complete[complete["system_id"] == system_id].sort_values("variation_id")
        times[row_idx] = group["time_unbound"].to_numpy(dtype=np.float64)
    stable = times[:, :, None] >= horizons[None, None, :]
    return system_ids, times, stable


def make_feature_names(base_names: Sequence[str], representation: str) -> list[str]:
    left = [f"left__{name}" for name in base_names]
    right = [f"right__{name}" for name in base_names]
    absdiff = [f"absdiff__{name}" for name in base_names]
    if representation == "pair_ic54":
        return left + right + absdiff
    signed = [f"signed_diff__{name}" for name in base_names]
    return left + right + signed + absdiff


def extract_ic18_by_system(
    result_paths: Sequence[Path],
    complete: pd.DataFrame,
    system_ids: np.ndarray,
    group_size: int,
    feature_keys: Sequence[str] = INITIAL_FEATURE_KEYS,
) -> tuple[np.ndarray, list[str]]:
    rows_per_batch: dict[int, int] = {}
    path_by_batch: dict[int, Path] = {}
    for path in result_paths:
        batch_id = parse_batch_id(path)
        if batch_id in path_by_batch:
            raise ValueError(
                f"Multiple HDF5 shards resolve to batch {batch_id}: "
                f"{path_by_batch[batch_id]} and {path}."
            )
        path_by_batch[batch_id] = path

    expected_batches = set(complete["num_batch"].unique().astype(int).tolist())
    missing_batches = sorted(expected_batches - set(path_by_batch))
    if missing_batches:
        raise ValueError(
            "Completion table references batches with no matching HDF5 shard: "
            f"{missing_batches}."
        )
    for batch_id, path in path_by_batch.items():
        with h5py.File(path, "r") as handle:
            rows_per_batch[batch_id] = int(handle[feature_keys[0]].shape[0])

    for system_id in system_ids:
        group = complete[complete["system_id"] == system_id].sort_values("variation_id")
        if group["num_batch"].nunique() != 1:
            raise ValueError(f"System {system_id} spans multiple batches.")

    base_names: list[str] | None = None
    system_pos = {int(system_id): pos for pos, system_id in enumerate(system_ids)}
    by_batch = {int(batch_id): batch_df for batch_id, batch_df in complete.groupby("num_batch", sort=True)}
    out: np.ndarray | None = None
    written = np.zeros(len(system_ids), dtype=bool)
    for batch_id, batch_df in by_batch.items():
        path = path_by_batch[batch_id]
        rows_in_batch = rows_per_batch[batch_id]
        with h5py.File(path, "r") as handle:
            shard = {key: np.asarray(handle[key][:, 0, ...], dtype=np.float64) for key in feature_keys}
        for system_id, group in batch_df.groupby("system_id", sort=True):
            group = group.sort_values("variation_id")
            local_rows = group["index"].to_numpy(dtype=np.int64) - batch_id * rows_in_batch
            if np.any(local_rows < 0) or np.any(local_rows >= rows_in_batch):
                raise ValueError(f"System {system_id} has local rows outside batch {batch_id}.")
            blocks: list[np.ndarray] = []
            names: list[str] = []
            for key in feature_keys:
                data = shard[key][local_rows].reshape(group_size, -1)
                blocks.append(data)
                names.extend(f"{key}_{idx}" for idx in range(data.shape[1]))
            matrix = np.concatenate(blocks, axis=1)
            if out is None:
                out = np.empty((len(system_ids), group_size, matrix.shape[1]), dtype=np.float64)
                base_names = names
            system_idx = system_pos[int(system_id)]
            out[system_idx] = matrix
            written[system_idx] = True
    if out is None or base_names is None:
        raise ValueError("No IC features were extracted.")
    if not np.all(written):
        missing_systems = system_ids[~written].tolist()
        raise ValueError(f"IC features were not extracted for systems: {missing_systems}.")
    if out.shape[2] != 18:
        raise ValueError(f"Expected 18-dimensional IC features, got {out.shape[2]}.")
    return out, base_names


def expand_pair_dataset(
    system_ids: np.ndarray,
    ic18: np.ndarray,
    stable: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    representation: str,
    base_feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    sample_system_ids: list[int] = []
    pair_ids: list[int] = []
    for s_pos, system_id in enumerate(system_ids):
        for pair_id, (left, right) in enumerate(pairs):
            left_ic = ic18[s_pos, left]
            right_ic = ic18[s_pos, right]
            diff = right_ic - left_ic
            if representation == "pair_ic54":
                feat = np.concatenate([left_ic, right_ic, np.abs(diff)])
            elif representation == "pair_ic72":
                feat = np.concatenate([left_ic, right_ic, diff, np.abs(diff)])
            else:
                raise ValueError(f"Unsupported representation: {representation}")
            x_rows.append(feat)
            y_rows.append((stable[s_pos, left] != stable[s_pos, right]).astype(np.int64))
            sample_system_ids.append(int(system_id))
            pair_ids.append(pair_id)
    feature_names = make_feature_names(base_feature_names, representation)
    return (
        np.vstack(x_rows).astype(np.float64),
        np.vstack(y_rows).astype(np.int64),
        np.asarray(sample_system_ids, dtype=np.int64),
        np.asarray(pair_ids, dtype=np.int64),
        feature_names,
    )


def build_dataset_summary(
    times: np.ndarray,
    stable: np.ndarray,
    y: np.ndarray,
    sample_system_ids: np.ndarray,
    horizons: np.ndarray,
    group_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    num_systems = int(times.shape[0])
    for h_idx, horizon in enumerate(horizons):
        y_h = y[:, h_idx]
        unstable_count = (times < horizon).sum(axis=1)
        p_unstable = unstable_count / float(group_size)
        eps = 1e-12
        entropy = -(p_unstable * np.log(p_unstable + eps) + (1.0 - p_unstable) * np.log(1.0 - p_unstable + eps))
        per_system_dis = pd.Series(y_h).groupby(sample_system_ids).sum().to_numpy()
        rows.append(
            {
                "horizon": float(horizon),
                "pair_disagreement_rate": float(y_h.mean()),
                "disagree_pairs": int(y_h.sum()),
                "total_pairs": int(y_h.size),
                "fraction_systems_any_pair_disagreement": float(np.mean(per_system_dis > 0)),
                "mean_unstable_variations": float(unstable_count.mean()),
                "mean_sensitivity": float(np.mean(p_unstable * (1.0 - p_unstable))),
                "mean_entropy": float(np.mean(entropy)),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "num_systems": num_systems,
        "num_pairs": int(y.shape[0]),
        "num_features": None,
        "horizons": horizons.tolist(),
        "per_horizon": rows,
    }
    return summary, frame


def split_system_ids(system_ids: np.ndarray, seed: int, max_systems: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(system_ids, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    if max_systems is None and n >= 4998:
        n_train, n_val, n_test = 3598, 400, 1000
    else:
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.08)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Not enough systems to split: {n}")
    train = np.sort(shuffled[:n_train])
    val = np.sort(shuffled[n_train : n_train + n_val])
    test = np.sort(shuffled[n_train + n_val : n_train + n_val + n_test])
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    return train, val, test


def mask_for_systems(sample_system_ids: np.ndarray, split_ids: np.ndarray) -> np.ndarray:
    return np.isin(sample_system_ids, split_ids)


def safe_pair_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    pred = (prob >= threshold).astype(np.int64)
    has_two_classes = np.unique(y_true).size == 2
    return {
        "positive_rate": float(y_true.mean()),
        "AUROC": float(roc_auc_score(y_true, prob)) if has_two_classes else math.nan,
        "AUPRC": float(average_precision_score(y_true, prob)) if has_two_classes else math.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if has_two_classes else math.nan,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])) if has_two_classes else math.nan,
    }


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.asarray(prob, dtype=np.float64), np.asarray([0.5])]))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        score = f1_score(y_true, prob >= thr, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr


def pair_metric_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    horizons: np.ndarray,
    thresholds: Sequence[float],
    threshold_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = ["AUROC", "AUPRC", "balanced_accuracy", "F1", "Brier"]
    for h_idx, horizon in enumerate(horizons):
        metrics = safe_pair_metrics(y_true[:, h_idx], prob[:, h_idx], float(thresholds[h_idx]))
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "threshold_mode": threshold_mode,
                "horizon": float(horizon),
                "threshold": float(thresholds[h_idx]),
                **metrics,
            }
        )
    mean_row = {
        "model": model_name,
        "split": split_name,
        "threshold_mode": threshold_mode,
        "horizon": "mean",
        "threshold": math.nan,
    }
    for key in rows[0]:
        if key in {"model", "split", "threshold_mode", "horizon", "threshold"}:
            continue
        mean_row[key] = float(np.nanmean([row[key] for row in rows]))
    for metric in metric_names:
        mean_row[f"mean_{metric}"] = mean_row.get(metric, math.nan)
    rows.append(mean_row)
    return rows


def corr_safe(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    if method == "spearman":
        if spearmanr is not None:
            value = spearmanr(x, y).statistic
            return float(value) if value is not None else math.nan
        return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    if pearsonr is not None:
        return float(pearsonr(x, y).statistic)
    return float(np.corrcoef(x, y)[0, 1])


def system_metric_rows(
    model_name: str,
    split_name: str,
    sample_system_ids: np.ndarray,
    y_true: np.ndarray,
    prob: np.ndarray,
    horizons: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_systems = np.asarray(sorted(np.unique(sample_system_ids)), dtype=np.int64)
    for h_idx, horizon in enumerate(horizons):
        true_sens = []
        pred_sens = []
        true_any = []
        pred_any = []
        for system_id in unique_systems:
            mask = sample_system_ids == system_id
            y_group = y_true[mask, h_idx]
            p_group = np.clip(prob[mask, h_idx], 0.0, 1.0)
            true_sens.append(float(y_group.mean()))
            pred_sens.append(float(p_group.mean()))
            true_any.append(int(np.any(y_group == 1)))
            pred_any.append(float(1.0 - np.prod(1.0 - p_group)))
        true_sens_arr = np.asarray(true_sens, dtype=np.float64)
        pred_sens_arr = np.asarray(pred_sens, dtype=np.float64)
        true_any_arr = np.asarray(true_any, dtype=np.int64)
        pred_any_arr = np.asarray(pred_any, dtype=np.float64)
        k = max(1, int(math.ceil(0.2 * len(unique_systems))))
        true_top = set(np.argsort(-true_sens_arr)[:k].tolist())
        pred_top = set(np.argsort(-pred_sens_arr)[:k].tolist())
        any_two_classes = np.unique(true_any_arr).size == 2
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "horizon": float(horizon),
                "MAE": float(mean_absolute_error(true_sens_arr, pred_sens_arr)),
                "RMSE": float(np.sqrt(mean_squared_error(true_sens_arr, pred_sens_arr))),
                "Spearman": corr_safe(true_sens_arr, pred_sens_arr, "spearman"),
                "Pearson": corr_safe(true_sens_arr, pred_sens_arr, "pearson"),
                "top20_recall": float(len(true_top & pred_top) / len(true_top)),
                "any_AUROC": float(roc_auc_score(true_any_arr, pred_any_arr)) if any_two_classes else math.nan,
                "any_AUPRC": float(average_precision_score(true_any_arr, pred_any_arr)) if any_two_classes else math.nan,
                "true_any_rate": float(true_any_arr.mean()),
            }
        )
    return rows


def predict_constant(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    rates = np.clip(y_train.mean(axis=0), 1e-7, 1.0 - 1e-7)
    return np.tile(rates[None, :], (n_rows, 1))


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    x_train_z = scaler.transform(x_train)
    outputs = {name: np.zeros((x.shape[0], y_train.shape[1]), dtype=np.float64) for name, x in x_eval.items()}
    eval_z = {name: scaler.transform(x) for name, x in x_eval.items()}
    for h_idx in range(y_train.shape[1]):
        if np.unique(y_train[:, h_idx]).size < 2:
            rate = float(np.clip(y_train[:, h_idx].mean(), 1e-7, 1.0 - 1e-7))
            for name in outputs:
                outputs[name][:, h_idx] = rate
            continue
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        model.fit(x_train_z, y_train[:, h_idx])
        for name, x_z in eval_z.items():
            outputs[name][:, h_idx] = model.predict_proba(x_z)[:, 1]
    return outputs


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: dict[str, np.ndarray],
    config: TaskGConfig,
) -> dict[str, np.ndarray]:
    outputs = {name: np.zeros((x.shape[0], y_train.shape[1]), dtype=np.float64) for name, x in x_eval.items()}
    for h_idx in range(y_train.shape[1]):
        if np.unique(y_train[:, h_idx]).size < 2:
            rate = float(np.clip(y_train[:, h_idx].mean(), 1e-7, 1.0 - 1e-7))
            for name in outputs:
                outputs[name][:, h_idx] = rate
            continue
        model = RandomForestClassifier(
            n_estimators=config.rf_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.seed,
        )
        model.fit(x_train, y_train[:, h_idx])
        for name, x in x_eval.items():
            outputs[name][:, h_idx] = model.predict_proba(x)[:, 1]
    return outputs


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: dict[str, np.ndarray],
    config: TaskGConfig,
) -> dict[str, np.ndarray] | None:
    if XGBClassifier is None:
        warnings.warn("xgboost is not installed; skipping xgboost.", RuntimeWarning)
        return None
    outputs = {name: np.zeros((x.shape[0], y_train.shape[1]), dtype=np.float64) for name, x in x_eval.items()}
    for h_idx in range(y_train.shape[1]):
        y_h = y_train[:, h_idx]
        if np.unique(y_h).size < 2:
            rate = float(np.clip(y_h.mean(), 1e-7, 1.0 - 1e-7))
            for name in outputs:
                outputs[name][:, h_idx] = rate
            continue
        n_pos = max(1, int(y_h.sum()))
        n_neg = max(1, int((1 - y_h).sum()))
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=config.xgb_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos,
            random_state=config.seed,
            n_jobs=-1,
        )
        try:
            model.fit(x_train, y_h, eval_set=[(x_val, y_val[:, h_idx])], verbose=False, early_stopping_rounds=30)
        except TypeError:
            model.fit(x_train, y_h, eval_set=[(x_val, y_val[:, h_idx])], verbose=False)
        for name, x in x_eval.items():
            outputs[name][:, h_idx] = model.predict_proba(x)[:, 1]
    return outputs


class BoundaryMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mean_average_precision(y_true: np.ndarray, prob: np.ndarray) -> float:
    scores = []
    for h_idx in range(y_true.shape[1]):
        if np.unique(y_true[:, h_idx]).size == 2:
            scores.append(average_precision_score(y_true[:, h_idx], prob[:, h_idx]))
    return float(np.mean(scores)) if scores else math.nan


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: dict[str, np.ndarray],
    config: TaskGConfig,
) -> dict[str, np.ndarray] | None:
    if torch is None:
        warnings.warn("PyTorch is not installed; skipping mlp.", RuntimeWarning)
        return None
    scaler = StandardScaler().fit(x_train)
    x_train_z = scaler.transform(x_train).astype(np.float32)
    x_val_z = scaler.transform(x_val).astype(np.float32)
    eval_z = {name: scaler.transform(x).astype(np.float32) for name, x in x_eval.items()}
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = BoundaryMLP(x_train.shape[1], y_train.shape[1], config.dropout).to(device)
    n_pos = np.maximum(y_train.sum(axis=0), 1.0)
    n_neg = np.maximum((1 - y_train).sum(axis=0), 1.0)
    pos_weight = torch.tensor(np.sqrt(n_neg / n_pos), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_train_z), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    val_x_t = torch.from_numpy(x_val_z).to(device)

    best_state = None
    best_score = -math.inf
    stale = 0
    for _ in range(config.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_x_t)).cpu().numpy()
        score = mean_average_precision(y_val, val_prob)
        if not np.isfinite(score):
            score = -float(np.mean((val_prob - y_val) ** 2))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, x in eval_z.items():
            outputs[name] = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
    return outputs


def positive_rate_rows(split_name: str, y: np.ndarray, horizons: np.ndarray) -> list[str]:
    rates = ", ".join(f"{h:g}:{y[:, idx].mean():.4f}" for idx, h in enumerate(horizons))
    return [f"{split_name} positive rates: {rates}"]


def save_summary_md(path: Path, config: TaskGConfig, dataset_summary: dict[str, Any], test_pair: pd.DataFrame) -> None:
    lines = [
        "# Task G Boundary Sensitivity",
        "",
        f"- representation: `{config.representation}`",
        f"- systems: {dataset_summary['num_systems']}",
        f"- pairs: {dataset_summary['num_pairs']}",
        f"- features: {dataset_summary['num_features']}",
        f"- horizons: {', '.join(f'{h:g}' for h in config.horizons)}",
        "",
        "## Test Pair Metrics",
        "",
    ]
    mean_rows = test_pair[(test_pair["horizon"] == "mean") & (test_pair["threshold_mode"] == "val_tuned")]
    if mean_rows.empty:
        mean_rows = test_pair[test_pair["horizon"] == "mean"]
    for row in mean_rows.itertuples(index=False):
        lines.append(
            f"- {row.model}: mean AUPRC={getattr(row, 'AUPRC', math.nan):.4f}, "
            f"mean AUROC={getattr(row, 'AUROC', math.nan):.4f}, mean F1={getattr(row, 'F1', math.nan):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: TaskGConfig) -> None:
    np.random.seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    write_json(output_dir / "config.json", asdict(config))
    result_paths = discover_result_paths(config.data_glob)
    complete, dropped = load_complete_systems(config.completed_systems, config.group_size)
    if config.max_systems is not None:
        keep = np.asarray(sorted(complete["system_id"].unique()), dtype=np.int64)[: config.max_systems]
        complete = complete[complete["system_id"].isin(keep)].copy()

    horizons = np.asarray(config.horizons, dtype=np.float64)
    system_ids, times, stable = build_system_arrays(complete, horizons, config.group_size)
    ic18, base_feature_names = extract_ic18_by_system(result_paths, complete, system_ids, config.group_size)
    x, y, sample_system_ids, pair_ids, feature_names = expand_pair_dataset(
        system_ids, ic18, stable, DEFAULT_VALID_PAIRS, config.representation, base_feature_names
    )
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Targets must be binary.")
    per_system_counts = pd.Series(pair_ids).groupby(sample_system_ids).size()
    if not np.all(per_system_counts.to_numpy() == len(DEFAULT_VALID_PAIRS)):
        raise ValueError("Every system must contribute exactly 16 pairs.")

    dataset_summary, dataset_summary_frame = build_dataset_summary(
        times, stable, y, sample_system_ids, horizons, config.group_size
    )
    dataset_summary["num_features"] = int(x.shape[1])
    dataset_summary["dropped"] = dropped
    dataset_summary["feature_names"] = feature_names
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    dataset_summary_frame.to_csv(output_dir / "dataset_summary.csv", index=False)

    train_ids, val_ids, test_ids = split_system_ids(system_ids, config.seed, config.max_systems)
    write_json(
        output_dir / "split_system_ids.json",
        {"train": train_ids.tolist(), "val": val_ids.tolist(), "test": test_ids.tolist()},
    )
    train_mask = mask_for_systems(sample_system_ids, train_ids)
    val_mask = mask_for_systems(sample_system_ids, val_ids)
    test_mask = mask_for_systems(sample_system_ids, test_ids)
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"Loaded systems: {complete['system_id'].nunique()}")
    print(f"Valid systems: {len(system_ids)}")
    print(f"Dropped systems and reasons: {dropped}")
    print(f"Pair count: {len(y)}")
    print(f"Feature shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Train/val/test system counts: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Train/val/test pair counts: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    for line in positive_rate_rows("train", y_train, horizons) + positive_rate_rows("val", y_val, horizons) + positive_rate_rows("test", y_test, horizons):
        print(line)

    predictions_val: dict[str, np.ndarray] = {}
    predictions_test: dict[str, np.ndarray] = {}
    models = tuple(config.models)
    if "constant" in models:
        predictions_val["constant"] = predict_constant(y_train, len(y_val))
        predictions_test["constant"] = predict_constant(y_train, len(y_test))
    if "logistic" in models:
        out = train_logistic(x_train, y_train, {"val": x_val, "test": x_test}, config.seed)
        predictions_val["logistic"], predictions_test["logistic"] = out["val"], out["test"]
    if "random_forest" in models:
        out = train_random_forest(x_train, y_train, {"val": x_val, "test": x_test}, config)
        predictions_val["random_forest"], predictions_test["random_forest"] = out["val"], out["test"]
    if "xgboost" in models:
        out = train_xgboost(x_train, y_train, x_val, y_val, {"val": x_val, "test": x_test}, config)
        if out is not None:
            predictions_val["xgboost"], predictions_test["xgboost"] = out["val"], out["test"]
    if "mlp" in models:
        out = train_mlp(x_train, y_train, x_val, y_val, {"val": x_val, "test": x_test}, config)
        if out is not None:
            predictions_val["mlp"], predictions_test["mlp"] = out["val"], out["test"]

    thresholds: dict[str, dict[str, list[float]]] = {}
    pair_rows_val: list[dict[str, Any]] = []
    pair_rows_test: list[dict[str, Any]] = []
    system_rows_val: list[dict[str, Any]] = []
    system_rows_test: list[dict[str, Any]] = []
    for model_name, val_prob in predictions_val.items():
        test_prob = predictions_test[model_name]
        tuned = [best_f1_threshold(y_val[:, h_idx], val_prob[:, h_idx]) for h_idx in range(y.shape[1])]
        thresholds[model_name] = {"default_0p5": [0.5] * y.shape[1], "val_tuned": tuned}
        pair_rows_val.extend(pair_metric_rows(model_name, "val", y_val, val_prob, horizons, [0.5] * y.shape[1], "default_0p5"))
        pair_rows_val.extend(pair_metric_rows(model_name, "val", y_val, val_prob, horizons, tuned, "val_tuned"))
        pair_rows_test.extend(pair_metric_rows(model_name, "test", y_test, test_prob, horizons, [0.5] * y.shape[1], "default_0p5"))
        pair_rows_test.extend(pair_metric_rows(model_name, "test", y_test, test_prob, horizons, tuned, "val_tuned"))
        system_rows_val.extend(system_metric_rows(model_name, "val", sample_system_ids[val_mask], y_val, val_prob, horizons))
        system_rows_test.extend(system_metric_rows(model_name, "test", sample_system_ids[test_mask], y_test, test_prob, horizons))

    write_json(output_dir / "thresholds.json", thresholds)
    pair_val_df = pd.DataFrame(pair_rows_val)
    pair_test_df = pd.DataFrame(pair_rows_test)
    system_val_df = pd.DataFrame(system_rows_val)
    system_test_df = pd.DataFrame(system_rows_test)
    pair_val_df.to_csv(output_dir / "pair_metrics_val.csv", index=False)
    pair_test_df.to_csv(output_dir / "pair_metrics_test.csv", index=False)
    system_val_df.to_csv(output_dir / "system_metrics_val.csv", index=False)
    system_test_df.to_csv(output_dir / "system_metrics_test.csv", index=False)

    np.savez_compressed(
        output_dir / "predictions_val.npz",
        system_ids=sample_system_ids[val_mask],
        pair_ids=pair_ids[val_mask],
        y_true=y_val,
        horizons=horizons,
        feature_representation=np.asarray(config.representation),
        **{f"prob_{name}": prob for name, prob in predictions_val.items()},
    )
    np.savez_compressed(
        output_dir / "predictions_test.npz",
        system_ids=sample_system_ids[test_mask],
        pair_ids=pair_ids[test_mask],
        y_true=y_test,
        horizons=horizons,
        feature_representation=np.asarray(config.representation),
        **{f"prob_{name}": prob for name, prob in predictions_test.items()},
    )
    save_summary_md(output_dir / "summary.md", config, dataset_summary, pair_test_df)
    print(f"Task G complete. Outputs: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task G: stability boundary sensitivity prediction.")
    parser.add_argument("--data-glob", "--data_glob", dest="data_glob", default="dataset_v3/results_batch_*_sampling_2.hdf5")
    parser.add_argument("--completed-systems", "--completed_systems", dest="completed_systems", type=Path, default=ROOT / "dataset_v3" / "completed_systems_sampling_2.txt")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=ROOT / "results" / "task_g_boundary_sensitivity")
    parser.add_argument("--representation", choices=["pair_ic54", "pair_ic72"], default="pair_ic54")
    parser.add_argument("--models", nargs="+", default=["constant", "logistic", "random_forest", "xgboost", "mlp"])
    parser.add_argument("--horizons", nargs="+", type=float, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-systems", "--max_systems", dest="max_systems", type=int, default=None)
    parser.add_argument("--rf-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TaskGConfig(
        data_glob=args.data_glob,
        completed_systems=args.completed_systems,
        output_dir=args.output_dir,
        representation=args.representation,
        models=tuple(args.models),
        horizons=tuple(args.horizons),
        group_size=args.group_size,
        seed=args.seed,
        max_systems=args.max_systems,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        xgb_estimators=args.xgb_estimators,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_max_depth=args.xgb_max_depth,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
    )
    run(config)


if __name__ == "__main__":
    main()
