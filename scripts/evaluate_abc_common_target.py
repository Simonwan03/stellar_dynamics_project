#!/usr/bin/env python3
"""Evaluate Tasks A/B/C on the same system stable-fraction target at t=10000.

The script consumes saved predictions/checkpoints only; it does not retrain any
model.  Task B trajectory probabilities are averaged within each 32-variation
system, while Task C survival curves are read at their final bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.clip(np.asarray(pred, dtype=np.float64).reshape(-1), 0.0, 1.0)
    error = pred - y
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else float("nan"),
        "bias": float(np.mean(error)),
    }


def bootstrap_intervals(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    values = {name: np.empty(samples, dtype=np.float64) for name in ("mae", "mse", "r2")}
    for index in range(samples):
        rows = rng.integers(0, y.size, size=y.size)
        result = metrics(y[rows], pred[rows])
        for name in values:
            values[name][index] = result[name]
    output: dict[str, float] = {}
    for name, array in values.items():
        output[f"{name}_ci_low"] = float(np.nanpercentile(array, 2.5))
        output[f"{name}_ci_high"] = float(np.nanpercentile(array, 97.5))
    return output


def paired_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    left = left.sort_values("raw_case_id")
    right = right.sort_values("raw_case_id")
    if not np.array_equal(left["raw_case_id"].to_numpy(), right["raw_case_id"].to_numpy()):
        raise ValueError("Paired comparison requires identical system IDs.")
    y = left["target_stable_fraction_h10000"].to_numpy(dtype=np.float64)
    if not np.allclose(y, right["target_stable_fraction_h10000"].to_numpy(), atol=1e-7):
        raise ValueError("Paired comparison requires identical targets.")
    left_pred = left["predicted_stable_fraction_h10000"].to_numpy(dtype=np.float64)
    right_pred = right["predicted_stable_fraction_h10000"].to_numpy(dtype=np.float64)
    differences = {
        "mae": np.abs(right_pred - y) - np.abs(left_pred - y),
        "mse": (right_pred - y) ** 2 - (left_pred - y) ** 2,
    }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples, dtype=np.float64) for name in differences}
    for index in range(samples):
        rows = rng.integers(0, y.size, size=y.size)
        for name, values in differences.items():
            draws[name][index] = values[rows].mean()
    output: dict[str, float] = {}
    for name, values in differences.items():
        output[f"delta_{name}"] = float(values.mean())
        output[f"delta_{name}_ci_low"] = float(np.percentile(draws[name], 2.5))
        output[f"delta_{name}_ci_high"] = float(np.percentile(draws[name], 97.5))
        output[f"bootstrap_probability_improved_{name}"] = float(np.mean(draws[name] < 0.0))
    return output


def load_task_a(task_dir: Path) -> list[pd.DataFrame]:
    import torch
    import xgboost as xgb

    from tasks.task_a.main import MLPRegressor

    config = json.loads((task_dir / "config.json").read_text())
    split = np.load(task_dir / "split_indices.npz")
    case_ids = split["case_ids"].astype(np.int64)
    test_idx = split["test_idx"].astype(np.int64)
    labels = pd.read_csv(task_dir / "group_labels.csv")
    if not np.array_equal(labels["raw_case_id"].to_numpy(dtype=np.int64), case_ids):
        raise ValueError("Task A group_labels.csv is not aligned with split_indices.npz.")

    features = [item.strip() for item in config["features"].split(",") if item.strip()]
    arrays = []
    with h5py.File(config["tokenized_path"], "r") as handle:
        for name in features:
            values = np.asarray(handle[name][case_ids], dtype=np.float32)
            arrays.append(values[:, 0, ...].reshape(values.shape[0], -1))
    x = np.concatenate(arrays, axis=1).astype(np.float32)
    normalization = np.load(task_dir / "normalization_stats.npz")
    x = (x - normalization["mean"]) / normalization["std"]
    x_test = x[test_idx]
    y_test = labels["stable_fraction"].to_numpy(dtype=np.float64)[test_idx]
    test_case_ids = case_ids[test_idx]

    frames: list[pd.DataFrame] = []
    checkpoint = torch.load(task_dir / "mlp_fraction_model.pt", map_location="cpu", weights_only=False)
    model = MLPRegressor(
        in_dim=x_test.shape[1],
        hidden=tuple(config["mlp_hidden"]),
        dropout=float(config["mlp_dropout"]),
        output_mode=config["mlp_loss"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        mlp_pred = model(torch.from_numpy(x_test).float()).numpy().reshape(-1)
    frames.append(prediction_frame("Initial conditions", "A", "mlp", test_case_ids, y_test, mlp_pred))

    booster = xgb.Booster()
    booster.load_model(task_dir / "xgboost_fraction_model.json")
    xgb_pred = booster.predict(xgb.DMatrix(x_test))
    frames.append(prediction_frame("Initial conditions", "A", "xgboost", test_case_ids, y_test, xgb_pred))
    return frames


def prediction_frame(
    setting: str,
    task: str,
    model: str,
    case_ids: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    representation: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_setting": setting,
            "task": task,
            "model": model,
            "representation": representation,
            "raw_case_id": np.asarray(case_ids, dtype=np.int64),
            "target_stable_fraction_h10000": np.asarray(target, dtype=np.float64),
            "predicted_stable_fraction_h10000": np.asarray(prediction, dtype=np.float64),
        }
    )


def load_task_b(task_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(task_dir.glob("*_test_predictions.csv")):
        model = path.name.removesuffix("_test_predictions.csv")
        data = pd.read_csv(path)
        required = {"raw_case_id", "target_stable", "prob_stable"}
        if not required.issubset(data.columns):
            raise ValueError(f"Missing Task B columns in {path}: {sorted(required - set(data.columns))}")
        grouped = data.groupby("raw_case_id", sort=True).agg(
            target_stable_fraction_h10000=("target_stable", "mean"),
            predicted_stable_fraction_h10000=("prob_stable", "mean"),
            variation_count=("target_stable", "size"),
        )
        if len(grouped) != 750 or not np.all(grouped["variation_count"].to_numpy() == 32):
            raise ValueError(f"{path} does not contain exactly 750 complete 32-variation test systems.")
        frames.append(
            prediction_frame(
                "Individual prefixes",
                "B",
                model,
                grouped.index.to_numpy(),
                grouped["target_stable_fraction_h10000"].to_numpy(),
                grouped["predicted_stable_fraction_h10000"].to_numpy(),
                representation="raw orbital15, prefix=96",
            )
        )
    return frames


def load_task_b_all_predictions(path: Path, model: str, representation: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"index", "split", "target_unbound", "prob_stable"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing Task B columns in {path}: {sorted(required - set(data.columns))}")
    data = data[data["split"] == "test"].copy()
    data["raw_case_id"] = data["index"].to_numpy(dtype=np.int64) // 32
    data["target_stable"] = 1 - data["target_unbound"].to_numpy(dtype=np.int64)
    grouped = data.groupby("raw_case_id", sort=True).agg(
        target_stable_fraction_h10000=("target_stable", "mean"),
        predicted_stable_fraction_h10000=("prob_stable", "mean"),
        variation_count=("index", "size"),
    )
    if len(grouped) != 750 or not np.all(grouped["variation_count"].to_numpy() == 32):
        raise ValueError(f"{path} does not contain exactly 750 complete 32-variation test systems.")
    return prediction_frame(
        "Individual prefixes",
        "B",
        model,
        grouped.index.to_numpy(),
        grouped["target_stable_fraction_h10000"].to_numpy(),
        grouped["predicted_stable_fraction_h10000"].to_numpy(),
        representation=representation,
    )


def load_task_c(task_dir: Path, representation: str) -> list[pd.DataFrame]:
    predictions = np.load(task_dir / "test_predictions.npz")
    split = np.load(task_dir / "split_indices.npz")
    case_ids = split["case_ids"][split["test_idx"]].astype(np.int64)
    bin_edges = predictions["bin_edges"]
    if not np.isclose(float(bin_edges[-1]), 10000.0):
        raise ValueError(f"Task C final horizon in {task_dir} is {bin_edges[-1]}, not 10000.")
    frames: list[pd.DataFrame] = []
    for key in predictions.files:
        if not key.endswith("_preds") or "variation" in key or "old_" in key:
            continue
        model = key.removesuffix("_preds")
        target_key = f"{model}_targets"
        pred = predictions[key][:, -1]
        target = predictions[target_key][:, -1]
        frames.append(
            prediction_frame(
                "Perturbation ensemble",
                "C",
                model,
                case_ids,
                target,
                pred,
                representation=representation,
            )
        )
    return frames


def validate_common_targets(predictions: pd.DataFrame) -> None:
    reference = None
    reference_ids = None
    for (_, _, model, representation), frame in predictions.groupby(
        ["input_setting", "task", "model", "representation"], sort=False
    ):
        ordered = frame.sort_values("raw_case_id")
        ids = ordered["raw_case_id"].to_numpy()
        target = ordered["target_stable_fraction_h10000"].to_numpy()
        if reference is None:
            reference_ids, reference = ids, target
        elif not np.array_equal(ids, reference_ids) or not np.allclose(target, reference, atol=1e-7):
            raise ValueError(f"Common-target mismatch for {model} ({representation}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/abc_common_target_h10000_prefix96"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = load_task_a(Path("results/task_a_common/stable_fraction"))
    frames += load_task_b(
        Path("experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/raw_orbital15_prefix96")
    )
    frames.append(
        load_task_b_all_predictions(
            Path(
                "experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/"
                "physics_summary_xgb_prefix96/predictions_all.csv"
            ),
            model="xgboost",
            representation="physical summary, prefix=96",
        )
    )
    c_sources = [
        (Path("results/task_c_survival20_common_prefix96_raw"), "raw orbital15, prefix=96"),
        (Path("results/task_c_survival20_common_prefix96_panda"), "frozen Panda, prefix=96"),
        (Path("results/task_c_survival20_common_prefix96_chronos"), "frozen Chronos-2, prefix=96"),
    ]
    for path, representation in c_sources:
        if path.exists():
            frames += load_task_c(path, representation)

    predictions = pd.concat(frames, ignore_index=True)
    validate_common_targets(predictions)

    metric_rows = []
    group_columns = ["input_setting", "task", "model", "representation"]
    for keys, frame in predictions.groupby(group_columns, sort=False):
        y = frame["target_stable_fraction_h10000"].to_numpy()
        pred = frame["predicted_stable_fraction_h10000"].to_numpy()
        metric_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_systems": int(len(frame)),
                **metrics(y, pred),
                **bootstrap_intervals(y, pred, seed=args.seed, samples=args.bootstrap_samples),
            }
        )
    metric_frame = pd.DataFrame(metric_rows).sort_values(["task", "mse", "mae"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "system_predictions.csv", index=False)
    metric_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)

    strict_keys = {
        ("A", "mlp", ""),
        ("B", "tcn", "raw orbital15, prefix=96"),
        ("C", "tcn", "raw orbital15, prefix=96"),
    }
    strict = metric_frame[
        metric_frame.apply(lambda row: (row["task"], row["model"], row["representation"]) in strict_keys, axis=1)
    ].copy()
    strict.to_csv(args.output_dir / "strict_raw_comparison.csv", index=False)

    best_observed = metric_frame.sort_values(["task", "mse"]).groupby("task", sort=True).head(1).copy()
    best_observed.to_csv(args.output_dir / "best_observed_by_setting.csv", index=False)

    def select(task: str, model: str, representation: str) -> pd.DataFrame:
        return predictions[
            (predictions["task"] == task)
            & (predictions["model"] == model)
            & (predictions["representation"] == representation)
        ]

    comparison_specs = [
        ("A MLP", "B raw TCN", select("A", "mlp", ""), select("B", "tcn", "raw orbital15, prefix=96")),
        (
            "A XGBoost",
            "B physical-summary XGBoost",
            select("A", "xgboost", ""),
            select("B", "xgboost", "physical summary, prefix=96"),
        ),
        (
            "B raw TCN",
            "C raw TCN",
            select("B", "tcn", "raw orbital15, prefix=96"),
            select("C", "tcn", "raw orbital15, prefix=96"),
        ),
        (
            "B raw TCN",
            "C frozen-Panda direct",
            select("B", "tcn", "raw orbital15, prefix=96"),
            select("C", "embedding_direct", "frozen Panda, prefix=96"),
        ),
        (
            "B physical-summary XGBoost",
            "C frozen-Panda direct",
            select("B", "xgboost", "physical summary, prefix=96"),
            select("C", "embedding_direct", "frozen Panda, prefix=96"),
        ),
    ]
    paired_rows = [
        {
            "left": left_name,
            "right": right_name,
            "delta_definition": "right minus left; negative is improvement",
            **paired_difference(left, right, seed=args.seed, samples=args.bootstrap_samples),
        }
        for left_name, right_name, left, right in comparison_specs
    ]
    pd.DataFrame(paired_rows).to_csv(args.output_dir / "paired_comparisons.csv", index=False)

    metadata = {
        "target": "mean_j 1[time_unbound_ij >= 10000] over 32 variations",
        "test_systems": 750,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "unit": "system"},
        "strict_raw_comparison": "Predeclared A=MLP, B=raw TCN, C=raw TCN; B/C both use orbital15 prefix=96.",
        "best_observed_warning": "Descriptive minimum test MSE among available saved models; not validation-selected.",
        "task_c_architecture_note": "The saved event-survival C models encode variations independently and aggregate their survival curves; they do not use explicit cross-variation attention.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metric_frame.to_string(index=False))
    print(f"\nSaved common-target evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
