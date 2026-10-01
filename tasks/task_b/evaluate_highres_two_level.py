#!/usr/bin/env python3
"""Evaluate sampling_3 Task B predictions at trajectory and system levels.

The high-resolution Task B runs predict one stable probability per trajectory.
For the system-level evaluation, this script averages the 32 probabilities for
each raw_case_id and compares that value with the observed stable fraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREFIXES = (64, 128, 256, 512)
GROUP_SIZE = 32
TIME_HORIZON = 10_000.0
EXPECTED_TEST_TRAJECTORIES = 24_000
EXPECTED_TEST_SYSTEMS = 750
OUTPUT = Path("results/task_b/sampling_3/task_b_sampling3_two_level_evaluation")


def sources(prefix: int) -> list[tuple[str, Path, str]]:
    base = Path(f"results/task_b/sampling_3/task_b_sampling3_prefix{prefix}")
    return [
        ("MLP", base / "mlp_test_predictions.csv", "test"),
        ("CNN", base / "cnn_test_predictions.csv", "test"),
        ("TCN", base / "tcn_test_predictions.csv", "test"),
        (
            "Panda-72M + linear",
            Path(f"results/task_b/sampling_3/task_b_sampling3_prefix{prefix}_panda_linear/predictions_all.csv"),
            "all",
        ),
        (
            "Chronos-2 + linear",
            Path(f"results/task_b/sampling_3/task_b_sampling3_prefix{prefix}_chronos2/predictions_all.csv"),
            "all",
        ),
        (
            "Stats XGBoost",
            Path(f"results/task_b/sampling_3/task_b_sampling3_prefix{prefix}_stats_xgboost/predictions_all.csv"),
            "all",
        ),
    ]


def load_test_predictions(prefix: int, path: Path, layout: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if layout == "all":
        frame = frame[frame["split"] == "test"].copy()
        labels_path = Path(f"results/task_b/sampling_3/task_b_sampling3_prefix{prefix}/labels_filtered.csv")
        labels = pd.read_csv(
            labels_path,
            usecols=["index", "raw_case_id", "split", "time_unbound"],
        ).rename(
            columns={
                "split": "_label_split",
                "time_unbound": "_label_time_unbound",
            }
        )
        frame = frame.merge(labels, on="index", how="left", validate="one_to_one")
        if frame[["raw_case_id", "_label_split", "_label_time_unbound"]].isna().any().any():
            raise ValueError(f"Could not align every prediction in {path} with {labels_path}.")
        if not (frame["_label_split"] == "test").all():
            raise ValueError(f"Prediction and label splits disagree in {path}.")
        if not np.allclose(
            frame["time_unbound"].to_numpy(dtype=np.float64),
            frame["_label_time_unbound"].to_numpy(dtype=np.float64),
        ):
            raise ValueError(f"Prediction and label times disagree in {path}.")
    elif layout != "test":
        raise ValueError(f"Unsupported prediction layout: {layout}")

    target = (frame["time_unbound"].to_numpy(dtype=np.float64) >= TIME_HORIZON).astype(np.int64)
    if "target_stable" in frame and not np.array_equal(
        target, frame["target_stable"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(f"Stable labels are not aligned in {path}.")
    if "target_unbound" in frame and not np.array_equal(
        1 - target, frame["target_unbound"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(f"Unbound labels are not aligned in {path}.")
    frame["target_stable"] = target

    if len(frame) != EXPECTED_TEST_TRAJECTORIES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_TRAJECTORIES:,} test trajectories in {path}, got {len(frame):,}."
        )
    counts = frame.groupby("raw_case_id").size()
    if len(counts) != EXPECTED_TEST_SYSTEMS or not (counts == GROUP_SIZE).all():
        raise ValueError(
            f"Expected {EXPECTED_TEST_SYSTEMS} systems with {GROUP_SIZE} trajectories each in {path}."
        )
    probability = frame["prob_stable"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"Invalid stable probabilities in {path}.")
    return frame


def evaluate(prefix: int, model: str, path: Path, layout: str) -> tuple[dict[str, object], pd.DataFrame]:
    frame = load_test_predictions(prefix, path, layout)
    grouped = (
        frame.groupby("raw_case_id", sort=True)
        .agg(
            target_stable_fraction=("target_stable", "mean"),
            predicted_stable_fraction=("prob_stable", "mean"),
        )
        .reset_index()
    )
    target = grouped["target_stable_fraction"].to_numpy(dtype=np.float64)
    prediction = grouped["predicted_stable_fraction"].to_numpy(dtype=np.float64)
    metrics = {
        "prefix": prefix,
        "model": model,
        "test_trajectories": len(frame),
        "test_systems": len(grouped),
        "system_mae": float(mean_absolute_error(target, prediction)),
        "system_mse": float(mean_squared_error(target, prediction)),
        "system_r2": float(r2_score(target, prediction)),
    }
    grouped.insert(0, "model", model)
    grouped.insert(0, "prefix", prefix)
    return metrics, grouped


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Prefix | Model | System MAE ↓ | System MSE ↓ | System R² ↑ |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.prefix} | {row.model} | {row.system_mae:.4f} | "
            f"{row.system_mse:.4f} | {row.system_r2:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    metric_rows: list[dict[str, object]] = []
    system_frames: list[pd.DataFrame] = []
    reference_targets: np.ndarray | None = None
    reference_ids: np.ndarray | None = None

    for prefix in PREFIXES:
        for model, path, layout in sources(prefix):
            metrics, systems = evaluate(prefix, model, path, layout)
            target = systems["target_stable_fraction"].to_numpy(dtype=np.float64)
            case_ids = systems["raw_case_id"].to_numpy(dtype=np.int64)
            if reference_targets is None:
                reference_targets = target
                reference_ids = case_ids
            elif not np.array_equal(case_ids, reference_ids) or not np.array_equal(target, reference_targets):
                raise ValueError(f"The test systems or targets differ for prefix={prefix}, model={model}.")
            metric_rows.append(metrics)
            system_frames.append(systems)

    metrics_frame = pd.DataFrame(metric_rows)
    systems_frame = pd.concat(system_frames, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(OUTPUT / "task_b_sampling3_two_level_metrics.csv", index=False)
    systems_frame.to_csv(OUTPUT / "task_b_sampling3_system_predictions.csv", index=False)
    (OUTPUT / "task_b_sampling3_system_table.md").write_text(
        markdown_table(metrics_frame), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved system-level evaluation to {OUTPUT}")


if __name__ == "__main__":
    main()
