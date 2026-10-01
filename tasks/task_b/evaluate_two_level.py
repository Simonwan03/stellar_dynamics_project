#!/usr/bin/env python3
"""Build the paper-facing two-level Task B evaluation table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path("experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common")
OUTPUT = Path("results/task_b/sampling_2/task_b_common/two_level_evaluation")


def sources(prefix: int) -> list[tuple[str, str, Path, str]]:
    return [
        ("Panda encoder", "Linear", ROOT / f"panda_prefix{prefix}_linear/predictions_all.csv", "all"),
        ("Chronos-2 encoder", "Linear", ROOT / f"chronos2_prefix{prefix}_linear/predictions_all.csv", "all"),
        (
            "Dynamical summaries",
            "XGBoost",
            ROOT / f"physics_summary_xgb_prefix{prefix}/predictions_all.csv",
            "all",
        ),
        ("Raw sequence", "SVM", ROOT / f"raw_orbital15_prefix{prefix}/svm_test_predictions.csv", "test"),
        ("Raw sequence", "CNN", ROOT / f"raw_orbital15_prefix{prefix}/cnn_test_predictions.csv", "test"),
        ("Raw sequence", "TCN", ROOT / f"raw_orbital15_prefix{prefix}/tcn_test_predictions.csv", "test"),
    ]


def load_test_predictions(path: Path, layout: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if layout == "all":
        frame = frame[frame["split"] == "test"].copy()
        frame["raw_case_id"] = frame["index"].to_numpy(dtype=np.int64) // 32
        frame["target_stable"] = 1 - frame["target_unbound"].to_numpy(dtype=np.int64)
    elif layout != "test":
        raise ValueError(f"Unsupported prediction layout: {layout}")
    expected_target = (frame["time_unbound"].to_numpy(dtype=np.float64) >= 10000.0).astype(np.int64)
    if not np.array_equal(expected_target, frame["target_stable"].to_numpy(dtype=np.int64)):
        raise ValueError(f"Test labels are not aligned with metadata in {path}.")
    if len(frame) != 24000 or frame["raw_case_id"].nunique() != 750:
        raise ValueError(f"Expected 24,000 trajectories from 750 systems in {path}, got {len(frame)} rows.")
    counts = frame.groupby("raw_case_id").size().to_numpy()
    if not np.all(counts == 32):
        raise ValueError(f"Expected exactly 32 test trajectories per system in {path}.")
    return frame


def evaluate(prefix: int, representation: str, model: str, path: Path, layout: str) -> dict[str, object]:
    frame = load_test_predictions(path, layout)
    target = frame["target_stable"].to_numpy(dtype=np.int64)
    probability = frame["prob_stable"].to_numpy(dtype=np.float64)
    prediction = probability >= 0.5
    grouped = frame.assign(_target=target, _probability=probability).groupby("raw_case_id", sort=True).agg(
        target_stable_fraction=("_target", "mean"),
        predicted_stable_fraction=("_probability", "mean"),
    )
    system_target = grouped["target_stable_fraction"].to_numpy()
    system_prediction = grouped["predicted_stable_fraction"].to_numpy()
    return {
        "prefix": prefix,
        "representation": representation,
        "model": model,
        "trajectory_accuracy": float(accuracy_score(target, prediction)),
        "trajectory_f1": float(f1_score(target, prediction)),
        "trajectory_auroc": float(roc_auc_score(target, probability)),
        "trajectory_auprc": float(average_precision_score(target, probability)),
        "system_mae": float(mean_absolute_error(system_target, system_prediction)),
        "system_mse": float(mean_squared_error(system_target, system_prediction)),
        "system_r2": float(r2_score(system_target, system_prediction)),
    }


def best_mask(frame: pd.DataFrame, prefix: int, column: str, minimize: bool) -> np.ndarray:
    subset = frame[frame["prefix"] == prefix]
    best = subset[column].min() if minimize else subset[column].max()
    return np.isclose(frame[column].to_numpy(), best) & (frame["prefix"].to_numpy() == prefix)


def formatted(value: float, bold: bool) -> str:
    text = f"{value:.4f}"
    return f"\\textbf{{{text}}}" if bold else text


def latex_table(frame: pd.DataFrame) -> str:
    metric_columns = [
        ("trajectory_accuracy", False),
        ("trajectory_f1", False),
        ("trajectory_auroc", False),
        ("trajectory_auprc", False),
        ("system_mae", True),
        ("system_mse", True),
        ("system_r2", False),
    ]
    best = {
        (prefix, column): best_mask(frame, prefix, column, minimize)
        for prefix in (96, 512)
        for column, minimize in metric_columns
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of frozen foundation-model encoders, physics-informed features, and raw-sequence baselines on the common test split at both trajectory and system levels. Trajectory-binary metrics use 24,000 trajectories from 750 systems. For system stability, the 32 predicted survival probabilities in each system are averaged and compared with the empirical stable fraction at $H=10{,}000$. Best results within each prefix length are shown in bold.}",
        r"\label{tab:task_b_two_level}",
        r"\begin{tabular}{cllccccccc}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{Trajectory binary} & \multicolumn{3}{c}{System stability} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
        r"Prefix & Representation & Model & Accuracy & F1 & AUROC & AUPRC & MAE $\downarrow$ & MSE $\downarrow$ & $R^2$ $\uparrow$ \\",
        r"\midrule",
    ]
    for prefix in (96, 512):
        subset = frame[frame["prefix"] == prefix].reset_index()
        for local_index, row in subset.iterrows():
            global_index = int(row["index"])
            prefix_cell = rf"\multirow{{6}}{{*}}{{{prefix}}}" if local_index == 0 else ""
            values = [
                formatted(float(row[column]), bool(best[(prefix, column)][global_index]))
                for column, _ in metric_columns
            ]
            lines.append(
                " & ".join([prefix_cell, str(row["representation"]), str(row["model"]), *values]) + r" \\"
            )
        if prefix == 96:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    rename = {
        "prefix": "Prefix",
        "representation": "Representation",
        "model": "Model",
        "trajectory_accuracy": "Trajectory Accuracy",
        "trajectory_f1": "Trajectory F1",
        "trajectory_auroc": "Trajectory AUROC",
        "trajectory_auprc": "Trajectory AUPRC",
        "system_mae": "System MAE ↓",
        "system_mse": "System MSE ↓",
        "system_r2": "System R² ↑",
    }
    display = display.rename(columns=rename)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [evaluate(prefix, representation, model, path, layout) for prefix in (96, 512) for representation, model, path, layout in sources(prefix)]
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "task_b_two_level_metrics.csv", index=False)
    (OUTPUT / "task_b_two_level_table.tex").write_text(latex_table(frame))
    (OUTPUT / "task_b_two_level_table.md").write_text(markdown_table(frame))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved table outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
