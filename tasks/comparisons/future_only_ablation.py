#!/usr/bin/env python3
"""Evaluate system- and trajectory-level at-risk prefix ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


ABC_PREDICTIONS = Path("results/comparisons/dataset_unknown/abc_common_target_h10000_prefix96/system_predictions.csv")
TASK_B_ROOT = Path("experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common")
OUTPUT = Path("results/comparisons/dataset_unknown/task_ab_future_only_ablation")
GROUP_SIZE = 32
FINAL_HORIZON = 10000.0
TIME_PER_STEP = 5.0
BOOTSTRAP_SAMPLES = 10000
SEED = 42

TRAJECTORY_MODELS = (
    ("Panda frozen encoder", "Linear probe", "panda_prefix{prefix}_linear"),
    ("Chronos-2 frozen encoder", "Linear probe", "chronos2_prefix{prefix}_linear"),
    ("Dynamical summaries", "XGBoost", "physics_summary_xgb_prefix{prefix}"),
)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        "bias": float(np.mean(error)),
    }


def load_task_a() -> pd.DataFrame:
    frame = pd.read_csv(ABC_PREDICTIONS)
    frame["representation"] = frame["representation"].fillna("")
    frame = frame[
        (frame["task"] == "A")
        & (frame["model"] == "xgboost")
        & (frame["representation"] == "")
    ].copy()
    if len(frame) != 750:
        raise ValueError(f"Expected 750 Task A test predictions, got {len(frame)}.")
    return frame[["raw_case_id", "target_stable_fraction_h10000", "predicted_stable_fraction_h10000"]].rename(
        columns={"predicted_stable_fraction_h10000": "task_a_initial_xgboost"}
    )


def load_task_b(prefix: int) -> pd.DataFrame:
    path = TASK_B_ROOT / f"physics_summary_xgb_prefix{prefix}/predictions_all.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == "test"].copy()
    frame["raw_case_id"] = frame["index"].to_numpy(dtype=np.int64) // GROUP_SIZE
    frame["target_stable"] = (frame["time_unbound"].to_numpy(dtype=np.float64) >= FINAL_HORIZON).astype(
        np.int64
    )
    grouped = frame.groupby("raw_case_id", sort=True).agg(
        min_time_unbound=("time_unbound", "min"),
        target_stable_fraction_h10000=("target_stable", "mean"),
        prediction=("prob_stable", "mean"),
        variation_count=("index", "size"),
    )
    if len(grouped) != 750 or not np.all(grouped["variation_count"].to_numpy() == GROUP_SIZE):
        raise ValueError(f"Task B prefix={prefix} does not contain 750 complete test systems.")
    return grouped.reset_index().rename(columns={"prediction": f"task_b_prefix{prefix}_summary_xgboost"})


def bootstrap_metric_intervals(target: np.ndarray, prediction: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64) for name in ("mae", "mse", "r2")}
    for draw in range(BOOTSTRAP_SAMPLES):
        rows = rng.integers(0, target.size, size=target.size)
        result = regression_metrics(target[rows], prediction[rows])
        for name in draws:
            draws[name][draw] = result[name]
    output: dict[str, float] = {}
    for name, values in draws.items():
        output[f"{name}_ci_low"] = float(np.nanpercentile(values, 2.5))
        output[f"{name}_ci_high"] = float(np.nanpercentile(values, 97.5))
    return output


def paired_bootstrap(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = {
        "mae": np.abs(right - target) - np.abs(left - target),
        "mse": (right - target) ** 2 - (left - target) ** 2,
    }
    output: dict[str, float] = {}
    for name, per_system in differences.items():
        draws = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
        for draw in range(BOOTSTRAP_SAMPLES):
            rows = rng.integers(0, target.size, size=target.size)
            draws[draw] = per_system[rows].mean()
        output[f"delta_{name}"] = float(per_system.mean())
        output[f"delta_{name}_ci_low"] = float(np.percentile(draws, 2.5))
        output[f"delta_{name}_ci_high"] = float(np.percentile(draws, 97.5))
        output[f"probability_right_improves_{name}"] = float(np.mean(draws < 0.0))
    return output


def load_trajectory_predictions(prefix: int, directory_pattern: str) -> pd.DataFrame:
    path = TASK_B_ROOT / directory_pattern.format(prefix=prefix) / "predictions_all.csv"
    frame = pd.read_csv(path)
    required = {"index", "split", "time_unbound", "target_unbound", "prob_stable"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    frame = frame[frame["split"] == "test"].copy()
    frame = frame.sort_values("index").reset_index(drop=True)
    expected_unbound = frame["time_unbound"].to_numpy(dtype=np.float64) < FINAL_HORIZON
    if not np.array_equal(frame["target_unbound"].to_numpy(dtype=bool), expected_unbound):
        raise ValueError(f"Inconsistent target_unbound values in {path}.")
    if len(frame) != 750 * GROUP_SIZE or frame["index"].nunique() != len(frame):
        raise ValueError(f"Expected 24,000 unique test trajectories in {path}, got {len(frame)}.")
    return frame[["index", "time_unbound", "prob_stable"]]


def trajectory_classification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = (frame["time_unbound"].to_numpy(dtype=np.float64) >= FINAL_HORIZON).astype(np.int64)
    probability = frame["prob_stable"].to_numpy(dtype=np.float64)
    prediction = probability >= 0.5
    return {
        "n_trajectories": int(len(frame)),
        "stable_fraction": float(target.mean()),
        "accuracy": float(accuracy_score(target, prediction)),
        "f1_stable": float(f1_score(target, prediction, zero_division=0)),
        "auroc_stable": float(roc_auc_score(target, probability)),
        "auprc_stable": float(average_precision_score(target, probability)),
    }


def build_trajectory_at_risk_table() -> pd.DataFrame:
    """Evaluate native and matched at-risk cohorts using stable-positive metrics."""
    specs = ((480.0, (96,)), (2560.0, (96, 512)))
    rows: list[dict[str, object]] = []
    cohort_identities: dict[float, np.ndarray] = {}
    cohort_targets: dict[float, np.ndarray] = {}
    for cutoff, prefixes in specs:
        for prefix in prefixes:
            for representation, model, directory_pattern in TRAJECTORY_MODELS:
                frame = load_trajectory_predictions(prefix, directory_pattern)
                # Strictly exclude events at the prefix endpoint because that
                # state is already visible in the observed prefix.
                subset = frame[frame["time_unbound"] > cutoff].copy()
                identities = subset["index"].to_numpy(dtype=np.int64)
                targets = (subset["time_unbound"].to_numpy(dtype=np.float64) >= FINAL_HORIZON).astype(np.int64)
                if cutoff in cohort_identities:
                    if not np.array_equal(identities, cohort_identities[cutoff]):
                        raise ValueError(f"At-risk cohort T>{cutoff:g} is not aligned across models/prefixes.")
                    if not np.array_equal(targets, cohort_targets[cutoff]):
                        raise ValueError(f"At-risk targets T>{cutoff:g} are not aligned across models/prefixes.")
                else:
                    cohort_identities[cutoff] = identities
                    cohort_targets[cutoff] = targets
                rows.append(
                    {
                        "at_risk_cohort": f"T>{cutoff:g}",
                        "at_risk_time": cutoff,
                        "prefix": prefix,
                        "representation": representation,
                        "model": model,
                        **trajectory_classification_metrics(subset),
                    }
                )
    return pd.DataFrame(rows)


def write_trajectory_tables(metrics: pd.DataFrame) -> None:
    markdown_lines = [
        "| At-risk cohort | Prefix | Representation | Model | Trajectories | Stable % | Accuracy | F1 | AUROC | AUPRC |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        markdown_lines.append(
            f"| ${row.at_risk_cohort}$ | {row.prefix} | {row.representation} | {row.model} | "
            f"{row.n_trajectories:,} | {100.0 * row.stable_fraction:.2f} | {row.accuracy:.5f} | "
            f"{row.f1_stable:.5f} | {row.auroc_stable:.5f} | {row.auprc_stable:.5f} |"
        )
    (OUTPUT / "trajectory_at_risk_table.md").write_text("\n".join(markdown_lines) + "\n")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{At-risk trajectory-level stability classification. Evaluation is restricted to trajectories that remain bound through the specified at-risk time. The $T>2560$ cohort enables a matched comparison between 96- and 512-step prefixes on exactly the same trajectories.}",
        r"\label{tab:trajectory_at_risk}",
        r"\small",
        r"\begin{tabular}{ccllrrrrrr}",
        r"\toprule",
        r"At-risk cohort & Prefix & Representation & Model & Traj. & Stable \% & Acc. & F1 & AUROC & AUPRC \\",
        r"\midrule",
    ]
    previous_cohort = None
    for row in metrics.itertuples(index=False):
        if previous_cohort is not None and row.at_risk_cohort != previous_cohort:
            latex_lines.append(r"\midrule")
        latex_lines.append(
            f"${row.at_risk_cohort}$ & {row.prefix} & {row.representation} & {row.model} & "
            f"{row.n_trajectories:,} & {100.0 * row.stable_fraction:.2f} & {row.accuracy:.5f} & "
            f"{row.f1_stable:.5f} & {row.auroc_stable:.5f} & {row.auprc_stable:.5f} \\\\"
        )
        previous_cohort = row.at_risk_cohort
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT / "trajectory_at_risk_table.tex").write_text("\n".join(latex_lines) + "\n")


def main() -> None:
    task_a = load_task_a()
    task_b_96 = load_task_b(96)
    task_b_512 = load_task_b(512)
    system = task_a.merge(task_b_96, on=["raw_case_id", "target_stable_fraction_h10000"], validate="one_to_one")
    system = system.merge(
        task_b_512.drop(columns=["min_time_unbound", "variation_count"]),
        on=["raw_case_id", "target_stable_fraction_h10000"],
        validate="one_to_one",
    )
    if len(system) != 750:
        raise ValueError(f"Expected 750 aligned systems, got {len(system)}.")

    cohort_specs = [
        (
            "bound_through_prefix96",
            96,
            ["task_a_initial_xgboost", "task_b_prefix96_summary_xgboost"],
        ),
        (
            "bound_through_prefix512",
            512,
            [
                "task_a_initial_xgboost",
                "task_b_prefix96_summary_xgboost",
                "task_b_prefix512_summary_xgboost",
            ],
        ),
    ]
    metric_rows: list[dict[str, object]] = []
    cohort_frames: dict[str, pd.DataFrame] = {}
    for cohort, prefix, models in cohort_specs:
        cutoff = prefix * TIME_PER_STEP
        subset = system[system["min_time_unbound"] >= cutoff].copy()
        cohort_frames[cohort] = subset
        target = subset["target_stable_fraction_h10000"].to_numpy(dtype=np.float64)
        for model_index, model in enumerate(models):
            prediction = subset[model].to_numpy(dtype=np.float64)
            metric_rows.append(
                {
                    "cohort": cohort,
                    "eligibility": f"min_j time_unbound_j >= {cutoff:g}",
                    "prefix_cutoff_time": cutoff,
                    "n_systems": len(subset),
                    "target_mean": float(target.mean()),
                    "model": model,
                    **regression_metrics(target, prediction),
                    **bootstrap_metric_intervals(target, prediction, SEED + prefix + model_index),
                }
            )

    comparison_specs = [
        (
            "bound_through_prefix96",
            "task_a_initial_xgboost",
            "task_b_prefix96_summary_xgboost",
        ),
        (
            "bound_through_prefix512",
            "task_a_initial_xgboost",
            "task_b_prefix96_summary_xgboost",
        ),
        (
            "bound_through_prefix512",
            "task_b_prefix96_summary_xgboost",
            "task_b_prefix512_summary_xgboost",
        ),
    ]
    comparison_rows: list[dict[str, object]] = []
    for comparison_index, (cohort, left, right) in enumerate(comparison_specs):
        subset = cohort_frames[cohort]
        target = subset["target_stable_fraction_h10000"].to_numpy(dtype=np.float64)
        comparison_rows.append(
            {
                "cohort": cohort,
                "n_systems": len(subset),
                "left": left,
                "right": right,
                "delta_definition": "right minus left; negative is improvement",
                **paired_bootstrap(
                    target,
                    subset[left].to_numpy(dtype=np.float64),
                    subset[right].to_numpy(dtype=np.float64),
                    SEED + comparison_index,
                ),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "cohort_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(OUTPUT / "paired_comparisons.csv", index=False)
    for cohort, subset in cohort_frames.items():
        subset.to_csv(OUTPUT / f"{cohort}_system_predictions.csv", index=False)
    trajectory_metrics = build_trajectory_at_risk_table()
    trajectory_metrics.to_csv(OUTPUT / "trajectory_at_risk_metrics.csv", index=False)
    write_trajectory_tables(trajectory_metrics)
    protocol = {
        "final_target": "mean_j 1[time_unbound_ij >= 10000] over 32 variations",
        "already_unbound_system_definition": "min_j time_unbound_ij < prefix_steps * 5",
        "prefix_cutoffs": {"96": 480.0, "512": 2560.0},
        "models": {
            "task_a": "initial-condition XGBoost",
            "task_b": "dynamical-summary XGBoost, independently applied to each trajectory then averaged",
        },
        "bootstrap": {"unit": "system", "samples": BOOTSTRAP_SAMPLES, "seed_base": SEED},
        "trajectory_at_risk": {
            "eligibility": "time_unbound > at_risk_time (strict; endpoint events are excluded)",
            "cohorts": {"T>480": [96], "T>2560": [96, 512]},
            "target": "stable-positive: 1[time_unbound >= 10000]",
            "threshold": 0.5,
            "matched_cohort": "All T>2560 rows use exactly the same trajectory indices and targets.",
        },
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPaired comparisons (right minus left):")
    print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTrajectory-level at-risk classification:")
    print(trajectory_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved ablation to {OUTPUT}")


if __name__ == "__main__":
    main()
