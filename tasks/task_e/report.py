"""Aggregate Task E E1/E2 run outputs and create comparison reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Task E E1/E2 model comparison reports.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Per-model Task E output directories containing test_metrics.json.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_report(args.run_dirs, args.output_dir)


def build_report(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [load_run(run_dir) for run_dir in run_dirs]
    if not records:
        raise ValueError("At least one Task E run directory is required.")
    protocol_pairs = {(record["context_len"], record["pred_len"]) for record in records}
    if len(protocol_pairs) != 1:
        raise ValueError(
            "All Task E model runs in one report must use the same context_len/pred_len; "
            f"got {sorted(protocol_pairs)}."
        )
    window_protocols = {
        (
            json.dumps(record["split_trajectory_counts"], sort_keys=True),
            json.dumps(record["used_window_counts"], sort_keys=True),
            json.dumps(record.get("sampling_metadata", {}), sort_keys=True),
            json.dumps(record["resolved_source_paths"], sort_keys=True),
            record["scaler_fit_scope"],
        )
        for record in records
    }
    if len(window_protocols) != 1:
        raise ValueError(
            "Task E model runs do not share the same split/window/source/scaler protocol."
        )
    records.sort(key=lambda record: record["model"])

    summary = {
        "models": [record["model"] for record in records],
        "runs": records,
    }
    _save_json(output_dir / "summary.json", summary)
    _save_summary_csv(output_dir / "summary.csv", records)
    _save_curve_csv(output_dir / "one_shot_horizon_mae.csv", records, "e1_horizon_mae")
    _save_curve_csv(output_dir / "rolling_horizon_mae.csv", records, "e2_rolling_horizon_mae")
    _save_markdown(output_dir / "summary.md", records)
    _plot_one_shot(output_dir / "one_shot_horizon_mae.png", records)
    _plot_rolling(output_dir / "rolling_horizon_mae.png", records)
    return summary


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        raise FileNotFoundError(f"Missing {test_metrics_path}.")
    e1 = _load_json(test_metrics_path)
    rolling_path = run_dir / "rolling_metrics.json"
    e2 = _load_json(rolling_path) if rolling_path.exists() else None
    horizon_mae = e1.get("horizon_mae")
    if horizon_mae is None:
        horizon_mae = _read_horizon_mae(run_dir / "horizon_metrics.csv")
    rolling_horizon_mae = e2.get("rolling_horizon_mae") if e2 else None
    return {
        "model": str(config.get("model", run_dir.name)),
        "run_dir": str(run_dir),
        "context_len": config.get("context_len"),
        "pred_len": config.get("pred_len"),
        "split_trajectory_counts": config.get("split_trajectory_counts"),
        "used_window_counts": config.get("used_window_counts"),
        "sampling_metadata": config.get("sampling_metadata", {}),
        "resolved_source_paths": config.get("resolved_source_paths"),
        "scaler_fit_scope": config.get("scaler_fit_scope"),
        "mse_z": e1.get("mse_z"),
        "mae_z": e1.get("mae_z"),
        "rmse_z": e1.get("rmse_z"),
        "mse_z_window_weighted": e1.get("mse_z_window_weighted"),
        "mae_z_window_weighted": e1.get("mae_z_window_weighted"),
        "rmse_z_window_weighted": e1.get("rmse_z_window_weighted"),
        "mse_z_system_balanced": e1.get("mse_z_system_balanced", e1.get("mse_z")),
        "mae_z_system_balanced": e1.get("mae_z_system_balanced", e1.get("mae_z")),
        "rmse_z_system_balanced": e1.get("rmse_z_system_balanced", e1.get("rmse_z")),
        "pearson": e1.get("pearson", e1.get("pearson_z")),
        "bootstrap_ci95": e1.get("bootstrap_ci95", {}),
        "unit_circle_error": e1.get("unit_circle_error"),
        "e1_horizon_mae": [float(value) for value in (horizon_mae or [])],
        "block_1_mae": e2.get("block_1_mae") if e2 else None,
        "block_2_mae": e2.get("block_2_mae") if e2 else None,
        "block_3_mae": e2.get("block_3_mae") if e2 else None,
        "block_4_mae": e2.get("block_4_mae") if e2 else None,
        "error_amplification_ratio": e2.get("error_amplification_ratio") if e2 else None,
        "e2_bootstrap_ci95": e2.get("bootstrap_ci95", {}) if e2 else {},
        "e2_rolling_horizon_mae": [float(value) for value in (rolling_horizon_mae or [])],
        "e2_num_windows": e2.get("num_windows") if e2 else None,
    }


def _read_horizon_mae(path: Path) -> list[float]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["horizon_step"]))
    return [float(row["mae_z"]) for row in rows]


def _save_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "run_dir",
        "context_len",
        "pred_len",
        "mse_z",
        "mae_z",
        "rmse_z",
        "mse_z_system_balanced",
        "mae_z_system_balanced",
        "rmse_z_system_balanced",
        "mae_z_ci95_low",
        "mae_z_ci95_high",
        "pearson",
        "unit_circle_error",
        "candidate_train_windows",
        "candidate_val_windows",
        "candidate_test_windows",
        "sampled_train_windows_per_epoch",
        "sampled_val_windows",
        "sampled_e1_test_windows",
        "sampled_e2_rolling_windows",
        "block_1_mae",
        "block_2_mae",
        "block_3_mae",
        "block_4_mae",
        "error_amplification_ratio",
        "e2_num_windows",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            ci = record.get("bootstrap_ci95", {}).get("mae_z", {})
            record["mae_z_ci95_low"] = ci.get("ci95_low")
            record["mae_z_ci95_high"] = ci.get("ci95_high")
            sampling = record.get("sampling_metadata", {})
            for name in fields:
                if name.startswith("candidate_") or name.startswith("sampled_"):
                    record[name] = sampling.get(name)
            writer.writerow({field: record.get(field) for field in fields})


def _save_curve_csv(path: Path, records: list[dict[str, Any]], key: str) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        for step, value in enumerate(record[key], start=1):
            rows.append({"model": record["model"], "horizon_step": step, "mae_z": value})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "horizon_step", "mae_z"])
        writer.writeheader()
        writer.writerows(rows)


def _save_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Task E E1/E2 model summary",
        "",
        "| Model | E1 MSE_z (system) | E1 MAE_z (system) | E1 RMSE_z (system) | MAE 95% CI | Pearson | Unit-circle error | E2 block 1 MAE | E2 block 4 MAE | Amplification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        ci = record.get("bootstrap_ci95", {}).get("mae_z", {})
        ci_text = f"[{ci.get('ci95_low')}, {ci.get('ci95_high')}]"
        lines.append(
            "| {model} | {mse_z_system_balanced} | {mae_z_system_balanced} | "
            "{rmse_z_system_balanced} | {ci_text} | {pearson} | {unit_circle_error} | "
            "{block_1_mae} | {block_4_mae} | {error_amplification_ratio} |".format(
                **record,
                ci_text=ci_text,
            )
        )
    lines.extend(
        [
            "",
            "E1 measures one-shot 512 -> 128 immediate-future accuracy.",
            "E2 measures recursive 128-step blocks up to global horizon 512 without ground-truth context updates.",
            "Primary scalar and horizon metrics are system-balanced: system -> trajectory -> window.",
            "Confidence intervals use system-level bootstrap resampling.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_one_shot(path: Path, records: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for record in records:
        values = np.asarray(record["e1_horizon_mae"], dtype=np.float64)
        if values.size == 0:
            continue
        axis.plot(np.arange(1, values.size + 1), values, linewidth=1.8, label=record["model"])
        plotted = True
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("MAE_z")
    axis.set_title("Task E E1 one-shot horizon MAE")
    if plotted:
        axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_rolling(path: Path, records: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for record in records:
        values = np.asarray(record["e2_rolling_horizon_mae"], dtype=np.float64)
        if values.size == 0:
            continue
        axis.plot(np.arange(1, values.size + 1), values, linewidth=1.8, label=record["model"])
        plotted = True
    for boundary in (128, 256, 384):
        axis.axvline(boundary, color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Global forecast horizon")
    axis.set_ylabel("MAE_z")
    axis.set_title("Task E E2 recursive rolling horizon MAE")
    if plotted:
        axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
