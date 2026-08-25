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


MODEL_DISPLAY_NAMES = {
    "linear_extrapolation": "Linear extrapolation",
    "chronos2": "Chronos2",
    "panda_zero_shot": "PANDA zero-shot",
    "itransformer": "iTransformer",
    "toto2": "TOTO2",
    "tirex2": "TiRex2",
}
MODEL_ORDER = (
    "linear_extrapolation",
    "chronos2",
    "panda_zero_shot",
    "itransformer",
    "toto2",
    "tirex2",
)
MODEL_COLORS = {
    "linear_extrapolation": "#000000",
    "chronos2": "#0072B2",
    "panda_zero_shot": "#D55E00",
    "itransformer": "#E69F00",
    "toto2": "#009E73",
    "tirex2": "#CC79A7",
}

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
            json.dumps(
                {
                    key: value
                    for key, value in record.get("sampling_metadata", {}).items()
                    if key not in {"last_train_epoch", "last_train_epoch_windows"}
                },
                sort_keys=True,
            ),
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
    _plot_rolling(output_dir / "rolling_horizon_mae.pdf", records)
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
        ordered_records = sorted(
            records,
            key=lambda record: MODEL_ORDER.index(record["model"])
            if record["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
        )
        for record in ordered_records:
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
        model = record["model"]
        axis.plot(
            np.arange(1, values.size + 1),
            values,
            color=MODEL_COLORS.get(model),
            linewidth=1.8,
            label=MODEL_DISPLAY_NAMES.get(model, model),
        )
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
    try:
        import scienceplots  # noqa: F401
    except ModuleNotFoundError:
        style_context: list[str] = []
    else:
        style_context = ["science", "no-latex"]

    with plt.style.context(style_context):
        plt.rcParams.update(
            {
                "font.size": 8,
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "axes.labelsize": 8.5,
                "axes.titlesize": 9,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
                "legend.fontsize": 7.5,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "lines.linewidth": 1.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figure, axis = plt.subplots(figsize=(7.0, 3.0))
        plotted = False
        max_horizon = 0
        ordered_records = sorted(
            records,
            key=lambda record: MODEL_ORDER.index(record["model"])
            if record["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
        )
        for record in ordered_records:
            values = np.asarray(record["e2_rolling_horizon_mae"], dtype=np.float64)
            if values.size == 0:
                continue
            horizon = np.arange(1, values.size + 1)
            model = record["model"]
            axis.plot(
                horizon,
                values,
                color=MODEL_COLORS.get(model),
                linewidth=1.8,
                label=MODEL_DISPLAY_NAMES.get(model, model),
            )
            max_horizon = max(max_horizon, values.size)
            plotted = True

        if not plotted:
            plt.close(figure)
            return

        for boundary in (128, 256, 384):
            if boundary < max_horizon:
                axis.axvline(boundary, color="0.5", linestyle="--", linewidth=0.7, alpha=0.6)
        axis.set_xlim(1, max_horizon)
        axis.set_ylim(bottom=0.0)
        axis.set_xticks([step for step in (128, 256, 384, 512) if step <= max_horizon])
        axis.set_xlabel("Forecast horizon [steps]")
        axis.set_ylabel("MAE (standardized z-scale)")
        axis.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.5)
        axis.grid(axis="y", alpha=0.15, linewidth=0.5)
        axis.grid(axis="x", visible=False)
        figure.tight_layout()
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if path.suffix.lower() == ".png":
            save_kwargs["dpi"] = 300
        figure.savefig(path, **save_kwargs)
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
