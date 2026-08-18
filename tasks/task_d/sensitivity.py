from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_d.pipeline import PipelineConfig, discover_result_paths, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task D threshold and pair-aggregation sensitivity grid."
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/task_d_proxy_sensitivity")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(1e3, 3e3, 1e4, 3e4),
    )
    parser.add_argument(
        "--aggregations",
        choices=["min", "median"],
        nargs="+",
        default=("min", "median"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _threshold_slug(value: float) -> str:
    return f"{value:g}".replace("+", "").replace(".", "p")


def _first_positive_stored_time(dataset_dir: Path, sampling_id: int) -> float:
    paths = discover_result_paths(dataset_dir, sampling_id)
    if not paths:
        return np.nan
    with h5py.File(paths[0], "r") as handle:
        times = np.asarray(handle["times_complete"][0], dtype=np.float64)
    positive = times[np.isfinite(times) & (times > 0.0)]
    return float(positive.min()) if positive.size else np.nan


def _summarize_cell(cell_dir: Path, first_time: float) -> list[dict[str, object]]:
    with open(cell_dir / "metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    labels = pd.read_csv(cell_dir / "task_d_labels.csv")
    finite = labels["tau_system"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    uncensored = ~labels["system_censored"].astype(bool).to_numpy()
    common = {
        "threshold": float(metrics["config"]["threshold"]),
        "aggregation": str(metrics["config"]["aggregation"]),
        "num_systems": int(len(labels)),
        "num_uncensored": int(uncensored.sum()),
        "censored_fraction": float(1.0 - uncensored.mean()),
        "first_stored_time": first_time,
        "first_time_fraction": (
            float(np.mean(finite == first_time)) if finite.size and np.isfinite(first_time) else np.nan
        ),
        "within_three_steps_fraction": (
            float(np.mean(finite <= 3.0 * first_time))
            if finite.size and np.isfinite(first_time)
            else np.nan
        ),
        "median_tau": float(np.median(finite)) if finite.size else np.nan,
        "p90_tau": float(np.quantile(finite, 0.90)) if finite.size else np.nan,
        "p99_tau": float(np.quantile(finite, 0.99)) if finite.size else np.nan,
    }
    rows: list[dict[str, object]] = []
    for model in metrics["model_summaries"]:
        row = dict(common)
        row.update(
            {
                "model": model["model_name"],
                "mae_log_tau": model["mae_log_tau"],
                "rmse_log_tau": model["rmse_log_tau"],
                "spearman": model["spearman"],
                "factor_2_accuracy": model["factor_2_accuracy"],
                "factor_3_accuracy": model["factor_3_accuracy"],
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_time = _first_positive_stored_time(args.dataset_dir, args.sampling_id)
    summary_rows: list[dict[str, object]] = []

    for threshold in args.thresholds:
        for aggregation in args.aggregations:
            cell_dir = (
                args.output_dir
                / f"threshold_{_threshold_slug(threshold)}"
                / f"aggregation_{aggregation}"
            )
            metrics_path = cell_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"Reusing completed cell: {cell_dir}", flush=True)
            else:
                print(
                    f"Running threshold={threshold:g}, aggregation={aggregation} -> {cell_dir}",
                    flush=True,
                )
                config = PipelineConfig(
                    task_setting="initial",
                    target_mode="lyapunov_proxy",
                    label_observable="smax",
                    dataset_dir=args.dataset_dir,
                    sampling_id=args.sampling_id,
                    threshold=float(threshold),
                    aggregation=aggregation,
                    censoring_rule="all_pairs" if aggregation == "min" else "median_pair",
                    test_size=args.test_size,
                    random_state=args.seed,
                    output_dir=cell_dir,
                    max_shards=args.max_shards,
                    max_systems=args.max_systems,
                    run_mlp=not args.no_mlp,
                    mlp_epochs=args.mlp_epochs,
                    nn_batch_size=args.batch_size,
                    device=args.device,
                    force_recompute_labels=args.force,
                )
                run_pipeline(config)
            summary_rows.extend(_summarize_cell(cell_dir, first_time))
            pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["threshold", "aggregation", "model"])
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Sensitivity summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
