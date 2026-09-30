"""Run Task D crossing-threshold sensitivity experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from tasks.task_d.quantile import RegressionConfig, run_regression_pipeline


ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task D single-target sensitivity over e^k crossing labels."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset_sampling_3_prefix1024",
    )
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--threshold-powers", type=float, nargs="+", default=(4.0, 5.0, 6.0))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/task_d_median_decorrelation_sensitivity",
    )
    parser.add_argument("--models", default="constant,xgboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    for power in args.threshold_powers:
        cell = args.output_dir / f"e{power:g}".replace(".", "p")
        if (cell / "metrics.json").exists() and not args.force:
            print(f"[skip] {cell}")
            continue
        print(f"[run] threshold=e^{power:g} -> {cell}")
        run_regression_pipeline(
            RegressionConfig(
                dataset_dir=args.dataset_dir,
                output_dir=cell,
                sampling_id=args.sampling_id,
                threshold_power=power,
                models=models,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
                device=args.device,
                max_shards=args.max_shards,
                max_systems=args.max_systems,
            )
        )


if __name__ == "__main__":
    main()
