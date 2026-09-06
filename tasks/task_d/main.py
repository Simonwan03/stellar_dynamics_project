from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_d.quantile import QuantileConfig, run_quantile_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task D: predict q25/q50/q75 of the system-level semimajor-axis-ratio "
            "Lyapunov-time proxy from initial conditions only."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset_sampling_3_prefix1024",
    )
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument(
        "--completion-table-name",
        default="completed_systems_sampling_3.txt",
    )
    parser.add_argument(
        "--threshold-power",
        type=float,
        default=5.0,
        help="Use the crossing G=exp(k); the pair proxy is t_cross/k (default: k=5).",
    )
    parser.add_argument(
        "--min-valid-pairs",
        type=int,
        default=8,
        help="Minimum observed pair crossings required for a system label.",
    )
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Use the first stored crossing instead of log-growth interpolation.",
    )
    parser.add_argument(
        "--models",
        default="constant,gbr,mlp,deepsets",
        help="Comma-separated subset of constant,gbr,xgboost,mlp,deepsets.",
    )
    parser.add_argument(
        "--tabular-features",
        choices=("summary", "first"),
        default="summary",
        help=(
            "Input for tabular models and MLP: mean/std/min/max across 16 nominal "
            "variations, or only the first variation's 15 features."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/task_d_initial_quantiles",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    config = QuantileConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sampling_id=args.sampling_id,
        completion_table_name=args.completion_table_name,
        threshold_power=args.threshold_power,
        min_valid_pairs=args.min_valid_pairs,
        interpolate_crossings=not args.no_interpolation,
        models=models,
        tabular_features=args.tabular_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        max_shards=args.max_shards,
        max_systems=args.max_systems,
    )
    result = run_quantile_pipeline(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Task D complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
