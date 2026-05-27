from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_d_lyapunov import DEFAULT_VALID_PAIRS, PipelineConfig, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task D: system-level semimajor-axis divergence regression.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1e3)
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--censoring-rule", choices=["median_pair", "all_pairs"], default="median_pair")
    parser.add_argument("--feature-aggregation", choices=["first", "mean_std"], default="first")
    parser.add_argument("--exclude-censored", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task_d_lyapunov")
    parser.add_argument("--allow-incomplete-valid-group", action="store_true")
    parser.add_argument("--completion-table-name", type=str, default="completed_systems_sampling_2.txt")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--include-prefix-features", action="store_true")
    parser.add_argument("--prefix-len", type=int, default=200)
    parser.add_argument("--include-positions-velocities", action="store_true")
    parser.add_argument("--sequence-aggregation", choices=["mean_std", "flatten"], default="mean_std")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--cnn-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PipelineConfig(
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        group_size=args.group_size,
        valid_pairs=DEFAULT_VALID_PAIRS,
        threshold=args.threshold,
        aggregation=args.aggregation,
        censoring_rule=args.censoring_rule,
        feature_aggregation=args.feature_aggregation,
        exclude_censored=args.exclude_censored,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=args.output_dir,
        require_complete_valid_group=not args.allow_incomplete_valid_group,
        completion_table_name=args.completion_table_name,
        max_shards=args.max_shards,
        max_systems=args.max_systems,
        include_prefix_features=args.include_prefix_features,
        prefix_len=args.prefix_len,
        include_positions_velocities=args.include_positions_velocities,
        sequence_aggregation=args.sequence_aggregation,
        run_mlp=not args.no_mlp,
        run_cnn=not args.no_cnn,
        mlp_epochs=args.mlp_epochs,
        cnn_epochs=args.cnn_epochs,
        nn_batch_size=args.batch_size,
        nn_lr=args.lr,
        nn_weight_decay=args.weight_decay,
        nn_dropout=args.dropout,
        nn_val_fraction=args.val_fraction,
        nn_patience=args.patience,
        device=args.device,
    )
    run_pipeline(config)
    print(f"Task D complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
