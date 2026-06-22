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
    parser = argparse.ArgumentParser(description="Task D: system-level phase-space divergence regression.")
    parser.add_argument("--task-setting", choices=["initial", "prefix"], default="initial")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1e3)
    parser.add_argument("--growth-lower", type=float, default=1.0)
    parser.add_argument("--growth-upper", type=float, default=100.0)
    parser.add_argument("--min-growth-points", type=int, default=5)
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--censoring-rule", choices=["median_pair", "all_pairs"], default="median_pair")
    parser.add_argument("--feature-aggregation", choices=["first", "mean_std"], default="first")
    parser.add_argument("--exclude-censored", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete-valid-group", action="store_true")
    parser.add_argument("--completion-table-name", type=str, default="completed_systems_sampling_2.txt")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=200)
    parser.add_argument("--prefix-time-stride", type=int, default=1)
    parser.add_argument("--sequence-aggregation", choices=["mean", "mean_std"], default="mean_std")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--no-cnn-orbital", action="store_true")
    parser.add_argument("--no-cnn-phase-space", action="store_true")
    parser.add_argument("--no-set-cnn", action="store_true")
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--cnn-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        suffix = "initial" if args.task_setting == "initial" else f"prefix_tobs_{args.prefix_len}"
        output_dir = ROOT / "results" / f"task_d_{suffix}"
    config = PipelineConfig(
        task_setting=args.task_setting,
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        group_size=args.group_size,
        valid_pairs=DEFAULT_VALID_PAIRS,
        threshold=args.threshold,
        growth_lower=args.growth_lower,
        growth_upper=args.growth_upper,
        min_growth_points=args.min_growth_points,
        aggregation=args.aggregation,
        censoring_rule=args.censoring_rule,
        feature_aggregation=args.feature_aggregation,
        exclude_censored=args.exclude_censored,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=output_dir,
        require_complete_valid_group=not args.allow_incomplete_valid_group,
        completion_table_name=args.completion_table_name,
        max_shards=args.max_shards,
        max_systems=args.max_systems,
        prefix_len=args.prefix_len,
        prefix_time_stride=args.prefix_time_stride,
        sequence_aggregation=args.sequence_aggregation,
        run_mlp=not args.no_mlp,
        run_cnn_orbital=not args.no_cnn_orbital,
        run_cnn_phase_space=not args.no_cnn_phase_space,
        run_set_cnn=not args.no_set_cnn,
        mlp_epochs=args.mlp_epochs,
        cnn_epochs=args.cnn_epochs,
        nn_batch_size=args.batch_size,
        nn_lr=args.lr,
        nn_weight_decay=args.weight_decay,
        nn_dropout=args.dropout,
        nn_val_fraction=args.val_fraction,
        nn_patience=args.patience,
        loss=args.loss,
        huber_beta=args.huber_beta,
        device=args.device,
    )
    run_pipeline(config)
    print(f"Task D complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
