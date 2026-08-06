from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_d.pipeline import DEFAULT_VALID_PAIRS, PipelineConfig, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task D: system-level divergence regression.")
    parser.add_argument(
        "--task-setting",
        choices=["initial", "initial_prefix", "prefix"],
        default="initial",
        help="initial uses IC only; initial_prefix combines orbital15 with an early trajectory and future-only label.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["min_instability", "lambda_fast", "lyapunov_proxy", "crossing_upper"],
        default="min_instability",
        help="Regression target: minimum instability time, log(lambda_fast), median proxy, or crossing upper bound.",
    )
    parser.add_argument(
        "--label-observable",
        choices=["aratio", "phase_space"],
        default="aratio",
        help="Observable used to construct the label; aratio uses aratios_complete only.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1e3)
    parser.add_argument("--growth-lower", type=float, default=1.0)
    parser.add_argument("--growth-upper", type=float, default=100.0)
    parser.add_argument(
        "--growth-sensitivity-uppers",
        type=float,
        nargs="+",
        default=(10.0, 30.0, 100.0),
        help="Upper R(t) bounds for slope-sensitivity fits; lower bound is --growth-lower.",
    )
    parser.add_argument("--min-growth-points", type=int, default=5)
    parser.add_argument(
        "--lambda-r2-min",
        type=float,
        default=None,
        help="Minimum pair-fit R^2; defaults to 0 for aratio and 0.8 for phase_space.",
    )
    parser.add_argument(
        "--min-valid-lambda-pairs",
        type=int,
        default=None,
        help="Minimum valid pair count; defaults to 1 for min_instability and 8 otherwise.",
    )
    parser.add_argument(
        "--max-pair-log-mad",
        type=float,
        default=None,
        help="Maximum pair log-slope MAD; defaults to 2 for aratio and 0.75 for phase_space.",
    )
    parser.add_argument(
        "--lyapunov-aggregation",
        choices=["median", "mean", "max"],
        default=None,
        help="How to aggregate valid pair growth rates; defaults to max for min_instability, otherwise median.",
    )
    parser.add_argument(
        "--require-pair-consistency",
        action="store_true",
        help="Require pair log-slope consistency; enabled by default for lyapunov_proxy only.",
    )
    parser.add_argument("--aggregation", choices=["median", "mean", "min"], default="median")
    parser.add_argument("--censoring-rule", choices=["median_pair", "all_pairs"], default="median_pair")
    parser.add_argument(
        "--initial-representation",
        choices=["orbital15", "full"],
        default="orbital15",
        help="Initial-condition input representation; orbital15 uses the 15-dimensional IC vector.",
    )
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
    parser.add_argument("--force-recompute", action="store_true", help="Recompute labels even if cached CSV exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        suffix = (
            "initial"
            if args.task_setting == "initial"
            else (
                f"initial_prefix_{args.prefix_len}"
                if args.task_setting == "initial_prefix"
                else f"prefix_tobs_{args.prefix_len}"
            )
        )
        output_dir = ROOT / "results" / f"task_d_{suffix}"
    lambda_r2_min = (
        args.lambda_r2_min
        if args.lambda_r2_min is not None
        else (0.0 if args.label_observable == "aratio" else 0.8)
    )
    max_pair_log_mad = (
        args.max_pair_log_mad
        if args.max_pair_log_mad is not None
        else (2.0 if args.label_observable == "aratio" else 0.75)
    )
    lyapunov_aggregation = (
        args.lyapunov_aggregation
        if args.lyapunov_aggregation is not None
        else ("max" if args.target_mode in {"min_instability", "lambda_fast"} else "median")
    )
    min_valid_lambda_pairs = (
        args.min_valid_lambda_pairs
        if args.min_valid_lambda_pairs is not None
        else (1 if args.target_mode in {"min_instability", "lambda_fast"} else 8)
    )
    require_pair_consistency = (
        args.require_pair_consistency or args.target_mode == "lyapunov_proxy"
    )
    config = PipelineConfig(
        task_setting=args.task_setting,
        target_mode=args.target_mode,
        label_observable=args.label_observable,
        lyapunov_aggregation=lyapunov_aggregation,
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        group_size=args.group_size,
        valid_pairs=DEFAULT_VALID_PAIRS,
        threshold=args.threshold,
        growth_lower=args.growth_lower,
        growth_upper=args.growth_upper,
        growth_sensitivity_uppers=tuple(args.growth_sensitivity_uppers),
        min_growth_points=args.min_growth_points,
        lambda_r2_min=lambda_r2_min,
        min_valid_lambda_pairs=min_valid_lambda_pairs,
        max_pair_log_mad=max_pair_log_mad,
        require_pair_consistency=require_pair_consistency,
        aggregation=args.aggregation,
        censoring_rule=args.censoring_rule,
        initial_representation=args.initial_representation,
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
        force_recompute_labels=args.force_recompute,
    )
    run_pipeline(config)
    print(f"Task D complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
