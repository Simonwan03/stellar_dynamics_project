from __future__ import annotations

import argparse
from pathlib import Path

from .cache import build_preparation_cache_signature, save_prepared_cache
from .data import prepare_forecasting_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and cache Task E HDF5 window records and the train-only scaler."
    )
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cache files.")
    parser.add_argument("--data-glob", "--data_glob", dest="data_glob", type=str, default="dataset_v3/results_batch_*_sampling_2.hdf5")
    parser.add_argument("--representation", choices=["orbital15", "raw_state18", "trajectory"], default="orbital15")
    parser.add_argument("--context-len", "--context_len", dest="context_len", type=int, default=512)
    parser.add_argument("--pred-len", "--pred_len", dest="pred_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--split-seed", "--split_seed", dest="split_seed", type=int, default=42)
    parser.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=32)
    parser.add_argument(
        "--completion-table",
        "--completion_table",
        dest="completion_table",
        type=Path,
        default=Path("dataset_v3/completed_systems_sampling_2.txt"),
    )
    parser.add_argument(
        "--require-complete-systems",
        "--require_complete_systems",
        dest="require_complete_systems",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-frac", "--train_frac", dest="train_frac", type=float, default=0.7)
    parser.add_argument("--val-frac", "--val_frac", dest="val_frac", type=float, default=0.15)
    parser.add_argument("--max-trajectories", "--max_trajectories", dest="max_trajectories", type=int, default=0)
    parser.add_argument(
        "--max-variations-per-system",
        "--max_variations_per_system",
        dest="max_variations_per_system",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-train-variations-per-system",
        "--max_train_variations_per_system",
        dest="max_train_variations_per_system",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-windows-per-trajectory",
        "--max_windows_per_trajectory",
        dest="max_windows_per_trajectory",
        type=int,
        default=0,
    )
    parser.add_argument("--window-sampling", "--window_sampling", dest="window_sampling", choices=["random", "sequential"], default="sequential")
    parser.add_argument(
        "--drop-nan-windows",
        "--drop_nan_windows",
        dest="drop_nan_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-scaler-trajectories", "--max_scaler_trajectories", dest="max_scaler_trajectories", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of worker threads for finite-window scanning and scaler fitting.",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preparation = build_preparation_cache_signature(
        data_glob=args.data_glob,
        representation=args.representation,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        group_size=args.group_size,
        completion_table=args.completion_table,
        require_complete_systems=args.require_complete_systems,
        max_trajectories=args.max_trajectories,
        max_variations_per_system=args.max_variations_per_system,
        max_train_variations_per_system=args.max_train_variations_per_system,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        window_sampling=args.window_sampling,
        drop_nan_windows=args.drop_nan_windows,
        max_scaler_trajectories=args.max_scaler_trajectories,
    )
    prepared = prepare_forecasting_data(
        data_glob=args.data_glob,
        representation=args.representation,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        group_size=args.group_size,
        completion_table=args.completion_table,
        require_complete_systems=args.require_complete_systems,
        max_trajectories=args.max_trajectories,
        max_variations_per_system=args.max_variations_per_system,
        max_train_variations_per_system=args.max_train_variations_per_system,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        window_sampling=args.window_sampling,
        drop_nan_windows=args.drop_nan_windows,
        max_scaler_trajectories=args.max_scaler_trajectories,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    try:
        save_prepared_cache(prepared, args.cache_dir, preparation, overwrite=args.overwrite)
    finally:
        for dataset in (prepared.train_dataset, prepared.val_dataset, prepared.test_dataset):
            dataset.close()
    print(f"[data] prepared cache written to {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
