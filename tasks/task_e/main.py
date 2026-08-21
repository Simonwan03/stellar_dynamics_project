from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_e.foundation import (  # noqa: E402
    build_chronos2_predictor,
    build_chronos_predictor,
    build_panda_predictor,
    build_tirex2_predictor,
    build_toto2_predictor,
    build_timesfm_predictor,
    is_chronos2_model,
)
from tasks.task_e.data import (  # noqa: E402
    DynamicTrajectoryWindowDataset,
    PreparedForecastData,
    RollingWindowDataset,
    count_rolling_window_candidates,
    prepare_forecasting_data,
    select_deterministic_windows,
)
from tasks.task_e.cache import (  # noqa: E402
    build_preparation_cache_signature,
    load_prepared_cache,
)
from tasks.task_e.metrics import (  # noqa: E402
    HierarchicalErrorAccumulator,
    bootstrap_block_ci,
    compute_forecast_metrics,
    save_rows,
    summarize_horizon_trends,
)
from tasks.task_e.models import build_supervised_model  # noqa: E402
from tasks.task_e.panda_dependency import panda_dependency_metadata  # noqa: E402


SUPERVISED_MODELS = {"tcn", "patchtst", "transformer", "itransformer"}
FOUNDATION_MODELS = {"chronos2", "panda_zero_shot", "tirex2", "toto2"}
# Kept out of the public model choices. The implementation and CLI arguments
# remain import-compatible with historical adapter checkpoints, but the
# current Task E protocol deliberately does not train or evaluate adapters.
LEGACY_ADAPTER_MODEL = "panda_adapter"
ALL_MODELS = {
    "ridge",
    *SUPERVISED_MODELS,
    *FOUNDATION_MODELS,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Task E past-window -> future-window forecasting benchmark.")
    parser.add_argument("--data_glob", type=str, default="dataset_v3/results_batch_*_sampling_2.hdf5")
    parser.add_argument(
        "--representation",
        choices=["orbital15", "raw_state18", "trajectory"],
        default="orbital15",
    )
    parser.add_argument("--model", choices=sorted(ALL_MODELS), default="ridge")
    parser.add_argument("--context_len", "--context-len", dest="context_len", type=int, default=512)
    parser.add_argument("--pred_len", "--pred-len", dest="pred_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument(
        "--completion_table",
        type=Path,
        default=Path("dataset_v3/completed_systems_sampling_2.txt"),
        help="Completion metadata used for valid-trajectory filtering and pre-event window cutoffs.",
    )
    parser.add_argument(
        "--require_complete_systems",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only systems whose full variation group is valid in the completion table.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument(
        "--max_variations_per_system",
        type=int,
        default=0,
        help=(
            "Randomly keep at most this many eligible trajectories from every grouped "
            "system before train/validation/test splitting (0 keeps all eligible variations)."
        ),
    )
    parser.add_argument(
        "--max_train_variations_per_system",
        type=int,
        default=0,
        help=(
            "Randomly keep at most this many training trajectories from each grouped "
            "system after splitting (0 keeps all training variations)."
        ),
    )
    parser.add_argument("--max_windows_per_trajectory", type=int, default=0)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=0)
    parser.add_argument("--max_test_windows", type=int, default=0)
    parser.add_argument(
        "--train-windows-per-trajectory",
        "--train_windows_per_trajectory",
        dest="train_windows_per_trajectory",
        type=int,
        default=8,
        help="Random candidate windows sampled per training trajectory each epoch; 0 keeps all.",
    )
    parser.add_argument(
        "--val-windows-per-trajectory",
        "--val_windows_per_trajectory",
        dest="val_windows_per_trajectory",
        type=int,
        default=4,
        help="Deterministic validation windows selected per trajectory; 0 keeps all.",
    )
    parser.add_argument(
        "--test-windows-per-trajectory",
        "--test_windows_per_trajectory",
        dest="test_windows_per_trajectory",
        type=int,
        default=4,
        help="Deterministic E1 test windows selected per trajectory; 0 keeps all.",
    )
    parser.add_argument(
        "--rolling-windows-per-trajectory",
        "--rolling_windows_per_trajectory",
        dest="rolling_windows_per_trajectory",
        type=int,
        default=2,
        help="Deterministic E2 rolling starts selected per trajectory; 0 keeps all.",
    )
    parser.add_argument("--max_scaler_trajectories", type=int, default=0)
    parser.add_argument("--window_sampling", choices=["random", "sequential"], default="sequential")
    parser.add_argument("--drop_nan_windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--num_eval_windows",
        type=int,
        default=0,
        help="Legacy shared validation/test cap; 0 keeps all prepared windows for every model.",
    )
    parser.add_argument(
        "--rolling-blocks",
        "--rolling_blocks",
        dest="rolling_blocks",
        type=int,
        default=4,
        help="Number of recursive prediction blocks for E2 rolling evaluation.",
    )
    parser.add_argument(
        "--run-rolling",
        "--run_rolling",
        dest="run_rolling",
        action="store_true",
        help="Run E2 rolling evaluation after the E1 test evaluation without retraining.",
    )
    parser.add_argument(
        "--rolling-batch-size",
        "--rolling_batch_size",
        dest="rolling_batch_size",
        type=int,
        default=0,
        help="Rolling evaluation batch size; 0 uses the model's normal batch size.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap_samples",
        dest="bootstrap_samples",
        type=int,
        default=1000,
        help="System-level bootstrap resamples for reported 95%% confidence intervals; 0 disables.",
    )
    parser.add_argument("--foundation_batch_size", type=int, default=64)
    parser.add_argument("--timesfm_model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--chronos_model_id", type=str, default="amazon/chronos-bolt-small")
    parser.add_argument("--chronos_variant", choices=["bolt", "chronos2"], default="bolt")
    parser.add_argument("--chronos_num_samples", type=int, default=20)
    parser.add_argument("--panda_model_id", type=str, default="GilpinLab/panda-72M")
    parser.add_argument("--tirex2_model_id", type=str, default="NX-AI/TiRex-2")
    parser.add_argument("--toto2_model_id", type=str, default="Datadog/Toto-2.0-22m")
    parser.add_argument("--freeze_panda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_hidden_dim", type=int, default=None)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument(
        "--adapter_decoder",
        "--adapter-decoder",
        dest="adapter_decoder",
        choices=["mlp", "gru"],
        default="mlp",
        help="Panda adapter decoder; GRU supports arbitrary output_len at model inference.",
    )
    parser.add_argument("--adapter_gru_layers", "--adapter-gru-layers", dest="adapter_gru_layers", type=int, default=1)
    parser.add_argument("--num_train_windows_for_adapter", type=int, default=4096)
    parser.add_argument(
        "--hf-local-files-only",
        "--hf_local_files_only",
        dest="hf_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument(
        "--distribution_bins",
        type=int,
        default=64,
        help="Number of pooled quantile bins for KL divergence and Hellinger distance metrics.",
    )
    parser.add_argument(
        "--distribution_epsilon",
        type=float,
        default=1e-12,
        help="Additive smoothing used for KL divergence and Hellinger distance histograms.",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Checkpoint to restore before training. --epochs is the number of additional epochs.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Restore --resume_from and run validation/test evaluation without updating model weights.",
    )
    parser.add_argument("--prediction_samples", type=int, default=16)
    parser.add_argument("--log_every_batches", type=int, default=100)
    parser.add_argument(
        "--prepared-cache",
        "--prepared_cache",
        dest="prepared_cache",
        type=Path,
        default=None,
        help="Load window records/scaler from a standalone prepared cache.",
    )
    return parser


def _preparation_signature_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_preparation_cache_signature(
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


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.rolling_blocks <= 0:
        raise ValueError(f"--rolling-blocks must be positive, got {args.rolling_blocks}.")
    if args.run_rolling and args.pred_len <= 0:
        raise ValueError("--pred-len must be positive when running rolling evaluation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(
        f"[task_e] starting model={args.model} representation={args.representation} "
        f"device={device} amp={amp_enabled} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[task_e] CUDA device: {torch.cuda.get_device_name(device)}", flush=True)

    prepared = (
        load_prepared_cache(
            args.prepared_cache,
            _preparation_signature_from_args(args),
            verbose=True,
        )
        if args.prepared_cache is not None
        else prepare_forecasting_data(
            data_glob=args.data_glob,
            representation=args.representation,
            context_len=args.context_len,
            pred_len=args.pred_len,
            stride=args.stride,
            split_seed=args.split_seed,
            group_size=args.group_size,
            completion_table=args.completion_table,
            require_complete_systems=args.require_complete_systems,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_trajectories=args.max_trajectories,
            max_variations_per_system=args.max_variations_per_system,
            max_train_variations_per_system=args.max_train_variations_per_system,
            max_windows_per_trajectory=args.max_windows_per_trajectory,
            window_sampling=args.window_sampling,
            drop_nan_windows=args.drop_nan_windows,
            max_scaler_trajectories=args.max_scaler_trajectories,
            verbose=True,
        )
    )
    feature_names = list(prepared.feature_metadata["channel_names"])
    num_features = int(prepared.feature_metadata["num_features"])
    if min(
        args.train_windows_per_trajectory,
        args.val_windows_per_trajectory,
        args.test_windows_per_trajectory,
        args.rolling_windows_per_trajectory,
    ) < 0:
        raise ValueError("Per-trajectory window counts must be non-negative.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")

    # Legacy global caps are applied to the candidate pools first.  The new
    # protocol then samples train/validation/test windows from those pools;
    # with the default zero caps, every eligible candidate remains available.
    candidate_datasets = {
        "train": cap_dataset(prepared.train_dataset, args.max_train_windows, args.seed + 1),
        "val": cap_dataset(prepared.val_dataset, args.max_val_windows, args.seed + 2),
        "test": cap_dataset(prepared.test_dataset, args.max_test_windows, args.seed + 3),
    }
    if args.train_windows_per_trajectory > 0:
        train_dataset: Dataset = DynamicTrajectoryWindowDataset(
            candidate_datasets["train"],
            windows_per_trajectory=args.train_windows_per_trajectory,
            seed=args.seed + 1001,
        )
    else:
        train_dataset = candidate_datasets["train"]
    datasets = {
        "train": train_dataset,
        "val": select_deterministic_windows(
            candidate_datasets["val"],
            args.val_windows_per_trajectory,
            target_positions=(0.10, 0.35, 0.65, 0.90),
        ),
        "test": select_deterministic_windows(
            candidate_datasets["test"],
            args.test_windows_per_trajectory,
            target_positions=(0.10, 0.35, 0.65, 0.90),
        ),
    }
    # This legacy option remains a shared validation/test cap. Applying it
    # after deterministic selection preserves identical samples for all models.
    if args.num_eval_windows > 0:
        datasets["val"] = cap_dataset(datasets["val"], args.num_eval_windows, args.seed + 4)
        datasets["test"] = cap_dataset(datasets["test"], args.num_eval_windows, args.seed + 5)

    rolling_candidate_dataset: Dataset = candidate_datasets["test"]
    if args.num_eval_windows > 0:
        rolling_candidate_dataset = cap_dataset(
            rolling_candidate_dataset,
            args.num_eval_windows,
            args.seed + 5,
        )
    rolling_candidate_count = count_rolling_window_candidates(
        rolling_candidate_dataset,
        context_len=args.context_len,
        block_len=args.pred_len,
        num_blocks=args.rolling_blocks,
        windows_per_trajectory=args.rolling_windows_per_trajectory,
        target_positions=(0.25, 0.75),
    )

    prepared.scaler.save(args.output_dir / "scaler.npz", feature_names)
    save_json(args.output_dir / "feature_metadata.json", prepared.feature_metadata)
    effective_limits = {
        "train": args.max_train_windows,
        "val": args.max_val_windows,
        "test": args.max_test_windows,
    }
    config = build_config(
        args,
        prepared,
        datasets,
        candidate_datasets,
        effective_limits,
        device,
        amp_enabled,
        rolling_candidate_count,
    )
    save_json(args.output_dir / "config.json", config)
    print_dataset_summary(config, feature_names, prepared)
    print("[task_e] building DataLoaders", flush=True)

    loaders = {
        name: build_loader(
            dataset,
            args.batch_size if args.model not in FOUNDATION_MODELS else args.foundation_batch_size,
            args.num_workers,
            shuffle=name == "train" and args.model in SUPERVISED_MODELS,
            device=device,
        )
        for name, dataset in datasets.items()
    }

    forecast_fn: Callable[[np.ndarray], np.ndarray]
    if args.model == "ridge":
        print("[task_e] materializing train windows for ridge", flush=True)
        ridge = Ridge(alpha=args.ridge_alpha)
        train_context, train_target = materialize_dataset(loaders["train"])
        print(f"[task_e] fitting ridge on {train_context.shape[0]} windows", flush=True)
        ridge.fit(train_context.reshape(train_context.shape[0], -1), train_target.reshape(train_target.shape[0], -1))
        with open(args.output_dir / "best_model.pkl", "wb") as handle:
            pickle.dump(ridge, handle)
        val_pred, val_target, _, val_trajectory_ids, _ = predict_ridge(
            ridge, loaders["val"], args.pred_len, num_features, return_metadata=True
        )
        test_pred, test_target, sample_context, test_trajectory_ids, _ = predict_ridge(
            ridge, loaders["test"], args.pred_len, num_features, return_metadata=True
        )
        forecast_fn = lambda context: forecast_ridge_batch(ridge, context, args.pred_len, num_features)
    elif args.model in SUPERVISED_MODELS:
        print(f"[task_e] building supervised model {args.model}", flush=True)
        model = build_supervised_model(
            model_name=args.model,
            num_features=num_features,
            context_len=args.context_len,
            pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
        ).to(device)
        train_model(model, loaders, args, config, prepared, device, amp_enabled)
        val_pred, val_target, _, val_trajectory_ids, _ = predict_model(
            model,
            loaders["val"],
            device,
            amp_enabled,
            args.log_every_batches,
            "val",
            return_metadata=True,
        )
        test_pred, test_target, sample_context, test_trajectory_ids, _ = predict_model(
            model,
            loaders["test"],
            device,
            amp_enabled,
            args.log_every_batches,
            "test",
            return_metadata=True,
        )
        forecast_fn = lambda context: forecast_model_batch(model, context, device, amp_enabled)
    elif args.model == LEGACY_ADAPTER_MODEL:
        raise ValueError(
            "panda_adapter is disabled in the current Task E protocol; "
            "use panda_zero_shot for the native 512 -> 128 Panda baseline."
        )
    else:
        print(f"[task_e] loading zero-shot foundation predictor {args.model}", flush=True)
        predictor, multivariate = build_foundation_predictor(args, device)
        print("[task_e] evaluating validation split", flush=True)
        val_pred, val_target, _, val_trajectory_ids, _ = predict_foundation(
            predictor,
            loaders["val"],
            args.pred_len,
            multivariate,
            args.log_every_batches,
            "val",
            return_metadata=True,
            context_limit=1,
        )
        print("[task_e] evaluating test split", flush=True)
        test_pred, test_target, sample_context, test_trajectory_ids, _ = predict_foundation(
            predictor,
            loaders["test"],
            args.pred_len,
            multivariate,
            args.log_every_batches,
            "test",
            return_metadata=True,
            context_limit=max(int(args.prediction_samples), 1),
        )
        forecast_fn = lambda context: forecast_foundation_batch(
            predictor, context, args.pred_len, multivariate
        )

    print("[task_e] saving validation/test metrics and prediction sample", flush=True)
    val_metrics = save_evaluation(
        args.output_dir,
        "val",
        val_pred,
        val_target,
        prepared,
        feature_names,
        save_detail=False,
        distribution_bins=args.distribution_bins,
        distribution_epsilon=args.distribution_epsilon,
        trajectory_ids=val_trajectory_ids,
        group_size=args.group_size,
        bootstrap_samples=0,
        bootstrap_seed=args.seed + 7001,
    )
    del val_pred, val_target, val_trajectory_ids
    test_metrics = save_evaluation(
        args.output_dir,
        "test",
        test_pred,
        test_target,
        prepared,
        feature_names,
        save_detail=True,
        distribution_bins=args.distribution_bins,
        distribution_epsilon=args.distribution_epsilon,
        trajectory_ids=test_trajectory_ids,
        group_size=args.group_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 7001,
    )
    save_prediction_sample(
        args.output_dir / "predictions_sample.npz",
        sample_context,
        test_target,
        test_pred,
        prepared,
        feature_names,
        args.prediction_samples,
    )
    del test_pred, test_target, sample_context, test_trajectory_ids
    rolling_metrics: dict[str, Any] | None = None
    if args.run_rolling:
        rolling_batch_size = args.rolling_batch_size or (
            args.foundation_batch_size if args.model in FOUNDATION_MODELS else args.batch_size
        )
        print(
            f"[task_e] running E2 rolling evaluation: blocks={args.rolling_blocks}, "
            f"block_len={args.pred_len}, total_horizon={args.pred_len * args.rolling_blocks}",
            flush=True,
        )
        rolling_metrics = run_rolling_evaluation(
            rolling_candidate_dataset,
            forecast_fn,
            context_len=args.context_len,
            block_len=args.pred_len,
            num_blocks=args.rolling_blocks,
            windows_per_trajectory=args.rolling_windows_per_trajectory,
            target_positions=(0.25, 0.75),
            group_size=args.group_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed + 7101,
            batch_size=rolling_batch_size,
            num_workers=args.num_workers,
            log_every_batches=args.log_every_batches,
        )
        save_rolling_evaluation(args.output_dir, rolling_metrics)
        config["rolling_test_window_count"] = rolling_metrics["num_windows"]
        config["sampling_metadata"]["sampled_e2_rolling_windows"] = rolling_metrics["num_windows"]
        save_json(args.output_dir / "config.json", config)
    save_run_summary(args.output_dir, args.model, test_metrics, rolling_metrics, config=config)
    print(
        f"Finished {args.model}: val_mse_z={val_metrics['mse_z']:.6g}, "
        f"test_mse_z={test_metrics['mse_z']:.6g}, "
        f"test_smape_transformed={test_metrics['smape_percent_transformed']:.6g}%, "
        f"test_pearson={test_metrics['pearson']:.6g}, "
        f"test_unit_circle_error={test_metrics['unit_circle_error']}"
    )
    close_prepared(prepared)


def build_config(
    args: argparse.Namespace,
    prepared: PreparedForecastData,
    datasets: dict[str, Dataset],
    candidate_datasets: dict[str, Dataset],
    effective_limits: dict[str, int],
    device: torch.device,
    amp_enabled: bool,
    rolling_candidate_count: int,
) -> dict[str, Any]:
    model_family = "baseline"
    training_mode = "none"
    if args.model in SUPERVISED_MODELS:
        model_family, training_mode = "supervised", "full_supervised"
    elif args.model in FOUNDATION_MODELS:
        model_family, training_mode = "foundation", "zero_shot"
    adapter_gradient_note = None
    sampling_metadata = {
        "num_train_systems": int(np.unique(prepared.split_indices["train"] // args.group_size).size),
        "num_val_systems": int(np.unique(prepared.split_indices["val"] // args.group_size).size),
        "num_test_systems": int(np.unique(prepared.split_indices["test"] // args.group_size).size),
        "num_train_trajectories": int(prepared.split_indices["train"].size),
        "num_val_trajectories": int(prepared.split_indices["val"].size),
        "num_test_trajectories": int(prepared.split_indices["test"].size),
        "candidate_train_windows": int(len(prepared.train_dataset)),
        "candidate_val_windows": int(len(prepared.val_dataset)),
        "candidate_test_windows": int(len(prepared.test_dataset)),
        "candidate_train_windows_after_legacy_cap": int(len(candidate_datasets["train"])),
        "candidate_val_windows_after_legacy_cap": int(len(candidate_datasets["val"])),
        "candidate_test_windows_after_legacy_cap": int(len(candidate_datasets["test"])),
        "sampled_train_windows_per_epoch": int(len(datasets["train"])),
        "sampled_val_windows": int(len(datasets["val"])),
        "sampled_e1_test_windows": int(len(datasets["test"])),
        "planned_e2_rolling_windows": int(rolling_candidate_count),
        "sampled_e2_rolling_windows": int(rolling_candidate_count) if args.run_rolling else 0,
        "train_windows_per_trajectory": int(args.train_windows_per_trajectory),
        "val_windows_per_trajectory": int(args.val_windows_per_trajectory),
        "test_windows_per_trajectory": int(args.test_windows_per_trajectory),
        "rolling_windows_per_trajectory": int(args.rolling_windows_per_trajectory),
        "train_sampling": "dynamic_random_per_trajectory" if args.train_windows_per_trajectory > 0 else "all_candidates",
        "validation_sampling": "deterministic_positions_10_35_65_90" if args.val_windows_per_trajectory > 0 else "all_candidates",
        "e1_test_sampling": "deterministic_positions_10_35_65_90" if args.test_windows_per_trajectory > 0 else "all_candidates",
        "e2_rolling_sampling": "deterministic_positions_25_75" if args.rolling_windows_per_trajectory > 0 else "all_candidates",
        "aggregation_unit": "system",
        "hierarchy": "system -> trajectory -> forecasting_window",
        "bootstrap_unit": "system",
    }
    config = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "resolved_source_paths": prepared.source_paths,
        "resolved_representation": prepared.feature_metadata["representation"],
        "num_features": prepared.feature_metadata["num_features"],
        "feature_names": prepared.feature_metadata["channel_names"],
        "trajectory_count": prepared.trajectory_count,
        "time_steps": prepared.time_steps,
        "split_trajectory_counts": {
            name: int(indices.size) for name, indices in prepared.split_indices.items()
        },
        "split_system_counts": {
            name: int(np.unique(indices // args.group_size).size)
            for name, indices in prepared.split_indices.items()
        },
        "full_window_counts": {
            "train": len(prepared.train_dataset),
            "val": len(prepared.val_dataset),
            "test": len(prepared.test_dataset),
        },
        "used_window_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "sampling_metadata": sampling_metadata,
        "effective_window_limits": effective_limits,
        "scaler_fit_scope": "selected_finite_train_windows_before_completion_event",
        "completion_filter": prepared.feature_metadata.get("completion_filter"),
        "split_unit": prepared.feature_metadata.get("split_unit"),
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "model_family": model_family,
        "training_mode": training_mode,
        "adapter_gradient_note": adapter_gradient_note,
        "itransformer_variant": "simple_variable_token_encoder" if args.model.startswith("itransformer") else None,
        "evaluation_protocol": {
            "E1": {
                "name": "one_shot_future_trajectory_forecasting",
                "context_len": int(args.context_len),
                "pred_len": int(args.pred_len),
            },
            "E2": {
                "name": "recursive_block_rolling_forecasting",
                "enabled": bool(args.run_rolling),
                "blocks": int(args.rolling_blocks),
                "total_horizon": int(args.pred_len * args.rolling_blocks),
            },
            "aggregation": {
                "primary_unit": "system",
                "hierarchy": "system -> trajectory -> forecasting_window",
                "bootstrap_samples": int(args.bootstrap_samples),
                "bootstrap_unit": "system",
            },
        },
    }
    if args.model == "chronos2":
        config["chronos_model_id"] = (
            args.chronos_model_id
            if is_chronos2_model(args.chronos_model_id)
            else "amazon/chronos-2"
        )
        config["chronos_variant"] = "chronos2"
    if args.model == "panda_zero_shot":
        config["panda_dependency"] = panda_dependency_metadata(
            args.panda_model_id,
            args.hf_local_files_only,
        )
    if args.model == "tirex2":
        config["tirex2_model_id"] = getattr(args, "tirex2_model_id", "NX-AI/TiRex-2")
        config["tirex2_inference_mode"] = "native_multivariate"
    if args.model == "toto2":
        config["toto2_model_id"] = getattr(args, "toto2_model_id", "Datadog/Toto-2.0-22m")
        config["toto2_inference_mode"] = "native_multivariate"
    return config


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {requested} was requested, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_dataset_summary(config: dict[str, Any], feature_names: list[str], prepared: PreparedForecastData) -> None:
    print(f"Sources: {len(prepared.source_paths)}, trajectories: {prepared.trajectory_count}, T={prepared.time_steps}")
    print(f"Split trajectories: {config['split_trajectory_counts']}")
    print(f"Candidate windows: {config['sampling_metadata']['candidate_train_windows']}, "
          f"{config['sampling_metadata']['candidate_val_windows']}, "
          f"{config['sampling_metadata']['candidate_test_windows']}")
    print(f"Sampled windows: {config['used_window_counts']}")
    print(f"Sampling metadata: {config['sampling_metadata']}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Scaler mean/std shapes: {prepared.scaler.mean.shape} / {prepared.scaler.std.shape}")


def save_model_config(output_dir: Path, config: dict[str, Any], model: nn.Module) -> None:
    metadata = getattr(model, "adapter_metadata", None)
    if metadata is not None:
        config["adapter_metadata"] = metadata
        save_json(output_dir / "config.json", config)


def cap_dataset(dataset: Dataset, limit: int, seed: int) -> Dataset:
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=limit, replace=False))
    return Subset(dataset, indices.tolist())


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        # DynamicTrajectoryWindowDataset changes its selected indices in the
        # parent process at every epoch, so persistent worker copies would
        # retain stale indices.
        persistent_workers=num_workers > 0 and not isinstance(dataset, DynamicTrajectoryWindowDataset),
    )


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    config: dict[str, Any],
    prepared: PreparedForecastData,
    device: torch.device,
    amp_enabled: bool,
    lr: float | None = None,
) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found.")
    log_parameter_counts(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr or args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext
    criterion = nn.MSELoss()
    history = []
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 1

    if args.resume_from is not None:
        checkpoint_path = args.resume_from.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_checkpoint_state(model, checkpoint)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(f"Resume checkpoint has no optimizer state: {checkpoint_path}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val = float(checkpoint.get("val_mse_z", float("inf")))
        best_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = best_epoch + 1
        print(
            f"Resumed from {checkpoint_path} at epoch {best_epoch}; "
            f"best val_mse_z={best_val:.6g}. Training {args.epochs} additional epoch(s).",
            flush=True,
        )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        set_epoch = getattr(loaders["train"].dataset, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
            config["sampling_metadata"]["last_train_epoch"] = int(epoch)
            config["sampling_metadata"]["last_train_epoch_windows"] = int(
                len(loaders["train"].dataset)
            )
        model.train()
        train_sum = 0.0
        train_count = 0
        started = time.perf_counter()
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            context = batch["context"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                prediction = model(context)
                loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                print(
                    f"Skipping non-finite loss at epoch {epoch}, batch {batch_idx}: "
                    f"loss={float(loss.detach().cpu())}",
                    flush=True,
                )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            else:
                grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
                grad_norm = (
                    torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in grads])
                    )
                    if grads
                    else torch.zeros((), device=device)
                )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                print(
                    f"Skipping non-finite gradients at epoch {epoch}, batch {batch_idx}: "
                    f"grad_norm={float(grad_norm.detach().cpu())}",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * context.shape[0]
            train_count += context.shape[0]
            if should_log(batch_idx, len(loaders["train"]), args.log_every_batches):
                print(
                    f"train epoch {epoch:03d} [{batch_idx}/{len(loaders['train'])}] "
                    f"mse_z={train_sum / train_count:.6g} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        val_pred, val_target, _, val_trajectory_ids, _ = predict_model(
            model,
            loaders["val"],
            device,
            amp_enabled,
            args.log_every_batches,
            f"epoch {epoch:03d} val",
            return_metadata=True,
        )
        val_metrics, _, _ = compute_forecast_metrics(
            val_pred,
            val_target,
            prepared.scaler,
            prepared.feature_metadata["channel_names"],
            distribution_bins=args.distribution_bins,
            distribution_epsilon=args.distribution_epsilon,
            trajectory_ids=val_trajectory_ids,
            group_size=args.group_size,
            bootstrap_samples=0,
            bootstrap_seed=args.seed + 7001,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_mse_z": val_metrics["mse_z"],
            "val_mae_z": val_metrics["mae_z"],
            "val_pearson_z": val_metrics["pearson_z"],
            "val_smape_percent_z": val_metrics["smape_percent_z"],
            "val_spearman_z": val_metrics["spearman_z"],
            "val_mse_raw": val_metrics["mse_raw"],
            "val_mae_raw": val_metrics["mae_raw"],
            "val_pearson_raw": val_metrics["pearson_raw"],
            "val_smape_percent_raw": val_metrics["smape_percent_raw"],
            "val_spearman_raw": val_metrics["spearman_raw"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        save_rows(args.output_dir / "train_log.csv", history)
        print(
            f"Epoch {epoch:03d} | train={row['train_loss']:.6g} | "
            f"val_mse_z={row['val_mse_z']:.6g} | val_mse_raw={row['val_mse_raw']:.6g}",
            flush=True,
        )
        if np.isfinite(val_metrics["mse_z"]) and val_metrics["mse_z"] < best_val:
            best_val = val_metrics["mse_z"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_state, excluded_prefixes = checkpoint_state_dict(model)
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse_z": best_val,
                    "model_state_dict": checkpoint_state,
                    "excluded_frozen_prefixes": excluded_prefixes,
                    "checkpoint_format": "adapter_only" if excluded_prefixes else "full_model",
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": to_jsonable(config),
                },
                args.output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break
    if best_epoch < 0:
        raise RuntimeError("No finite validation checkpoint was produced.")
    checkpoint = torch.load(
        args.output_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    load_checkpoint_state(model, checkpoint)


def checkpoint_state_dict(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[str]]:
    metadata = getattr(model, "adapter_metadata", None)
    if not metadata or bool(metadata.get("backbone_trainable", True)):
        return model.state_dict(), []
    backbone = str(metadata.get("backbone", ""))
    prefix_map = {
        "panda": ["panda."],
    }
    excluded = prefix_map.get(backbone, [])
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not any(name.startswith(prefix) for prefix in excluded)
    }
    return state, excluded


def load_checkpoint_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    excluded = list(checkpoint.get("excluded_frozen_prefixes", []))
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=not excluded)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in excluded)
    ]
    if invalid_missing:
        raise RuntimeError(f"Missing non-backbone checkpoint keys: {invalid_missing}")


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    log_every_batches: int = 0,
    split: str = "eval",
    return_metadata: bool = False,
) -> tuple[Any, ...]:
    model.eval()
    autocast = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext
    predictions, targets, contexts = [], [], []
    trajectory_ids, starts = [], []
    started = time.perf_counter()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        context = batch["context"].to(device, non_blocking=True)
        with autocast():
            prediction = model(context)
        predictions.append(prediction.detach().cpu().float().numpy())
        targets.append(batch["target"].numpy())
        contexts.append(batch["context"].numpy())
        if return_metadata:
            if "trajectory_index" not in batch or "start" not in batch:
                raise ValueError("Evaluation dataset does not provide trajectory/start metadata.")
            trajectory_ids.append(batch["trajectory_index"].numpy())
            starts.append(batch["start"].numpy())
        if should_log(batch_idx, total_batches, log_every_batches):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = batch_idx / elapsed
            eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
            print(
                f"[task_e] model {split} [{batch_idx}/{total_batches}] "
                f"elapsed={elapsed:.1f}s eta={format_duration(eta)}",
                flush=True,
            )
    return concatenate_predictions(
        predictions,
        targets,
        contexts,
        trajectory_ids=trajectory_ids if return_metadata else None,
        starts=starts if return_metadata else None,
    )


def materialize_dataset(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    contexts, targets = [], []
    for batch in loader:
        contexts.append(batch["context"].numpy())
        targets.append(batch["target"].numpy())
    if not contexts:
        raise ValueError("Dataset produced zero windows.")
    return np.concatenate(contexts), np.concatenate(targets)


def forecast_ridge_batch(
    model: Ridge,
    context: np.ndarray,
    pred_len: int,
    num_features: int,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3:
        raise ValueError(f"Expected ridge context [B,T,C], got {context.shape}.")
    prediction = model.predict(context.reshape(context.shape[0], -1))
    prediction = np.asarray(prediction, dtype=np.float32).reshape(
        context.shape[0], pred_len, num_features
    )
    if prediction.shape != (context.shape[0], pred_len, num_features):
        raise ValueError(f"Ridge returned unexpected prediction shape {prediction.shape}.")
    if not np.isfinite(prediction).all():
        raise ValueError("Ridge produced non-finite predictions.")
    return prediction


@torch.no_grad()
def forecast_model_batch(
    model: nn.Module,
    context: np.ndarray,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3:
        raise ValueError(f"Expected model context [B,T,C], got {context.shape}.")
    model.eval()
    autocast = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext
    context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device)
    with autocast():
        prediction = model(context_tensor)
    prediction = prediction.detach().cpu().float().numpy()
    if prediction.ndim != 3 or prediction.shape[0] != context.shape[0] or prediction.shape[2] != context.shape[2]:
        raise ValueError(
            f"Supervised model returned shape {prediction.shape} for context {context.shape}."
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Supervised model produced non-finite predictions.")
    return prediction


def forecast_foundation_batch(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    context: np.ndarray,
    pred_len: int,
    multivariate: bool,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3:
        raise ValueError(f"Expected foundation context [B,T,C], got {context.shape}.")
    if multivariate:
        prediction = predictor(context, pred_len)
    else:
        batch_count, context_len, channels = context.shape
        flattened = context.transpose(0, 2, 1).reshape(batch_count * channels, context_len)
        flat_prediction = predictor(flattened, pred_len)
        prediction = flat_prediction.reshape(batch_count, channels, pred_len).transpose(0, 2, 1)
    prediction = np.asarray(prediction, dtype=np.float32)
    expected = (context.shape[0], pred_len, context.shape[2])
    if prediction.shape != expected:
        raise ValueError(f"Foundation predictor returned shape {prediction.shape}, expected {expected}.")
    if not np.isfinite(prediction).all():
        raise ValueError("Foundation predictor produced non-finite predictions.")
    return prediction


def predict_ridge(
    model: Ridge,
    loader: DataLoader,
    pred_len: int,
    num_features: int,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _predict_ridge(
        model,
        loader,
        pred_len,
        num_features,
        return_metadata=return_metadata,
    )


def _predict_ridge(
    model: Ridge,
    loader: DataLoader,
    pred_len: int,
    num_features: int,
    *,
    return_metadata: bool = False,
) -> tuple[Any, ...]:
    predictions, targets, contexts = [], [], []
    trajectory_ids, starts = [], []
    for batch in loader:
        context = batch["context"].numpy()
        predictions.append(forecast_ridge_batch(model, context, pred_len, num_features))
        targets.append(batch["target"].numpy())
        contexts.append(context)
        if return_metadata:
            trajectory_ids.append(batch["trajectory_index"].numpy())
            starts.append(batch["start"].numpy())
    return concatenate_predictions(
        predictions,
        targets,
        contexts,
        trajectory_ids=trajectory_ids if return_metadata else None,
        starts=starts if return_metadata else None,
    )


def build_foundation_predictor(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Callable[[np.ndarray, int], np.ndarray], bool]:
    chronos_model_id = args.chronos_model_id
    chronos_variant = getattr(args, "chronos_variant", "bolt")
    if args.model == "chronos2":
        if not is_chronos2_model(chronos_model_id):
            chronos_model_id = "amazon/chronos-2"
        chronos_variant = "chronos2"
    foundation_args = SimpleNamespace(
        timesfm_model_id=args.timesfm_model_id,
        context_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.foundation_batch_size,
        freq=0,
        hf_local_files_only=args.hf_local_files_only,
        chronos_model_id=chronos_model_id,
        chronos_variant=chronos_variant,
        chronos_num_samples=args.chronos_num_samples,
        panda_model_id=args.panda_model_id,
        tirex2_model_id=getattr(args, "tirex2_model_id", "NX-AI/TiRex-2"),
        toto2_model_id=getattr(args, "toto2_model_id", "Datadog/Toto-2.0-22m"),
        torch_dtype=args.torch_dtype,
    )
    if args.model == "timesfm":
        return build_timesfm_predictor(foundation_args, str(device)), False
    if args.model in {"chronos", "chronos2"}:
        if is_chronos2_model(chronos_model_id) or chronos_variant == "chronos2":
            if not is_chronos2_model(chronos_model_id):
                raise ValueError(
                    "--chronos_variant chronos2 requires a Chronos-2 checkpoint, "
                    "for example --chronos_model_id amazon/chronos-2."
                )
            return build_chronos2_predictor(foundation_args, str(device)), True
        return build_chronos_predictor(foundation_args, str(device)), False
    if args.model == "panda_zero_shot":
        return build_panda_predictor(foundation_args, str(device)), True
    if args.model == "tirex2":
        return build_tirex2_predictor(foundation_args, str(device)), True
    if args.model == "toto2":
        return build_toto2_predictor(foundation_args, str(device)), True
    raise ValueError(args.model)


def predict_foundation(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    loader: DataLoader,
    pred_len: int,
    multivariate: bool,
    log_every_batches: int = 100,
    split: str = "eval",
    return_metadata: bool = False,
    context_limit: int | None = None,
) -> tuple[Any, ...]:
    predictions, targets, contexts = [], [], []
    trajectory_ids, starts = [], []
    context_limit = None if context_limit is None else max(int(context_limit), 1)
    context_collected = 0
    started = time.perf_counter()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        context = batch["context"].numpy()
        predictions.append(forecast_foundation_batch(predictor, context, pred_len, multivariate))
        targets.append(batch["target"].numpy())
        if context_limit is None:
            contexts.append(context)
        elif context_collected < context_limit:
            take = min(context_limit - context_collected, context.shape[0])
            contexts.append(context[:take])
            context_collected += take
        if return_metadata:
            if "trajectory_index" not in batch or "start" not in batch:
                raise ValueError("Evaluation dataset does not provide trajectory/start metadata.")
            trajectory_ids.append(batch["trajectory_index"].numpy())
            starts.append(batch["start"].numpy())
        if should_log(batch_idx, total_batches, log_every_batches):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = batch_idx / elapsed
            eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
            print(
                f"[task_e] foundation {split} [{batch_idx}/{total_batches}] "
                f"elapsed={elapsed:.1f}s eta={format_duration(eta)}",
                flush=True,
            )
    return concatenate_predictions(
        predictions,
        targets,
        contexts,
        trajectory_ids=trajectory_ids if return_metadata else None,
        starts=starts if return_metadata else None,
    )


def log_parameter_counts(model: nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    print(
        f"Parameters: total={total:,} trainable={trainable:,} frozen={frozen:,}",
        flush=True,
    )



def build_panda_adapter(args: argparse.Namespace, num_features: int, device: torch.device) -> nn.Module:
    from tasks.task_e.foundation import load_panda_model
    from tasks.task_e.panda_adapter import PandaTemporalAdapter

    panda_args = SimpleNamespace(
        panda_model_id=args.panda_model_id,
        hf_local_files_only=args.hf_local_files_only,
    )
    panda = load_panda_model(panda_args, str(device))
    panda_channels = int(getattr(panda.config, "num_input_channels", num_features))
    if panda_channels not in {1, num_features}:
        raise ValueError(
            f"Panda config reports {panda_channels} fixed channels, but "
            f"{args.representation} produced {num_features}."
        )
    if panda_channels == 1 and num_features != 1:
        print(
            "Panda config reports num_input_channels=1; continuing with "
            f"{args.representation} multivariate input ({num_features} channels) because "
            "Panda shares its embedding and prediction head across channels.",
            flush=True,
        )
    panda_context_len = int(getattr(panda.config, "context_length", args.context_len))
    panda_pred_len = int(getattr(panda.config, "prediction_length", args.pred_len))
    adapter = PandaTemporalAdapter(
        panda=panda,
        input_len=args.context_len,
        panda_context_len=panda_context_len,
        panda_pred_len=panda_pred_len,
        output_len=args.pred_len,
        hidden_len=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        freeze_panda=args.freeze_panda,
        decoder=getattr(args, "adapter_decoder", "mlp"),
        num_features=num_features,
        gru_layers=int(getattr(args, "adapter_gru_layers", 1)),
    ).to(device)
    adapter.adapter_metadata = {
        "backbone": "panda",
        "backbone_model_id": args.panda_model_id,
        "backbone_trainable": not args.freeze_panda,
        "pre_adapter_trainable": True,
        "post_adapter_trainable": True,
        "decoder": getattr(args, "adapter_decoder", "mlp"),
        "output_len": args.pred_len,
        "num_features": num_features,
        "gru_layers": int(getattr(args, "adapter_gru_layers", 1)),
    }
    return adapter


def save_evaluation(
    output_dir: Path,
    split: str,
    prediction: np.ndarray,
    target: np.ndarray,
    prepared: PreparedForecastData,
    feature_names: list[str],
    save_detail: bool,
    distribution_bins: int,
    distribution_epsilon: float,
    trajectory_ids: np.ndarray,
    group_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics, horizon_rows, channel_rows = compute_forecast_metrics(
        prediction,
        target,
        prepared.scaler,
        feature_names,
        distribution_bins=distribution_bins,
        distribution_epsilon=distribution_epsilon,
        trajectory_ids=trajectory_ids,
        group_size=group_size,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    horizon_trend = summarize_horizon_trends(horizon_rows)
    metrics["horizon_trend"] = horizon_trend
    save_json(output_dir / f"{split}_metrics.json", metrics)
    save_json(output_dir / f"{split}_horizon_trend.json", horizon_trend)
    if save_detail:
        save_rows(output_dir / "horizon_metrics.csv", horizon_rows)
        save_rows(output_dir / "channel_metrics.csv", channel_rows)
        save_json(output_dir / "e1_metrics.json", metrics)
    return metrics


def run_rolling_evaluation(
    dataset: Dataset,
    forecast_fn: Callable[[np.ndarray], np.ndarray],
    *,
    context_len: int,
    block_len: int,
    num_blocks: int,
    windows_per_trajectory: int = 0,
    target_positions: tuple[float, ...] = (0.25, 0.75),
    group_size: int = 32,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    log_every_batches: int = 0,
) -> dict[str, Any]:
    """Evaluate recursive block forecasts without feeding ground truth back in."""
    rolling_dataset = RollingWindowDataset(
        dataset,
        context_len=context_len,
        block_len=block_len,
        num_blocks=num_blocks,
        windows_per_trajectory=windows_per_trajectory,
        target_positions=target_positions,
    )
    loader = DataLoader(
        rolling_dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    total_horizon = block_len * num_blocks
    num_channels = int(rolling_dataset.records[0][0].feature_metadata["num_features"])
    accumulator = HierarchicalErrorAccumulator(total_horizon, group_size=group_size)
    started = time.perf_counter()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        context = batch["context"].numpy().astype(np.float32, copy=False)
        target = batch["target"].numpy().astype(np.float32, copy=False)
        expected_context = (context.shape[0], context_len, num_channels)
        expected_target = (context.shape[0], total_horizon, num_channels)
        if context.shape != expected_context or target.shape != expected_target:
            raise ValueError(
                f"Rolling batch shapes {context.shape} and {target.shape}; "
                f"expected {expected_context} and {expected_target}."
            )
        if not np.isfinite(context).all() or not np.isfinite(target).all():
            raise ValueError("Rolling evaluation received non-finite context or target values.")

        current_context = context
        block_predictions: list[np.ndarray] = []
        for block_index in range(num_blocks):
            prediction = np.asarray(forecast_fn(current_context), dtype=np.float32)
            expected_prediction = (context.shape[0], block_len, num_channels)
            if prediction.shape != expected_prediction:
                raise ValueError(
                    f"Rolling block {block_index + 1} returned {prediction.shape}; "
                    f"expected {expected_prediction}."
                )
            if not np.isfinite(prediction).all():
                raise ValueError(f"Rolling block {block_index + 1} produced non-finite predictions.")
            block_predictions.append(prediction)
            if block_index + 1 < num_blocks:
                # Only the model prediction is appended. The true future target
                # is never used to refresh the context.
                current_context = np.concatenate([current_context, prediction], axis=1)
                current_context = current_context[:, -context_len:, :]
                if current_context.shape != expected_context:
                    raise ValueError(f"Unexpected recursive context shape {current_context.shape}.")

        prediction = np.concatenate(block_predictions, axis=1)
        if prediction.shape != expected_target:
            raise ValueError(f"Rolling prediction shape {prediction.shape} != {expected_target}.")
        error = prediction - target
        if not np.isfinite(error).all():
            raise ValueError("Rolling evaluation produced a non-finite error array.")
        if "trajectory_index" not in batch:
            raise ValueError("Rolling dataset did not provide trajectory metadata.")
        accumulator.add(error, batch["trajectory_index"].numpy())
        if should_log(batch_idx, total_batches, log_every_batches):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = batch_idx / elapsed
            eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
            print(
                f"[task_e] rolling [{batch_idx}/{total_batches}] "
                f"elapsed={elapsed:.1f}s eta={format_duration(eta)}",
                flush=True,
            )

    hierarchy, hierarchy_internals = accumulator.finalize(
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rolling_horizon_mae = np.asarray(hierarchy["rolling_horizon_mae_system_balanced"], dtype=np.float64)
    weighted_horizon_mae = np.asarray(
        hierarchy["rolling_horizon_mae_window_weighted"],
        dtype=np.float64,
    )
    system_horizon_mae = hierarchy_internals["system_horizon_mae"]
    system_block_mae = np.asarray(
        [
            np.mean(system_horizon_mae[:, index * block_len : (index + 1) * block_len], axis=1)
            for index in range(num_blocks)
        ],
        dtype=np.float64,
    ).T
    block_mae = np.mean(system_block_mae, axis=0, dtype=np.float64).tolist()
    weighted_block_mae = [
        float(np.mean(weighted_horizon_mae[index * block_len : (index + 1) * block_len]))
        for index in range(num_blocks)
    ]
    ratio_epsilon = 1e-12
    block_ci = bootstrap_block_ci(
        system_block_mae,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
        ratio_epsilon=ratio_epsilon,
    )
    amplification_ratio = block_ci.get("error_amplification_ratio", {}).get("mean")
    weighted_ratio = (
        float(weighted_block_mae[-1] / weighted_block_mae[0])
        if weighted_block_mae and weighted_block_mae[0] > ratio_epsilon
        else (1.0 if weighted_block_mae and weighted_block_mae[-1] <= ratio_epsilon else None)
    )
    metrics = {
        **hierarchy,
        "context_len": int(context_len),
        "block_len": int(block_len),
        "num_blocks": int(num_blocks),
        "rolling_horizon": int(total_horizon),
        "candidate_num_windows": int(rolling_dataset.candidate_count),
        "windows_per_trajectory": int(windows_per_trajectory),
        "target_positions": [float(value) for value in target_positions],
        "rolling_horizon_mae_window_weighted": weighted_horizon_mae.tolist(),
        "rolling_horizon_mae_system_balanced": rolling_horizon_mae.tolist(),
        "rolling_horizon_mae": rolling_horizon_mae.tolist(),
        **{
            f"block_{index + 1}_mae": float(block_mae[index]) if index < len(block_mae) else None
            for index in range(max(num_blocks, 4))
        },
        **{
            f"block_{index + 1}_mae_window_weighted": (
                float(weighted_block_mae[index]) if index < len(weighted_block_mae) else None
            )
            for index in range(max(num_blocks, 4))
        },
        "error_amplification_ratio": amplification_ratio,
        "error_amplification_ratio_window_weighted": weighted_ratio,
        "error_amplification_epsilon": ratio_epsilon,
    }
    metrics["bootstrap_ci95"].update(block_ci)
    return metrics


def save_rolling_evaluation(output_dir: Path, metrics: dict[str, Any]) -> None:
    save_json(output_dir / "rolling_metrics.json", metrics)
    save_json(output_dir / "e2_metrics.json", metrics)
    block_len = int(metrics["block_len"])
    horizon_rows = []
    for index, value in enumerate(metrics["rolling_horizon_mae"], start=1):
        horizon_rows.append(
            {
                "horizon_step": index,
                "mae_z": float(value),
                "mae_z_system_balanced": float(value),
                "mae_z_window_weighted": float(metrics["rolling_horizon_mae_window_weighted"][index - 1]),
                "block": int((index - 1) // block_len + 1),
            }
        )
    save_rows(output_dir / "rolling_horizon_metrics.csv", horizon_rows)


def save_run_summary(
    output_dir: Path,
    model_name: str,
    e1_metrics: dict[str, Any],
    e2_metrics: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> None:
    summary = {
        "model": model_name,
        "sampling_metadata": (config or {}).get("sampling_metadata", {}),
        "evaluation_protocol": (config or {}).get("evaluation_protocol", {}),
        "E1": {
            "mse_z": e1_metrics.get("mse_z"),
            "mae_z": e1_metrics.get("mae_z"),
            "rmse_z": e1_metrics.get("rmse_z"),
            "mse_z_window_weighted": e1_metrics.get("mse_z_window_weighted"),
            "mae_z_window_weighted": e1_metrics.get("mae_z_window_weighted"),
            "rmse_z_window_weighted": e1_metrics.get("rmse_z_window_weighted"),
            "mse_z_system_balanced": e1_metrics.get("mse_z_system_balanced"),
            "mae_z_system_balanced": e1_metrics.get("mae_z_system_balanced"),
            "rmse_z_system_balanced": e1_metrics.get("rmse_z_system_balanced"),
            "pearson": e1_metrics.get("pearson"),
            "horizon_mae": e1_metrics.get("horizon_mae", []),
            "horizon_mae_window_weighted": e1_metrics.get("horizon_mae_window_weighted", []),
            "horizon_mae_system_balanced": e1_metrics.get("horizon_mae_system_balanced", []),
            "unit_circle_error": e1_metrics.get("unit_circle_error"),
            "bootstrap_ci95": e1_metrics.get("bootstrap_ci95", {}),
        },
        "E2": e2_metrics,
    }
    save_json(output_dir / "summary.json", summary)
    row = {
        "model": model_name,
        "mse_z": e1_metrics.get("mse_z"),
        "mae_z": e1_metrics.get("mae_z"),
        "rmse_z": e1_metrics.get("rmse_z"),
        "pearson": e1_metrics.get("pearson"),
        "unit_circle_error": e1_metrics.get("unit_circle_error"),
        "horizon_mae": json.dumps(e1_metrics.get("horizon_mae", [])),
        "mse_z_ci95_low": e1_metrics.get("bootstrap_ci95", {}).get("mse_z", {}).get("ci95_low"),
        "mse_z_ci95_high": e1_metrics.get("bootstrap_ci95", {}).get("mse_z", {}).get("ci95_high"),
        "mae_z_ci95_low": e1_metrics.get("bootstrap_ci95", {}).get("mae_z", {}).get("ci95_low"),
        "mae_z_ci95_high": e1_metrics.get("bootstrap_ci95", {}).get("mae_z", {}).get("ci95_high"),
        "rmse_z_ci95_low": e1_metrics.get("bootstrap_ci95", {}).get("rmse_z", {}).get("ci95_low"),
        "rmse_z_ci95_high": e1_metrics.get("bootstrap_ci95", {}).get("rmse_z", {}).get("ci95_high"),
        "block_1_mae": e2_metrics.get("block_1_mae") if e2_metrics else None,
        "block_2_mae": e2_metrics.get("block_2_mae") if e2_metrics else None,
        "block_3_mae": e2_metrics.get("block_3_mae") if e2_metrics else None,
        "block_4_mae": e2_metrics.get("block_4_mae") if e2_metrics else None,
        "error_amplification_ratio": e2_metrics.get("error_amplification_ratio") if e2_metrics else None,
        "candidate_train_windows": (config or {}).get("sampling_metadata", {}).get("candidate_train_windows"),
        "candidate_val_windows": (config or {}).get("sampling_metadata", {}).get("candidate_val_windows"),
        "candidate_test_windows": (config or {}).get("sampling_metadata", {}).get("candidate_test_windows"),
        "sampled_train_windows_per_epoch": (config or {}).get("sampling_metadata", {}).get("sampled_train_windows_per_epoch"),
        "sampled_val_windows": (config or {}).get("sampling_metadata", {}).get("sampled_val_windows"),
        "sampled_e1_test_windows": (config or {}).get("sampling_metadata", {}).get("sampled_e1_test_windows"),
        "sampled_e2_rolling_windows": (config or {}).get("sampling_metadata", {}).get("sampled_e2_rolling_windows"),
    }
    save_rows(output_dir / "summary.csv", [row])
    lines = [
        f"# Task E summary: `{model_name}`",
        "",
        "## E1 one-shot 512 -> 128",
        "",
        f"- MSE (z): {e1_metrics.get('mse_z')}",
        f"- MAE (z): {e1_metrics.get('mae_z')}",
        f"- RMSE (z): {e1_metrics.get('rmse_z')}",
        f"- Pearson: {e1_metrics.get('pearson')}",
        f"- Unit-circle error: {e1_metrics.get('unit_circle_error')}",
        "- Primary aggregation: equal-weight system -> trajectory -> window.",
        f"- System bootstrap 95% CI: {e1_metrics.get('bootstrap_ci95', {})}",
        f"- Sampling metadata: {(config or {}).get('sampling_metadata', {})}",
    ]
    if e2_metrics is not None:
        lines.extend(
            [
                "",
                "## E2 recursive rolling",
                "",
                f"- Windows: {e2_metrics.get('num_windows')}",
                f"- Horizon: {e2_metrics.get('rolling_horizon')}",
                f"- Block 1 MAE: {e2_metrics.get('block_1_mae')}",
                f"- Block 2 MAE: {e2_metrics.get('block_2_mae')}",
                f"- Block 3 MAE: {e2_metrics.get('block_3_mae')}",
                f"- Block 4 MAE: {e2_metrics.get('block_4_mae')}",
                f"- Error amplification ratio: {e2_metrics.get('error_amplification_ratio')}",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_prediction_sample(
    path: Path,
    context_z: np.ndarray,
    target_z: np.ndarray,
    pred_z: np.ndarray,
    prepared: PreparedForecastData,
    feature_names: list[str],
    limit: int,
) -> None:
    count = min(max(limit, 1), pred_z.shape[0])
    np.savez_compressed(
        path,
        context_raw=prepared.scaler.inverse_transform(context_z[:count]),
        target_raw=prepared.scaler.inverse_transform(target_z[:count]),
        pred_raw=prepared.scaler.inverse_transform(pred_z[:count]),
        context_z=context_z[:count],
        target_z=target_z[:count],
        pred_z=pred_z[:count],
        feature_names=np.asarray(feature_names, dtype=object),
    )


def concatenate_predictions(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    contexts: list[np.ndarray],
    trajectory_ids: list[np.ndarray] | None = None,
    starts: list[np.ndarray] | None = None,
) -> tuple[Any, ...]:
    if not predictions:
        raise ValueError("Evaluation produced zero batches.")
    prediction = np.concatenate(predictions).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    context = np.concatenate(contexts).astype(np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != target shape {target.shape}")
    if trajectory_ids is None and starts is None:
        return prediction, target, context
    if trajectory_ids is None or starts is None:
        raise ValueError("Trajectory IDs and starts must be supplied together.")
    concatenated_trajectory_ids = np.concatenate(trajectory_ids).astype(np.int64)
    concatenated_starts = np.concatenate(starts).astype(np.int64)
    if concatenated_trajectory_ids.shape != (prediction.shape[0],):
        raise ValueError(
            f"Trajectory metadata shape {concatenated_trajectory_ids.shape} does not match "
            f"prediction batch count {prediction.shape[0]}."
        )
    if concatenated_starts.shape != concatenated_trajectory_ids.shape:
        raise ValueError("Trajectory IDs and start metadata have different lengths.")
    return prediction, target, context, concatenated_trajectory_ids, concatenated_starts


def should_log(batch_idx: int, total_batches: int, every: int) -> bool:
    return every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % every == 0)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        scalar = value.item()
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.bool_):
        return value.item()
    return value


def close_prepared(prepared: PreparedForecastData) -> None:
    prepared.train_dataset.close()
    prepared.val_dataset.close()
    prepared.test_dataset.close()


if __name__ == "__main__":
    main()
