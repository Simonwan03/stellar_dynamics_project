from __future__ import annotations

import argparse
import json
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task_e_foundation_forecasters import (  # noqa: E402
    build_chronos_predictor,
    build_panda_predictor,
    build_timesfm_predictor,
)
from src.task_e.data import PreparedForecastData, prepare_forecasting_data  # noqa: E402
from src.task_e.metrics import compute_forecast_metrics, save_rows  # noqa: E402
from src.task_e.models import build_supervised_model  # noqa: E402


SUPERVISED_MODELS = {"tcn", "patchtst", "transformer", "itransformer", "itransformer_simple"}
FOUNDATION_MODELS = {"timesfm", "chronos", "panda_zero_shot"}
ALL_MODELS = {
    "persistence",
    "ridge",
    *SUPERVISED_MODELS,
    *FOUNDATION_MODELS,
    "panda_adapter",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Task E past-window -> future-window forecasting benchmark.")
    parser.add_argument("--data_glob", type=str, default="dataset_v3/results_batch_*_sampling_2.hdf5")
    parser.add_argument(
        "--representation",
        choices=["orbital15", "raw_state18", "trajectory"],
        default="orbital15",
    )
    parser.add_argument("--model", choices=sorted(ALL_MODELS), default="persistence")
    parser.add_argument("--context_len", type=int, default=200)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--max_windows_per_trajectory", type=int, default=0)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=0)
    parser.add_argument("--max_test_windows", type=int, default=0)
    parser.add_argument("--max_scaler_trajectories", type=int, default=0)
    parser.add_argument("--window_sampling", choices=["random", "sequential"], default="sequential")
    parser.add_argument("--drop_nan_windows", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--num_eval_windows", type=int, default=256)
    parser.add_argument("--foundation_batch_size", type=int, default=64)
    parser.add_argument("--timesfm_model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--timesfm_backend", type=str, default="auto")
    parser.add_argument("--chronos_model_id", type=str, default="amazon/chronos-bolt-small")
    parser.add_argument("--chronos_variant", choices=["bolt", "chronos2"], default="bolt")
    parser.add_argument("--chronos_num_samples", type=int, default=20)
    parser.add_argument("--panda_model_id", type=str, default="GilpinLab/panda-72M")
    parser.add_argument("--freeze_panda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_hidden_dim", type=int, default=None)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--num_train_windows_for_adapter", type=int, default=4096)
    parser.add_argument("--hf_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prediction_samples", type=int, default=16)
    parser.add_argument("--log_every_batches", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(args.amp and device.type == "cuda")

    prepared = prepare_forecasting_data(
        data_glob=args.data_glob,
        representation=args.representation,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        max_trajectories=args.max_trajectories,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        window_sampling=args.window_sampling,
        drop_nan_windows=args.drop_nan_windows,
        max_scaler_trajectories=args.max_scaler_trajectories,
    )
    feature_names = list(prepared.feature_metadata["channel_names"])
    num_features = int(prepared.feature_metadata["num_features"])
    max_train_windows = args.max_train_windows
    max_val_windows = args.max_val_windows
    max_test_windows = args.max_test_windows
    if args.model == "ridge" and max_train_windows <= 0:
        max_train_windows = 5000
        max_val_windows = max_val_windows or 2000
        max_test_windows = max_test_windows or 2000
        print(
            "Ridge materializes flattened windows in memory; applying automatic limits "
            "train=5000, val=2000, test=2000. Override with --max_*_windows.",
            flush=True,
        )
    datasets = {
        "train": cap_dataset(prepared.train_dataset, max_train_windows, args.seed + 1),
        "val": cap_dataset(prepared.val_dataset, max_val_windows, args.seed + 2),
        "test": cap_dataset(prepared.test_dataset, max_test_windows, args.seed + 3),
    }
    if args.model in FOUNDATION_MODELS and args.num_eval_windows > 0:
        datasets["val"] = cap_dataset(datasets["val"], args.num_eval_windows, args.seed + 4)
        datasets["test"] = cap_dataset(datasets["test"], args.num_eval_windows, args.seed + 5)
    if args.model == "panda_adapter" and args.num_train_windows_for_adapter > 0:
        datasets["train"] = cap_dataset(
            datasets["train"], args.num_train_windows_for_adapter, args.seed + 6
        )

    prepared.scaler.save(args.output_dir / "scaler.npz", feature_names)
    save_json(args.output_dir / "feature_metadata.json", prepared.feature_metadata)
    effective_limits = {
        "train": max_train_windows,
        "val": max_val_windows,
        "test": max_test_windows,
    }
    config = build_config(args, prepared, datasets, effective_limits, device, amp_enabled)
    save_json(args.output_dir / "config.json", config)
    print_dataset_summary(config, feature_names, prepared)

    loaders = {
        name: build_loader(
            dataset,
            args.batch_size if args.model not in FOUNDATION_MODELS else args.foundation_batch_size,
            args.num_workers,
            shuffle=name == "train" and args.model in SUPERVISED_MODELS | {"panda_adapter"},
            device=device,
        )
        for name, dataset in datasets.items()
    }

    if args.model == "persistence":
        predictor = lambda context: context[:, -1:, :].repeat(1, args.pred_len, 1)
        val_pred, val_target, _ = predict_torch_callable(predictor, loaders["val"], device)
        test_pred, test_target, sample_context = predict_torch_callable(predictor, loaders["test"], device)
    elif args.model == "ridge":
        ridge = Ridge(alpha=args.ridge_alpha)
        train_context, train_target = materialize_dataset(loaders["train"])
        ridge.fit(train_context.reshape(train_context.shape[0], -1), train_target.reshape(train_target.shape[0], -1))
        with open(args.output_dir / "best_model.pkl", "wb") as handle:
            pickle.dump(ridge, handle)
        val_pred, val_target, _ = predict_ridge(ridge, loaders["val"], args.pred_len, num_features)
        test_pred, test_target, sample_context = predict_ridge(ridge, loaders["test"], args.pred_len, num_features)
    elif args.model in SUPERVISED_MODELS:
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
        val_pred, val_target, _ = predict_model(model, loaders["val"], device, amp_enabled)
        test_pred, test_target, sample_context = predict_model(model, loaders["test"], device, amp_enabled)
    elif args.model == "panda_adapter":
        model = build_panda_adapter(args, num_features, device)
        train_model(model, loaders, args, config, prepared, device, amp_enabled, lr=args.adapter_lr)
        val_pred, val_target, _ = predict_model(model, loaders["val"], device, amp_enabled)
        test_pred, test_target, sample_context = predict_model(model, loaders["test"], device, amp_enabled)
    else:
        predictor, multivariate = build_foundation_predictor(args, device)
        val_pred, val_target, _ = predict_foundation(
            predictor, loaders["val"], args.pred_len, multivariate
        )
        test_pred, test_target, sample_context = predict_foundation(
            predictor, loaders["test"], args.pred_len, multivariate
        )

    val_metrics = save_evaluation(
        args.output_dir,
        "val",
        val_pred,
        val_target,
        prepared,
        feature_names,
        save_detail=False,
    )
    test_metrics = save_evaluation(
        args.output_dir,
        "test",
        test_pred,
        test_target,
        prepared,
        feature_names,
        save_detail=True,
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
    print(
        f"Finished {args.model}: val_mse_z={val_metrics['mse_z']:.6g}, "
        f"test_mse_z={test_metrics['mse_z']:.6g}, test_mse_raw={test_metrics['mse_raw']:.6g}"
    )
    close_prepared(prepared)


def build_config(
    args: argparse.Namespace,
    prepared: PreparedForecastData,
    datasets: dict[str, Dataset],
    effective_limits: dict[str, int],
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    model_family = "baseline"
    training_mode = "none"
    if args.model in SUPERVISED_MODELS:
        model_family, training_mode = "supervised", "full_supervised"
    elif args.model in FOUNDATION_MODELS:
        model_family, training_mode = "foundation", "zero_shot"
    elif args.model == "panda_adapter":
        model_family, training_mode = "foundation_adapter", "adapter_supervised"
    return {
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
        "full_window_counts": {
            "train": len(prepared.train_dataset),
            "val": len(prepared.val_dataset),
            "test": len(prepared.test_dataset),
        },
        "used_window_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "effective_window_limits": effective_limits,
        "scaler_fit_scope": "train_trajectories_full_series_only",
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "model_family": model_family,
        "training_mode": training_mode,
        "itransformer_variant": "simple_variable_token_encoder" if args.model.startswith("itransformer") else None,
    }


def print_dataset_summary(config: dict[str, Any], feature_names: list[str], prepared: PreparedForecastData) -> None:
    print(f"Sources: {len(prepared.source_paths)}, trajectories: {prepared.trajectory_count}, T={prepared.time_steps}")
    print(f"Split trajectories: {config['split_trajectory_counts']}")
    print(f"Used windows: {config['used_window_counts']}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Scaler mean/std shapes: {prepared.scaler.mean.shape} / {prepared.scaler.std.shape}")


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
        persistent_workers=num_workers > 0,
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
    optimizer = torch.optim.AdamW(parameters, lr=lr or args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext
    criterion = nn.MSELoss()
    history = []
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
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
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
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
        val_pred, val_target, _ = predict_model(model, loaders["val"], device, amp_enabled)
        val_metrics, _, _ = compute_forecast_metrics(
            val_pred, val_target, prepared.scaler, prepared.feature_metadata["channel_names"]
        )
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_mse_z": val_metrics["mse_z"],
            "val_mae_z": val_metrics["mae_z"],
            "val_mse_raw": val_metrics["mse_raw"],
            "val_mae_raw": val_metrics["mae_raw"],
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
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse_z": best_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
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
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    autocast = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext
    predictions, targets, contexts = [], [], []
    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        with autocast():
            prediction = model(context)
        predictions.append(prediction.detach().cpu().float().numpy())
        targets.append(batch["target"].numpy())
        contexts.append(batch["context"].numpy())
    return concatenate_predictions(predictions, targets, contexts)


@torch.no_grad()
def predict_torch_callable(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, targets, contexts = [], [], []
    for batch in loader:
        context = batch["context"].to(device)
        predictions.append(predictor(context).detach().cpu().numpy())
        targets.append(batch["target"].numpy())
        contexts.append(batch["context"].numpy())
    return concatenate_predictions(predictions, targets, contexts)


def materialize_dataset(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    contexts, targets = [], []
    for batch in loader:
        contexts.append(batch["context"].numpy())
        targets.append(batch["target"].numpy())
    if not contexts:
        raise ValueError("Dataset produced zero windows.")
    return np.concatenate(contexts), np.concatenate(targets)


def predict_ridge(
    model: Ridge,
    loader: DataLoader,
    pred_len: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, targets, contexts = [], [], []
    for batch in loader:
        context = batch["context"].numpy()
        pred = model.predict(context.reshape(context.shape[0], -1))
        predictions.append(pred.reshape(context.shape[0], pred_len, num_features).astype(np.float32))
        targets.append(batch["target"].numpy())
        contexts.append(context)
    return concatenate_predictions(predictions, targets, contexts)


def build_foundation_predictor(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Callable[[np.ndarray, int], np.ndarray], bool]:
    foundation_args = SimpleNamespace(
        timesfm_model_id=args.timesfm_model_id,
        context_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.foundation_batch_size,
        freq=0,
        hf_local_files_only=args.hf_local_files_only,
        chronos_model_id=args.chronos_model_id,
        chronos_num_samples=args.chronos_num_samples,
        panda_model_id=args.panda_model_id,
        torch_dtype=args.torch_dtype,
    )
    if args.model == "timesfm":
        return build_timesfm_predictor(foundation_args, str(device)), False
    if args.model == "chronos":
        return build_chronos_predictor(foundation_args, str(device)), False
    if args.model == "panda_zero_shot":
        if args.representation != "orbital15":
            raise ValueError("panda_zero_shot requires --representation orbital15 for channel compatibility.")
        return build_panda_predictor(foundation_args, str(device)), True
    raise ValueError(args.model)


def predict_foundation(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    loader: DataLoader,
    pred_len: int,
    multivariate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, targets, contexts = [], [], []
    for batch in loader:
        context = batch["context"].numpy()
        if multivariate:
            prediction = predictor(context, pred_len)
        else:
            batch_count, context_len, channels = context.shape
            flattened = context.transpose(0, 2, 1).reshape(batch_count * channels, context_len)
            flat_prediction = predictor(flattened, pred_len)
            prediction = flat_prediction.reshape(batch_count, channels, pred_len).transpose(0, 2, 1)
        predictions.append(np.asarray(prediction, dtype=np.float32))
        targets.append(batch["target"].numpy())
        contexts.append(context)
    return concatenate_predictions(predictions, targets, contexts)


def build_panda_adapter(args: argparse.Namespace, num_features: int, device: torch.device) -> nn.Module:
    from scripts.task_e_foundation_forecasters import load_panda_model
    from scripts.train_panda_temporal_adapter import PandaTemporalAdapter

    if args.representation != "orbital15":
        raise ValueError("The current panda_adapter implementation requires --representation orbital15.")
    panda_args = SimpleNamespace(
        panda_model_id=args.panda_model_id,
        hf_local_files_only=args.hf_local_files_only,
    )
    panda = load_panda_model(panda_args, str(device))
    panda_channels = int(getattr(panda.config, "num_input_channels", num_features))
    if panda_channels != num_features:
        raise ValueError(f"Panda expects {panda_channels} channels, but orbital15 produced {num_features}.")
    panda_context_len = int(getattr(panda.config, "context_length", args.context_len))
    panda_pred_len = int(getattr(panda.config, "prediction_length", args.pred_len))
    return PandaTemporalAdapter(
        panda=panda,
        input_len=args.context_len,
        panda_context_len=panda_context_len,
        panda_pred_len=panda_pred_len,
        output_len=args.pred_len,
        hidden_len=args.adapter_hidden_dim,
        dropout=args.dropout,
        freeze_panda=args.freeze_panda,
    ).to(device)


def save_evaluation(
    output_dir: Path,
    split: str,
    prediction: np.ndarray,
    target: np.ndarray,
    prepared: PreparedForecastData,
    feature_names: list[str],
    save_detail: bool,
) -> dict[str, Any]:
    metrics, horizon_rows, channel_rows = compute_forecast_metrics(
        prediction, target, prepared.scaler, feature_names
    )
    save_json(output_dir / f"{split}_metrics.json", metrics)
    if save_detail:
        save_rows(output_dir / "horizon_metrics.csv", horizon_rows)
        save_rows(output_dir / "channel_metrics.csv", channel_rows)
    return metrics


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("Evaluation produced zero batches.")
    prediction = np.concatenate(predictions).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    context = np.concatenate(contexts).astype(np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != target shape {target.shape}")
    return prediction, target, context


def should_log(batch_idx: int, total_batches: int, every: int) -> bool:
    return every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % every == 0)


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
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def close_prepared(prepared: PreparedForecastData) -> None:
    prepared.train_dataset.close()
    prepared.val_dataset.close()
    prepared.test_dataset.close()


if __name__ == "__main__":
    main()
