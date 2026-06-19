from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task_e_foundation_forecasters import (  # noqa: E402
    DEFAULT_ORBITAL_FEATURES,
    DEFAULT_TRAJECTORY_KEYS,
    load_eval_data,
    load_panda_model,
)
from scripts.train_panda_temporal_adapter import TemporalMLPAdapter, filter_window_values  # noqa: E402
from src.train_utils import save_json, set_random_seed  # noqa: E402


class ChannelMLPAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, zero_output: bool = False) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(64, in_dim, out_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_output:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}")
        return self.net(x)


class PandaTrajectoryAdapter(nn.Module):
    def __init__(
        self,
        panda: nn.Module,
        input_len: int,
        output_len: int,
        input_channels: int,
        output_channels: int,
        panda_context_len: int,
        panda_pred_len: int,
        panda_channels: int,
        temporal_hidden_len: int | None,
        channel_hidden_dim: int | None,
        dropout: float,
        freeze_panda: bool,
    ) -> None:
        super().__init__()
        self.pre_time = TemporalMLPAdapter(input_len, panda_context_len, hidden_len=temporal_hidden_len, dropout=dropout)
        self.pre_channel = ChannelMLPAdapter(input_channels, panda_channels, channel_hidden_dim, zero_output=True)
        self.panda = panda
        self.post_channel = ChannelMLPAdapter(panda_channels, output_channels, channel_hidden_dim, zero_output=True)
        self.post_time = TemporalMLPAdapter(panda_pred_len, output_len, hidden_len=temporal_hidden_len, dropout=dropout)
        self.panda_pred_len = int(panda_pred_len)
        self.freeze_panda = bool(freeze_panda)

        if freeze_panda:
            for parameter in self.panda.parameters():
                parameter.requires_grad = False
            self.panda.eval()

    def train(self, mode: bool = True) -> "PandaTrajectoryAdapter":
        super().train(mode)
        if self.freeze_panda:
            self.panda.eval()
        return self

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.pre_time(context)
        x = self.pre_channel(x).float()
        observed = torch.ones_like(x, dtype=torch.bool)
        with torch.amp.autocast("cuda", enabled=False):
            outputs = self.panda(past_values=x, past_observed_mask=observed, future_values=None)
        panda_prediction = outputs.prediction_outputs
        if panda_prediction.shape[1] != self.panda_pred_len:
            raise ValueError(
                f"Panda returned prediction length {panda_prediction.shape[1]}, expected {self.panda_pred_len}."
            )
        y = self.post_channel(panda_prediction.to(dtype=context.dtype))
        return self.post_time(y)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune temporal and channel adapters around Panda for position/velocity forecasting."
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--trajectory-keys", type=str, default=",".join(DEFAULT_TRAJECTORY_KEYS))
    parser.add_argument(
        "--reference-frame",
        choices=["com", "body0", "absolute"],
        default="com",
        help="Transform position/velocity channels into a relative frame before training.",
    )
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--panda-context-len", type=int, default=None)
    parser.add_argument("--panda-pred-len", type=int, default=None)
    parser.add_argument("--panda-channels", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=4096)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-abs-value", type=float, default=5e7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-hidden-len", type=int, default=None)
    parser.add_argument("--channel-hidden-dim", type=int, default=128)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--freeze-panda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "panda_trajectory_adapter")
    parser.add_argument("--panda-model-id", type=str, default="GilpinLab/panda-72M")
    parser.add_argument("--hf-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every-batches", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(args.amp and device.type == "cuda")

    trajectory_keys = [item.strip() for item in args.trajectory_keys.split(",") if item.strip()]
    data = load_eval_data(
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        representation="trajectory",
        trajectory_keys=trajectory_keys,
        orbital_features=list(DEFAULT_ORBITAL_FEATURES),
        context_len=args.input_len,
        pred_len=args.output_len,
        num_windows=args.num_windows,
        seed=args.seed,
    )
    data.context = apply_reference_frame(data.context, args.reference_frame)
    data.target = apply_reference_frame(data.target, args.reference_frame)
    raw_window_count = int(data.context.shape[0])
    data.context, data.target, kept_indices = filter_window_values(data.context, data.target, args.max_abs_value)
    data.window_metadata = [data.window_metadata[int(index)] for index in kept_indices]

    split_indices = split_indices_for_count(
        count=data.context.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    stats = compute_normalization_stats(data.context, data.target, split_indices["train"])
    norm_context = normalize(data.context, stats)
    norm_target = normalize(data.target, stats)

    panda = load_panda_model(args, str(device))
    panda_context_len = int(args.panda_context_len or getattr(panda.config, "context_length", 512))
    panda_pred_len = int(args.panda_pred_len or getattr(panda.config, "prediction_length", 128))
    panda_channels = int(args.panda_channels or getattr(panda.config, "num_input_channels", 15))
    model = PandaTrajectoryAdapter(
        panda=panda,
        input_len=args.input_len,
        output_len=args.output_len,
        input_channels=norm_context.shape[-1],
        output_channels=norm_target.shape[-1],
        panda_context_len=panda_context_len,
        panda_pred_len=panda_pred_len,
        panda_channels=panda_channels,
        temporal_hidden_len=args.temporal_hidden_len,
        channel_hidden_dim=args.channel_hidden_dim,
        dropout=args.adapter_dropout,
        freeze_panda=args.freeze_panda,
    ).to(device)

    loaders = build_loaders(norm_context, norm_target, split_indices, args.batch_size, args.num_workers)
    config = {
        "dataset_dir": str(args.dataset_dir),
        "sampling_id": args.sampling_id,
        "representation": "trajectory",
        "reference_frame": args.reference_frame,
        "trajectory_keys": trajectory_keys,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "panda_context_len": panda_context_len,
        "panda_pred_len": panda_pred_len,
        "panda_channels": panda_channels,
        "num_windows": args.num_windows,
        "raw_window_count": raw_window_count,
        "filtered_window_count": int(data.context.shape[0]),
        "max_abs_value": args.max_abs_value,
        "split_window_counts": {name: int(indices.size) for name, indices in split_indices.items()},
        "num_channels": len(data.channel_names),
        "channel_names": data.channel_names,
        "normalization_mean": stats["mean"].reshape(-1).tolist(),
        "normalization_std": stats["std"].reshape(-1).tolist(),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temporal_hidden_len": args.temporal_hidden_len,
        "channel_hidden_dim": args.channel_hidden_dim,
        "adapter_dropout": args.adapter_dropout,
        "freeze_panda": args.freeze_panda,
        "amp": args.amp,
        "amp_enabled": amp_enabled,
        "device": str(device),
        "panda_model_id": args.panda_model_id,
        "hf_local_files_only": args.hf_local_files_only,
        "seed": args.seed,
    }
    save_json(args.output_dir / "config.json", config)
    np.savez(args.output_dir / "normalization_stats.npz", mean=stats["mean"], std=stats["std"])

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer, scaler, amp_enabled, args.log_every_batches, epoch)
        val_metrics = evaluate(model, loaders["val"], device, amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_mse": val_metrics["mse"], "val_mae": val_metrics["mae"]}
        history.append(row)
        write_rows(args.output_dir / "train_log.csv", history)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | val_mae={val_metrics['mae']:.6f}",
            flush=True,
        )
        if np.isfinite(val_metrics["mse"]) and val_metrics["mse"] < best_val:
            best_val = float(val_metrics["mse"])
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse": best_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                args.output_dir / "best_model.pt",
            )

    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config},
        args.output_dir / "final_model.pt",
    )
    if best_epoch < 0:
        raise RuntimeError("No finite validation checkpoint was saved.")

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    norm_prediction, norm_truth = predict_all(model, loaders["test"], device, amp_enabled)
    raw_prediction = denormalize(norm_prediction, stats)
    raw_truth = denormalize(norm_truth, stats)
    metrics = compute_metrics(norm_prediction, norm_truth, raw_prediction, raw_truth)
    metrics["best_epoch"] = best_epoch
    save_json(args.output_dir / "test_metrics.json", metrics)
    save_horizon_csv(args.output_dir / "horizon_metrics.csv", metrics)
    print(
        f"TEST | norm_mse={metrics['norm_mse']:.6g} norm_mae={metrics['norm_mae']:.6g} "
        f"raw_mse={metrics['raw_mse']:.6g} raw_mae={metrics['raw_mae']:.6g}",
        flush=True,
    )


def split_indices_for_count(count: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, np.ndarray]:
    if count < 3:
        raise ValueError(f"Need at least 3 windows for train/val/test splits, got {count}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    return {
        "train": order[:train_count],
        "val": order[train_count : train_count + val_count],
        "test": order[train_count + val_count :],
    }


def compute_normalization_stats(context: np.ndarray, target: np.ndarray, train_indices: np.ndarray) -> dict[str, np.ndarray]:
    train_values = np.concatenate([context[train_indices], target[train_indices]], axis=1)
    mean = train_values.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_values.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def normalize(array: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((array - stats["mean"]) / stats["std"]).astype(np.float32)


def denormalize(array: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (array * stats["std"] + stats["mean"]).astype(np.float32)


def apply_reference_frame(array: np.ndarray, reference_frame: str) -> np.ndarray:
    if reference_frame == "absolute":
        return array.astype(np.float32, copy=False)
    if array.shape[-1] != 18:
        raise ValueError(f"Reference-frame transform expects 18 channels, got {array.shape[-1]}.")

    pos = array[:, :, :9].reshape(array.shape[0], array.shape[1], 3, 3)
    vel = array[:, :, 9:].reshape(array.shape[0], array.shape[1], 3, 3)
    if reference_frame == "com":
        pos_ref = pos.mean(axis=2, keepdims=True)
        vel_ref = vel.mean(axis=2, keepdims=True)
    elif reference_frame == "body0":
        pos_ref = pos[:, :, :1, :]
        vel_ref = vel[:, :, :1, :]
    else:
        raise ValueError(f"Unsupported reference frame: {reference_frame}")

    rel_pos = pos - pos_ref
    rel_vel = vel - vel_ref
    transformed = np.concatenate(
        [
            rel_pos.reshape(array.shape[0], array.shape[1], 9),
            rel_vel.reshape(array.shape[0], array.shape[1], 9),
        ],
        axis=-1,
    )
    return transformed.astype(np.float32)


def build_loaders(
    context: np.ndarray,
    target: np.ndarray,
    split_indices: dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split_name, indices in split_indices.items():
        dataset = TensorDataset(torch.from_numpy(context[indices]), torch.from_numpy(target[indices]))
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    log_every_batches: int,
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    started = time.perf_counter()
    for batch_idx, (context, target) in enumerate(loader, start=1):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction = model(context)
            loss = nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch} batch={batch_idx}.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(context.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        if should_log(batch_idx, len(loader), log_every_batches):
            print_progress("train", epoch, batch_idx, len(loader), started, loss_sum / sample_count)
    return loss_sum / max(sample_count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    for context, target in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction = model(context)
        error = prediction - target
        sq_sum += float(error.square().sum().item())
        abs_sum += float(error.abs().sum().item())
        count += int(error.numel())
    return {"mse": sq_sum / max(count, 1), "mae": abs_sum / max(count, 1)}


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for context, target in loader:
        context = context.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction = model(context)
        predictions.append(prediction.detach().cpu().float().numpy())
        targets.append(target.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def compute_metrics(norm_prediction: np.ndarray, norm_target: np.ndarray, raw_prediction: np.ndarray, raw_target: np.ndarray) -> dict[str, Any]:
    norm_error = norm_prediction - norm_target
    raw_error = raw_prediction - raw_target
    return {
        "norm_mse": float(np.square(norm_error).mean()),
        "norm_mae": float(np.abs(norm_error).mean()),
        "raw_mse": float(np.square(raw_error).mean()),
        "raw_mae": float(np.abs(raw_error).mean()),
        "horizon_norm_mse": np.square(norm_error).mean(axis=(0, 2)).tolist(),
        "horizon_norm_mae": np.abs(norm_error).mean(axis=(0, 2)).tolist(),
        "horizon_raw_mse": np.square(raw_error).mean(axis=(0, 2)).tolist(),
        "horizon_raw_mae": np.abs(raw_error).mean(axis=(0, 2)).tolist(),
    }


def save_horizon_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for idx, (norm_mse, norm_mae, raw_mse, raw_mae) in enumerate(
        zip(
            metrics["horizon_norm_mse"],
            metrics["horizon_norm_mae"],
            metrics["horizon_raw_mse"],
            metrics["horizon_raw_mae"],
        ),
        start=1,
    ):
        rows.append(
            {
                "horizon_step": idx,
                "norm_mse": norm_mse,
                "norm_mae": norm_mae,
                "raw_mse": raw_mse,
                "raw_mae": raw_mae,
            }
        )
    write_rows(path, rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def should_log(batch_idx: int, total_batches: int, log_every_batches: int) -> bool:
    return log_every_batches > 0 and (
        batch_idx == 1 or batch_idx == total_batches or batch_idx % log_every_batches == 0
    )


def print_progress(phase: str, epoch: int, batch_idx: int, total_batches: int, started: float, loss: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = batch_idx / elapsed
    eta = int((total_batches - batch_idx) / rate) if rate > 0 else 0
    print(f"{phase} epoch {epoch:03d} [{batch_idx}/{total_batches}] loss={loss:.6f} eta={eta}s", flush=True)


if __name__ == "__main__":
    main()
