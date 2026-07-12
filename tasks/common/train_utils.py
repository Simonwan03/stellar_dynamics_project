from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_horizon_metrics_csv(
    path: str | Path,
    horizon_mse: np.ndarray,
    horizon_mae: np.ndarray,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon_step", "mse", "mae"])
        writer.writeheader()
        for index, (mse_value, mae_value) in enumerate(zip(horizon_mse, horizon_mae), start=1):
            writer.writerow(
                {
                    "horizon_step": index,
                    "mse": float(mse_value),
                    "mae": float(mae_value),
                }
            )


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    log_every_batches: int = 0,
    epoch: int | None = None,
    phase: str = "train",
) -> float:
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    loss_sum = 0.0
    sample_count = 0
    autocast_context = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext

    total_batches = len(dataloader)
    start_time = time.perf_counter()
    for batch_index, batch in enumerate(dataloader, start=1):
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            prediction = model(context)
            loss = criterion(prediction, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = context.size(0)
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        if _should_log_progress(batch_index, total_batches, log_every_batches):
            _print_progress(
                phase=phase,
                epoch=epoch,
                batch_index=batch_index,
                total_batches=total_batches,
                start_time=start_time,
                metric_name="loss",
                metric_value=loss_sum / sample_count,
            )

    if sample_count == 0:
        raise ValueError("Training dataloader produced zero batches.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    feature_metadata: dict[str, Any],
    amp_enabled: bool,
    collect_example: bool = False,
    log_every_batches: int = 0,
    epoch: int | None = None,
    phase: str = "eval",
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_count = 0
    example_payload = None
    horizon_mse_accum = None
    horizon_mae_accum = None
    horizon_count = 0

    group_sq: dict[str, float] = {group["name"]: 0.0 for group in feature_metadata["groups"]}
    group_count: dict[str, int] = {group["name"]: 0 for group in feature_metadata["groups"]}

    autocast_context = partial(torch.amp.autocast, device_type="cuda", enabled=True) if amp_enabled else nullcontext

    total_batches = len(dataloader)
    start_time = time.perf_counter()
    for batch_index, batch in enumerate(dataloader, start=1):
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast_context():
            prediction = model(context)

        error = prediction - target
        sq = error.square()
        abs_err = error.abs()

        total_sq += float(sq.sum().item())
        total_abs += float(abs_err.sum().item())
        total_count += int(sq.numel())

        batch_horizon_sq = sq.sum(dim=(0, 2)).detach().cpu().numpy()
        batch_horizon_abs = abs_err.sum(dim=(0, 2)).detach().cpu().numpy()
        if horizon_mse_accum is None:
            horizon_mse_accum = np.zeros_like(batch_horizon_sq, dtype=np.float64)
            horizon_mae_accum = np.zeros_like(batch_horizon_abs, dtype=np.float64)
        horizon_mse_accum += batch_horizon_sq
        horizon_mae_accum += batch_horizon_abs
        horizon_count += int(prediction.size(0) * prediction.size(2))

        for group in feature_metadata["groups"]:
            start = group["channel_start"]
            end = group["channel_end"]
            group_sq[group["name"]] += float(sq[:, :, start:end].sum().item())
            group_count[group["name"]] += int(sq[:, :, start:end].numel())

        if _should_log_progress(batch_index, total_batches, log_every_batches):
            _print_progress(
                phase=phase,
                epoch=epoch,
                batch_index=batch_index,
                total_batches=total_batches,
                start_time=start_time,
                metric_name="mse",
                metric_value=total_sq / total_count,
            )

        if collect_example and example_payload is None:
            example_payload = {
                "prediction": prediction[0].detach().cpu().numpy(),
                "target": target[0].detach().cpu().numpy(),
            }

    if total_count == 0 or horizon_mse_accum is None or horizon_mae_accum is None:
        raise ValueError("Evaluation dataloader produced zero batches.")

    metrics: dict[str, Any] = {
        "mse": total_sq / total_count,
        "mae": total_abs / total_count,
        "horizon_mse": horizon_mse_accum / horizon_count,
        "horizon_mae": horizon_mae_accum / horizon_count,
    }
    for group in feature_metadata["groups"]:
        name = group["name"]
        metrics[f"{name}_mse"] = group_sq[name] / max(group_count[name], 1)

    return metrics, example_payload


def _should_log_progress(batch_index: int, total_batches: int, log_every_batches: int) -> bool:
    if log_every_batches <= 0:
        return False
    return batch_index == 1 or batch_index == total_batches or batch_index % log_every_batches == 0


def _print_progress(
    phase: str,
    epoch: int | None,
    batch_index: int,
    total_batches: int,
    start_time: float,
    metric_name: str,
    metric_value: float,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    batches_per_second = batch_index / elapsed
    remaining = (total_batches - batch_index) / batches_per_second if batches_per_second > 0 else 0.0
    prefix = f"{phase}"
    if epoch is not None:
        prefix = f"{prefix} epoch {epoch:03d}"
    print(
        f"{prefix} [{batch_index}/{total_batches}] "
        f"{metric_name}={metric_value:.6f} "
        f"{batches_per_second:.2f} batch/s eta={_format_duration(remaining)}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def plot_loss_curve(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_horizon_mse(horizon_mse: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    steps = np.arange(1, len(horizon_mse) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, horizon_mse)
    ax.set_xlabel("Forecast Horizon Step")
    ax.set_ylabel("MSE")
    ax.set_title("Test MSE vs Forecast Horizon")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example_predictions(
    example_payload: dict[str, np.ndarray] | None,
    feature_metadata: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    selections = [
        ("aratios", "aratios"),
        ("eccs", "eccs"),
        ("APs", "APs"),
    ]

    if example_payload is None:
        for ax in axes:
            ax.text(0.5, 0.5, "No example prediction available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    prediction = example_payload["prediction"]
    target = example_payload["target"]

    for ax, (group_name, label) in zip(axes, selections):
        channel_index, channel_name = _find_group_channel(feature_metadata, group_name)
        if channel_index is None:
            ax.text(0.5, 0.5, f"{label} channel not available", ha="center", va="center")
            ax.axis("off")
            continue
        ax.plot(target[:, channel_index], label="target")
        ax.plot(prediction[:, channel_index], label="prediction")
        ax.set_title(f"{label}: {channel_name}")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Forecast Step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _find_group_channel(feature_metadata: dict[str, Any], group_name: str) -> tuple[int | None, str | None]:
    for group in feature_metadata["groups"]:
        if group["name"] == group_name and group["channel_names"]:
            return int(group["channel_start"]), str(group["channel_names"][0])
    return None, None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
