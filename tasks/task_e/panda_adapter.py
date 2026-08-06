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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_e.foundation import (  # noqa: E402
    DEFAULT_ORBITAL_FEATURES,
    DEFAULT_TRAJECTORY_KEYS,
    compute_metrics,
    compute_metric_scale,
    load_eval_data,
    load_panda_model,
)
from tasks.common.train_utils import save_json, set_random_seed  # noqa: E402


class TemporalMLPAdapter(nn.Module):
    def __init__(self, in_len: int, out_len: int, hidden_len: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_len = int(hidden_len or max(in_len, out_len))
        self.resample = nn.Linear(in_len, out_len)
        self.delta = nn.Sequential(
            nn.Linear(in_len, hidden_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_len, out_len),
        )
        self._init_as_resample(in_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.resample(x) + self.delta(x)
        return x.transpose(1, 2)

    def _init_as_resample(self, in_len: int, out_len: int) -> None:
        with torch.no_grad():
            self.resample.weight.copy_(linear_resample_weight(in_len, out_len))
            self.resample.bias.zero_()
            final = self.delta[-1]
            if isinstance(final, nn.Linear):
                final.weight.zero_()
                final.bias.zero_()


def linear_resample_weight(in_len: int, out_len: int) -> torch.Tensor:
    if in_len <= 0 or out_len <= 0:
        raise ValueError(f"Sequence lengths must be positive, got {in_len=} {out_len=}")
    weight = torch.zeros(out_len, in_len, dtype=torch.float32)
    if out_len == 1:
        weight[0, 0] = 1.0
        return weight
    if in_len == 1:
        weight[:, 0] = 1.0
        return weight
    positions = torch.linspace(0, in_len - 1, out_len)
    left = torch.floor(positions).long()
    right = torch.clamp(left + 1, max=in_len - 1)
    frac = positions - left.float()
    weight[torch.arange(out_len), left] += 1.0 - frac
    weight[torch.arange(out_len), right] += frac
    return weight


class GRUForecastDecoder(nn.Module):
    """Encode a foundation forecast and decode an arbitrary output horizon."""

    def __init__(
        self,
        num_features: int,
        hidden_len: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.hidden_len = int(hidden_len)
        self.num_layers = int(num_layers)
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}.")
        if self.hidden_len <= 0:
            raise ValueError(f"hidden_len must be positive, got {self.hidden_len}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        gru_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.num_features,
            hidden_size=self.hidden_len,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=self.num_features + 1,
            hidden_size=self.hidden_len,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_len, self.num_features)

    def forward(self, foundation_prediction: torch.Tensor, output_len: int) -> torch.Tensor:
        if foundation_prediction.ndim != 3:
            raise ValueError(
                f"Expected foundation prediction [B, T, C], got {tuple(foundation_prediction.shape)}."
            )
        output_len = int(output_len)
        if output_len <= 0:
            raise ValueError(f"output_len must be positive, got {output_len}.")
        batch, _, channels = foundation_prediction.shape
        if channels != self.num_features:
            raise ValueError(
                f"GRU decoder expected {self.num_features} channels, got {channels}."
            )
        _, hidden = self.encoder(foundation_prediction)
        last_value = foundation_prediction[:, -1:, :].expand(-1, output_len, -1)
        time_query = torch.linspace(
            0.0,
            1.0,
            output_len,
            device=foundation_prediction.device,
            dtype=foundation_prediction.dtype,
        ).view(1, output_len, 1).expand(batch, -1, -1)
        decoder_input = torch.cat([last_value, time_query], dim=-1)
        decoded, _ = self.decoder(decoder_input, hidden)
        return self.output(decoded)


class PandaTemporalAdapter(nn.Module):
    def __init__(
        self,
        panda: nn.Module,
        input_len: int,
        panda_context_len: int,
        panda_pred_len: int,
        output_len: int,
        hidden_len: int | None,
        dropout: float,
        freeze_panda: bool,
        decoder: str = "mlp",
        num_features: int | None = None,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.pre = TemporalMLPAdapter(input_len, panda_context_len, hidden_len=hidden_len, dropout=dropout)
        self.panda = panda
        self.decoder_type = str(decoder).lower()
        if self.decoder_type not in {"mlp", "gru"}:
            raise ValueError(f"decoder must be 'mlp' or 'gru', got {decoder!r}.")
        self.output_len = int(output_len)
        if self.output_len <= 0:
            raise ValueError(f"output_len must be positive, got {self.output_len}.")
        self.panda_pred_len = int(panda_pred_len)
        self.freeze_panda = bool(freeze_panda)
        self.num_features = int(
            num_features or getattr(getattr(panda, "config", None), "num_input_channels", 0)
        )
        self.gru_layers = int(gru_layers)
        if self.decoder_type == "mlp":
            self.post: nn.Module | None = TemporalMLPAdapter(
                panda_pred_len, self.output_len, hidden_len=hidden_len, dropout=dropout
            )
            self.gru_decoder: GRUForecastDecoder | None = None
        else:
            if self.num_features <= 0:
                raise ValueError("GRU decoder requires a positive num_features value.")
            gru_hidden_len = int(hidden_len or max(32, panda_pred_len, self.output_len))
            self.post = None
            self.gru_decoder = GRUForecastDecoder(
                num_features=self.num_features,
                hidden_len=gru_hidden_len,
                num_layers=self.gru_layers,
                dropout=dropout,
            )

        if freeze_panda:
            for parameter in self.panda.parameters():
                parameter.requires_grad = False
            self.panda.eval()

    def train(self, mode: bool = True) -> "PandaTemporalAdapter":
        super().train(mode)
        if self.freeze_panda:
            self.panda.eval()
        return self

    def forward(self, context: torch.Tensor, output_len: int | None = None) -> torch.Tensor:
        requested_len = self.output_len if output_len is None else int(output_len)
        if requested_len <= 0:
            raise ValueError(f"output_len must be positive, got {requested_len}.")
        adapted_context = self.pre(context)
        panda_context = adapted_context.float()
        observed = torch.ones_like(panda_context, dtype=torch.bool)
        # Panda can produce NaNs under CUDA autocast on this data, so keep the
        # backbone in FP32 while still allowing AMP around the lightweight adapters.
        with torch.amp.autocast("cuda", enabled=False):
            outputs = self.panda(
                past_values=panda_context,
                past_observed_mask=observed,
                future_values=None,
            )
        panda_prediction = outputs.prediction_outputs.to(dtype=adapted_context.dtype)
        if panda_prediction.ndim != 3:
            raise ValueError(
                f"Panda returned shape {tuple(panda_prediction.shape)}, expected [B, T, C]."
            )
        if panda_prediction.shape[1] != self.panda_pred_len:
            raise ValueError(
                f"Panda returned prediction length {panda_prediction.shape[1]}, expected {self.panda_pred_len}."
            )
        if self.decoder_type == "mlp":
            if requested_len != self.output_len:
                raise ValueError(
                    "The MLP adapter decoder has a fixed output length; use decoder='gru' "
                    "to request arbitrary output_len values."
                )
            if self.post is None:
                raise RuntimeError("MLP adapter decoder was not initialized.")
            return self.post(panda_prediction)
        if self.gru_decoder is None:
            raise RuntimeError("GRU adapter decoder was not initialized.")
        return self.gru_decoder(panda_prediction, requested_len)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune temporal MLP window adapters around a Panda forecaster.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--representation", choices=["trajectory", "orbital15"], default="orbital15")
    parser.add_argument("--trajectory-keys", type=str, default=",".join(DEFAULT_TRAJECTORY_KEYS))
    parser.add_argument("--orbital-features", type=str, default=",".join(DEFAULT_ORBITAL_FEATURES))
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--panda-context-len", type=int, default=None)
    parser.add_argument("--panda-pred-len", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=4096)
    parser.add_argument(
        "--max-abs-value",
        type=float,
        default=1e4,
        help="Drop windows whose context or target has any absolute channel value above this threshold. Use <=0 to disable.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-hidden-len", type=int, default=None)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument(
        "--adapter-decoder",
        "--adapter_decoder",
        dest="adapter_decoder",
        choices=["mlp", "gru"],
        default="mlp",
        help="Temporal adapter decoder. GRU supports requesting arbitrary output lengths at inference.",
    )
    parser.add_argument("--adapter-gru-layers", "--adapter_gru_layers", dest="adapter_gru_layers", type=int, default=1)
    parser.add_argument("--freeze-panda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "panda_temporal_adapter")
    parser.add_argument("--panda-model-id", type=str, default="GilpinLab/panda-72M")
    parser.add_argument(
        "--hf-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load Panda from the local Hugging Face cache unless --no-hf-local-files-only is passed.",
    )
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
    orbital_features = [item.strip() for item in args.orbital_features.split(",") if item.strip()]
    data = load_eval_data(
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        representation=args.representation,
        trajectory_keys=trajectory_keys,
        orbital_features=orbital_features,
        context_len=args.input_len,
        pred_len=args.output_len,
        num_windows=args.num_windows,
        seed=args.seed,
    )
    raw_window_count = int(data.context.shape[0])
    data.context, data.target, kept_indices = filter_window_values(
        data.context,
        data.target,
        max_abs_value=args.max_abs_value,
    )
    data.window_metadata = [data.window_metadata[int(index)] for index in kept_indices]

    panda = load_panda_model(args, str(device))
    panda_context_len = int(args.panda_context_len or getattr(panda.config, "context_length", 512))
    panda_pred_len = int(args.panda_pred_len or getattr(panda.config, "prediction_length", 128))
    model = PandaTemporalAdapter(
        panda=panda,
        input_len=args.input_len,
        panda_context_len=panda_context_len,
        panda_pred_len=panda_pred_len,
        output_len=args.output_len,
        hidden_len=args.adapter_hidden_len,
        dropout=args.adapter_dropout,
        freeze_panda=args.freeze_panda,
        decoder=args.adapter_decoder,
        num_features=len(data.channel_names),
        gru_layers=args.adapter_gru_layers,
    ).to(device)

    train_loader, val_loader, test_loader, split_counts = build_loaders(
        context=data.context,
        target=data.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    config = {
        "dataset_dir": str(args.dataset_dir),
        "sampling_id": args.sampling_id,
        "representation": args.representation,
        "trajectory_keys": trajectory_keys,
        "orbital_features": orbital_features,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "panda_context_len": panda_context_len,
        "panda_pred_len": panda_pred_len,
        "num_windows": args.num_windows,
        "raw_window_count": raw_window_count,
        "filtered_window_count": int(data.context.shape[0]),
        "max_abs_value": args.max_abs_value,
        "split_window_counts": split_counts,
        "num_channels": len(data.channel_names),
        "channel_names": data.channel_names,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "adapter_hidden_len": args.adapter_hidden_len,
        "adapter_dropout": args.adapter_dropout,
        "adapter_decoder": args.adapter_decoder,
        "adapter_gru_layers": args.adapter_gru_layers,
        "freeze_panda": args.freeze_panda,
        "amp": args.amp,
        "amp_enabled": amp_enabled,
        "device": str(device),
        "panda_model_id": args.panda_model_id,
        "hf_local_files_only": args.hf_local_files_only,
        "seed": args.seed,
    }
    save_json(args.output_dir / "config.json", config)

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
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_every_batches=args.log_every_batches,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        append_csv(args.output_dir / "train_log.csv", history)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | val_mae={val_metrics['mae']:.6f}",
            flush=True,
        )
        if val_metrics["mse"] < best_val:
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
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        args.output_dir / "final_model.pt",
    )

    if best_epoch < 0:
        raise RuntimeError(
            "No finite validation checkpoint was saved. Check the first non-finite loss error above, "
            "lower --lr, disable --amp, or tighten --max-abs-value."
        )

    best_checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    prediction, target = predict_all(model, test_loader, device, amp_enabled)
    metric_scale = compute_metric_scale(data.context)
    metrics = compute_metrics(prediction, target, metric_scale)
    metrics["best_epoch"] = best_epoch
    save_json(args.output_dir / "test_metrics.json", metrics)
    save_horizon_csv(args.output_dir / "horizon_metrics.csv", metrics)
    print(
        f"TEST | raw_mse={metrics['raw_mse']:.6g} raw_mae={metrics['raw_mae']:.6g} "
        f"z_mse={metrics['z_mse']:.6g}",
        flush=True,
    )


def build_loaders(
    context: np.ndarray,
    target: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    count = context.shape[0]
    if count < 3:
        raise ValueError(f"Need at least 3 windows for train/val/test splits, got {count}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    train_idx = order[:train_count]
    val_idx = order[train_count : train_count + val_count]
    test_idx = order[train_count + val_count :]

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(context[indices]), torch.from_numpy(target[indices]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_idx, True),
        make_loader(val_idx, False),
        make_loader(test_idx, False),
        {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
    )


def filter_window_values(
    context: np.ndarray,
    target: np.ndarray,
    max_abs_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_abs_value <= 0:
        indices = np.arange(context.shape[0], dtype=np.int64)
        return context, target, indices
    context_flat = context.reshape(context.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    finite_mask = np.isfinite(context_flat).all(axis=1) & np.isfinite(target_flat).all(axis=1)
    range_mask = (np.abs(context_flat) <= max_abs_value).all(axis=1) & (
        np.abs(target_flat) <= max_abs_value
    ).all(axis=1)
    keep_mask = finite_mask & range_mask
    indices = np.nonzero(keep_mask)[0].astype(np.int64)
    if indices.size < 3:
        raise ValueError(
            f"Only {indices.size} windows remain after max_abs_value={max_abs_value} filtering. "
            "Increase --num-windows, raise --max-abs-value, or pass --max-abs-value 0 to disable filtering."
        )
    dropped = int(context.shape[0] - indices.size)
    if dropped:
        print(
            f"Dropped {dropped} / {context.shape[0]} windows with non-finite values or "
            f"abs(value) > {max_abs_value:g}.",
            flush=True,
        )
    return context[indices], target[indices], indices


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
            finite_context = bool(torch.isfinite(context).all().item())
            finite_target = bool(torch.isfinite(target).all().item())
            finite_prediction = bool(torch.isfinite(prediction).all().item())
            raise FloatingPointError(
                "Non-finite training loss at "
                f"epoch={epoch} batch={batch_idx}. "
                f"finite_context={finite_context} finite_target={finite_target} "
                f"finite_prediction={finite_prediction}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = context.shape[0]
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
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
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


def save_horizon_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for idx, (raw_mse, raw_mae, z_mse) in enumerate(
        zip(metrics["horizon_raw_mse"], metrics["horizon_raw_mae"], metrics["horizon_z_mse"]), start=1
    ):
        rows.append({"horizon_step": idx, "raw_mse": raw_mse, "raw_mae": raw_mae, "z_mse": z_mse})
    write_rows(path, rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def print_progress(
    phase: str,
    epoch: int,
    batch_idx: int,
    total_batches: int,
    started: float,
    loss: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = batch_idx / elapsed
    eta = int((total_batches - batch_idx) / rate) if rate > 0 else 0
    print(f"{phase} epoch {epoch:03d} [{batch_idx}/{total_batches}] loss={loss:.6f} eta={eta}s", flush=True)


if __name__ == "__main__":
    main()
