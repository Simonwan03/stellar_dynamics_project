from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data_trajectory import (
    DEFAULT_TRAJECTORY_KEYS,
    parse_trajectory_keys,
    prepare_trajectory_datasets,
    save_feature_metadata,
    save_normalization_stats,
)
from src.models.transformer_forecaster import TransformerForecaster
from src.train_utils import (
    ensure_dir,
    evaluate_model,
    plot_horizon_mse,
    plot_loss_curve,
    save_checkpoint,
    save_history_csv,
    save_horizon_metrics_csv,
    save_json,
    set_random_seed,
    train_one_epoch,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer baseline on direct trajectory forecasting.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trajectory_keys", type=str, default=",".join(DEFAULT_TRAJECTORY_KEYS))
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=0)
    parser.add_argument("--max_test_windows", type=int, default=0)
    parser.add_argument("--max_normalization_trajectories", type=int, default=0)
    parser.add_argument("--log_every_batches", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    trajectory_keys = parse_trajectory_keys(args.trajectory_keys)
    set_random_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(args.amp and device.type == "cuda" and torch.cuda.is_available())
    if args.amp and not amp_enabled:
        print("AMP was requested but CUDA is unavailable or device is not CUDA; continuing without AMP.")

    prepared = prepare_trajectory_datasets(
        data_path=args.data_path,
        trajectory_keys=trajectory_keys,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        seed=args.seed,
        max_normalization_trajectories=args.max_normalization_trajectories,
    )

    save_feature_metadata(output_dir / "feature_metadata.json", prepared.feature_metadata)
    save_normalization_stats(output_dir / "normalization_stats.npz", prepared.normalization_stats)

    train_dataset = maybe_subsample_dataset(prepared.train_dataset, args.max_train_windows, args.seed)
    val_dataset = maybe_subsample_dataset(prepared.val_dataset, args.max_val_windows, args.seed + 1)
    test_dataset = maybe_subsample_dataset(prepared.test_dataset, args.max_test_windows, args.seed + 2)

    config = {
        "data_path": args.data_path,
        "resolved_source_paths": prepared.source_paths,
        "output_dir": str(output_dir),
        "trajectory_keys": trajectory_keys,
        "context_len": args.context_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "amp": bool(args.amp),
        "amp_enabled": amp_enabled,
        "device": str(device),
        "log_every_batches": args.log_every_batches,
        "trajectory_count": prepared.trajectory_count,
        "time_steps": prepared.time_steps,
        "split_trajectory_counts": {
            split_name: int(indices.size) for split_name, indices in prepared.split_indices.items()
        },
        "split_window_counts": {
            "train": len(prepared.train_dataset),
            "val": len(prepared.val_dataset),
            "test": len(prepared.test_dataset),
        },
        "used_window_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "max_train_windows": args.max_train_windows,
        "max_val_windows": args.max_val_windows,
        "max_test_windows": args.max_test_windows,
        "max_normalization_trajectories": args.max_normalization_trajectories,
        "normalization_trajectory_count": int(prepared.normalization_stats.get("trajectory_count", -1)),
        "num_features": int(prepared.feature_metadata["num_features"]),
        "per_step_shapes": prepared.feature_metadata["per_step_shapes"],
    }
    save_json(output_dir / "config.json", config)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )

    model = TransformerForecaster(
        num_features=prepared.feature_metadata["num_features"],
        context_len=args.context_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            log_every_batches=args.log_every_batches,
            epoch=epoch,
            phase="train",
        )
        val_metrics, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            feature_metadata=prepared.feature_metadata,
            amp_enabled=amp_enabled,
            collect_example=False,
            log_every_batches=args.log_every_batches,
            epoch=epoch,
            phase="val",
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["mse"]),
            "val_mae": float(val_metrics["mae"]),
        }
        for group in prepared.feature_metadata["groups"]:
            row[f"val_{group['name']}_mse"] = float(val_metrics[f"{group['name']}_mse"])
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['mse']:.6f} | val_mae={val_metrics['mae']:.6f}"
        )

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = float(val_metrics["mse"])
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_loss,
                config=config,
            )

    save_history_csv(output_dir / "train_log.csv", history)
    plot_loss_curve(history, output_dir / "loss_curve.png")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_dir / "final_model.pt",
    )

    best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, example_payload = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        feature_metadata=prepared.feature_metadata,
        amp_enabled=amp_enabled,
        collect_example=True,
        log_every_batches=args.log_every_batches,
        phase="test",
    )
    test_metrics["best_epoch"] = best_epoch
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_horizon_metrics_csv(
        output_dir / "horizon_metrics.csv",
        horizon_mse=test_metrics["horizon_mse"],
        horizon_mae=test_metrics["horizon_mae"],
    )
    plot_horizon_mse(test_metrics["horizon_mse"], output_dir / "horizon_mse.png")
    plot_example_trajectory_predictions(
        example_payload=example_payload,
        feature_metadata=prepared.feature_metadata,
        normalization_stats=prepared.normalization_stats,
        path=output_dir / "example_predictions.png",
    )

    print(f"Finished. Best epoch: {best_epoch}, test_mse={test_metrics['mse']:.6f}, test_mae={test_metrics['mae']:.6f}")

    prepared.train_dataset.close()
    prepared.val_dataset.close()
    prepared.test_dataset.close()


def build_dataloader(
    dataset: torch.utils.data.Dataset,
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


def maybe_subsample_dataset(
    dataset: torch.utils.data.Dataset,
    max_windows: int,
    seed: int,
) -> torch.utils.data.Dataset:
    if max_windows <= 0 or max_windows >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_windows, replace=False)
    return Subset(dataset, indices.astype(np.int64).tolist())


def plot_example_trajectory_predictions(
    example_payload: dict[str, np.ndarray] | None,
    feature_metadata: dict[str, object],
    normalization_stats: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if example_payload is None:
        for ax in axes:
            ax.text(0.5, 0.5, "No example prediction available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    mean = normalization_stats["mean"].reshape(1, -1)
    std = normalization_stats["std"].reshape(1, -1)
    position_group = next(
        group for group in feature_metadata["groups"] if group["source_key"] == "positions_complete"  # type: ignore[index]
    )
    start = int(position_group["channel_start"])
    end = int(position_group["channel_end"])
    per_step_shape = tuple(position_group["per_step_shape"])
    prediction = (example_payload["prediction"][:, start:end] * std[:, start:end] + mean[:, start:end]).reshape(
        -1, *per_step_shape
    )
    target = (example_payload["target"][:, start:end] * std[:, start:end] + mean[:, start:end]).reshape(
        -1, *per_step_shape
    )
    body_count = prediction.shape[1] if prediction.ndim >= 3 else 1
    labels = [f"body {idx}" for idx in range(body_count)]

    for body_idx, label in enumerate(labels):
        axes[0].plot(target[:, body_idx, 0], target[:, body_idx, 1], label=f"{label} target")
        axes[0].plot(prediction[:, body_idx, 0], prediction[:, body_idx, 1], linestyle="--", label=f"{label} pred")
        axes[1].plot(target[:, body_idx, 0], label=f"{label} target x")
        axes[1].plot(prediction[:, body_idx, 0], linestyle="--", label=f"{label} pred x")

    axes[0].set_title("Future XY Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Future X Coordinate")
    axes[1].set_xlabel("Forecast Step")
    axes[1].set_ylabel("x")
    axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
