from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data_orbital import (
    DEFAULT_FEATURES,
    parse_feature_list,
    prepare_orbital_datasets,
    save_feature_metadata,
    save_normalization_stats,
)
from src.models.transformer_forecaster import TransformerForecaster
from src.train_utils import (
    ensure_dir,
    evaluate_model,
    plot_example_predictions,
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
    parser = argparse.ArgumentParser(description="Train a Transformer baseline on orbital feature forecasting.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    selected_features = parse_feature_list(args.features)
    set_random_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(args.amp and device.type == "cuda" and torch.cuda.is_available())
    if args.amp and not amp_enabled:
        print("AMP was requested but CUDA is unavailable or device is not CUDA; continuing without AMP.")

    prepared = prepare_orbital_datasets(
        data_path=args.data_path,
        selected_features=selected_features,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        seed=args.seed,
    )

    save_feature_metadata(output_dir / "feature_metadata.json", prepared.feature_metadata)
    save_normalization_stats(output_dir / "normalization_stats.npz", prepared.normalization_stats)

    config = {
        "data_path": args.data_path,
        "resolved_source_paths": prepared.source_paths,
        "output_dir": str(output_dir),
        "features": selected_features,
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
        "num_features": int(prepared.feature_metadata["num_features"]),
    }
    save_json(output_dir / "config.json", config)

    train_loader = build_dataloader(
        prepared.train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = build_dataloader(
        prepared.val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    test_loader = build_dataloader(
        prepared.test_dataset,
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
        )
        val_metrics, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            feature_metadata=prepared.feature_metadata,
            amp_enabled=amp_enabled,
            collect_example=False,
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
    )
    test_metrics["best_epoch"] = best_epoch
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_horizon_metrics_csv(
        output_dir / "horizon_metrics.csv",
        horizon_mse=test_metrics["horizon_mse"],
        horizon_mae=test_metrics["horizon_mae"],
    )
    plot_horizon_mse(test_metrics["horizon_mse"], output_dir / "horizon_mse.png")
    plot_example_predictions(example_payload, prepared.feature_metadata, output_dir / "example_predictions.png")

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


if __name__ == "__main__":
    main()
