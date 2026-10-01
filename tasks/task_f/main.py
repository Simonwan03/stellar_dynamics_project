"""Task F: stability prediction from randomly masked high-resolution prefixes."""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_b.highres import (  # noqa: E402
    FEATURE_NAMES,
    calculate_normalization,
    discover_result_paths,
    load_aligned_labels,
    make_cache,
)
from tasks.common.case_split import case_split_indices, ensure_case_split_manifest  # noqa: E402
from tasks.common.experiment_utils import (  # noqa: E402
    binary_metrics_from_probs,
    ensure_dir,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.task_b.data import batch_id_from_path  # noqa: E402
from tasks.task_f.masking import (  # noqa: E402
    DEFAULT_MASK_RATIOS,
    apply_observation_mask,
    random_observation_mask,
    validate_mask_ratio,
)
from tasks.task_f.models import build_model  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    ratios = tuple(validate_mask_ratio(float(item.strip())) for item in value.split(",") if item.strip())
    if not ratios:
        raise ValueError("At least one mask ratio is required.")
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"Mask ratios must be unique, got {ratios}.")
    return ratios


def ratio_tag(mask_ratio: float) -> str:
    return f"mask{int(round(mask_ratio * 100)):02d}"


class RandomMaskedMemmapDataset(Dataset):
    """Read normalized trajectories and independently mask random time positions."""

    def __init__(
        self,
        features: np.memmap,
        rows: np.ndarray,
        targets: np.ndarray,
        sample_ids: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        mask_ratio: float,
        seed: int,
        dynamic: bool,
        append_mask_channel: bool = True,
    ) -> None:
        self.features = features
        self.rows = np.asarray(rows, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mask_ratio = validate_mask_ratio(mask_ratio)
        self.seed = int(seed)
        self.dynamic = bool(dynamic)
        self.append_mask_channel = bool(append_mask_channel)
        self.epoch = 0
        if not (len(self.rows) == len(self.targets) == len(self.sample_ids)):
            raise ValueError("rows, targets, and sample_ids must have equal length.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.asarray(self.features[self.rows[index]], dtype=np.float32)
        values = np.array((values - self.mean[0]) / self.std[0], dtype=np.float32, copy=True)
        observed = random_observation_mask(
            values.shape[0],
            self.mask_ratio,
            seed=self.seed,
            sample_id=int(self.sample_ids[index]),
            epoch=self.epoch if self.dynamic else 0,
        )
        masked = apply_observation_mask(values, observed, append_mask_channel=self.append_mask_channel)
        target = np.asarray([self.targets[index]], dtype=np.float32)
        return torch.from_numpy(masked), torch.from_numpy(target)


def make_loader(
    features: np.memmap,
    local_indices: np.ndarray,
    global_indices: np.ndarray,
    targets: np.ndarray,
    sample_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask_ratio: float,
    seed: int,
    batch_size: int,
    *,
    train: bool,
    num_workers: int,
    append_mask_channel: bool = True,
) -> tuple[RandomMaskedMemmapDataset, DataLoader]:
    local_indices = np.asarray(local_indices, dtype=np.int64)
    dataset = RandomMaskedMemmapDataset(
        features,
        global_indices[local_indices],
        targets[local_indices],
        sample_ids[local_indices],
        mean,
        std,
        mask_ratio,
        seed,
        dynamic=train,
        append_mask_channel=append_mask_channel,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )
    return dataset, loader


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for xb, yb in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(xb.to(device, non_blocking=use_amp))
        targets.append(yb.numpy().reshape(-1))
        probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(targets), np.concatenate(probabilities)


def train_model(
    model: nn.Module,
    train_dataset: RandomMaskedMemmapDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    best_auc = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_amp)
            yb = yb.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(xb), yb)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            count += len(yb)
        y_val, p_val = predict(model, val_loader, device, use_amp)
        val_metrics = binary_metrics_from_probs(y_val, p_val)
        auc = float(val_metrics["auc"])
        history.append(
            {"epoch": epoch, "loss": loss_sum / max(count, 1), "val_auc": auc, "val_f1": float(val_metrics["f1"])}
        )
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val_auc={auc:.4f}")
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.asarray(probability), np.asarray([0.5])]))
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def default_split_path(dataset_dir: Path, sampling_id: int, group_size: int, horizon: float, seed: int) -> Path:
    name = f"common_case_split_sampling_{sampling_id}_g{group_size}_h{horizon:g}_r70-15-15_seed{seed}.csv"
    previous = ROOT / "results" / "task_b_sampling3_prefix512" / name
    return previous if previous.exists() else dataset_dir / name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task F: random-masked high-resolution trajectory stability prediction."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sampling_3_prefix1024"))
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_f/sampling_3/task_f_masked_prefix"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument(
        "--mask-ratios",
        type=str,
        default=",".join(str(int(round(100 * ratio))) for ratio in DEFAULT_MASK_RATIOS),
    )
    parser.add_argument("--models", type=str, default="tcn")
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tcn-hidden", type=int, default=160)
    parser.add_argument("--tcn-levels", type=int, default=5)
    parser.add_argument("--attention-hidden", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-channel",
        dest="append_mask_channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append observed_mask as an input channel (disable to use all-zero masked tokens).",
    )
    return parser


def normalize_ratio_args(value: str) -> tuple[float, ...]:
    ratios = parse_float_list(value)
    if any(ratio > 1.0 for ratio in ratios):
        raise AssertionError("parse_float_list should reject ratios above one.")
    return ratios


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_ratio_values = [float(item.strip()) for item in args.mask_ratios.split(",") if item.strip()]
    ratios = tuple(value / 100.0 if value >= 1.0 else value for value in raw_ratio_values)
    ratios = tuple(validate_mask_ratio(value) for value in ratios)
    if len(set(ratios)) != len(ratios):
        raise ValueError("--mask-ratios contains duplicate values.")
    models = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    unsupported = sorted(set(models) - {"tcn", "attention"})
    if unsupported or not models:
        raise ValueError(f"Choose one or more models from tcn,attention; invalid={unsupported}.")

    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    batch_ids = [batch_id_from_path(path) for path in paths]
    row_counts: list[int] = []
    import h5py

    for path in paths:
        with h5py.File(path, "r") as handle:
            row_counts.append(int(handle["times_complete"].shape[0]))
    cache_path = (
        args.cache_path
        or Path("/tmp") / f"task_f_sampling_{args.sampling_id}_prefix_{args.prefix_len}_orbital15.f32"
    ).resolve()
    shape, finite = make_cache(paths, cache_path, args.prefix_len, args.time_stride, args.rebuild_cache)
    features = np.memmap(cache_path, mode="r", dtype=np.float32, shape=shape)

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    labels = load_aligned_labels(labels_path, batch_ids, row_counts)
    all_case_ids = labels["index"].to_numpy(dtype=np.int64) // args.group_size
    split_manifest = args.split_manifest or default_split_path(
        args.dataset_dir, args.sampling_id, args.group_size, args.time_horizon, args.seed
    )
    manifest = ensure_case_split_manifest(
        labels_path,
        split_manifest,
        group_size=args.group_size,
        time_horizon=args.time_horizon,
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    valid = (
        np.isin(all_case_ids, manifest["raw_case_id"].to_numpy(dtype=np.int64))
        & (labels["time_unbound"].to_numpy(dtype=np.float64) >= 0.0)
        & (labels["diagnosis"].to_numpy(dtype=np.int64) == 0)
        & finite
    )
    filtered = labels.loc[valid].reset_index(drop=True)
    global_indices = np.flatnonzero(valid).astype(np.int64)
    sample_ids = filtered["index"].to_numpy(dtype=np.int64)
    sample_case_ids = sample_ids // args.group_size
    targets = (filtered["time_unbound"].to_numpy(dtype=np.float64) >= args.time_horizon).astype(np.float32)
    splits = case_split_indices(sample_case_ids, manifest)
    mean, std = calculate_normalization(
        features, global_indices, splits["train"], args.batch_size, shape[1], shape[2]
    )
    np.savez(output_dir / "sequence_normalization.npz", mean=mean, std=std)
    split_names = np.empty(len(filtered), dtype=object)
    for name, indices in splits.items():
        split_names[indices] = name
    filtered["raw_case_id"] = sample_case_ids
    filtered["split"] = split_names
    filtered.to_csv(output_dir / "labels_filtered.csv", index=False)
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=splits["train"], val_idx=splits["val"], test_idx=splits["test"],
        case_ids=sample_case_ids, cache_indices=global_indices,
    )
    config: dict[str, Any] = {
        **vars(args),
        "device": str(device),
        "mask_ratios": ratios,
        "mask_semantics": (
            "ratio of randomly hidden time positions; all feature channels hidden together"
            if args.append_mask_channel
            else "ratio of randomly hidden time positions; hidden feature vectors are all zero"
        ),
        "training_mask_policy": "resampled each epoch from seed, sample index, and epoch; label independent",
        "evaluation_mask_policy": "fixed per sample and ratio using epoch zero",
        "input_shape": [shape[1], shape[2] + int(args.append_mask_channel)],
        "feature_names": [*FEATURE_NAMES, "observed_mask"] if args.append_mask_channel else list(FEATURE_NAMES),
        "num_samples": len(targets),
        "stable_rate": float(targets.mean()),
        "split_manifest": split_manifest,
        "split_counts": {name: len(indices) for name, indices in splits.items()},
    }
    save_json(output_dir / "config.json", config)

    all_metrics: dict[str, Any] = {}
    for ratio in ratios:
        tag = ratio_tag(ratio)
        all_metrics[tag] = {}
        train_ds, train_loader = make_loader(
            features, splits["train"], global_indices, targets, sample_ids, mean, std,
            ratio, args.seed, args.batch_size, train=True, num_workers=args.num_workers,
            append_mask_channel=args.append_mask_channel,
        )
        _, val_loader = make_loader(
            features, splits["val"], global_indices, targets, sample_ids, mean, std,
            ratio, args.seed + 10_000, args.batch_size, train=False, num_workers=args.num_workers,
            append_mask_channel=args.append_mask_channel,
        )
        _, test_loader = make_loader(
            features, splits["test"], global_indices, targets, sample_ids, mean, std,
            ratio, args.seed + 20_000, args.batch_size, train=False, num_workers=args.num_workers,
            append_mask_channel=args.append_mask_channel,
        )
        stable_prior = float(targets[splits["train"]].mean())
        y_test = targets[splits["test"]]
        all_metrics[tag]["constant"] = binary_metrics_from_probs(
            y_test, np.full(len(y_test), stable_prior, dtype=np.float64)
        )
        for model_name in models:
            set_random_seed(args.seed)
            model = build_model(
                model_name,
                shape[2],
                dropout=args.dropout,
                tcn_hidden=args.tcn_hidden,
                tcn_levels=args.tcn_levels,
                attention_hidden=args.attention_hidden,
                attention_heads=args.attention_heads,
                attention_layers=args.attention_layers,
                max_len=shape[1],
                append_mask_channel=args.append_mask_channel,
            )
            model, history = train_model(
                model, train_ds, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                patience=args.early_stopping_patience, device=device, use_amp=use_amp,
            )
            y_val, p_val = predict(model, val_loader, device, use_amp)
            threshold = best_f1_threshold(y_val, p_val)
            y_test_model, p_test = predict(model, test_loader, device, use_amp)
            all_metrics[tag][model_name] = {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "mask_ratio": ratio,
                "validation_threshold": threshold,
                "test_at_0.5": binary_metrics_from_probs(y_test_model, p_test),
                "test_val_f1_threshold": binary_metrics_from_probs(y_test_model, p_test, threshold),
            }
            save_csv(output_dir / f"{tag}_{model_name}_train_log.csv", history)
            prediction_frame = filtered.iloc[splits["test"]][
                ["index", "raw_case_id", "time_unbound", "diagnosis", "split"]
            ].copy()
            prediction_frame["target_stable"] = y_test_model.astype(np.int64)
            prediction_frame["prob_stable"] = p_test
            prediction_frame["mask_ratio"] = ratio
            prediction_frame.to_csv(output_dir / f"{tag}_{model_name}_test_predictions.csv", index=False)
            torch.save({"model_state_dict": model.state_dict()}, output_dir / f"{tag}_{model_name}_model.pt")
            save_binary_curves(
                output_dir / f"{tag}_{model_name}_test_curves.png",
                y_test_model,
                p_test,
                f"Task F random mask={ratio:.0%} {model_name}",
            )
            save_json(output_dir / "metrics.json", all_metrics)
    save_json(output_dir / "metrics.json", all_metrics)
    print(f"Task F complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
