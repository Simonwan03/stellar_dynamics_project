#!/usr/bin/env python3
"""Run Task B MLP/CNN/TCN on the sampling_3 high-resolution prefix dataset.

The regular Task B entry point keeps all decoded arrays in RAM, which is useful
for the small sampling_2 experiments but too large for the 1024-step shards on
the current machine.  This runner converts only the requested orbital15
prefix into a float32 memmap and feeds normalized batches from that memmap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from tasks.common.case_split import (
    case_split_indices,
    ensure_case_split_manifest,
)
from tasks.common.experiment_utils import (
    binary_metrics_from_probs,
    ensure_dir,
    save_binary_curves,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.common.orbital_schema import ANGLE_RESULT_KEYS, ORBITAL15_RESULT_KEYS
from tasks.task_b.data import batch_id_from_path, discover_result_paths, load_aligned_labels
from tasks.task_b.main import (
    MLPClassifier,
    CNN1DClassifier,
    TCNClassifier,
    predict_classifier,
    save_test_predictions,
    train_classifier,
)


FEATURE_NAMES = (
    "aratios_complete_0",
    "mratios_complete_0",
    "mratios_complete_1",
    "eccs_complete_0",
    "eccs_complete_1",
    "incs_complete_0",
    "incs_complete_1",
    "APs_complete_0",
    "APs_complete_1",
    "APs_complete_2",
    "APs_complete_3",
    "LANs_complete_0",
    "LANs_complete_1",
    "LANs_complete_2",
    "LANs_complete_3",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory-efficient Task B MLP/CNN/TCN runner for sampling_3."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sampling_3_prefix1024"))
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_b_sampling3_prefix512"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--models", type=str, default="mlp,cnn,tcn")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision when running on a CUDA device.",
    )
    parser.add_argument("--no-preload", action="store_true", help="Keep normalized batches on the memmap instead of loading them into RAM.")
    return parser


def source_signature(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    ]


def cache_paths(cache_path: Path) -> tuple[Path, Path]:
    return (
        Path(str(cache_path) + ".valid.npy"),
        Path(str(cache_path) + ".meta.json"),
    )


def read_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def make_cache(
    paths: list[Path],
    cache_path: Path,
    prefix_len: int,
    time_stride: int,
    rebuild: bool,
) -> tuple[tuple[int, int, int], np.ndarray]:
    if prefix_len <= 0 or time_stride <= 0:
        raise ValueError("prefix_len and time_stride must be positive.")
    cache_path = cache_path.resolve()
    ensure_dir(cache_path.parent)
    valid_path, meta_path = cache_paths(cache_path)

    with h5py.File(paths[0], "r") as handle:
        source_steps = int(handle["times_complete"].shape[1])
        if prefix_len > source_steps:
            raise ValueError(
                f"Requested prefix_len={prefix_len}, but {paths[0]} has only {source_steps} steps."
            )
        steps = len(range(0, prefix_len, time_stride))
        first_shapes = {
            key: tuple(handle[key].shape[2:]) for key in ORBITAL15_RESULT_KEYS
        }

    row_counts: list[int] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if int(handle["times_complete"].shape[1]) < prefix_len:
                raise ValueError(f"{path} is shorter than the requested prefix.")
            row_counts.append(int(handle["times_complete"].shape[0]))
    total_rows = int(sum(row_counts))
    shape = (total_rows, steps, len(FEATURE_NAMES))
    expected_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.float32).itemsize)
    signature = source_signature(paths)
    expected_meta = {
        "shape": list(shape),
        "dtype": "float32",
        "prefix_len": int(prefix_len),
        "time_stride": int(time_stride),
        "source_signature": signature,
        "feature_names": list(FEATURE_NAMES),
    }

    meta = read_cache_meta(meta_path)
    cache_ok = (
        not rebuild
        and cache_path.exists()
        and valid_path.exists()
        and meta == expected_meta
        and cache_path.stat().st_size == expected_bytes
    )
    if cache_ok:
        finite = np.load(valid_path)
        if finite.shape == (total_rows,):
            print(f"Reusing feature cache: {cache_path} ({expected_bytes / 2**30:.2f} GiB)")
            return shape, finite.astype(bool, copy=False)
        print("Feature-cache validity mask has the wrong shape; rebuilding.")

    tmp_path = Path(str(cache_path) + ".part")
    tmp_path.unlink(missing_ok=True)
    print(f"Building feature cache: {cache_path} ({expected_bytes / 2**30:.2f} GiB)")
    features = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=shape)
    finite = np.zeros(total_rows, dtype=bool)
    time_sel = slice(0, prefix_len, time_stride)
    offset = 0

    for shard_no, (path, row_count) in enumerate(zip(paths, row_counts), start=1):
        with h5py.File(path, "r") as handle:
            target = features[offset : offset + row_count]
            cursor = 0
            for key in ORBITAL15_RESULT_KEYS:
                arr = np.asarray(
                    handle[key][slice(0, row_count), time_sel, ...],
                    dtype=np.float32,
                ).reshape(row_count, steps, -1)
                width = int(arr.shape[-1])
                if key in ANGLE_RESULT_KEYS:
                    np.sin(arr, out=target[..., cursor : cursor + width])
                    np.cos(arr, out=target[..., cursor + width : cursor + 2 * width])
                    cursor += 2 * width
                else:
                    target[..., cursor : cursor + width] = arr
                    cursor += width
            if cursor != len(FEATURE_NAMES):
                raise RuntimeError(f"Feature width mismatch for {path}: wrote {cursor}.")
            finite[offset : offset + row_count] = np.isfinite(target).all(axis=(1, 2))
        offset += row_count
        print(f"  cached shard {shard_no:02d}/{len(paths):02d}: {path.name} rows={row_count}")

    features.flush()
    del features
    os.replace(tmp_path, cache_path)
    np.save(valid_path, finite)
    meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")
    return shape, finite


def calculate_normalization(
    features: np.memmap,
    global_indices: np.ndarray,
    train_indices: np.ndarray,
    batch_size: int,
    steps: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = 0
    train_global = global_indices[train_indices]
    for start in range(0, len(train_global), batch_size):
        batch = np.asarray(features[train_global[start : start + batch_size]], dtype=np.float32)
        total += batch.sum(axis=(0, 1), dtype=np.float64)
        total_sq += np.square(batch, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
        count += int(batch.shape[0] * batch.shape[1])
    if count == 0:
        raise ValueError("The training split is empty.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.reshape(1, 1, channels).astype(np.float32), std.reshape(1, 1, channels).astype(np.float32)


class MemmapBatchDataset(IterableDataset):
    def __init__(
        self,
        features: np.memmap,
        global_indices: np.ndarray,
        targets: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.features = features
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = mean
        self.std = std
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.global_indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = np.arange(len(self.global_indices), dtype=np.int64)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for start in range(0, len(order), self.batch_size):
            positions = order[start : start + self.batch_size]
            rows = self.global_indices[positions]
            batch = np.asarray(self.features[rows], dtype=np.float32)
            batch = np.ascontiguousarray((batch - self.mean) / self.std, dtype=np.float32)
            target = np.asarray(self.targets[positions], dtype=np.float32).reshape(-1, 1)
            yield torch.from_numpy(batch), torch.from_numpy(target)


def identity_collate(batch):
    return batch


def make_loader(
    features: np.memmap,
    local_indices: np.ndarray,
    global_indices: np.ndarray,
    targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = MemmapBatchDataset(
        features=features,
        global_indices=global_indices[local_indices],
        targets=targets[local_indices],
        mean=mean,
        std=std,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        collate_fn=identity_collate,
    )


def read_existing_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    models = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    supported = {"mlp", "cnn", "tcn"}
    unknown = sorted(set(models) - supported)
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}; choose from {sorted(supported)}")
    if not models:
        raise ValueError("At least one model is required.")

    paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    batch_ids = [batch_id_from_path(path) for path in paths]
    row_counts = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            row_counts.append(int(handle["times_complete"].shape[0]))
    shape, finite = make_cache(
        paths,
        args.cache_path
        or Path("/tmp") / f"task_b_sampling_{args.sampling_id}_prefix_{args.prefix_len}_orbital15.f32",
        args.prefix_len,
        args.time_stride,
        args.rebuild_cache,
    )
    cache_path = (
        args.cache_path
        or Path("/tmp") / f"task_b_sampling_{args.sampling_id}_prefix_{args.prefix_len}_orbital15.f32"
    ).resolve()
    features = np.memmap(cache_path, mode="r", dtype=np.float32, shape=shape)

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    labels = load_aligned_labels(labels_path, batch_ids, row_counts)
    if len(labels) != shape[0] or len(finite) != len(labels):
        raise ValueError(f"Cache/label row mismatch: cache={shape[0]} labels={len(labels)} finite={len(finite)}")

    manifest_path = output_dir / (
        f"common_case_split_sampling_{args.sampling_id}_g{args.group_size}_"
        f"h{args.time_horizon:g}_r{int(args.train_ratio * 100)}-"
        f"{int(args.val_ratio * 100)}-{int(args.test_ratio * 100)}_seed{args.seed}.csv"
    )
    manifest = ensure_case_split_manifest(
        labels_path,
        manifest_path,
        group_size=args.group_size,
        time_horizon=args.time_horizon,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    all_case_ids = (labels["index"].to_numpy(dtype=np.int64) // args.group_size).astype(np.int64)
    known_case_mask = np.isin(all_case_ids, manifest["raw_case_id"].to_numpy(dtype=np.int64))
    label_valid = (
        (labels["time_unbound"].to_numpy(dtype=np.float32) >= 0.0)
        & (labels["diagnosis"].to_numpy(dtype=np.int64) == 0)
    )
    sample_mask = known_case_mask & label_valid & finite
    if args.max_shards is None:
        retained_case_ids = all_case_ids[sample_mask]
        expected_case_ids = manifest["raw_case_id"].to_numpy(dtype=np.int64)
        counts = pd.Series(retained_case_ids).value_counts()
        incomplete = expected_case_ids[
            counts.reindex(expected_case_ids, fill_value=0).to_numpy() != args.group_size
        ]
        if incomplete.size:
            raise ValueError(
                "The full dataset does not contain all complete systems after filtering; "
                f"missing/incomplete systems={incomplete[:10].tolist()} (count={incomplete.size})."
            )

    filtered_labels = labels.loc[sample_mask].reset_index(drop=True)
    global_indices = np.flatnonzero(sample_mask).astype(np.int64)
    targets = (
        filtered_labels["time_unbound"].to_numpy(dtype=np.float32) >= float(args.time_horizon)
    ).astype(np.float32)
    sample_case_ids = (filtered_labels["index"].to_numpy(dtype=np.int64) // args.group_size).astype(np.int64)
    split_indices = case_split_indices(sample_case_ids, manifest)
    split_membership = np.empty(len(filtered_labels), dtype=object)
    for split_name, split_rows in split_indices.items():
        split_membership[split_rows] = split_name
    filtered_labels = filtered_labels.copy()
    filtered_labels["raw_case_id"] = sample_case_ids
    filtered_labels["split"] = split_membership
    filtered_labels.to_csv(output_dir / "labels_filtered.csv", index=False)
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
        case_ids=sample_case_ids,
        cache_indices=global_indices,
    )

    mean, std = calculate_normalization(
        features,
        global_indices,
        split_indices["train"],
        batch_size=args.batch_size,
        steps=shape[1],
        channels=shape[2],
    )
    np.savez(output_dir / "sequence_normalization.npz", mean=mean, std=std)

    if not args.no_preload:
        print(f"Preloading normalized features into RAM: {len(global_indices)} rows")
        normalized_features = np.empty(
            (len(global_indices), shape[1], shape[2]), dtype=np.float32
        )
        if len(global_indices):
            breaks = np.flatnonzero(np.diff(global_indices) != 1) + 1
            bounds = np.concatenate(
                (np.array([0], dtype=np.int64), breaks, np.array([len(global_indices)], dtype=np.int64))
            )
            print(f"  loading {len(bounds) - 1} contiguous cache blocks")
            for left, right in zip(bounds[:-1], bounds[1:]):
                left = int(left)
                right = int(right)
                source_left = int(global_indices[left])
                source_right = int(global_indices[right - 1]) + 1
                normalized_features[left:right] = np.asarray(
                    features[source_left:source_right], dtype=np.float32
                )
        normalized_features -= mean
        normalized_features /= std
        del features
        features = normalized_features
        global_indices = np.arange(len(targets), dtype=np.int64)
        mean = np.zeros_like(mean)
        std = np.ones_like(std)
        print(f"Preloaded normalized features: {features.nbytes / 2**30:.2f} GiB")
    effective_scale = {
        "mlp": {"hidden": [256, 128, 64]},
        "cnn": {"hidden": 192},
        "tcn": {"hidden": 160, "levels": 5},
    }
    config = {
        **vars(args),
        "device": str(device),
        "result_paths": paths,
        "cache_path": cache_path,
        "cache_shape": shape,
        "cache_dtype": "float32",
        "preload_normalized_features": not args.no_preload,
        "input_shape": [shape[1], shape[2]],
        "feature_names": list(FEATURE_NAMES),
        "num_samples": int(len(targets)),
        "target_mean": float(targets.mean()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "split_counts": {name: int(rows.size) for name, rows in split_indices.items()},
        "split_case_counts": {
            name: int(np.unique(sample_case_ids[rows]).size)
            for name, rows in split_indices.items()
        },
        "split_manifest": manifest_path,
        "split_unit": "complete raw_case_id system",
        "split_horizon": float(args.time_horizon),
        "effective_model_scale": effective_scale,
    }
    save_json(output_dir / "config.json", config)

    metrics_path = output_dir / "metrics.json"
    metrics = read_existing_metrics(metrics_path)
    for name in models:
        set_random_seed(args.seed)
        if name == "mlp":
            factory = lambda: MLPClassifier(
                shape[1] * shape[2], hidden=(256, 128, 64), dropout=args.dropout
            )
        elif name == "cnn":
            factory = lambda: CNN1DClassifier(shape[2], hidden=192, dropout=args.dropout)
        else:
            factory = lambda: TCNClassifier(shape[2], hidden=160, levels=5, dropout=args.dropout)

        print(
            f"Starting {name}: train={len(split_indices['train'])} "
            f"val={len(split_indices['val'])} test={len(split_indices['test'])} "
            f"input={shape[1:]} device={device}"
        )
        train_loader = make_loader(
            features, split_indices["train"], global_indices, targets, mean, std,
            args.batch_size, True, args.seed
        )
        val_loader = make_loader(
            features, split_indices["val"], global_indices, targets, mean, std,
            args.batch_size, False, args.seed
        )
        test_loader = make_loader(
            features, split_indices["test"], global_indices, targets, mean, std,
            args.batch_size, False, args.seed
        )
        model = factory()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model, history = train_classifier(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            device,
            early_stopping_patience=args.early_stopping_patience,
            amp=use_amp,
        )
        y_eval, probability = predict_classifier(model, test_loader, device, amp=use_amp)
        metrics[name] = {
            "parameters": int(parameter_count),
            "test": binary_metrics_from_probs(y_eval, probability),
        }
        save_csv(output_dir / f"{name}_train_log.csv", history)
        save_test_predictions(
            output_dir / f"{name}_test_predictions.csv",
            filtered_labels,
            split_indices["test"],
            y_eval,
            probability,
        )
        torch.save({"model_state_dict": model.state_dict()}, output_dir / f"{name}_model.pt")
        save_binary_curves(
            output_dir / f"{name}_test_curves.png",
            y_eval,
            probability,
            f"Task B sampling_3 prefix={args.prefix_len} {name}",
        )
        save_json(metrics_path, metrics)
        test_metrics = metrics[name]["test"]
        print(
            f"Finished {name}: f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f} ap={test_metrics['average_precision']:.4f}"
        )


if __name__ == "__main__":
    main()
