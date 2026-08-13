from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "vendor" / "panda"))

from adapters.project_data import (  # noqa: E402
    build_representation,
    case_summary,
    choose_default_dataset,
    choose_dataset_from_root,
    clean_and_group_trajectories,
    ensure_dir,
    load_labels,
    load_result_arrays,
    load_yaml,
    required_keys_for_representation,
    stratified_case_split,
    summarize_audit,
    write_json,
)
from panda.patchtst.patchtst import PatchTSTForPrediction  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


POOLING_MODES = ("global_mean", "patch_mean_keep_channels")


def pool_encoder_features(encoded: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool Panda output [B, C, patches, d_model] into classifier features."""
    if encoded.ndim != 4:
        raise ValueError(f"Expected encoded features [B, C, patches, d_model], got {tuple(encoded.shape)}.")
    if pooling == "global_mean":
        return encoded.mean(dim=(1, 2))
    if pooling == "patch_mean_keep_channels":
        # Average only the patch axis: [B, C, patches, D] -> [B, C, D].
        # Flattening preserves a separate D-dimensional block for every channel.
        return encoded.mean(dim=2).flatten(start_dim=1)
    raise ValueError(f"Unknown pooling mode {pooling!r}; choose one of {POOLING_MODES}.")


class PandaUnboundClassifier(nn.Module):
    def __init__(self, checkpoint: str, freeze_backbone: bool, num_channels: int, pooling: str):
        super().__init__()
        self.backbone = PatchTSTForPrediction.from_pretrained(checkpoint)
        hidden_dim = int(self.backbone.config.d_model)
        self.num_channels = int(num_channels)
        self.pooling = str(pooling)
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}.")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling mode {self.pooling!r}; choose one of {POOLING_MODES}.")
        classifier_dim = hidden_dim if self.pooling == "global_mean" else self.num_channels * hidden_dim
        self.classifier = nn.Linear(classifier_dim, 1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Use Panda's encoder representation directly. Shape: [B, C, patches, d_model].
        observed = torch.ones_like(x)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            encoded = self.backbone.model(
                past_values=x,
                past_observed_mask=observed,
                return_dict=True,
            ).last_hidden_state
        if encoded.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} encoded channels, got shape {tuple(encoded.shape)}."
            )
        return pool_encoder_features(encoded, self.pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x)).squeeze(-1)


class CachedFeatureDataset(Dataset):
    """Read selected rows from a memory-mapped frozen-backbone feature file."""

    def __init__(self, features: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.features = features
        self.y = np.asarray(y, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[idx])
        feature = torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))
        return feature, torch.tensor(self.y[row], dtype=torch.float32)


class CachedLinearProbe(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.classifier = nn.Linear(int(feature_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(-1)


@dataclass
class SplitMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    positive_rate: float
    n: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict[str, Any], allow_cpu_flag: bool) -> torch.device:
    requested = str(cfg.get("device", "cuda")).lower()
    allow_cpu = bool(cfg.get("allow_cpu_fallback", False)) or allow_cpu_flag
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if allow_cpu:
            print("WARNING: CUDA is not available; using CPU fallback because allow_cpu_fallback is enabled.")
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is not available in this environment. Re-run where PyTorch can see a GPU, "
            "or pass --allow-cpu for a slow CPU smoke test."
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def metric_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float, threshold: float) -> SplitMetrics:
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true_i = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_true_i == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_i == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_i == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_i == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return SplitMetrics(
        loss=float(loss),
        accuracy=float((tp + tn) / max(1, len(y_true_i))),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=metric_auroc(y_true_i, y_prob),
        positive_rate=float(y_true_i.mean()) if len(y_true_i) else float("nan"),
        n=int(len(y_true_i)),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[SplitMetrics, np.ndarray]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()) * len(yb))
            ys.append(yb.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.float32)
    y_prob = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)
    mean_loss = sum(losses) / max(1, len(y_true))
    return compute_metrics(y_true, y_prob, mean_loss, threshold), y_prob


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_cached_features_if_valid(
    cache_path: Path,
    metadata_path: Path,
    *,
    rows: int,
    input_shape: tuple[int, int],
    checkpoint: str,
    pooling: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Return a read-only feature memmap only when file and metadata agree."""
    if not cache_path.exists() or not metadata_path.exists():
        return None, None
    try:
        metadata = read_json(metadata_path)
        expected = {
            "rows": int(rows),
            "input_shape": [int(input_shape[0]), int(input_shape[1])],
            "checkpoint": str(checkpoint),
            "pooling": str(pooling),
            "dtype": "float16",
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            print(f"Ignoring incompatible Panda feature cache metadata: {mismatches}")
            return None, None
        feature_dim = int(metadata["feature_dim"])
        features = np.load(cache_path, mmap_mode="r")
        if features.shape != (rows, feature_dim) or features.dtype != np.float16:
            print(
                "Ignoring incompatible Panda feature cache array: "
                f"shape={features.shape}, dtype={features.dtype}"
            )
            return None, None
        return features, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable Panda feature cache: {exc}")
        return None, None


def extract_or_load_cached_features(
    model: PandaUnboundClassifier,
    x: np.ndarray,
    cache_path: Path,
    metadata_path: Path,
    *,
    checkpoint: str,
    pooling: str,
    batch_size: int,
    device: torch.device,
    force: bool = False,
    log_every: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the frozen Panda backbone once and atomically cache float16 features."""
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("Backbone feature caching requires freeze_backbone=true.")

    rows = int(x.shape[0])
    input_shape = (int(x.shape[1]), int(x.shape[2]))
    if not force:
        cached, metadata = load_cached_features_if_valid(
            cache_path,
            metadata_path,
            rows=rows,
            input_shape=input_shape,
            checkpoint=checkpoint,
            pooling=pooling,
        )
        if cached is not None and metadata is not None:
            print(f"Reusing cached Panda features: {cache_path}")
            return cached, metadata

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npy")
    feature_dim = int(model.classifier.in_features)
    output = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float16,
        shape=(rows, feature_dim),
    )
    loader = DataLoader(
        SequenceDataset(x, np.zeros(rows, dtype=np.float32)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    started = time.monotonic()
    cursor = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, (xb, _) in enumerate(loader, start=1):
                xb = xb.to(device, non_blocking=(device.type == "cuda"))
                features = model.encode(xb).cpu().numpy().astype(np.float16, copy=False)
                next_cursor = cursor + len(features)
                output[cursor:next_cursor] = features
                cursor = next_cursor
                if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
                    elapsed = time.monotonic() - started
                    eta = elapsed / batch_idx * max(0, len(loader) - batch_idx)
                    print(
                        f"feature_cache batch={batch_idx}/{len(loader)} rows={cursor}/{rows} "
                        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                        flush=True,
                    )
    except BaseException:
        output.flush()
        raise
    output.flush()
    del output
    os.replace(temp_path, cache_path)

    metadata = {
        "rows": rows,
        "input_shape": list(input_shape),
        "feature_dim": feature_dim,
        "checkpoint": str(checkpoint),
        "pooling": str(pooling),
        "dtype": "float16",
        "batch_size": int(batch_size),
        "seconds": float(time.monotonic() - started),
    }
    write_json(metadata_path, metadata)
    features = np.load(cache_path, mmap_mode="r")
    print(f"Cached Panda features: {cache_path}")
    return features, metadata


def cached_run_inputs_match(
    cfg: dict[str, Any],
    run_config: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> bool:
    """Check the fields that determine labels, splits, and encoder inputs."""
    keys = (
        "dataset_root",
        "sampling_id_preference",
        "group_size",
        "representation",
        "prefix_steps",
        "max_cases",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
        "checkpoint",
        "pooling",
    )
    if any(run_config.get(key) != cfg.get(key) for key in keys):
        return False
    feature_shape = dataset_summary.get("feature_shape")
    return (
        isinstance(feature_shape, list)
        and len(feature_shape) == 3
        and int(feature_shape[1]) == int(cfg["prefix_steps"])
    )


def build_dataset(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Any, dict[str, np.ndarray], float, dict[str, Any]]:
    preferred_sampling_ids = cfg.get("sampling_id_preference", [0])
    if cfg.get("dataset_root"):
        dataset_root = Path(str(cfg["dataset_root"]))
        if not dataset_root.is_absolute():
            dataset_root = (EXPERIMENT_ROOT / dataset_root).resolve()
        choice = choose_dataset_from_root(dataset_root, int(preferred_sampling_ids[0]))
    else:
        choice = choose_default_dataset(preferred_sampling_ids)
    representation = str(cfg.get("representation", "orbital15"))
    group_size = int(cfg.get("group_size", 8))
    max_cases = cfg.get("max_cases")
    max_rows = None if max_cases is None else int(max_cases) * group_size
    prefix_steps = int(cfg["prefix_steps"])
    keys = required_keys_for_representation(representation)

    print("Dataset choice:", summarize_audit(choice))
    print("Representation:", representation)
    print("Prefix steps:", prefix_steps)

    labels = load_labels(choice, max_rows=max_rows, group_size=group_size)
    data = load_result_arrays(choice, keys=keys, time_steps=prefix_steps, max_rows=max_rows)
    x = build_representation(data, representation=representation)
    x, labels = clean_and_group_trajectories(x, labels, group_size=group_size)

    final_time = float(labels["time_unbound"].max())
    labels = labels.copy()
    labels["final_unbound"] = (labels["time_unbound"] < final_time).astype(int)
    case_df = case_summary(labels, group_size=group_size)
    case_df = stratified_case_split(
        case_df,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    labels = labels.merge(case_df[["case_id", "split", "tail_label", "stable_count"]], on="case_id", how="left")
    split_masks = {split: labels["split"].to_numpy() == split for split in ["train", "val", "test"]}
    y = labels["final_unbound"].to_numpy(dtype=np.float32)
    metadata = {
        "source_dataset": summarize_audit(choice),
        "representation": representation,
        "prefix_steps": prefix_steps,
        "group_size": group_size,
        "feature_shape": list(x.shape),
        "final_time": final_time,
        "label_definition": "final_unbound = time_unbound < max(time_unbound among retained valid dataset rows)",
        "split_counts": {split: int(mask.sum()) for split, mask in split_masks.items()},
        "positive_counts": {split: int(y[mask].sum()) for split, mask in split_masks.items()},
        "positive_rates": {
            split: (float(y[mask].mean()) if mask.any() else None)
            for split, mask in split_masks.items()
        },
    }
    return x.astype(np.float32), y, labels, split_masks, final_time, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/panda_unbound_dataset.yaml")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=None)
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(exp_root / args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_cases is not None:
        cfg["max_cases"] = args.max_cases
    if args.log_every_batches is not None:
        cfg["log_every_batches"] = args.log_every_batches

    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg, allow_cpu_flag=args.allow_cpu)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    output_root = exp_root / str(cfg["output_root"]) / str(cfg["run_name"])
    ensure_dir(output_root)
    pooling = str(cfg.get("pooling", "global_mean"))
    cache_enabled = bool(cfg.get("cache_backbone_features", False))
    force_cache = bool(cfg.get("force_feature_cache", False))
    if cache_enabled and not bool(cfg.get("freeze_backbone", True)):
        raise ValueError("cache_backbone_features=true requires freeze_backbone=true.")

    cache_path = output_root / "panda_patch_mean_features.float16.npy"
    cache_metadata_path = cache_path.with_suffix(".json")
    dataset_summary_path = output_root / "dataset_summary.json"
    manifest_path = output_root / "manifest_with_labels.csv"
    run_config_path = output_root / "run_config.json"

    x: np.ndarray | None = None
    features: np.ndarray | None = None
    metadata: dict[str, Any]
    labels: pd.DataFrame
    y: np.ndarray
    split_masks: dict[str, np.ndarray]

    # Fast path: reuse both the frozen features and the exact labels/splits that
    # produced them. This avoids loading the raw HDF5 trajectories or PANDA.
    artifacts_exist = all(
        path.exists()
        for path in (cache_path, cache_metadata_path, dataset_summary_path, manifest_path, run_config_path)
    )
    if cache_enabled and not force_cache and artifacts_exist:
        saved_run_config = read_json(run_config_path)
        saved_dataset_summary = read_json(dataset_summary_path)
        if cached_run_inputs_match(cfg, saved_run_config, saved_dataset_summary):
            labels = pd.read_csv(manifest_path)
            y = labels["final_unbound"].to_numpy(dtype=np.float32)
            split_masks = {
                split: labels["split"].to_numpy() == split
                for split in ("train", "val", "test")
            }
            feature_shape = saved_dataset_summary["feature_shape"]
            features, _ = load_cached_features_if_valid(
                cache_path,
                cache_metadata_path,
                rows=len(labels),
                input_shape=(int(feature_shape[1]), int(feature_shape[2])),
                checkpoint=str(cfg["checkpoint"]),
                pooling=pooling,
            )
            if features is not None:
                metadata = saved_dataset_summary
                print(f"Reusing cached dataset manifest and Panda features: {cache_path}", flush=True)

    if features is None:
        dataset_start = time.monotonic()
        x, y, labels, split_masks, _, metadata = build_dataset(cfg)
        print(f"Dataset built in {format_seconds(time.monotonic() - dataset_start)}; X={x.shape}", flush=True)
        write_json(dataset_summary_path, metadata)
        labels.to_csv(manifest_path, index=False)

    num_channels = int(metadata["feature_shape"][-1])
    if cache_enabled:
        if features is None:
            assert x is not None
            backbone_model = PandaUnboundClassifier(
                str(cfg["checkpoint"]),
                True,
                num_channels=num_channels,
                pooling=pooling,
            ).to(device)
            print(f"Panda checkpoint: {cfg['checkpoint']}")
            print(
                f"Encoder pooling: {pooling}; [B, {num_channels}, patches, "
                f"{backbone_model.backbone.config.d_model}] -> classifier input "
                f"{backbone_model.classifier.in_features}"
            )
            features, _ = extract_or_load_cached_features(
                backbone_model,
                x,
                cache_path,
                cache_metadata_path,
                checkpoint=str(cfg["checkpoint"]),
                pooling=pooling,
                batch_size=int(cfg.get("feature_cache_batch_size", cfg["batch_size"])),
                device=device,
                force=force_cache,
                log_every=int(cfg.get("log_every_batches", 20)),
            )
            del backbone_model
            del x
            x = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model = CachedLinearProbe(int(features.shape[1])).to(device)
        loaders = {
            split: DataLoader(
                CachedFeatureDataset(features, y, np.flatnonzero(mask)),
                batch_size=int(cfg["batch_size"]),
                shuffle=(split == "train"),
                num_workers=int(cfg.get("num_workers", 0)),
                pin_memory=(device.type == "cuda"),
            )
            for split, mask in split_masks.items()
        }
        print("Training input: cached Panda features")
    else:
        assert x is not None
        model = PandaUnboundClassifier(
            str(cfg["checkpoint"]),
            bool(cfg.get("freeze_backbone", True)),
            num_channels=num_channels,
            pooling=pooling,
        ).to(device)
        loaders = {
            split: DataLoader(
                SequenceDataset(x[mask], y[mask]),
                batch_size=int(cfg["batch_size"]),
                shuffle=(split == "train"),
                num_workers=int(cfg.get("num_workers", 0)),
                pin_memory=(device.type == "cuda"),
            )
            for split, mask in split_masks.items()
        }
        print(f"Panda checkpoint: {cfg['checkpoint']}")
        print(
            f"Encoder pooling: {pooling}; [B, {num_channels}, patches, {model.backbone.config.d_model}] "
            f"-> classifier input {model.classifier.in_features}"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}; trainable params: {trainable:,}")

    pos = float(y[split_masks["train"]].sum())
    neg = float(split_masks["train"].sum() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_val = -float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        total_batches = len(loaders["train"])
        log_every = int(cfg.get("log_every_batches", 20))
        for batch_idx, (xb, yb) in enumerate(loaders["train"], start=1):
            batch_start = time.monotonic()
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_n += len(yb)
            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.monotonic() - epoch_start
                avg_per_batch = elapsed / max(1, batch_idx)
                eta = avg_per_batch * max(0, total_batches - batch_idx)
                running_loss = train_loss_sum / max(1, train_n)
                print(
                    f"epoch={epoch}/{cfg['epochs']} batch={batch_idx}/{total_batches} "
                    f"loss={float(loss.item()):.6f} running_loss={running_loss:.6f} "
                    f"batch_time={format_seconds(time.monotonic() - batch_start)} "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                    flush=True,
                )
        print(f"epoch={epoch} training pass complete in {format_seconds(time.monotonic() - epoch_start)}", flush=True)
        val_start = time.monotonic()
        val_metrics, _ = evaluate(model, loaders["val"], criterion, device, float(cfg["threshold"]))
        print(f"epoch={epoch} validation complete in {format_seconds(time.monotonic() - val_start)}", flush=True)
        record = {"epoch": epoch, "train_loss": train_loss_sum / max(1, train_n), "val": asdict(val_metrics)}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"val_auc={val_metrics.auroc:.4f} val_f1={val_metrics.f1:.4f} val_acc={val_metrics.accuracy:.4f}"
        )
        score = val_metrics.auroc if np.isfinite(val_metrics.auroc) else val_metrics.f1
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_val_score": best_val,
                    "panda_checkpoint": cfg["checkpoint"],
                    "pooling": pooling,
                    "cached_backbone_features": cache_enabled,
                },
                output_root / "checkpoint_best.pt",
            )

    checkpoint = torch.load(output_root / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    for split in ["train", "val", "test"]:
        split_metrics, y_prob = evaluate(model, loaders[split], criterion, device, float(cfg["threshold"]))
        metrics[split] = asdict(split_metrics)
        pred_df = labels.loc[
            split_masks[split],
            ["case_id", "perturb_id", "num_batch", "index", "time_unbound", "final_unbound", "tail_label"],
        ].copy()
        pred_df["prob_unbound"] = y_prob
        pred_df["pred_unbound"] = (y_prob >= float(cfg["threshold"])).astype(int)
        pred_df.to_csv(output_root / f"predictions_{split}.csv", index=False)

    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "history.json", history)
    write_json(output_root / "run_config.json", cfg)
    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {output_root}")


if __name__ == "__main__":
    main()
