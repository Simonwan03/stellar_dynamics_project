from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "vendor" / "panda"))
sys.path.insert(0, str(REPO_ROOT))

from adapters.project_data import build_representation, required_keys_for_representation  # noqa: E402
from scripts.train_panda_unbound_classifier import (  # noqa: E402
    CachedLinearProbe,
    PandaUnboundClassifier,
)


class ChannelDeepSetsHead(nn.Module):
    """Shared nonlinear channel encoder followed by permutation-style pooling."""

    def __init__(self, channels: int = 15, channel_dim: int = 768, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.channels = int(channels)
        self.channel_dim = int(channel_dim)
        self.input_norm = nn.LayerNorm(channel_dim)
        self.project = nn.Linear(channel_dim, hidden)
        self.channel_embedding = nn.Parameter(torch.empty(channels, hidden))
        self.phi = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.normal_(self.channel_embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels = features.reshape(-1, self.channels, self.channel_dim)
        encoded = self.project(self.input_norm(channels)) + self.channel_embedding.unsqueeze(0)
        pooled = self.phi(encoded).mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


class ChannelAttentionHead(nn.Module):
    """Shared channel projection plus learned channel identities and self-attention."""

    def __init__(
        self,
        channels: int = 15,
        channel_dim: int = 768,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.channels = int(channels)
        self.channel_dim = int(channel_dim)
        self.input_norm = nn.LayerNorm(channel_dim)
        self.project = nn.Linear(channel_dim, hidden)
        self.channel_embedding = nn.Parameter(torch.empty(channels, hidden))
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.normal_(self.channel_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels = features.reshape(-1, self.channels, self.channel_dim)
        tokens = self.project(self.input_norm(channels)) + self.channel_embedding.unsqueeze(0)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.classifier(encoded[:, 0]).squeeze(-1)


def build_head(name: str, hidden: int, dropout: float, attention_heads: int, attention_layers: int) -> nn.Module:
    if name == "linear":
        return CachedLinearProbe(15 * 768)
    if name == "deepsets":
        return ChannelDeepSetsHead(hidden=hidden, dropout=dropout)
    if name == "attention":
        return ChannelAttentionHead(
            hidden=hidden,
            heads=attention_heads,
            layers=attention_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown head: {name}")


class TaskSplitFeatureDataset(Dataset):
    def __init__(
        self,
        cached_features: np.ndarray,
        cache_rows: np.ndarray,
        missing_features: np.ndarray,
        missing_rows: np.ndarray,
        targets: np.ndarray,
        task_rows: np.ndarray,
    ) -> None:
        self.cached_features = cached_features
        self.cache_rows = np.asarray(cache_rows, dtype=np.int64)
        self.missing_features = np.asarray(missing_features, dtype=np.float16)
        self.missing_rows = np.asarray(missing_rows, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.task_rows = np.asarray(task_rows, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.task_rows.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        task_row = int(self.task_rows[item])
        cache_row = int(self.cache_rows[task_row])
        if cache_row >= 0:
            feature = self.cached_features[cache_row]
        else:
            missing_row = int(self.missing_rows[task_row])
            if missing_row < 0:
                raise KeyError(f"No Panda feature for Task B row {task_row}.")
            feature = self.missing_features[missing_row]
        return (
            torch.from_numpy(np.asarray(feature, dtype=np.float32)),
            torch.tensor(self.targets[task_row], dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def task_label_fingerprint(labels: pd.DataFrame) -> str:
    identity_columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        identity_columns.append("row_in_batch")
    identity = labels[identity_columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def extract_task_features(
    task_labels: pd.DataFrame,
    dataset_dir: Path,
    checkpoint: str,
    prefix_steps: int,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Stream exact Task B rows through frozen Panda and cache 15x768 features."""
    if prefix_steps < 16 or prefix_steps % 16:
        raise ValueError("--prefix-steps must be a positive multiple of Panda's patch length 16.")
    metadata_path = output_path.with_suffix(".json")
    fingerprint = task_label_fingerprint(task_labels)
    expected_shape = (len(task_labels), 15 * 768)
    if output_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        cached = np.load(output_path, mmap_mode="r")
        expected_metadata = {
            "rows": len(task_labels),
            "feature_dim": 15 * 768,
            "prefix_steps": prefix_steps,
            "checkpoint": checkpoint,
            "label_fingerprint": fingerprint,
            "dtype": "float16",
        }
        if cached.shape == expected_shape and all(metadata.get(key) == value for key, value in expected_metadata.items()):
            print(f"Reusing exact Task B Panda features: {output_path}")
            return cached
        raise ValueError(f"Existing feature cache is incompatible with this run: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp.npy")
    features = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float16, shape=expected_shape)
    keys = required_keys_for_representation("orbital15")
    labels_all = pd.read_csv(
        dataset_dir / "completed_systems_sampling_2.txt",
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    encoder = PandaUnboundClassifier(
        checkpoint,
        freeze_backbone=True,
        num_channels=15,
        pooling="patch_mean_keep_channels",
    ).to(device)
    encoder.eval()

    completed = 0
    try:
        for batch_id in sorted(task_labels["num_batch"].unique()):
            task_positions = np.flatnonzero(task_labels["num_batch"].to_numpy() == batch_id)
            group = task_labels.iloc[task_positions]
            if "row_in_batch" in group.columns:
                local_rows = group["row_in_batch"].to_numpy(dtype=np.int64)
            else:
                batch_labels = labels_all.loc[labels_all["num_batch"] == batch_id].copy()
                batch_labels = batch_labels.sort_values("index").reset_index(drop=True)
                index_to_local_row = {
                    int(index): row for row, index in enumerate(batch_labels["index"].to_numpy())
                }
                local_rows = np.asarray(
                    [index_to_local_row[int(index)] for index in group["index"]], dtype=np.int64
                )
            read_order = np.argsort(local_rows)
            restore_order = np.argsort(read_order)
            shard = dataset_dir / f"results_batch_{int(batch_id):02d}_sampling_2.hdf5"
            data: dict[str, np.ndarray] = {}
            with h5py.File(shard, "r") as handle:
                for key in keys:
                    sorted_values = np.asarray(
                        handle[key][local_rows[read_order], :prefix_steps, ...], dtype=np.float32
                    )
                    data[key] = sorted_values[restore_order]
            x = build_representation(data, "orbital15")
            if not np.isfinite(x).all():
                bad = int((~np.isfinite(x).all(axis=(1, 2))).sum())
                raise ValueError(f"Found {bad} non-finite Task B inputs in batch {batch_id}.")
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    stop = min(start + batch_size, len(x))
                    encoded = encoder.encode(torch.from_numpy(x[start:stop]).to(device))
                    features[task_positions[start:stop]] = encoded.cpu().numpy().astype(np.float16)
            completed += len(x)
            features.flush()
            print(
                f"feature_cache batch={int(batch_id):02d} rows={completed}/{len(task_labels)}",
                flush=True,
            )
    except BaseException:
        features.flush()
        raise
    features.flush()
    del features
    os.replace(temp_path, output_path)
    write_json(
        metadata_path,
        {
            "rows": len(task_labels),
            "feature_dim": 15 * 768,
            "prefix_steps": prefix_steps,
            "num_patches": prefix_steps // 16,
            "patch_length": 16,
            "patch_stride": 16,
            "checkpoint": checkpoint,
            "pooling": "patch_mean_keep_channels",
            "label_fingerprint": fingerprint,
            "dtype": "float16",
            "extraction_batch_size": batch_size,
        },
    )
    print(f"Cached exact Task B Panda features: {output_path}")
    return np.load(output_path, mmap_mode="r")


def extract_missing_features(
    missing_labels: pd.DataFrame,
    dataset_dir: Path,
    checkpoint: str,
    output_path: Path,
    device: torch.device,
) -> np.ndarray:
    if output_path.exists():
        cached = np.load(output_path)
        if cached.shape == (len(missing_labels), 15 * 768):
            print(f"Reusing {len(cached)} supplemental Panda features: {output_path}")
            return cached

    keys = required_keys_for_representation("orbital15")
    labels_all = pd.read_csv(
        dataset_dir / "completed_systems_sampling_2.txt",
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
    )
    inputs: list[np.ndarray] = []
    for batch_id, group in missing_labels.groupby("num_batch", sort=True):
        batch_labels = labels_all.loc[labels_all["num_batch"] == batch_id].reset_index(drop=True)
        positions = {int(index): row for row, index in enumerate(batch_labels["index"].to_numpy())}
        local_rows = np.asarray([positions[int(index)] for index in group["index"]], dtype=np.int64)
        read_order = np.argsort(local_rows)
        restore_order = np.argsort(read_order)
        shard = dataset_dir / f"results_batch_{int(batch_id):02d}_sampling_2.hdf5"
        data: dict[str, np.ndarray] = {}
        with h5py.File(shard, "r") as handle:
            for key in keys:
                sorted_values = np.asarray(handle[key][local_rows[read_order], :512, ...], dtype=np.float32)
                data[key] = sorted_values[restore_order]
        inputs.append(build_representation(data, "orbital15"))
    x = np.concatenate(inputs, axis=0)
    encoder = PandaUnboundClassifier(
        checkpoint,
        freeze_backbone=True,
        num_channels=15,
        pooling="patch_mean_keep_channels",
    ).to(device)
    encoder.eval()
    with torch.no_grad():
        features = encoder.encode(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float16)
    np.save(output_path, features)
    print(f"Encoded {len(features)} supplemental Panda features: {output_path}")
    return features


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=True))
        targets.append(target.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen Panda 15x768 probe on the exact Task B split.")
    parser.add_argument("--prefix-dir", type=Path, default=REPO_ROOT / "results/task_b_prefix_ablation/prefix_512")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset_v3")
    parser.add_argument(
        "--panda-cache-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs/panda_unbound_dataset/panda_unbound_orbital15_patch_mean_keep_channels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs/panda_unbound_dataset/panda_task_b_split_prefix_512",
    )
    parser.add_argument("--checkpoint", default="GilpinLab/panda-72M")
    parser.add_argument("--prefix-steps", type=int, default=512)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Exact Task B Panda feature cache; generated streaming when missing.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head", choices=["linear", "deepsets", "attention"], default="linear")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--supplemental-features", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run.")
    device = torch.device(args.device)

    task_labels = pd.read_csv(args.prefix_dir / "labels_filtered.csv")
    split_file = np.load(args.prefix_dir / "split_indices.npz")
    split_rows = {name: split_file[f"{name}_idx"] for name in ("train", "val", "test")}
    target_unbound = (task_labels["time_unbound"].to_numpy(dtype=np.float32) < 10000.0).astype(np.float32)

    if args.feature_cache is not None:
        feature_cache = extract_task_features(
            task_labels=task_labels,
            dataset_dir=args.dataset_dir,
            checkpoint=args.checkpoint,
            prefix_steps=args.prefix_steps,
            output_path=args.feature_cache,
            device=device,
            batch_size=args.feature_batch_size,
        )
        cache_rows = np.arange(len(task_labels), dtype=np.int64)
        missing_task_rows = np.zeros(0, dtype=np.int64)
        missing_features = np.empty((0, 15 * 768), dtype=np.float16)
        missing_rows = np.full(len(task_labels), -1, dtype=np.int64)
    else:
        if args.prefix_steps != 512:
            raise ValueError("A non-512 prefix requires --feature-cache so its Panda features can be generated.")
        panda_manifest = pd.read_csv(args.panda_cache_dir / "manifest_with_labels.csv")
        feature_cache = np.load(args.panda_cache_dir / "panda_patch_mean_features.float16.npy", mmap_mode="r")
        feature_metadata = read_json(args.panda_cache_dir / "panda_patch_mean_features.float16.json")
        if feature_cache.shape[1] != 15 * 768 or int(feature_metadata["feature_dim"]) != 15 * 768:
            raise ValueError(f"Expected Panda 15x768 features, got {feature_cache.shape}.")

        index_to_cache_row = {int(index): row for row, index in enumerate(panda_manifest["index"].to_numpy())}
        cache_rows = np.asarray(
            [index_to_cache_row.get(int(index), -1) for index in task_labels["index"].to_numpy()],
            dtype=np.int64,
        )
        missing_task_rows = np.flatnonzero(cache_rows < 0)
        if missing_task_rows.size:
            missing_labels = task_labels.iloc[missing_task_rows].copy()
            supplemental_path = args.supplemental_features
            if supplemental_path is None:
                supplemental_path = args.output_dir / "supplemental_test_features.float16.npy"
            missing_features = extract_missing_features(
                missing_labels, args.dataset_dir, args.checkpoint, supplemental_path, device
            )
        else:
            missing_features = np.empty((0, 15 * 768), dtype=np.float16)
        missing_rows = np.full(len(task_labels), -1, dtype=np.int64)
        if missing_task_rows.size:
            missing_rows[missing_task_rows] = np.arange(len(missing_task_rows), dtype=np.int64)

    split_membership = np.full(len(task_labels), "", dtype=object)
    for split, rows in split_rows.items():
        split_membership[rows] = split
    if np.any(split_membership == ""):
        raise ValueError("Task B split does not cover every filtered sample.")

    loaders = {
        split: DataLoader(
            TaskSplitFeatureDataset(
                feature_cache,
                cache_rows,
                missing_features,
                missing_rows,
                target_unbound,
                rows,
            ),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=True,
        )
        for split, rows in split_rows.items()
    }
    # Training intentionally shuffles its loader.  Use separate deterministic
    # loaders when exporting predictions so each prediction remains aligned
    # with task_labels.iloc[split_rows[split]].
    eval_loaders = {
        split: DataLoader(
            TaskSplitFeatureDataset(
                feature_cache,
                cache_rows,
                missing_features,
                missing_rows,
                target_unbound,
                rows,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for split, rows in split_rows.items()
    }

    model = build_head(
        args.head,
        hidden=args.hidden,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Head={args.head}; trainable parameters={trainable_parameters:,}", flush=True)
    train_target = target_unbound[split_rows["train"]]
    positive = float(train_target.sum())
    negative = float(len(train_target) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, target in loaders["train"]:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            count += len(target)
        val_target, val_probability = predict(model, loaders["val"], device)
        val_auc = float(roc_auc_score(val_target, val_probability))
        record = {"epoch": epoch, "train_loss": loss_sum / count, "val_auroc": val_auc}
        history.append(record)
        print(f"epoch={epoch:03d} loss={record['train_loss']:.6f} val_auroc={val_auc:.6f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "feature_dim": 15 * 768,
            "checkpoint": args.checkpoint,
            "pooling": "patch_mean_keep_channels",
            "head": args.head,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "attention_heads": args.attention_heads,
            "attention_layers": args.attention_layers,
            "trainable_parameters": trainable_parameters,
            "best_val_auroc": best_auc,
        },
        args.output_dir / "checkpoint_best.pt",
    )

    all_metrics: dict[str, Any] = {}
    prediction_parts: list[pd.DataFrame] = []
    for split, rows in split_rows.items():
        y_unbound, prob_unbound = predict(model, eval_loaders[split], device)
        expected_target = target_unbound[rows]
        if not np.array_equal(y_unbound, expected_target):
            raise RuntimeError(f"Prediction export order is misaligned for the {split} split.")
        y_stable = 1.0 - y_unbound
        prob_stable = 1.0 - prob_unbound
        all_metrics[split] = {
            "unbound_positive": probability_metrics(y_unbound, prob_unbound),
            "stable_positive": probability_metrics(y_stable, prob_stable),
        }
        prediction = task_labels.iloc[rows][["index", "num_batch", "time_unbound", "diagnosis"]].copy()
        prediction["task_b_row"] = rows
        prediction["split"] = split
        prediction["target_unbound"] = y_unbound.astype(np.int64)
        prediction["prob_unbound"] = prob_unbound
        prediction["prob_stable"] = prob_stable
        prediction_parts.append(prediction)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(args.output_dir / "predictions_all.csv", index=False)
    test_predictions = predictions.loc[predictions["split"] == "test"]
    time_unbound = test_predictions["time_unbound"]
    prefix_end_time = float(args.prefix_steps * 5)
    segment_definitions = (
        (f"prefix {args.prefix_steps}内已 unbound", time_unbound < prefix_end_time, True),
        (
            f"prefix {args.prefix_steps}后才 unbound",
            (time_unbound >= prefix_end_time) & (time_unbound < 10000.0),
            True,
        ),
        ("一直到结束仍稳定", time_unbound >= 10000.0, False),
    )
    segmented_rows = []
    for segment, mask, expected_unbound in segment_definitions:
        probability = test_predictions.loc[mask, "prob_unbound"]
        segmented_rows.append(
            {
                "segment": segment,
                "count": int(mask.sum()),
                "correct_or_recall": float(((probability >= 0.5) == expected_unbound).mean()),
                "mean_prob_unbound": float(probability.mean()),
            }
        )
    pd.DataFrame(segmented_rows).to_csv(args.output_dir / "segmented_test_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    write_json(args.output_dir / "metrics.json", all_metrics)
    write_json(
        args.output_dir / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "prefix_dir": str(args.prefix_dir),
            "dataset_dir": str(args.dataset_dir),
            "panda_cache_dir": str(args.panda_cache_dir),
            "output_dir": str(args.output_dir),
            "representation": "orbital15 -> Panda patch mean keep channels (15x768)",
            "prefix_steps": args.prefix_steps,
            "num_patches": args.prefix_steps // 16,
            "trainable_parameters": trainable_parameters,
            "split_counts": {split: int(len(rows)) for split, rows in split_rows.items()},
            "supplemental_feature_count": int(len(missing_task_rows)),
        },
    )
    print(json.dumps(all_metrics["test"], ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
