from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.experiment_utils import binary_metrics_from_probs, save_json, select_device
from tasks.task_b.main import (
    AttentionClassifier,
    CNN1DClassifier,
    GRUClassifier,
    LinearSVM,
    SequenceDataset,
    TCNClassifier,
    build_prefix_features,
    discover_result_paths,
    filter_valid_samples,
    load_aligned_labels,
    load_result_prefixes,
    predict_classifier,
)


def evaluate_prefix(prefix_dir: Path, dataset_dir: Path, sampling_id: int, device: torch.device) -> dict:
    config = json.loads((prefix_dir / "config.json").read_text())
    prefix_len = int(config["prefix_len"])
    result_paths = discover_result_paths(dataset_dir, sampling_id)
    data, batch_ids, rows_loaded = load_result_prefixes(
        result_paths,
        result_keys=(
            "aratios_complete",
            "mratios_complete",
            "eccs_complete",
            "incs_complete",
            "APs_complete",
            "LANs_complete",
        ),
        prefix_len=prefix_len,
        time_stride=int(config["time_stride"]),
        rows_per_shard=None,
        include_geom=False,
        add_time=False,
    )
    labels = load_aligned_labels(dataset_dir / f"completed_systems_sampling_{sampling_id}.txt", batch_ids, rows_loaded)
    X, _ = build_prefix_features(
        data,
        result_keys=(
            "aratios_complete",
            "mratios_complete",
            "eccs_complete",
            "incs_complete",
            "APs_complete",
            "LANs_complete",
        ),
        add_time=False,
        include_geom=False,
    )
    X, y, labels, _ = filter_valid_samples(X, labels, data, float(config["time_horizon"]))
    split = np.load(prefix_dir / "split_indices.npz")
    test_idx = split["test_idx"]
    norm = np.load(prefix_dir / "sequence_normalization.npz")
    X_test = ((X[test_idx] - norm["mean"]) / norm["std"]).astype(np.float32)
    y_test = y[test_idx]

    scale = config["effective_model_scale"]
    input_dim = int(X_test.shape[-1])
    metrics = {}

    def eval_model(name: str, model: torch.nn.Module, test_X: np.ndarray) -> None:
        checkpoint = torch.load(prefix_dir / f"{name}_model.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        loader = DataLoader(SequenceDataset(test_X, y_test), batch_size=256, shuffle=False)
        _, prob = predict_classifier(model, loader, device)
        metrics[name] = {"test": binary_metrics_from_probs(y_test, prob)}

    if (prefix_dir / "svm_model.pt").exists():
        checkpoint = torch.load(prefix_dir / "svm_model.pt", map_location="cpu")
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        X_test_2d = ((X_test_2d - checkpoint["norm_mean"].numpy()) / checkpoint["norm_std"].numpy()).astype(np.float32)
        model = LinearSVM(X_test_2d.shape[1])
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        loader = DataLoader(torch.from_numpy(X_test_2d).float(), batch_size=4096, shuffle=False)
        model.eval()
        probs = []
        with torch.no_grad():
            for xb in loader:
                probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        metrics["svm"] = {"test": binary_metrics_from_probs(y_test, np.concatenate(probs))}

    if (prefix_dir / "cnn_model.pt").exists():
        eval_model("cnn", CNN1DClassifier(input_dim, hidden=int(scale["cnn"]["hidden"]), dropout=float(config["dropout"])), X_test)
    if (prefix_dir / "gru_model.pt").exists():
        eval_model(
            "gru",
            GRUClassifier(
                input_dim,
                hidden=int(scale["gru"]["hidden"]),
                layers=int(scale["gru"]["layers"]),
                dropout=float(config["dropout"]),
            ),
            X_test,
        )
    if (prefix_dir / "tcn_model.pt").exists():
        eval_model(
            "tcn",
            TCNClassifier(
                input_dim,
                hidden=int(scale["tcn"]["hidden"]),
                levels=int(scale["tcn"]["levels"]),
                dropout=float(config["dropout"]),
            ),
            X_test,
        )
    if (prefix_dir / "attention_model.pt").exists():
        eval_model(
            "attention",
            AttentionClassifier(
                input_dim,
                hidden=int(scale["attention"]["hidden"]),
                heads=int(scale["attention"]["heads"]),
                layers=int(scale["attention"]["layers"]),
                dropout=float(config["dropout"]),
            ),
            X_test,
        )
    save_json(prefix_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metrics = evaluate_prefix(args.prefix_dir, args.dataset_dir, args.sampling_id, select_device(args.device))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
