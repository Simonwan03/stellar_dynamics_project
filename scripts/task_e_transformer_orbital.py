from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUTS = [
    "config.json",
    "normalization_stats.npz",
    "feature_metadata.json",
    "train_log.csv",
    "best_model.pt",
    "final_model.pt",
    "test_metrics.json",
    "horizon_metrics.csv",
    "loss_curve.png",
    "horizon_mse.png",
    "example_predictions.png",
]
MIXED_KEY_MAPPING = {
    "aratios": "aratios_complete",
    "mratios_complete": "mratios_complete",
    "eccs": "eccs_complete",
    "incs_complete": "incs_complete",
    "APs": "APs_complete",
    "LANs_complete": "LANs_complete",
}


def build_debug_dataset(result_shard_path: Path, debug_data_path: Path, rows: int, overwrite: bool = False) -> None:
    if debug_data_path.exists() and not overwrite:
        print(f"Debug dataset already exists: {debug_data_path}")
        return
    debug_data_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(result_shard_path, "r") as src, h5py.File(debug_data_path, "w") as dst:
        for out_key, in_key in MIXED_KEY_MAPPING.items():
            dst.create_dataset(out_key, data=src[in_key][:rows], compression="gzip")
    print(f"Wrote debug dataset to {debug_data_path}")


def run_training(python_cmd: list[str], script_args: list[str]) -> None:
    cmd = python_cmd + [str(ROOT / "train_transformer_orbital.py")] + [str(arg) for arg in script_args]
    print("$", " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def validate_outputs(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs in {output_dir}: {missing}")
    print(f"All expected outputs exist in {output_dir}")


def print_metrics(output_dir: Path) -> None:
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    print(json.dumps(metrics, indent=2)[:4000])

    horizon_path = output_dir / "horizon_metrics.csv"
    with open(horizon_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    print("First horizon rows:")
    for row in rows[:5]:
        print(row)


def default_python_cmd(conda_env: str | None) -> list[str]:
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u"]
    return [sys.executable, "-u"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task E: run the Transformer orbital forecasting baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--mode", choices=["debug", "full", "both"], default="debug")
    parser.add_argument("--conda-env", type=str, default=None, help="Optional conda env for the training subprocess, e.g. stellar.")
    parser.add_argument("--debug-data-path", type=Path, default=Path("/tmp/orbital_smoke_sampling_2.hdf5"))
    parser.add_argument("--debug-output-dir", type=Path, default=ROOT / "runs" / "debug_transformer_orbital")
    parser.add_argument("--full-output-dir", type=Path, default=ROOT / "runs" / "transformer_orbital_baseline")
    parser.add_argument("--debug-rows", type=int, default=256)
    parser.add_argument("--overwrite-debug-data", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--full-epochs", type=int, default=50)
    parser.add_argument("--debug-epochs", type=int, default=3)
    parser.add_argument("--log-every-batches", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ics_path = args.dataset_dir / f"ICs_sampling_{args.sampling_id}.hdf5"
    result_shard_path = args.dataset_dir / f"results_batch_00_sampling_{args.sampling_id}.hdf5"
    if not ics_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {ics_path}")
    if not result_shard_path.exists():
        raise FileNotFoundError(f"Missing result shard: {result_shard_path}")

    python_cmd = default_python_cmd(args.conda_env)
    common = ["--seed", args.seed, "--num_workers", args.num_workers, "--log_every_batches", args.log_every_batches]
    if args.amp:
        common.append("--amp")
    if args.device:
        common.extend(["--device", args.device])

    if args.mode in {"debug", "both"}:
        build_debug_dataset(result_shard_path, args.debug_data_path, args.debug_rows, overwrite=args.overwrite_debug_data)
        run_training(
            python_cmd,
            [
                "--data_path",
                args.debug_data_path,
                "--context_len",
                128,
                "--pred_len",
                32,
                "--stride",
                32,
                "--batch_size",
                32,
                "--epochs",
                args.debug_epochs,
                "--d_model",
                128,
                "--n_heads",
                4,
                "--num_layers",
                2,
                "--dim_feedforward",
                512,
                "--dropout",
                0.1,
                "--output_dir",
                args.debug_output_dir,
                *common,
            ],
        )
        validate_outputs(args.debug_output_dir)
        print_metrics(args.debug_output_dir)

    if args.mode in {"full", "both"}:
        run_training(
            python_cmd,
            [
                "--data_path",
                ics_path,
                "--context_len",
                256,
                "--pred_len",
                64,
                "--stride",
                32,
                "--batch_size",
                64,
                "--epochs",
                args.full_epochs,
                "--lr",
                1e-4,
                "--weight_decay",
                1e-4,
                "--d_model",
                256,
                "--n_heads",
                8,
                "--num_layers",
                4,
                "--dim_feedforward",
                1024,
                "--dropout",
                0.1,
                "--output_dir",
                args.full_output_dir,
                *common,
            ],
        )
        validate_outputs(args.full_output_dir)
        print_metrics(args.full_output_dir)


if __name__ == "__main__":
    main()
