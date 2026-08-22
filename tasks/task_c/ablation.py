from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tasks.task_b.main import FULL_RESULT_KEYS, ORBITAL15_RESULT_KEYS
from tasks.task_c.cache import prepare_prefix_cache


ROOT = Path(__file__).resolve().parents[2]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task C prefix/variation-count ablation grid.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_c_ablation"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Shared prefix cache directory; defaults to <output-dir>/_cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the shared prefix cache even when a matching cache exists.",
    )
    parser.add_argument("--prefix-lens", type=str, default="10,20,50,100")
    parser.add_argument("--variation-counts", type=str, default="2,4,8,16,32")
    parser.add_argument("--models", type=str, default="baseline,direct,traj,dual")
    parser.add_argument("--variation-target", choices=["subset", "full32"], default="full32")
    parser.add_argument("--prefix-representation", choices=["full", "orbital15"], default="orbital15")
    parser.add_argument("--variation-subset-mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--variation-subset-seed", type=int, default=42)
    parser.add_argument("--future-times", type=str, default="500,800,1500,2500,4000,6500,10000")
    parser.add_argument(
        "--future-time-mode",
        choices=["absolute"],
        default="absolute",
        help="Task C labels use time_unbound horizons; absolute is the only valid mode for dataset_v3.",
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-horizon",
        type=float,
        default=10000.0,
        help="Fixed horizon used only to stratify the shared A/B/C system split.",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--system-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder-type", choices=["cnn", "gru", "attention"], default="cnn")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells that already contain metrics.json; otherwise the grid is resumable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_cell(
    args: argparse.Namespace,
    prefix_len: int,
    variation_count: int,
    cell_dir: Path,
    cache_dir: Path,
) -> None:
    metrics_path = cell_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print("SKIP", cell_dir, "(metrics.json already exists; use --force to re-run)")
        return

    command = [
        sys.executable,
        "-m",
        "tasks.task_c",
        "--dataset-dir",
        str(args.dataset_dir),
        "--sampling-id",
        str(args.sampling_id),
        "--output-dir",
        str(cell_dir),
        "--prefix-cache",
        str(cache_dir),
        "--prefix-len",
        str(prefix_len),
        "--time-stride",
        str(args.time_stride),
        "--prefix-representation",
        args.prefix_representation,
        "--variation-count",
        str(variation_count),
        "--variation-target",
        args.variation_target,
        "--variation-subset-mode",
        args.variation_subset_mode,
        "--variation-subset-seed",
        str(args.variation_subset_seed),
        "--group-size",
        str(args.group_size),
        "--future-times",
        args.future_times,
        "--future-time-mode",
        args.future_time_mode,
        "--models",
        args.models,
        "--encoder-type",
        args.encoder_type,
        "--test-ratio",
        str(args.test_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--train-ratio",
        str(args.train_ratio),
        "--split-horizon",
        str(args.split_horizon),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--system-loss-weight",
        str(args.system_loss_weight),
        "--consistency-weight",
        str(args.consistency_weight),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.max_shards is not None:
        command += ["--max-shards", str(args.max_shards)]
    if args.rows_per_shard is not None:
        command += ["--rows-per-shard", str(args.rows_per_shard)]
    if args.max_cases is not None:
        command += ["--max-cases", str(args.max_cases)]
    if args.device is not None:
        command += ["--device", args.device]
    if args.split_manifest is not None:
        command += ["--split-manifest", str(args.split_manifest)]
    if args.no_geom:
        command.append("--no-geom")
    if args.no_time:
        command.append("--no-time")

    print("RUN", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def collect_summary(output_dir: Path, prefix_lens: list[int], variation_counts: list[int], models: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    cells = {(int(prefix_len), int(variation_count)) for prefix_len in prefix_lens for variation_count in variation_counts}
    for metrics_path in output_dir.glob("prefix_*/variations_*/metrics.json"):
        prefix_part, variation_part = metrics_path.parts[-3:-1]
        cells.add((int(prefix_part.removeprefix("prefix_")), int(variation_part.removeprefix("variations_"))))

    for prefix_len, variation_count in sorted(cells):
        cell_dir = output_dir / f"prefix_{prefix_len}" / f"variations_{variation_count}"
        metrics_path = cell_dir / "metrics.json"
        config_path = cell_dir / "config.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        for model in models:
            if model not in metrics:
                continue
            payload = metrics[model]
            rows.append(
                {
                    "prefix_len": prefix_len,
                    "variation_count": variation_count,
                    "model": model,
                    "num_cases": config.get("num_cases"),
                    "overall_mae": payload.get("overall_mae"),
                    "final_mae": payload.get("final_mae"),
                    "variation_overall_mae": payload.get("variation_overall_mae"),
                    "mean_variation_overall_mae": payload.get("mean_variation_overall_mae"),
                    "system_loss": payload.get("system_loss"),
                    "variation_loss": payload.get("variation_loss"),
                    "consistency_loss": payload.get("consistency_loss"),
                }
            )
    summary_path = output_dir / "summary.csv"
    if rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {summary_path} with {len(rows)} model-cell rows.")


def main() -> None:
    args = build_arg_parser().parse_args()
    prefix_lens = parse_int_list(args.prefix_lens)
    variation_counts = parse_int_list(args.variation_counts)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or output_dir / "_cache"

    if not args.dry_run:
        if args.prefix_representation == "orbital15":
            result_keys = ORBITAL15_RESULT_KEYS
            include_geom = False
        else:
            result_keys = FULL_RESULT_KEYS
            include_geom = not args.no_geom
        prepare_prefix_cache(
            cache_dir,
            dataset_dir=args.dataset_dir,
            sampling_id=args.sampling_id,
            max_prefix_len=max(prefix_lens),
            time_stride=args.time_stride,
            result_keys=result_keys,
            include_geom=include_geom,
            max_shards=args.max_shards,
            rows_per_shard=args.rows_per_shard,
            force=args.rebuild_cache,
        )

    for prefix_len in prefix_lens:
        for variation_count in variation_counts:
            cell_dir = output_dir / f"prefix_{prefix_len}" / f"variations_{variation_count}"
            run_cell(args, prefix_len, variation_count, cell_dir, cache_dir)
    collect_summary(output_dir, prefix_lens, variation_counts, models)


if __name__ == "__main__":
    main()
