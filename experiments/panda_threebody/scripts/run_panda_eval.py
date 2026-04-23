from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXPERIMENT_ROOT, VENDOR_PANDA, choose_device, hydra_list, python_cmd, stream_subprocess, write_json
from adapters.project_data import get_export_root, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    cfg = load_yaml(EXPERIMENT_ROOT / args.config)
    split = args.split or cfg.get("split", "test")
    export_root = get_export_root(cfg, representation=cfg.get("representation", "orbital15"))
    data_root = export_root / split
    if not data_root.exists():
        raise FileNotFoundError(f"Evaluation split not found: {data_root}")

    run_dir = EXPERIMENT_ROOT / "outputs" / "eval" / args.run_name
    metrics_dir = run_dir / "metrics"
    forecast_dir = run_dir / "forecasts"
    labels_dir = run_dir / "labels"
    plot_dir = EXPERIMENT_ROOT / "outputs" / "figures" / args.run_name
    log_path = EXPERIMENT_ROOT / "outputs" / "logs" / f"{args.run_name}.log"

    command = [
        *python_cmd(),
        str(VENDOR_PANDA / "scripts" / "panda" / "evaluate.py"),
        f"hydra.run.dir={EXPERIMENT_ROOT / 'outputs' / 'logs' / 'hydra'}",
        f"run_name={args.run_name}",
        f"eval.mode=predict",
        f"eval.data_paths_lst={hydra_list([data_root])}",
        f"eval.checkpoint_path={args.checkpoint}",
        f"eval.device={choose_device(cfg.get('device', 'auto'))}",
        f"eval.batch_size={cfg['batch_size']}",
        "eval.dataloader_num_workers=0",
        f"eval.num_subdirs={cfg.get('num_subdirs')}",
        f"eval.num_samples_per_subdir={cfg.get('num_samples_per_subdir')}",
        f"eval.prediction_length={cfg['prediction_length']}",
        f"eval.window_style={cfg['window_style']}",
        f"eval.window_stride={cfg['window_stride']}",
        f"eval.num_test_instances={cfg['num_test_instances']}",
        f"eval.save_forecasts={str(cfg.get('save_forecasts', True)).lower()}",
        f"eval.save_contexts={str(cfg.get('save_contexts', True)).lower()}",
        f"eval.save_labels={str(cfg.get('save_labels', True)).lower()}",
        "eval.limit_prediction_length=false",
        "eval.sliding_context=true",
        "eval.overwrite=true",
        f"eval.metric_names={hydra_list(cfg['metric_names'])}",
        f"eval.metrics_save_dir={metrics_dir}",
        f"eval.metrics_fname={args.run_name}",
        f"eval.forecast_save_dir={forecast_dir}",
        f"eval.labels_save_dir={labels_dir}",
    ]

    print("Running:", " ".join(str(x) for x in command))
    stream_subprocess(command, log_path=log_path, cwd=EXPERIMENT_ROOT)

    from plot_forecasts import plot_run

    plot_run(run_dir=run_dir, out_dir=plot_dir)
    write_json(
        run_dir / "run_metadata.json",
        {
            "checkpoint": args.checkpoint,
            "split": split,
            "data_root": str(data_root),
            "metrics_dir": str(metrics_dir),
            "forecast_dir": str(forecast_dir),
            "labels_dir": str(labels_dir),
            "plot_dir": str(plot_dir),
        },
    )


if __name__ == "__main__":
    main()
