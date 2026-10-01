from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import FileDataset


def _load_arrow(path: Path) -> np.ndarray:
    entry = next(iter(FileDataset(path=path, freq="h", one_dim_target=False)))
    return np.asarray(entry["target"])


def _metric_mean(series: pd.Series) -> float:
    values: list[float] = []
    for item in series.dropna():
        if isinstance(item, (int, float, np.number)):
            values.append(float(item))
            continue
        text = str(item)
        matches = re.findall(r"np\.float64\(([-+0-9.eE]+)\)|([-+0-9.eE]+)", text)
        for wrapped, plain in matches:
            token = wrapped or plain
            try:
                values.append(float(token))
            except ValueError:
                pass
    return float(np.mean(values)) if values else np.nan


def plot_run(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    forecast_dir = run_dir / "forecasts"
    labels_dir = run_dir / "labels"

    metric_files = sorted(metrics_dir.glob("*_pred*.csv"))
    if metric_files:
        horizon_records = []
        for metric_file in metric_files:
            df = pd.read_csv(metric_file)
            pred = int(metric_file.stem.split("_pred")[-1])
            horizon_records.append(
                {
                    "prediction_length": pred,
                    "mse": _metric_mean(df["mse"]) if "mse" in df else np.nan,
                    "mae": _metric_mean(df["mae"]) if "mae" in df else np.nan,
                    "smape": _metric_mean(df["smape"]) if "smape" in df else np.nan,
                }
            )
        horizon_df = pd.DataFrame(horizon_records).sort_values("prediction_length")
        fig, ax = plt.subplots(figsize=(7, 4))
        for metric_name in ["mse", "mae", "smape"]:
            if metric_name in horizon_df:
                ax.plot(horizon_df["prediction_length"], horizon_df[metric_name], marker="o", label=metric_name)
        ax.set_xlabel("Prediction length")
        ax.set_ylabel("Mean metric across systems")
        ax.set_title(run_dir.name)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        fig.savefig(out_dir / "horizon_metrics.png", dpi=160)
        plt.close(fig)

    forecast_files = sorted(forecast_dir.glob("*/*.arrow"))
    label_files = sorted(labels_dir.glob("*/*.arrow"))
    if not forecast_files or not label_files:
        return

    by_rel = {p.relative_to(forecast_dir): p for p in forecast_files}
    matched = [(by_rel[rel], labels_dir / rel) for rel in by_rel if (labels_dir / rel).exists()]
    for idx, (pred_path, gt_path) in enumerate(matched[:3]):
        pred = _load_arrow(pred_path)
        gt = _load_arrow(gt_path)
        num_channels = min(3, pred.shape[0], gt.shape[0])
        fig, axes = plt.subplots(num_channels, 1, figsize=(8, 2.5 * num_channels), sharex=True)
        if num_channels == 1:
            axes = [axes]
        for ch in range(num_channels):
            axes[ch].plot(gt[ch], label="gt", linewidth=2)
            axes[ch].plot(pred[ch], label="pred", linestyle="--")
            axes[ch].set_ylabel(f"ch{ch}")
            axes[ch].grid(True, alpha=0.25, linestyle=":")
        axes[0].legend()
        axes[-1].set_xlabel("time step")
        fig.suptitle(str(pred_path.relative_to(forecast_dir)))
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_example_{idx}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    plot_run(Path(args.run_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
