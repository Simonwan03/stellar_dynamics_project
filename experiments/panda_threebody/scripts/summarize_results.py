from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def metric_mean(series: pd.Series) -> float:
    values: list[float] = []
    for item in series.dropna():
        if isinstance(item, (int, float, np.number)):
            values.append(float(item))
            continue
        matches = re.findall(r"np\.float64\(([-+0-9.eE]+)\)|([-+0-9.eE]+)", str(item))
        for wrapped, plain in matches:
            token = wrapped or plain
            try:
                values.append(float(token))
            except ValueError:
                pass
    return float(np.mean(values)) if values else np.nan


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_run(run_dir: Path) -> list[str]:
    metrics_dir = run_dir / "metrics"
    metric_files = sorted(metrics_dir.glob("*_pred*.csv"))
    if not metric_files:
        return [f"## {run_dir.name}", "", "- No metrics files found.", ""]
    lines = [f"## {run_dir.name}", ""]
    rows = []
    for metric_file in metric_files:
        df = pd.read_csv(metric_file)
        pred = int(metric_file.stem.split("_pred")[-1])
        rows.append(
            {
                "prediction_length": pred,
                "mse": metric_mean(df["mse"]) if "mse" in df else np.nan,
                "mae": metric_mean(df["mae"]) if "mae" in df else np.nan,
                "smape": metric_mean(df["smape"]) if "smape" in df else np.nan,
                "n_systems": int(len(df)),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("prediction_length")
    lines.append(dataframe_to_markdown(summary_df.round(6)))
    lines.append("")
    lines.append(f"- metrics dir: `{metrics_dir}`")
    lines.append(f"- figures dir: `outputs/figures/{run_dir.name}`")
    lines.append("")
    return lines


def main() -> None:
    exp_root = Path(__file__).resolve().parents[1]
    eval_root = exp_root / "outputs" / "eval"
    report_path = exp_root / "reports" / "results_summary.md"
    lines = ["# Results Summary", ""]
    for run_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
        lines.extend(summarize_run(run_dir))
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
