"""Summarize and plot divergence times from a completion table.

The completion table is expected to contain the five whitespace-separated
columns used by this project::

    index  batch  time_unbound  wall_time  diagnosis

Run from the repository root, for example::

    conda run -n stellar python -m tasks.common.analyze_divergence

``time_unbound == censor_time`` means that the trajectory reached the end of
the integration without diverging.  At system level, the representative time
is the earliest valid divergence time among the 32 variations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "dataset_v3" / "completed_systems_sampling_2.txt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "divergence_statistics"
COLUMNS = ["index", "batch", "time_unbound", "wall_time", "diagnosis"]


def figure_caption(group_size: int) -> str:
    """Return the paper-facing caption for the survival-distribution figure."""
    return (
        "Distribution of instability times in the dataset. "
        "(a) Empirical survival distribution of individual trajectories from the analyzed complete systems. "
        f"(b) System-level survival distribution, where the instability time of a system is defined as "
        f"the earliest instability among its {group_size} variations "
        f"(T_sys = min_{{j=1,...,{group_size}}} T_j). "
        "Dashed vertical lines indicate the 50th, 90th, and 99th percentiles of finite instability times. "
        "Trajectories that remain bound throughout the simulation horizon are treated as right-censored."
    )


def parse_percentiles(value: str) -> list[float]:
    percentiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not percentiles or any(not 0.0 < item < 100.0 for item in percentiles):
        raise argparse.ArgumentTypeError("Percentiles must be comma-separated values strictly between 0 and 100.")
    return percentiles


def load_completion_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if table.empty:
        raise ValueError(f"Completion table is empty: {path}")
    table = table.astype(
        {
            "index": "int64",
            "batch": "int64",
            "time_unbound": "float64",
            "wall_time": "float64",
            "diagnosis": "int64",
        }
    )
    duplicated = table["index"].duplicated(keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, "index"].head(5).tolist()
        raise ValueError(f"Trajectory indices must be unique; duplicates include {examples}.")
    return table


def select_valid_rows(table: pd.DataFrame) -> pd.DataFrame:
    valid = table[(table["time_unbound"] >= 0.0) & (table["diagnosis"] == 0)].copy()
    if valid.empty:
        raise ValueError("No valid rows have time_unbound >= 0 and diagnosis == 0.")
    return valid


def build_group_times(
    valid: pd.DataFrame,
    group_size: int,
    require_complete: bool = True,
) -> pd.DataFrame:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    grouped_rows = valid.assign(
        group_id=valid["index"] // group_size,
        variation_id=valid["index"] % group_size,
    )
    group_table = (
        grouped_rows.groupby("group_id", sort=True)
        .agg(
            valid_variations=("variation_id", "size"),
            unique_variations=("variation_id", "nunique"),
            time_unbound=("time_unbound", "min"),
        )
        .reset_index()
    )
    if require_complete:
        keep = (group_table["valid_variations"] == group_size) & (
            group_table["unique_variations"] == group_size
        )
        group_table = group_table[keep].copy()

    return group_table


def survival_curve(times: np.ndarray, censor_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Return x and N(T >= x), including the integration end point."""
    values = np.sort(np.asarray(times, dtype=np.float64))
    x = np.unique(np.concatenate(([0.0], values[values <= censor_time], [censor_time])))
    counts = values.size - np.searchsorted(values, x, side="left")
    return x, counts.astype(np.int64)


def summarize_population(
    times: np.ndarray,
    censor_time: float,
    percentiles: list[float],
) -> dict[str, Any]:
    values = np.asarray(times, dtype=np.float64)
    divergent = values < censor_time
    event_times = values[divergent]
    percentile_values = np.percentile(event_times, percentiles) if event_times.size else [None] * len(percentiles)
    return {
        "total": int(values.size),
        "divergent": int(divergent.sum()),
        "non_divergent_at_censor_time": int((~divergent).sum()),
        "divergent_fraction": float(divergent.mean()) if values.size else None,
        "divergence_time_percentiles": {
            f"p{percentile:g}": None if result is None else float(result)
            for percentile, result in zip(percentiles, percentile_values)
        },
    }


def save_curve(path: Path, times: np.ndarray, censor_time: float) -> None:
    x, counts = survival_curve(times, censor_time)
    pd.DataFrame({"time": x, "surviving_count": counts}).to_csv(path, index=False)


def plot_population(
    axis: Any,
    times: np.ndarray,
    censor_time: float,
    percentiles: list[float],
    title: str,
    population_label: str = "samples",
    show_ylabel: bool = True,
) -> None:
    values = np.asarray(times, dtype=np.float64)
    x, counts = survival_curve(values, censor_time)
    total = values.size
    survival_fraction = counts / total if total else np.zeros_like(counts, dtype=np.float64)
    positive = counts > 0
    axis.plot(x[positive], survival_fraction[positive], color="#0072B2", linewidth=1.8)
    axis.fill_between(x[positive], survival_fraction[positive], 0, color="#56B4E9", alpha=0.12)

    event_times = values[values < censor_time]
    if event_times.size:
        percentile_values = np.percentile(event_times, percentiles)
        for percentile, value in zip(percentiles, percentile_values):
            axis.axvline(value, color="0.35", linestyle="--", linewidth=1.0, alpha=0.65)
            axis.text(
                value,
                0.97,
                f"p{percentile:g}",
                transform=axis.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.5,
                color="0.25",
            )

    axis.set_xlim(left=0.0, right=censor_time)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title, loc="left", pad=3.0)
    axis.set_xlabel(r"Time $t$ [outer orbital periods]")
    if show_ylabel:
        axis.set_ylabel(r"Survival fraction $S(t)=P(T \geq t)$")
    else:
        axis.set_ylabel("")
    axis.text(
        0.98,
        0.86,
        f"N={total:,} {population_label}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    axis.grid(alpha=0.12, linewidth=0.5, which="major")


def analyze(
    table_path: Path,
    output_dir: Path,
    group_size: int,
    censor_time: float | None,
    percentiles: list[float],
    require_complete_groups: bool,
) -> dict[str, Any]:
    table = load_completion_table(table_path)
    valid = select_valid_rows(table)
    if censor_time is None:
        censor_time = float(valid["time_unbound"].max())
    if censor_time <= 0:
        raise ValueError(f"censor_time must be positive, got {censor_time}.")

    group_table = build_group_times(valid, group_size, require_complete_groups)
    if group_table.empty:
        raise ValueError("No groups remain after validity/completeness filtering.")

    included_group_ids = group_table["group_id"].to_numpy()
    trajectory_valid = valid[(valid["index"] // group_size).isin(included_group_ids)]
    trajectory_times = trajectory_valid["time_unbound"].to_numpy(dtype=np.float64)
    group_times = group_table["time_unbound"].to_numpy(dtype=np.float64)
    raw_group_count = int(table.assign(group_id=table["index"] // group_size)["group_id"].nunique())

    summary = {
        "input_table": str(table_path.resolve()),
        "censor_time": float(censor_time),
        "group_size": int(group_size),
        "group_representative": "minimum time_unbound among variations",
        "percentiles_use_divergent_samples_only": True,
        "figure_caption": figure_caption(group_size),
        "raw_rows": int(len(table)),
        "invalid_rows": int(len(table) - len(valid)),
        "valid_rows_excluded_as_incomplete_groups": int(len(valid) - len(trajectory_valid)),
        "raw_groups": raw_group_count,
        "groups_excluded_as_invalid_or_incomplete": int(raw_group_count - len(group_table)),
        "trajectory": summarize_population(trajectory_times, censor_time, percentiles),
        "group": summarize_population(group_times, censor_time, percentiles),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "divergence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    save_curve(output_dir / "trajectory_survival_curve.csv", trajectory_times, censor_time)
    save_curve(output_dir / "group_survival_curve.csv", group_times, censor_time)
    group_table.to_csv(output_dir / "group_representative_times.csv", index=False)

    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True, sharey=True)
    plot_population(
        axes[0],
        trajectory_times,
        censor_time,
        percentiles,
        "(a) Individual",
        population_label="trajectories",
    )
    plot_population(
        axes[1],
        group_times,
        censor_time,
        percentiles,
        "(b) Ensemble",
        population_label="systems",
        show_ylabel=False,
    )
    figure.savefig(output_dir / "divergence_time_distribution.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count instability times and plot trajectory-level/system-level survival curves."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Completion table path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--censor-time",
        type=float,
        default=None,
        help="Integration end time; defaults to the largest valid time_unbound.",
    )
    parser.add_argument("--percentiles", type=parse_percentiles, default=parse_percentiles("50,90,99"))
    parser.add_argument(
        "--include-incomplete-groups",
        action="store_true",
        help="Include groups with fewer than group-size valid variations (excluded by default).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(
        table_path=args.table,
        output_dir=args.output_dir,
        group_size=args.group_size,
        censor_time=args.censor_time,
        percentiles=args.percentiles,
        require_complete_groups=not args.include_incomplete_groups,
    )
    trajectory = summary["trajectory"]
    group = summary["group"]
    print(f"Censor time: {summary['censor_time']:g}")
    print(
        f"Trajectories: {trajectory['divergent']:,}/{trajectory['total']:,} divergent "
        f"({trajectory['divergent_fraction']:.2%})"
    )
    print(
        f"Systems: {group['divergent']:,}/{group['total']:,} divergent "
        f"({group['divergent_fraction']:.2%}); representative = earliest of {args.group_size} variations"
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
