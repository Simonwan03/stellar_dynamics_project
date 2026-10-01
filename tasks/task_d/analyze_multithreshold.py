#!/usr/bin/env python3
"""Estimate a semimajor-axis-ratio Lyapunov-time proxy from multiple crossings.

For each (i, i+16) perturbation pair, crossing times are measured for several
growth factors G = |Delta Q(t)| / |Delta Q(0)|.  A line

    t_cross / P_out,0 = intercept + (T_L,Q / P_out,0) * log(G)

is fitted to the observed crossings.  The slope is therefore an observable
finite-time Lyapunov-time proxy in initial outer-period units.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset_sampling_3_prefix1024",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/task_d/sampling_3/task_d_sampling3_q_multithreshold",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[10.0, 30.0, 100.0, 100.0 * math.e])
    parser.add_argument("--min-crossings", type=int, default=3)
    parser.add_argument("--r2-min", type=float, default=0.8)
    parser.add_argument("--min-span-periods", type=float, default=3.0)
    parser.add_argument("--min-valid-pairs", type=int, default=8)
    return parser.parse_args()


def first_crossing(
    growth: np.ndarray, times: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(growth) & np.isfinite(times)[None, :]
    hits = finite & (growth >= threshold)
    observed = hits.any(axis=1)
    index = np.argmax(hits, axis=1).astype(np.int64)
    crossing = np.full(growth.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(observed)
    crossing[rows] = times[index[rows]]

    previous = index[rows] - 1
    can_interpolate = (
        (previous >= 0)
        & np.isfinite(growth[rows, previous])
        & (growth[rows, previous] > 0.0)
        & np.isfinite(growth[rows, index[rows]])
        & (growth[rows, index[rows]] > 0.0)
        & (growth[rows, previous] < threshold)
    )
    selected = rows[can_interpolate]
    if selected.size:
        left_index = index[selected] - 1
        right_index = index[selected]
        left = np.log(growth[selected, left_index])
        right = np.log(growth[selected, right_index])
        denominator = right - left
        valid = np.isfinite(denominator) & (denominator > 0.0)
        selected = selected[valid]
        left_index = left_index[valid]
        right_index = right_index[valid]
        fraction = (math.log(threshold) - left[valid]) / denominator[valid]
        fraction = np.clip(fraction, 0.0, 1.0)
        crossing[selected] = times[left_index] + fraction * (
            times[right_index] - times[left_index]
        )
    return crossing, observed, index


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    result: dict[str, Any] = {"count": int(values.size)}
    if not values.size:
        return result
    result.update(
        {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": {
                f"p{p:g}": float(value)
                for p, value in zip(PERCENTILES, np.percentile(values, PERCENTILES))
            },
        }
    )
    return result


def fit_crossings(
    crossing_periods: np.ndarray,
    observed: np.ndarray,
    crossing_indices: np.ndarray,
    thresholds: np.ndarray,
    min_crossings: int,
) -> pd.DataFrame:
    log_thresholds = np.log(thresholds)
    rows: list[dict[str, Any]] = []
    for pair_index in range(len(crossing_periods)):
        mask = observed[pair_index] & np.isfinite(crossing_periods[pair_index])
        count = int(mask.sum())
        record: dict[str, Any] = {
            "num_crossings": count,
            "fit_intercept_periods": math.nan,
            "lyapunov_time_periods": math.nan,
            "lambda_per_period": math.nan,
            "fit_r2": math.nan,
            "crossing_span_periods": math.nan,
            "num_crossing_intervals": 0,
        }
        if count >= min_crossings:
            x = log_thresholds[mask]
            y = crossing_periods[pair_index, mask]
            design = np.column_stack((np.ones(count), x))
            intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
            prediction = intercept + slope * x
            sse = float(np.sum((y - prediction) ** 2))
            sst = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - sse / sst if sst > 0.0 else math.nan
            record.update(
                {
                    "fit_intercept_periods": float(intercept),
                    "lyapunov_time_periods": float(slope),
                    "lambda_per_period": float(1.0 / slope) if slope > 0.0 else math.nan,
                    "fit_r2": float(r2),
                    "crossing_span_periods": float(np.max(y) - np.min(y)),
                    "num_crossing_intervals": int(np.unique(crossing_indices[pair_index, mask]).size),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def save_plot(path: Path, pair_values: np.ndarray, system_values: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, title in (
        (axes[0], pair_values, "Pair-level"),
        (axes[1], system_values, "System median"),
    ):
        values = values[np.isfinite(values) & (values > 0.0)]
        axis.hist(values, bins=np.geomspace(values.min(), values.max(), 45), alpha=0.8)
        axis.set_xscale("log")
        axis.set_xlabel(r"$T_{L,Q}/P_{out,0}$")
        axis.set_ylabel("count")
        axis.set_title(f"{title} (N={len(values):,})")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    thresholds = np.asarray(sorted(set(args.thresholds)), dtype=np.float64)
    if thresholds.size < args.min_crossings or np.any(~np.isfinite(thresholds)) or np.any(thresholds <= 1.0):
        raise ValueError("Need at least min-crossings distinct finite thresholds greater than one")

    paths = sorted(args.dataset_dir.glob("results_batch_*_sampling_3.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No sampling_3 shards under {args.dataset_dir}")
    completion = pd.read_csv(
        args.dataset_dir / "completed_systems_sampling_3.txt",
        sep=r"\s+",
        header=None,
        names=["index", "batch", "time_unbound", "aux", "diagnosis"],
    )
    frames: list[pd.DataFrame] = []
    time_grid: np.ndarray | None = None

    for path in paths:
        match = RESULT_RE.fullmatch(path.name)
        if match is None:
            continue
        batch = int(match.group(1))
        batch_completion = completion.loc[completion["batch"] == batch].sort_values("index")
        with h5py.File(path, "r") as handle:
            ratios = np.asarray(handle["aratios_complete"][:, :, 0], dtype=np.float64)
            times = np.asarray(handle["times_complete"][0], dtype=np.float64)
            periods = np.asarray(handle["periods_complete"][:, 0, 1], dtype=np.float64)
        if ratios.shape[0] != len(batch_completion) or ratios.shape[0] % 32:
            raise ValueError(f"Unexpected row alignment in {path}")
        if time_grid is None:
            time_grid = times
        elif not np.allclose(times, time_grid, rtol=1e-12, atol=1e-9):
            raise ValueError(f"Time grid differs in {path}")

        groups = ratios.shape[0] // 32
        ratios = ratios.reshape(groups, 32, ratios.shape[1])
        periods = periods.reshape(groups, 32)
        valid_rows = (
            (batch_completion["time_unbound"].to_numpy(dtype=np.float64) >= 0.0)
            & (batch_completion["diagnosis"].to_numpy(dtype=np.int64) == 0)
        ).reshape(groups, 32)
        complete = valid_rows.all(axis=1)
        indices = batch_completion["index"].to_numpy(dtype=np.int64).reshape(groups, 32)
        system_ids = indices[:, 0] // 32

        separation = np.abs(ratios[:, :16] - ratios[:, 16:])
        initial = separation[:, :, 0]
        growth = np.full_like(separation, np.nan)
        denominator_ok = np.isfinite(initial) & (initial > 0.0)
        np.divide(separation, initial[:, :, None], out=growth, where=denominator_ok[:, :, None])
        flat_growth = growth.reshape(-1, growth.shape[-1])
        pair_period = np.nanmedian(
            np.stack((periods[:, :16], periods[:, 16:]), axis=-1), axis=-1
        ).reshape(-1)

        crossing_columns: list[np.ndarray] = []
        observed_columns: list[np.ndarray] = []
        index_columns: list[np.ndarray] = []
        for threshold in thresholds:
            crossing, observed, crossing_index = first_crossing(flat_growth, times, float(threshold))
            observed &= denominator_ok.reshape(-1) & np.isfinite(pair_period) & (pair_period > 0.0)
            crossing_columns.append(crossing / pair_period)
            observed_columns.append(observed)
            index_columns.append(crossing_index)
        crossing_matrix = np.column_stack(crossing_columns)
        observed_matrix = np.column_stack(observed_columns)
        index_matrix = np.column_stack(index_columns)
        fitted = fit_crossings(
            crossing_matrix, observed_matrix, index_matrix, thresholds, args.min_crossings
        )
        pair_number = np.tile(np.arange(16), groups)
        fitted.insert(0, "pair_index", pair_number)
        fitted.insert(0, "batch", batch)
        fitted.insert(0, "system_id", np.repeat(system_ids, 16))
        fitted["initial_separation"] = initial.reshape(-1)
        fitted["initial_outer_period"] = pair_period
        fitted["system_complete"] = np.repeat(complete, 16)
        for column, threshold in enumerate(thresholds):
            name = f"g{threshold:.8g}".replace(".", "p")
            fitted[f"crossed_{name}"] = observed_matrix[:, column]
            fitted[f"t_{name}_periods"] = crossing_matrix[:, column]
            fitted[f"index_{name}"] = index_matrix[:, column]
        frames.append(fitted)

    pair = pd.concat(frames, ignore_index=True).sort_values(["system_id", "pair_index"])
    pair = pair.loc[pair["system_complete"]].reset_index(drop=True)
    pair["fit_quality"] = (
        (pair["num_crossings"] >= args.min_crossings)
        & (pair["lyapunov_time_periods"] > 0.0)
        & (pair["fit_r2"] >= args.r2_min)
    )
    pair["time_resolved_quality"] = pair["fit_quality"] & (
        pair["crossing_span_periods"] >= args.min_span_periods
    )

    system_rows: list[dict[str, Any]] = []
    for system_id, group in pair.groupby("system_id", sort=True):
        record: dict[str, Any] = {"system_id": int(system_id), "num_pairs": int(len(group))}
        for quality_name in ("fit_quality", "time_resolved_quality"):
            selected = group.loc[group[quality_name] & np.isfinite(group["lambda_per_period"])]
            count = int(len(selected))
            lambda_median = float(selected["lambda_per_period"].median()) if count else math.nan
            suffix = "candidate" if quality_name == "fit_quality" else "resolved"
            record[f"num_valid_pairs_{suffix}"] = count
            record[f"lambda_system_{suffix}"] = lambda_median
            record[f"lyapunov_time_system_{suffix}_periods"] = (
                1.0 / lambda_median if count >= args.min_valid_pairs and lambda_median > 0.0 else math.nan
            )
        system_rows.append(record)
    system = pd.DataFrame(system_rows)

    threshold_summary: dict[str, Any] = {}
    for threshold in thresholds:
        name = f"g{threshold:.8g}".replace(".", "p")
        crossed = pair[f"crossed_{name}"].to_numpy(dtype=bool)
        system_crossed = pair.groupby("system_id")[f"crossed_{name}"].any()
        threshold_summary[f"{threshold:.12g}"] = {
            "pair_crossed": int(crossed.sum()),
            "pair_crossed_fraction": float(crossed.mean()),
            "system_any_pair_crossed": int(system_crossed.sum()),
            "system_any_pair_crossed_fraction": float(system_crossed.mean()),
        }

    pair_candidate = pair.loc[pair["fit_quality"], "lyapunov_time_periods"].to_numpy()
    pair_resolved = pair.loc[pair["time_resolved_quality"], "lyapunov_time_periods"].to_numpy()
    system_candidate = system["lyapunov_time_system_candidate_periods"].to_numpy()
    system_resolved = system["lyapunov_time_system_resolved_periods"].to_numpy()
    summary = {
        "definition": {
            "observable": "Q = a_in/a_out",
            "growth": "G(t) = abs(Delta Q(t)) / abs(Delta Q(0))",
            "thresholds": thresholds.tolist(),
            "fit": "OLS: t_cross/P_out,0 = intercept + (T_L,Q/P_out,0)*log(G)",
            "interpolation": "linear in log G between stored samples",
            "candidate_quality": {
                "min_crossings": args.min_crossings,
                "r2_min": args.r2_min,
                "positive_slope": True,
            },
            "time_resolved_quality_addition": {
                "min_crossing_span_periods": args.min_span_periods,
            },
            "system_aggregation": "median pair lambda, then reciprocal",
            "min_valid_pairs_per_system": args.min_valid_pairs,
        },
        "dataset": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "num_shards": len(paths),
            "num_complete_systems": int(pair["system_id"].nunique()),
            "num_pairs": int(len(pair)),
            "time_steps": int(len(time_grid)) if time_grid is not None else 0,
            "time_step_in_initial_outer_periods": float(
                np.median(np.diff(time_grid)) / pair["initial_outer_period"].median()
            )
            if time_grid is not None
            else None,
        },
        "threshold_crossings": threshold_summary,
        "quality_counts": {
            "pair_candidates": int(pair["fit_quality"].sum()),
            "pair_time_resolved": int(pair["time_resolved_quality"].sum()),
            "system_candidates": int(np.isfinite(system_candidate).sum()),
            "system_time_resolved": int(np.isfinite(system_resolved).sum()),
        },
        "pair_candidate_TL_over_Pout": distribution(pair_candidate),
        "pair_time_resolved_TL_over_Pout": distribution(pair_resolved),
        "system_candidate_TL_over_Pout": distribution(system_candidate),
        "system_time_resolved_TL_over_Pout": distribution(system_resolved),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(args.output_dir / "pair_multithreshold_labels.csv", index=False)
    system.to_csv(args.output_dir / "system_multithreshold_labels.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    if pair_resolved.size and np.isfinite(system_resolved).any():
        save_plot(
            args.output_dir / "multithreshold_lyapunov_time_distribution.png",
            pair_resolved,
            system_resolved,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
