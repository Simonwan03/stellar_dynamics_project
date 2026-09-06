#!/usr/bin/env python3
"""Estimate a paired e-folding Lyapunov-time proxy from sampling_3.

Each complete system contains 32 variations. Pair variation ``i`` with
``i + 16`` for ``i = 0, ..., 15`` and let ``Q = a_in / a_out`` be the scalar
semimajor-axis ratio stored in ``aratios_complete``. For each pair define

    D_i(t) = |Q_i(t) - Q_{i+16}(t)|,
    G_i(t) = D_i(t) / D_i(0).

The pair Lyapunov-time proxy is the first time at which ``G_i(t) >= e``.
Pairs without a crossing in the stored prefix are right-censored at the last
stored time. The primary output is the pair-level distribution; a system-level
summary aggregates the 16 pair times (minimum by default).

This is an observable-based e-folding proxy, not a variational-equation
maximum Lyapunov exponent.
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


ROOT = Path(__file__).resolve().parents[1]
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
VALID_PAIRS = tuple((i, i + 16) for i in range(16))
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the e-folding Lyapunov-time distribution from sampling_3."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset_sampling_3_prefix1024",
    )
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/task_d_sampling3_pair_efold_float64",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=math.e,
        help="Growth threshold for paired semimajor-axis-ratio separation (default: e).",
    )
    parser.add_argument(
        "--signed-growth",
        action="store_true",
        help="Deprecated compatibility option; paired separation is always non-negative.",
    )
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Keep the first sampled crossing time; otherwise interpolate linearly in log G.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["min", "median", "mean"],
        default="min",
        help="System aggregation for the 16 pair event/censor times.",
    )
    parser.add_argument("--completion-table-name", default="completed_systems_sampling_3.txt")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    return parser.parse_args()


def result_paths(dataset_dir: Path, sampling_id: int, max_shards: int | None) -> list[Path]:
    paths = sorted(dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No sampling_{sampling_id} HDF5 shards found under {dataset_dir}")
    if max_shards is not None:
        paths = paths[:max_shards]
    return paths


def batch_id(path: Path) -> int:
    match = RESULT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot parse batch id from {path}")
    return int(match.group(1))


def load_completion_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "batch", "time_unbound", "aux", "diagnosis"],
    )
    if table.empty:
        raise ValueError(f"Completion table is empty: {path}")
    table = table.astype(
        {
            "index": "int64",
            "batch": "int64",
            "time_unbound": "float64",
            "aux": "float64",
            "diagnosis": "int64",
        }
    )
    if table["index"].duplicated().any():
        raise ValueError("Completion-table trajectory indices are not unique.")
    return table.sort_values(["batch", "index"]).reset_index(drop=True)


def first_crossing(
    observable: np.ndarray,
    times: np.ndarray,
    threshold: float,
    interpolate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event/censor time, event flag, and first crossing index per row."""
    finite = np.isfinite(observable) & np.isfinite(times)[None, :]
    hits = finite & (observable >= threshold)
    observed = hits.any(axis=1)
    crossing_index = np.argmax(hits, axis=1).astype(np.int64)
    last_time = float(times[np.flatnonzero(np.isfinite(times))[-1]])
    sampled_time = np.full(observable.shape[0], last_time, dtype=np.float64)
    rows = np.flatnonzero(observed)
    sampled_time[rows] = times[crossing_index[rows]]
    event_time = sampled_time.copy()

    if interpolate and rows.size:
        previous = crossing_index[rows] - 1
        can_interpolate = (
            (previous >= 0)
            & np.isfinite(observable[rows, crossing_index[rows]])
            & np.isfinite(observable[rows, previous])
            & (observable[rows, previous] > 0.0)
            & (observable[rows, crossing_index[rows]] > 0.0)
            & (observable[rows, previous] < threshold)
            & (observable[rows, crossing_index[rows]] >= threshold)
        )
        if np.any(can_interpolate):
            selected_rows = rows[can_interpolate]
            selected_previous = previous[can_interpolate]
            selected_current = crossing_index[selected_rows]
            previous_log = np.log(observable[selected_rows, selected_previous])
            current_log = np.log(observable[selected_rows, selected_current])
            threshold_log = math.log(threshold)
            denominator = current_log - previous_log
            valid = np.isfinite(denominator) & (denominator > 0.0)
            if np.any(valid):
                valid_rows = selected_rows[valid]
                valid_previous = selected_previous[valid]
                fraction = (threshold_log - previous_log[valid]) / denominator[valid]
                fraction = np.clip(fraction, 0.0, 1.0)
                event_time[valid_rows] = times[valid_previous] + fraction * (
                    times[selected_current[valid]] - times[valid_previous]
                )
    return event_time, observed, crossing_index


def quantile_summary(values: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    events = values[observed & np.isfinite(values)]
    result: dict[str, Any] = {
        "total": int(values.size),
        "events": int(events.size),
        "censored": int(values.size - events.size),
        "censored_fraction": float(1.0 - events.size / values.size) if values.size else None,
        "event_time_quantiles": {},
    }
    if events.size:
        result["event_time_quantiles"] = {
            f"p{p:g}": float(v) for p, v in zip(PERCENTILES, np.percentile(events, PERCENTILES))
        }
        result["event_time_mean"] = float(np.mean(events))
        result["event_time_median"] = float(np.median(events))
        result["event_time_min"] = float(np.min(events))
        result["event_time_max"] = float(np.max(events))
    else:
        result.update(
            {
                "event_time_mean": None,
                "event_time_median": None,
                "event_time_min": None,
                "event_time_max": None,
            }
        )
    return result


def kaplan_meier(values: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    """Build a right-censored Kaplan–Meier survival table."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    finite = np.isfinite(values)
    values = values[finite]
    observed = observed[finite]
    if values.size == 0:
        return pd.DataFrame(columns=["time", "at_risk", "events", "censored", "survival"])

    event_times = np.unique(values[observed])
    censor_times = np.unique(values[~observed])
    all_times = np.unique(values)
    at_risk = values.size
    survival = 1.0
    rows: list[dict[str, Any]] = []
    for time in all_times:
        events = int(np.sum(observed & (values == time)))
        censored = int(np.sum((~observed) & (values == time)))
        if events:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "time": float(time),
                "at_risk": int(at_risk),
                "events": events,
                "censored": censored,
                "survival": float(survival),
            }
        )
        at_risk -= events + censored
    return pd.DataFrame(rows)


def km_summary(curve: pd.DataFrame) -> dict[str, Any]:
    if curve.empty:
        return {"median": None, "p90": None, "p95": None}
    survival = curve["survival"].to_numpy(dtype=np.float64)
    times = curve["time"].to_numpy(dtype=np.float64)

    def first_time_at_or_below(level: float) -> float | None:
        hit = np.flatnonzero(survival <= level)
        return float(times[hit[0]]) if hit.size else None

    return {
        "median": first_time_at_or_below(0.5),
        "p90": first_time_at_or_below(0.1),
        "p95": first_time_at_or_below(0.05),
    }


def aggregate_system_times(
    trajectory_times: np.ndarray,
    trajectory_observed: np.ndarray,
    system_ids: np.ndarray,
    aggregation: str,
    complete_system_ids: np.ndarray,
    max_systems: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    complete_set = set(int(value) for value in complete_system_ids)
    for system_id in sorted(complete_set):
        mask = system_ids == system_id
        times = trajectory_times[mask]
        observed = trajectory_observed[mask]
        if times.size == 0:
            continue
        if aggregation == "min":
            value = float(np.min(times))
            event = bool(np.any(observed))
        elif aggregation == "median":
            value = float(np.median(times))
            event = bool(np.all(observed))
        else:
            value = float(np.mean(times))
            event = bool(np.all(observed))
        rows.append(
            {
                "system_id": int(system_id),
                "tau_system": value,
                "event_observed": event,
                "num_trajectories": int(times.size),
                "num_trajectory_events": int(np.sum(observed)),
            }
        )
        if max_systems is not None and len(rows) >= max_systems:
            break
    frame = pd.DataFrame(rows)
    return frame, frame["system_id"].to_numpy(dtype=np.int64) if len(frame) else np.empty(0, dtype=np.int64)


def pair_initial_precision_diagnostic(
    smax_initial: np.ndarray,
    aratio_initial: np.ndarray,
    group_size: int,
) -> dict[str, Any]:
    if group_size < 32:
        return {"skipped": True, "reason": "group_size < 32"}
    s_deltas: list[float] = []
    q_deltas: list[float] = []
    for start in range(0, len(smax_initial), group_size):
        s_group = smax_initial[start : start + group_size]
        q_group = aratio_initial[start : start + group_size]
        if len(s_group) < group_size:
            continue
        for i, j in VALID_PAIRS:
            s_deltas.append(abs(float(s_group[i] - s_group[j])))
            q_deltas.append(abs(float(q_group[i] - q_group[j])))

    def summarize(values: list[float]) -> dict[str, Any]:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return {
            "count": int(finite.size),
            "zero_count": int(np.sum(finite == 0.0)),
            "zero_fraction": float(np.mean(finite == 0.0)) if finite.size else None,
            "quantiles": {
                f"p{p:g}": float(v) for p, v in zip(PERCENTILES, np.percentile(finite, PERCENTILES))
            }
            if finite.size
            else {},
        }

    return {
        "pairs": [list(pair) for pair in VALID_PAIRS],
        "smax_inner_delta0": summarize(s_deltas),
        "aratio_delta0": summarize(q_deltas),
        "interpretation": (
            "These paired denominators are not suitable for the primary label when their "
            "zero_fraction is large; the primary label uses each trajectory's Q(0) instead."
        ),
    }


def save_distribution_plot(
    output_path: Path,
    trajectory_times: np.ndarray,
    trajectory_observed: np.ndarray,
    system_times: np.ndarray,
    system_observed: np.ndarray,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, values, observed, title in zip(
        axes,
        (trajectory_times, system_times),
        (trajectory_observed, system_observed),
        ("Variation-pair level", "Complete-system level (minimum pair time)"),
    ):
        finite_events = values[observed & np.isfinite(values)]
        finite_all = values[np.isfinite(values)]
        if finite_events.size:
            lo = max(float(np.min(finite_events)), 1.0)
            hi = max(float(np.max(finite_events)), lo * 1.01)
            bins = np.geomspace(lo, hi, 45)
            axis.hist(finite_events, bins=bins, color="#4472C4", alpha=0.78)
            axis.set_xscale("log")
        axis.set_title(f"{title}\nN={finite_all.size:,}, events={finite_events.size:,}")
        axis.set_xlabel("paired e-folding time")
        axis.set_ylabel("count")
        axis.grid(alpha=0.2)
    fig.suptitle(f"sampling_3 paired Lyapunov-time proxy, threshold={threshold:.8g}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _legacy_single_trajectory_main() -> None:
    args = parse_args()
    if args.threshold <= 1.0 or not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite and greater than 1.")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")

    paths = result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    completion_path = args.dataset_dir / args.completion_table_name
    completion = load_completion_table(completion_path)
    all_trajectory_rows: list[dict[str, Any]] = []
    all_times: list[float] = []
    all_observed: list[bool] = []
    all_crossing_index: list[int] = []
    all_smax_initial: list[float] = []
    all_aratio_initial: list[float] = []
    time_grid: np.ndarray | None = None
    source_attrs: dict[str, Any] = {}

    for path in paths:
        current_batch = batch_id(path)
        batch_completion = completion.loc[completion["batch"] == current_batch].sort_values("index")
        with h5py.File(path, "r") as handle:
            ratio_ds = handle["aratios_complete"]
            time_ds = handle["times_complete"]
            smax_ds = handle["smaxes_complete"]
            row_count, time_steps = time_ds.shape[:2]
            if ratio_ds.shape[:2] != (row_count, time_steps):
                raise ValueError(f"{path}: aratio/time dimensions do not match")
            if len(batch_completion) != row_count:
                raise ValueError(
                    f"{path}: completion table has {len(batch_completion)} rows, expected {row_count}"
                )
            if ratio_ds.ndim != 3 or ratio_ds.shape[-1] != 1:
                raise ValueError(f"{path}: expected aratios_complete shape (N,T,1), got {ratio_ds.shape}")
            current_times = np.asarray(time_ds[0], dtype=np.float64)
            if time_grid is None:
                time_grid = current_times
                source_attrs = {str(key): value for key, value in handle.attrs.items()}
            elif current_times.shape != time_grid.shape or not np.allclose(
                current_times, time_grid, equal_nan=True, rtol=1e-7, atol=1e-5
            ):
                raise ValueError(f"{path}: time grid differs from the first shard")

            ratios = np.asarray(ratio_ds[:, :, 0], dtype=np.float64)
            initial = ratios[:, 0]
            denominator_ok = np.isfinite(initial) & (np.abs(initial) > 1e-12)
            growth = np.full_like(ratios, np.nan, dtype=np.float64)
            if args.signed_growth:
                np.divide(
                    ratios,
                    initial[:, None],
                    out=growth,
                    where=denominator_ok[:, None],
                )
            else:
                np.divide(
                    np.abs(ratios),
                    np.abs(initial)[:, None],
                    out=growth,
                    where=denominator_ok[:, None],
                )
            event_time, observed, crossing_index = first_crossing(
                growth,
                current_times,
                args.threshold,
                interpolate=not args.no_interpolation,
            )
            observed &= denominator_ok
            event_time[~denominator_ok] = current_times[np.flatnonzero(np.isfinite(current_times))[-1]]
            indices = batch_completion["index"].to_numpy(dtype=np.int64)
            valid = (
                batch_completion["time_unbound"].to_numpy(dtype=np.float64) >= 0.0
            ) & (batch_completion["diagnosis"].to_numpy(dtype=np.int64) == 0)
            all_times.extend(event_time.tolist())
            all_observed.extend(observed.tolist())
            all_crossing_index.extend(crossing_index.tolist())
            all_smax_initial.extend(np.asarray(smax_ds[:, 0, 0], dtype=np.float64).tolist())
            all_aratio_initial.extend(initial.tolist())
            for row, index, is_valid in zip(
                range(row_count), indices, valid, strict=True
            ):
                all_trajectory_rows.append(
                    {
                        "index": int(index),
                        "batch": current_batch,
                        "row_in_shard": int(row),
                        "system_id": int(index // args.group_size),
                        "valid_completion": bool(is_valid),
                        "tau_efold": float(event_time[row]),
                        "event_observed": bool(observed[row]),
                        "crossing_index": int(crossing_index[row]),
                    }
                )

    if time_grid is None:
        raise ValueError("No time grid was loaded")
    trajectory = pd.DataFrame(all_trajectory_rows).sort_values("index").reset_index(drop=True)
    if trajectory.empty:
        raise ValueError("No trajectory rows were loaded")
    trajectory["system_complete"] = trajectory.groupby("system_id")["valid_completion"].transform("all")
    complete_system_ids = np.sort(
        trajectory.loc[trajectory["system_complete"], "system_id"].unique().astype(np.int64)
    )
    if args.max_systems is not None:
        complete_system_ids = complete_system_ids[: args.max_systems]

    system, selected_system_ids = aggregate_system_times(
        trajectory_times=trajectory["tau_efold"].to_numpy(dtype=np.float64),
        trajectory_observed=trajectory["event_observed"].to_numpy(dtype=bool),
        system_ids=trajectory["system_id"].to_numpy(dtype=np.int64),
        aggregation=args.aggregation,
        complete_system_ids=complete_system_ids,
        max_systems=args.max_systems,
    )
    if system.empty:
        raise ValueError("No complete systems remain after completion filtering")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_dir / "trajectory_efold_labels.csv", index=False)
    system.to_csv(output_dir / "system_efold_labels.csv", index=False)

    trajectory_curve = kaplan_meier(
        trajectory.loc[trajectory["valid_completion"], "tau_efold"].to_numpy(dtype=np.float64),
        trajectory.loc[trajectory["valid_completion"], "event_observed"].to_numpy(dtype=bool),
    )
    system_curve = kaplan_meier(
        system["tau_system"].to_numpy(dtype=np.float64),
        system["event_observed"].to_numpy(dtype=bool),
    )
    trajectory_curve.to_csv(output_dir / "trajectory_kaplan_meier.csv", index=False)
    system_curve.to_csv(output_dir / "system_kaplan_meier.csv", index=False)

    trajectory_valid = trajectory[trajectory["valid_completion"]].copy()
    summary: dict[str, Any] = {
        "definition": {
            "observable": "aratios_complete = a_in/a_out",
            "growth_curve": "abs(Q(t))/abs(Q(0))" if not args.signed_growth else "Q(t)/Q(0)",
            "threshold": float(args.threshold),
            "threshold_description": "e-folding threshold" if np.isclose(args.threshold, math.e) else "custom growth threshold",
            "crossing": "first time growth_curve >= threshold",
            "interpolation": "linear in log growth between adjacent samples"
            if not args.no_interpolation
            else "none; first sampled crossing",
            "system_aggregation": args.aggregation,
            "censoring": "right-censored at last stored time when no crossing is observed",
            "group_size": int(args.group_size),
        },
        "dataset": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "sampling_id": int(args.sampling_id),
            "shards": [str(path.resolve()) for path in paths],
            "num_shards": len(paths),
            "num_trajectory_rows": int(len(trajectory)),
            "num_valid_completion_rows": int(trajectory["valid_completion"].sum()),
            "num_invalid_completion_rows": int((~trajectory["valid_completion"]).sum()),
            "num_complete_systems": int(len(complete_system_ids)),
            "time_steps": int(len(time_grid)),
            "time_start": float(time_grid[np.flatnonzero(np.isfinite(time_grid))[0]]),
            "time_end": float(time_grid[np.flatnonzero(np.isfinite(time_grid))[-1]]),
            "time_step_median": float(np.median(np.diff(time_grid[np.isfinite(time_grid)]))),
            "source_attrs": source_attrs,
        },
        "trajectory_level_valid_completion": quantile_summary(
            trajectory_valid["tau_efold"].to_numpy(dtype=np.float64),
            trajectory_valid["event_observed"].to_numpy(dtype=bool),
        ),
        "trajectory_level_all_rows": quantile_summary(
            trajectory["tau_efold"].to_numpy(dtype=np.float64),
            trajectory["event_observed"].to_numpy(dtype=bool),
        ),
        "system_level": quantile_summary(
            system["tau_system"].to_numpy(dtype=np.float64),
            system["event_observed"].to_numpy(dtype=bool),
        ),
        "trajectory_kaplan_meier": km_summary(trajectory_curve),
        "system_kaplan_meier": km_summary(system_curve),
        "pair_initial_precision_diagnostic": pair_initial_precision_diagnostic(
            np.asarray(all_smax_initial, dtype=np.float64),
            np.asarray(all_aratio_initial, dtype=np.float64),
            args.group_size,
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    save_distribution_plot(
        output_dir / "efold_time_distribution.png",
        trajectory_valid["tau_efold"].to_numpy(dtype=np.float64),
        trajectory_valid["event_observed"].to_numpy(dtype=bool),
        system["tau_system"].to_numpy(dtype=np.float64),
        system["event_observed"].to_numpy(dtype=bool),
        args.threshold,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    print(f"Outputs written to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.threshold <= 1.0 or not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite and greater than 1.")
    if args.group_size != 32:
        raise ValueError("Paired analysis requires --group-size 32 for pairs (i, i+16).")
    if args.signed_growth:
        raise ValueError("--signed-growth is not defined for absolute paired separation.")

    paths = result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
    completion = load_completion_table(args.dataset_dir / args.completion_table_name)
    pair_frames: list[pd.DataFrame] = []
    time_grid: np.ndarray | None = None
    source_attrs: dict[str, Any] = {}
    source_dtype: str | None = None

    for path in paths:
        current_batch = batch_id(path)
        batch_completion = completion.loc[completion["batch"] == current_batch].sort_values("index")
        with h5py.File(path, "r") as handle:
            ratio_ds = handle["aratios_complete"]
            time_ds = handle["times_complete"]
            row_count, time_steps = time_ds.shape[:2]
            if ratio_ds.shape != (row_count, time_steps, 1):
                raise ValueError(
                    f"{path}: expected aratios_complete shape {(row_count, time_steps, 1)}, "
                    f"got {ratio_ds.shape}"
                )
            if ratio_ds.dtype != np.dtype("float64"):
                raise ValueError(f"{path}: expected float64 aratios_complete, got {ratio_ds.dtype}")
            if row_count % args.group_size:
                raise ValueError(f"{path}: row count {row_count} is not divisible by 32")
            if len(batch_completion) != row_count:
                raise ValueError(
                    f"{path}: completion table has {len(batch_completion)} rows, expected {row_count}"
                )

            current_times = np.asarray(time_ds[0], dtype=np.float64)
            if time_grid is None:
                time_grid = current_times
                source_attrs = {str(key): value for key, value in handle.attrs.items()}
                source_dtype = str(ratio_ds.dtype)
            elif current_times.shape != time_grid.shape or not np.allclose(
                current_times, time_grid, equal_nan=True, rtol=1e-12, atol=1e-9
            ):
                raise ValueError(f"{path}: time grid differs from the first shard")

            indices = batch_completion["index"].to_numpy(dtype=np.int64)
            grouped_indices = indices.reshape(-1, args.group_size)
            system_ids = grouped_indices[:, 0] // args.group_size
            expected_indices = system_ids[:, None] * args.group_size + np.arange(args.group_size)
            if not np.array_equal(grouped_indices, expected_indices):
                raise ValueError(f"{path}: indices are not aligned in contiguous groups of 32")

            completion_valid = (
                (batch_completion["time_unbound"].to_numpy(dtype=np.float64) >= 0.0)
                & (batch_completion["diagnosis"].to_numpy(dtype=np.int64) == 0)
            ).reshape(-1, args.group_size)
            ratios = np.asarray(ratio_ds[:, :, 0], dtype=np.float64).reshape(
                -1, args.group_size, time_steps
            )
            separation = np.abs(ratios[:, :16, :] - ratios[:, 16:, :])
            initial_separation = separation[:, :, 0]
            denominator_ok = np.isfinite(initial_separation) & (initial_separation > 0.0)
            growth = np.full_like(separation, np.nan, dtype=np.float64)
            np.divide(
                separation,
                initial_separation[:, :, None],
                out=growth,
                where=denominator_ok[:, :, None],
            )
            event_time, observed, crossing_index = first_crossing(
                growth.reshape(-1, time_steps),
                current_times,
                args.threshold,
                interpolate=not args.no_interpolation,
            )
            denominator_flat = denominator_ok.reshape(-1)
            observed &= denominator_flat
            event_time[~denominator_flat] = current_times[
                np.flatnonzero(np.isfinite(current_times))[-1]
            ]

            num_systems = len(system_ids)
            pair_index = np.tile(np.arange(16, dtype=np.int64), num_systems)
            pair_frames.append(
                pd.DataFrame(
                    {
                        "system_id": np.repeat(system_ids, 16),
                        "batch": np.full(num_systems * 16, current_batch, dtype=np.int64),
                        "pair_index": pair_index,
                        "variation_i": pair_index,
                        "variation_j": pair_index + 16,
                        "index_i": grouped_indices[:, :16].reshape(-1),
                        "index_j": grouped_indices[:, 16:].reshape(-1),
                        "initial_separation": initial_separation.reshape(-1),
                        "tau_pair": event_time,
                        "event_observed": observed,
                        "crossing_index": crossing_index,
                        "valid_pair_completion": (
                            completion_valid[:, :16] & completion_valid[:, 16:]
                        ).reshape(-1),
                        "system_complete": np.repeat(completion_valid.all(axis=1), 16),
                    }
                )
            )

    if time_grid is None or not pair_frames:
        raise ValueError("No paired data were loaded")
    pair = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["system_id", "pair_index"]
    ).reset_index(drop=True)
    all_system_ids = np.sort(pair["system_id"].unique().astype(np.int64))
    complete_system_ids = np.sort(
        pair.loc[pair["system_complete"], "system_id"].unique().astype(np.int64)
    )
    system, selected_system_ids = aggregate_system_times(
        trajectory_times=pair["tau_pair"].to_numpy(dtype=np.float64),
        trajectory_observed=pair["event_observed"].to_numpy(dtype=bool),
        system_ids=pair["system_id"].to_numpy(dtype=np.int64),
        aggregation=args.aggregation,
        complete_system_ids=complete_system_ids,
        max_systems=args.max_systems,
    )
    if system.empty:
        raise ValueError("No complete systems remain after completion filtering")
    system = system.rename(
        columns={
            "num_trajectories": "num_pairs",
            "num_trajectory_events": "num_pair_events",
        }
    )
    pair_complete = pair.loc[pair["system_id"].isin(selected_system_ids)].copy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(output_dir / "pair_efold_labels.csv", index=False)
    system.to_csv(output_dir / "system_efold_labels.csv", index=False)
    pair_curve = kaplan_meier(
        pair_complete["tau_pair"].to_numpy(dtype=np.float64),
        pair_complete["event_observed"].to_numpy(dtype=bool),
    )
    system_curve = kaplan_meier(
        system["tau_system"].to_numpy(dtype=np.float64),
        system["event_observed"].to_numpy(dtype=bool),
    )
    pair_curve.to_csv(output_dir / "pair_kaplan_meier.csv", index=False)
    system_curve.to_csv(output_dir / "system_kaplan_meier.csv", index=False)

    pair_index_rows: list[dict[str, Any]] = []
    for pair_index_value, group in pair_complete.groupby("pair_index", sort=True):
        stats = quantile_summary(
            group["tau_pair"].to_numpy(dtype=np.float64),
            group["event_observed"].to_numpy(dtype=bool),
        )
        row = {
            "pair_index": int(pair_index_value),
            "variation_i": int(pair_index_value),
            "variation_j": int(pair_index_value) + 16,
            "total": stats["total"],
            "events": stats["events"],
            "censored": stats["censored"],
            "censored_fraction": stats["censored_fraction"],
            "mean": stats["event_time_mean"],
            "median": stats["event_time_median"],
            "min": stats["event_time_min"],
            "max": stats["event_time_max"],
        }
        row.update(stats["event_time_quantiles"])
        pair_index_rows.append(row)
    pair_index_summary = pd.DataFrame(pair_index_rows)
    pair_index_summary.to_csv(output_dir / "pair_index_summary.csv", index=False)

    valid_pair = pair.loc[pair["valid_pair_completion"]]
    initial_all = pair["initial_separation"].to_numpy(dtype=np.float64)
    initial_complete = pair_complete["initial_separation"].to_numpy(dtype=np.float64)

    def initial_summary(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        return {
            "count": int(values.size),
            "zero_count": int(np.sum(finite == 0.0)),
            "nonfinite_count": int(values.size - finite.size),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "quantiles": {
                f"p{p:g}": float(value)
                for p, value in zip(PERCENTILES, np.percentile(finite, PERCENTILES))
            }
            if finite.size
            else {},
        }

    summary: dict[str, Any] = {
        "definition": {
            "observable": "Q = aratios_complete = a_in/a_out",
            "pairing": "variation i paired with variation i+16 for i=0,...,15",
            "separation": "D_i(t) = abs(Q_i(t) - Q_(i+16)(t))",
            "growth_curve": "G_i(t) = D_i(t) / D_i(0)",
            "threshold": float(args.threshold),
            "crossing": "first time G_i(t) >= threshold",
            "interpolation": "linear in log growth between adjacent samples"
            if not args.no_interpolation
            else "none; first sampled crossing",
            "censoring": "right-censored at the last stored time",
            "system_aggregation": args.aggregation,
        },
        "dataset": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "sampling_id": int(args.sampling_id),
            "source_dtype": source_dtype,
            "num_shards": len(paths),
            "num_trajectory_rows": int(len(pair) * 2),
            "num_systems": int(len(all_system_ids)),
            "num_complete_systems": int(len(complete_system_ids)),
            "num_selected_complete_systems": int(len(selected_system_ids)),
            "num_pair_rows": int(len(pair)),
            "num_valid_pair_completion_rows": int(pair["valid_pair_completion"].sum()),
            "num_pairs_in_selected_complete_systems": int(len(pair_complete)),
            "time_steps": int(len(time_grid)),
            "time_start": float(time_grid[np.flatnonzero(np.isfinite(time_grid))[0]]),
            "time_end": float(time_grid[np.flatnonzero(np.isfinite(time_grid))[-1]]),
            "time_step_median": float(np.median(np.diff(time_grid[np.isfinite(time_grid)]))),
            "source_attrs": source_attrs,
        },
        "pair_level_complete_systems": quantile_summary(
            pair_complete["tau_pair"].to_numpy(dtype=np.float64),
            pair_complete["event_observed"].to_numpy(dtype=bool),
        ),
        "pair_level_valid_pairs": quantile_summary(
            valid_pair["tau_pair"].to_numpy(dtype=np.float64),
            valid_pair["event_observed"].to_numpy(dtype=bool),
        ),
        "pair_level_all_pairs": quantile_summary(
            pair["tau_pair"].to_numpy(dtype=np.float64),
            pair["event_observed"].to_numpy(dtype=bool),
        ),
        "system_level": quantile_summary(
            system["tau_system"].to_numpy(dtype=np.float64),
            system["event_observed"].to_numpy(dtype=bool),
        ),
        "pair_index_complete_systems": pair_index_rows,
        "pair_kaplan_meier": km_summary(pair_curve),
        "system_kaplan_meier": km_summary(system_curve),
        "initial_pair_separation_all": initial_summary(initial_all),
        "initial_pair_separation_complete_systems": initial_summary(initial_complete),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    save_distribution_plot(
        output_dir / "pair_efold_time_distribution.png",
        pair_complete["tau_pair"].to_numpy(dtype=np.float64),
        pair_complete["event_observed"].to_numpy(dtype=bool),
        system["tau_system"].to_numpy(dtype=np.float64),
        system["event_observed"].to_numpy(dtype=bool),
        args.threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    print(f"Outputs written to {output_dir}")

if __name__ == "__main__":
    main()
