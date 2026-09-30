#!/usr/bin/env python3
"""Create a system-level statistical audit for 32-variation triple systems.

The completion-table ``index`` is the global trajectory id.  Systems are formed
as ``index // group_size`` and variations as ``index % group_size``.  The
script keeps only systems with every valid variation and reports trajectory
events only within that common complete-system cohort.

Example:

    conda run -n stellar python analysis/audit_system_level_statistics.py \
      --dataset-dir /data/sampling_2 --sampling-id 2

The requested horizons (32, 64, ...) are interpreted on the absolute
``time_unbound`` scale.  They are not interpreted as numbers of stored samples:
the repository documents that the trajectory sample clock uses a different
unit from ``time_unbound``.
"""

from __future__ import annotations

import argparse
import json
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
GROUP_SIZE = 32
DEFAULT_TIME_HORIZON = 10000.0
DEFAULT_FAIL_HORIZONS = (32.0, 64.0, 128.0, 256.0, 512.0)
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
LABEL_COLUMNS = ["index", "batch", "time_unbound", "wall_time", "diagnosis"]
IC_FIELDS = ("aratios", "eccs", "incs", "LANs", "masses")


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.") from exc
    if not values or any(not np.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("Horizons must be finite, non-negative values.")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("Horizons must be unique and sorted in ascending order.")
    return values


def _load_labels(path: Path, group_size: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=LABEL_COLUMNS)
    if frame.empty:
        raise ValueError(f"Completion table is empty: {path}")
    frame = frame.astype(
        {
            "index": "int64",
            "batch": "int64",
            "time_unbound": "float64",
            "wall_time": "float64",
            "diagnosis": "int64",
        }
    )
    if frame["index"].duplicated().any():
        repeated = frame.loc[frame["index"].duplicated(keep=False), "index"].head(10).tolist()
        raise ValueError(f"Completion table has duplicate trajectory ids, including {repeated}.")
    frame["system_id"] = frame["index"] // group_size
    frame["variation_id"] = frame["index"] % group_size
    frame["valid"] = (
        np.isfinite(frame["time_unbound"])
        & (frame["time_unbound"] >= 0.0)
        & (frame["diagnosis"] == 0)
    )
    return frame


def _complete_system_ids(labels: pd.DataFrame, group_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    all_groups = labels.groupby("system_id", sort=True)
    raw_group = all_groups.agg(
        raw_rows=("index", "size"),
        raw_variations=("variation_id", "nunique"),
        raw_min_variation=("variation_id", "min"),
        raw_max_variation=("variation_id", "max"),
        batches=("batch", "nunique"),
    )
    valid = labels.loc[labels["valid"]]
    valid_group = valid.groupby("system_id", sort=True).agg(
        valid_rows=("index", "size"),
        valid_variations=("variation_id", "nunique"),
        valid_min_variation=("variation_id", "min"),
        valid_max_variation=("variation_id", "max"),
    )
    catalog = raw_group.join(valid_group, how="left").fillna(
        {
            "valid_rows": 0,
            "valid_variations": 0,
            "valid_min_variation": -1,
            "valid_max_variation": -1,
        }
    )
    catalog["valid_rows"] = catalog["valid_rows"].astype(int)
    catalog["valid_variations"] = catalog["valid_variations"].astype(int)
    complete = (
        (catalog["raw_rows"] == group_size)
        & (catalog["raw_variations"] == group_size)
        & (catalog["raw_min_variation"] == 0)
        & (catalog["raw_max_variation"] == group_size - 1)
        & (catalog["valid_rows"] == group_size)
        & (catalog["valid_variations"] == group_size)
        & (catalog["valid_min_variation"] == 0)
        & (catalog["valid_max_variation"] == group_size - 1)
        & (catalog["batches"] == 1)
    )
    catalog["complete_valid_system"] = complete
    catalog["invalid_rows"] = catalog["raw_rows"] - catalog["valid_rows"]
    catalog = catalog.reset_index()
    return catalog.loc[complete, "system_id"].to_numpy(dtype=np.int64), catalog


def _read_static_array(handle: h5py.File, name: str) -> np.ndarray:
    """Read an IC field, accepting static ``[N, d]`` or result ``[N, T, d]`` layout."""
    if name not in handle:
        raise KeyError(f"Missing initial-condition field {name!r}.")
    dataset = handle[name]
    if dataset.ndim >= 3:
        values = np.asarray(dataset[:, 0, ...], dtype=np.float64)
    else:
        values = np.asarray(dataset[:], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return values.reshape(values.shape[0], -1)


def _load_initial_conditions(
    dataset_dir: Path,
    sampling_id: int,
    labels: pd.DataFrame,
    group_size: int,
) -> tuple[dict[str, np.ndarray], str, int | None]:
    """Load only t=0 coarse ICs, using static ICs when available."""
    static_path = dataset_dir / f"ICs_sampling_{sampling_id}.hdf5"
    if static_path.is_file():
        with h5py.File(static_path, "r") as handle:
            fields = {name: _read_static_array(handle, name) for name in IC_FIELDS}
        row_counts = {name: value.shape[0] for name, value in fields.items()}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Static IC fields have inconsistent row counts: {row_counts}")
        return fields, f"static IC file: {static_path}", None

    paths: list[tuple[int, Path]] = []
    for path in dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"):
        match = RESULT_RE.fullmatch(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda item: item[0])
    if not paths:
        raise FileNotFoundError(f"No result shards found in {dataset_dir} for sampling {sampling_id}.")

    max_index = int(labels["index"].max())
    fields: dict[str, np.ndarray] | None = None
    step_count: int | None = None
    for batch_id, path in paths:
        batch_labels = labels.loc[labels["batch"] == batch_id]
        with h5py.File(path, "r") as handle:
            if "times_complete" not in handle:
                raise KeyError(f"Missing times_complete in {path}")
            row_count = int(handle["times_complete"].shape[0])
            time_steps = int(handle["times_complete"].shape[1])
            step_count = time_steps if step_count is None else step_count
            if step_count != time_steps:
                raise ValueError("Result shards have inconsistent trajectory lengths.")
            global_ids = batch_labels["index"].to_numpy(dtype=np.int64)
            local_ids = global_ids - batch_id * row_count
            if global_ids.size != row_count or not np.array_equal(np.sort(local_ids), np.arange(row_count)):
                raise ValueError(
                    f"Completion-table ids do not map one-to-one to rows in batch {batch_id}; "
                    "cannot safely align IC features."
                )
            values = {
                name: _read_static_array(handle, f"{name}_complete")
                for name in IC_FIELDS
            }
        if fields is None:
            fields = {
                name: np.full((max_index + 1, value.shape[1]), np.nan, dtype=np.float64)
                for name, value in values.items()
            }
        for name, value in values.items():
            fields[name][global_ids] = value[local_ids]

    assert fields is not None
    return fields, f"t=0 fields from {len(paths)} result shards under {dataset_dir}", step_count


def _load_time_axis_metadata(dataset_dir: Path, sampling_id: int) -> dict[str, Any] | None:
    paths = []
    for path in dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"):
        match = RESULT_RE.fullmatch(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda item: item[0])
    if not paths:
        return None

    source_path = paths[0][1]
    with h5py.File(source_path, "r") as handle:
        time_key = "times_complete" if "times_complete" in handle else "times"
        time_grid = np.asarray(handle[time_key][0], dtype=np.float64)
        source_steps = int(handle.attrs.get("source_time_steps", time_grid.size))
        source_end_attr = handle.attrs.get("source_time_end")

    finite_time = time_grid[np.isfinite(time_grid)]
    if finite_time.size == 0:
        raise ValueError(f"No finite time values in {source_path}")
    deltas = np.diff(finite_time)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    median_interval = float(np.median(deltas)) if deltas.size else None
    stored_start = float(finite_time[0])
    stored_end = float(finite_time[-1])
    inferred_source_end = (
        stored_start + median_interval * (source_steps - 1)
        if median_interval is not None and source_steps >= 1
        else stored_end
    )

    source_end_attr_value = None
    source_end_attr_used = False
    if source_end_attr is not None:
        candidate = float(np.asarray(source_end_attr).reshape(-1)[0])
        if np.isfinite(candidate):
            source_end_attr_value = candidate
            # Some older sampling_3 prefix files have a corrupt source endpoint
            # attribute. It must not precede the endpoint present in the prefix.
            source_end_attr_used = candidate >= stored_end
    source_end = source_end_attr_value if source_end_attr_used else inferred_source_end
    source_duration = source_end - stored_start
    stored_duration_fraction = (
        (stored_end - stored_start) / source_duration if source_duration > 0.0 else 1.0
    )

    return {
        "source_file": str(source_path),
        "stored_time_steps": int(time_grid.size),
        "source_time_steps": source_steps,
        "stored_time_start": stored_start,
        "stored_time_end": stored_end,
        "median_sampling_interval": median_interval,
        "source_time_end_attribute": source_end_attr_value,
        "source_time_end_attribute_used": source_end_attr_used,
        "estimated_source_time_end": float(source_end),
        "stored_sample_fraction": float(time_grid.size / source_steps) if source_steps else 1.0,
        "stored_duration_fraction": float(stored_duration_fraction),
    }


def _physical_features(fields: dict[str, np.ndarray]) -> pd.DataFrame:
    aratios = fields["aratios"][:, 0]
    eccentricity = fields["eccs"]
    inclination = fields["incs"]
    longitude = fields["LANs"]
    masses = fields["masses"]
    if eccentricity.shape[1] < 2 or inclination.shape[1] < 2 or longitude.shape[1] < 2 or masses.shape[1] < 3:
        raise ValueError("Expected two orbital channels and three masses for each trajectory.")

    delta_node = np.arctan2(
        np.sin(longitude[:, 0] - longitude[:, 1]),
        np.cos(longitude[:, 0] - longitude[:, 1]),
    )
    cosine_mutual = (
        np.cos(inclination[:, 0]) * np.cos(inclination[:, 1])
        + np.sin(inclination[:, 0])
        * np.sin(inclination[:, 1])
        * np.cos(delta_node)
    )
    i_mut = np.arccos(np.clip(cosine_mutual, -1.0, 1.0))
    e_in = eccentricity[:, 0]
    e_out = eccentricity[:, 1]
    q_out = masses[:, 2] / (masses[:, 0] + masses[:, 1])
    a_out_over_a_in = 1.0 / aratios

    # This is the same log(Rp / Rp_crit) margin used by
    # tasks.common.traditional_stability.triple_stability.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rp_actual = a_out_over_a_in * (1.0 - e_out)
        rp_crit = (
            2.8
            * np.power(
                (1.0 + q_out) * (1.0 + e_out) / np.sqrt(1.0 - e_out),
                2.0 / 5.0,
            )
            * (1.0 - 0.3 * i_mut / np.pi)
        )
        ma_margin = np.log(rp_actual / rp_crit)

    return pd.DataFrame(
        {
            "a_out_over_a_in": a_out_over_a_in,
            "e_in": e_in,
            "e_out": e_out,
            "i_mut_deg": np.degrees(i_mut),
            "q_out": q_out,
            "ma_margin": ma_margin,
        }
    )


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return {"q25": None, "median": None, "q75": None}
    q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def _km_curve(times: np.ndarray, event: np.ndarray, censor_time: float) -> pd.DataFrame:
    times = np.asarray(times, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    event_times = np.unique(times[event & np.isfinite(times) & (times < censor_time)])
    survival = 1.0
    rows: list[dict[str, Any]] = [
        {"time": 0.0, "at_risk": int(times.size), "events": 0, "survival": 1.0}
    ]
    for time in event_times:
        at_risk = int(np.count_nonzero(times >= time))
        events = int(np.count_nonzero(event & (times == time)))
        if at_risk:
            survival *= 1.0 - events / at_risk
        rows.append(
            {"time": float(time), "at_risk": at_risk, "events": events, "survival": survival}
        )
    rows.append(
        {
            "time": float(censor_time),
            "at_risk": int(np.count_nonzero(times >= censor_time)),
            "events": 0,
            "survival": survival,
        }
    )
    return pd.DataFrame(rows).drop_duplicates(subset=["time"], keep="last").sort_values("time")


def _save_figure(fig: Any, out_dir: Path, stem: str, dpi: int = 220) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_stable_fraction(systems: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.arange(-0.5, 33.5, 1.0) / GROUP_SIZE
    axes[0].hist(systems["p_stable"], bins=bins, color="#3178a8", edgecolor="white")
    axes[0].set(xlim=(-0.02, 1.02), xlabel=r"Stable fraction $p_s$", ylabel="Base systems", title="System stable-fraction distribution")
    axes[0].set_xticks(np.linspace(0, 1, 5))
    labels = ["0", "(0, .25)", "[.25, .5)", "[.5, .75)", "[.75, 1)", "1"]
    counts = systems["stable_fraction_bin"].value_counts().reindex(range(6), fill_value=0)
    axes[1].bar(labels, counts.to_numpy(), color="#dc8a39")
    axes[1].set(xlabel=r"$p_s$ bin", ylabel="Base systems", title="Requested coarse bins")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_figure(fig, out_dir, "stable_fraction_histogram")


def _plot_sensitivity(systems: pd.DataFrame, out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.8))
    axis.hist(systems["phase_sensitivity"], bins=np.linspace(0, 1, 21), color="#6d9f71", edgecolor="white")
    axis.set(xlim=(0, 1), xlabel=r"Phase sensitivity $S_s=4p_s(1-p_s)$", ylabel="Base systems", title="System phase-sensitivity distribution")
    fig.tight_layout()
    _save_figure(fig, out_dir, "phase_sensitivity_histogram")


def _plot_failure_times(event_times: np.ndarray, out_dir: Path, censor_time: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(event_times, bins=50, color="#b35b53", edgecolor="white")
    axes[0].set(xlabel=r"Failure time $T_{fail}$ (time_unbound units)", ylabel="Unstable variations", title="Failure-time histogram")
    positive = event_times[event_times > 0]
    zero_count = int(np.count_nonzero(event_times == 0))
    if positive.size:
        low = max(float(positive.min()), np.finfo(float).tiny)
        high = max(float(positive.max()), low * 1.001)
        log_bins = np.geomspace(low, high, 45) if high > low else np.array([low * 0.99, high * 1.01])
        axes[1].hist(positive, bins=log_bins, color="#80569b", edgecolor="white")
        axes[1].set_xscale("log")
    axes[1].set(xlabel=r"Failure time $T_{fail}$ (log scale)", ylabel="Unstable variations", title="Log-time histogram")
    axes[1].text(0.04, 0.95, f"T_fail = 0: {zero_count:,}\nCensor horizon: {censor_time:g}", transform=axes[1].transAxes, va="top")
    fig.tight_layout()
    _save_figure(fig, out_dir, "failure_time_histograms")


def _plot_failure_cdf_survival(event_times: np.ndarray, km: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if event_times.size:
        sorted_times = np.sort(event_times)
        cdf = np.arange(1, sorted_times.size + 1) / sorted_times.size
        axes[0].step(sorted_times, cdf, where="post", color="#b35b53")
    axes[0].set(xlabel=r"Failure time $T_{fail}$", ylabel="Empirical CDF among failures", ylim=(0, 1.02), title="Failure-time empirical CDF")
    axes[1].step(km["time"], km["survival"], where="post", color="#3178a8")
    axes[1].set(xlabel=r"Time_unbound", ylabel=r"$S(t)$", ylim=(0, 1.02), title="Kaplan–Meier survival (censored at horizon)")
    fig.tight_layout()
    _save_figure(fig, out_dir, "failure_cdf_and_survival")


def _plot_fail_count_histograms(system_counts: pd.DataFrame, horizons: tuple[float, ...], out_dir: Path) -> None:
    ncols = 2
    nrows = int(np.ceil(len(horizons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.0 * nrows), squeeze=False)
    for axis, horizon in zip(axes.flat, horizons):
        column = f"n_fail_t{horizon:g}"
        counts = system_counts[column].value_counts().reindex(range(GROUP_SIZE + 1), fill_value=0)
        axis.bar(np.arange(GROUP_SIZE + 1), counts.to_numpy(), color="#d48a46", width=0.9)
        axis.set(xlim=(-1, GROUP_SIZE + 1), xlabel=rf"$N_{{fail}}(T={horizon:g})$", ylabel="Base systems", title=f"Failures by absolute horizon T={horizon:g}")
    for axis in axes.flat[len(horizons) :]:
        axis.set_visible(False)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fail_count_histograms_by_horizon")


def _binned_median(axis: Any, x: np.ndarray, y: np.ndarray, bins: int = 20) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < bins or float(np.max(x)) <= float(np.min(x)):
        return
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    assignments = np.digitize(x, edges[1:-1], right=False)
    centers, medians, lows, highs = [], [], [], []
    min_count = max(20, int(x.size / bins * 0.10))
    for bin_id in range(bins):
        in_bin = assignments == bin_id
        if int(in_bin.sum()) < min_count:
            continue
        centers.append(float(np.median(x[in_bin])))
        medians.append(float(np.median(y[in_bin])))
        lows.append(float(np.quantile(y[in_bin], 0.25)))
        highs.append(float(np.quantile(y[in_bin], 0.75)))
    if centers:
        axis.plot(centers, medians, color="#d24e36", linewidth=2.0, marker="o", markersize=3.5, label="Equal-width-bin median")
        axis.fill_between(centers, lows, highs, color="#d24e36", alpha=0.16, linewidth=0)


def _plot_relation(axis: Any, systems: pd.DataFrame, x_name: str, y_name: str, y_label: str) -> None:
    x = systems[x_name].to_numpy(dtype=np.float64)
    y = systems[y_name].to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    axis.scatter(x[valid], y[valid], s=7, alpha=0.17, color="#3178a8", edgecolors="none", rasterized=True)
    _binned_median(axis, x, y)
    x_labels = {
        "ma_margin": r"Mardling–Aarseth margin $\delta_{MA}$",
        "a_out_over_a_in": r"$a_{out}/a_{in}$",
        "e_in": r"$e_{in}$",
        "e_out": r"$e_{out}$",
        "i_mut_deg": r"$i_{mut}$ (degrees)",
        "q_out": r"$q_{out}$",
    }
    axis.set(xlabel=x_labels.get(x_name, x_name.replace("_", " ")), ylabel=y_label)
    axis.grid(alpha=0.2)


def _plot_physical_relations(systems: pd.DataFrame, out_dir: Path) -> None:
    x_fields = [
        ("ma_margin", "Mardling–Aarseth margin"),
        ("a_out_over_a_in", r"$a_{out}/a_{in}$"),
        ("e_in", r"$e_{in}$"),
        ("e_out", r"$e_{out}$"),
        ("i_mut_deg", r"$i_{mut}$ (degrees)"),
        ("q_out", r"$q_{out}$"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(19, 13.5))
    for idx, (field, title) in enumerate(x_fields):
        row, pair = divmod(idx, 2)
        col = 2 * pair
        _plot_relation(axes[row, col], systems, field, "p_stable", r"Stable fraction $p_s$")
        axes[row, col].set_title(f"$p_s$ vs {title}")
        _plot_relation(axes[row, col + 1], systems, field, "phase_sensitivity", r"$S_s=4p_s(1-p_s)$")
        axes[row, col + 1].set_title(f"$S_s$ vs {title}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False)
        fig.subplots_adjust(top=0.93)
    else:
        fig.tight_layout()
    _save_figure(fig, out_dir, "stable_fraction_and_sensitivity_vs_initial_conditions")

    for y_name, y_label, stem in (
        ("p_stable", r"Stable fraction $p_s$", "stable_fraction_vs_ma_margin"),
        ("phase_sensitivity", r"Phase sensitivity $S_s$", "phase_sensitivity_vs_ma_margin"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        _plot_relation(axis, systems, "ma_margin", y_name, y_label)
        axis.set_title(f"{y_label} vs Mardling–Aarseth margin")
        fig.tight_layout()
        _save_figure(fig, out_dir, stem)


def _ma_distance_summary(systems: pd.DataFrame) -> pd.DataFrame:
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, np.inf]
    labels = ["[0,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.5)", "[0.5,0.75)", "[0.75,1)", "[1,+inf)"]
    frame = systems.copy()
    frame["ma_distance_bin"] = pd.cut(
        frame["abs_ma_margin"], bins=edges, labels=labels, right=False, include_lowest=True
    )
    result = (
        frame.groupby("ma_distance_bin", observed=True, sort=True)
        .agg(
            systems=("system_id", "size"),
            mean_p_stable=("p_stable", "mean"),
            mean_sensitivity=("phase_sensitivity", "mean"),
            median_sensitivity=("phase_sensitivity", "median"),
            phase_sensitive_fraction=("is_phase_sensitive", "mean"),
            sensitivity_gt_0_5_fraction=("phase_sensitivity", lambda values: float((values > 0.5).mean())),
        )
        .reset_index()
    )
    return result


def _plot_ma_distance_summary(distance: pd.DataFrame, out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(distance))
    bars = axis.bar(x, distance["mean_sensitivity"], color="#729c75", alpha=0.88)
    axis.set_ylabel(r"Mean phase sensitivity $S_s$")
    axis.set_xlabel(r"Distance to MA boundary $|\delta_{MA}|$")
    axis.set_xticks(x, distance["ma_distance_bin"].astype(str), rotation=20, ha="right")
    axis.set_ylim(bottom=0)
    twin = axis.twinx()
    twin.plot(x, distance["phase_sensitive_fraction"], color="#bd5c46", marker="o", linewidth=2)
    twin.set_ylabel("Phase-sensitive system fraction")
    twin.set_ylim(0, 1)
    for xpos, bar, count in zip(x, bars, distance["systems"]):
        axis.text(xpos, bar.get_height() + 0.007, f"n={count:,}", ha="center", va="bottom", fontsize=8)
    axis.set_title("Phase sensitivity is greatest near the MA boundary")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "phase_sensitivity_by_ma_distance")


def _write_summary_markdown(path: Path, summary: dict[str, Any], bins: pd.DataFrame, by_horizon: pd.DataFrame, correlations: pd.DataFrame, ma_distance: pd.DataFrame) -> None:
    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100.0 * value:.2f}%"

    lines = [
        f"# System-level statistical audit (sampling {summary['sampling_id']})",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Completion table: `{summary['completion_table']}`",
        f"- Grouping: `system_id = index // {summary['group_size']}`, `variation_id = index % {summary['group_size']}`",
        f"- Stable label: `time_unbound >= {summary['stable_horizon']:g}`; valid row requires `time_unbound >= 0` and `diagnosis == 0`.",
    ]
    if summary.get("sampling_relationship"):
        lines.extend(["", "## Relationship between sampling versions", "", summary["sampling_relationship"]])
    time_axis = summary.get("time_axis")
    if time_axis:
        lines.extend(
            [
                "",
                "## Stored trajectory time coverage",
                "",
                f"- Stored `times_complete` samples: {time_axis['stored_time_steps']:,} / {time_axis['source_time_steps']:,} source samples ({pct(time_axis['stored_sample_fraction'])}).",
                f"- Stored time window: `{time_axis['stored_time_start']:g}` to `{time_axis['stored_time_end']:g}`; median sample interval `{time_axis['median_sampling_interval']}`.",
                f"- Estimated full-source endpoint: `{time_axis['estimated_source_time_end']:g}` ({pct(time_axis['stored_duration_fraction'])} of the source time span is present locally).",
                "- The stable-fraction and failure-time statistics below are calculated from the completion table, not inferred by treating the local HDF5 prefix as a completed trajectory.",
            ]
        )
    lines.extend(
        [
            "",
            "## Cohort and labels",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Raw base systems | {summary['raw_systems']:,} |",
            f"| Raw trajectories | {summary['raw_trajectories']:,} |",
            f"| Invalid trajectories | {summary['invalid_trajectories']:,} |",
            f"| Complete valid systems | {summary['complete_valid_systems']:,} |",
            f"| Trajectories in audit cohort | {summary['cohort_trajectories']:,} |",
            f"| Valid rows excluded with incomplete systems | {summary['valid_rows_excluded_with_incomplete_systems']:,} |",
            f"| Mean stable trajectory rate | {pct(summary['stable_trajectory_rate'])} |",
            "",
            "## Stable fraction and phase sensitivity",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Fully unstable systems (`p=0`) | {summary['fully_unstable_systems']:,} ({pct(summary['fully_unstable_fraction'])}) |",
            f"| Phase-sensitive systems (`0<p<1`) | {summary['phase_sensitive_systems']:,} ({pct(summary['phase_sensitive_fraction'])}) |",
            f"| Fully stable systems (`p=1`) | {summary['fully_stable_systems']:,} ({pct(summary['fully_stable_fraction'])}) |",
            f"| Median `p_stable` | {summary['p_stable_quantiles']['median']:.5f} |",
            f"| Mean phase sensitivity | {summary['phase_sensitivity_mean']:.5f} |",
            f"| Median phase sensitivity | {summary['phase_sensitivity_quantiles']['median']:.5f} |",
            f"| Phase sensitivity `S>0.5` | {summary['sensitivity_gt_0_5_systems']:,} ({pct(summary['sensitivity_gt_0_5_fraction'])}) |",
            "",
            "| Stable-fraction bin | Systems | Fraction |",
            "|---|---:|---:|",
        ]
    )
    for row in bins.itertuples(index=False):
        lines.append(f"| {row.bin_label} | {row.systems:,} | {pct(row.fraction)} |")
    lines.extend(
        [
            "",
            "![Stable fraction histogram](stable_fraction_histogram.png)",
            "",
            "![Phase sensitivity histogram](phase_sensitivity_histogram.png)",
            "",
            "## Failure times and requested horizons",
            "",
            f"- Observed unstable variations in the audit cohort: {summary['unstable_variations']:,}/{summary['cohort_trajectories']:,} ({pct(summary['unstable_variation_fraction'])}).",
            f"- Failure-time quantiles among observed unstable variations: `{summary['failure_time_quantiles']}`.",
            "- The table below treats `T` as an absolute `time_unbound` threshold, matching the repository's survival targets.",
            "- Repository notes say the stored trajectory sample clock uses a different unit from `time_unbound`; these rows must not be read as `prefix_len=T` exposure rates without a validated clock conversion.",
            "",
            "| Absolute `time_unbound` horizon T | Failures by T | Cohort trajectories | Fraction failed by T | Fraction of all eventual failures |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_horizon.itertuples(index=False):
        lines.append(
            f"| {row.horizon:g} | {row.failed_by_horizon:,} | {row.cohort_trajectories:,} | "
            f"{pct(row.trajectory_fraction)} | {pct(row.eventual_failures_seen_fraction)} |"
        )
    lines.extend(
        [
            "",
            "![Failure-time distribution and survival](failure_cdf_and_survival.png)",
            "",
            "![Per-system early-failure counts](fail_count_histograms_by_horizon.png)",
            "",
            "## Coarse initial conditions and MA boundary",
            "",
            f"Initial conditions source: `{summary['initial_conditions_source']}`.",
            "For each system, physical coordinates are medians across its 32 valid initial variations. The MA margin follows the repository's existing log(Rp/Rp,crit) implementation.",
            "",
            "| Feature | Spearman with `p_stable` | Spearman with `S` | Systems used |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in correlations.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.spearman_p_stable:.4f} | {row.spearman_sensitivity:.4f} | {row.n_systems:,} |")
    lines.extend(
        [
            "",
            "![Stable fraction and sensitivity vs initial conditions](stable_fraction_and_sensitivity_vs_initial_conditions.png)",
            "",
            "### Phase sensitivity by distance to the MA boundary",
            "",
            "| Absolute MA-margin bin | Systems | Mean `S` | Median `S` | Phase-sensitive fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ma_distance.itertuples(index=False):
        lines.append(
            f"| {row.ma_distance_bin} | {row.systems:,} | {row.mean_sensitivity:.4f} | "
            f"{row.median_sensitivity:.4f} | {pct(row.phase_sensitive_fraction)} |"
        )
    lines.extend(
        [
            "",
            "![Phase sensitivity by distance to MA boundary](phase_sensitivity_by_ma_distance.png)",
            "",
            "Separate signed-margin figures are also provided as `stable_fraction_vs_ma_margin.png` and `phase_sensitivity_vs_ma_margin.png`. These are descriptive summaries without confidence intervals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    dataset_dir: Path,
    sampling_id: int,
    output_dir: Path,
    group_size: int = GROUP_SIZE,
    stable_horizon: float = DEFAULT_TIME_HORIZON,
    fail_horizons: tuple[float, ...] = DEFAULT_FAIL_HORIZONS,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = dataset_dir / f"completed_systems_sampling_{sampling_id}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Completion table does not exist: {label_path}")

    labels = _load_labels(label_path, group_size)
    complete_ids, catalog = _complete_system_ids(labels, group_size)
    if not complete_ids.size:
        raise ValueError("No complete valid systems were found.")
    valid = labels.loc[labels["valid"]].copy()
    cohort = valid.loc[valid["system_id"].isin(complete_ids)].copy()
    cohort["stable"] = cohort["time_unbound"] >= stable_horizon
    cohort["failed"] = cohort["time_unbound"] < stable_horizon

    stable_count = cohort.groupby("system_id", sort=True)["stable"].sum().reindex(complete_ids).astype(int)
    systems = pd.DataFrame(
        {
            "system_id": complete_ids,
            "stable_count": stable_count.to_numpy(dtype=np.int64),
        }
    )
    systems["valid_variations"] = group_size
    systems["p_stable"] = systems["stable_count"] / float(group_size)
    systems["phase_sensitivity"] = 4.0 * systems["p_stable"] * (1.0 - systems["p_stable"])
    systems["is_phase_sensitive"] = (systems["p_stable"] > 0.0) & (systems["p_stable"] < 1.0)
    systems["unstable_count"] = group_size - systems["stable_count"]
    systems["stable_fraction_bin"] = np.select(
        [
            systems["p_stable"] == 0.0,
            (systems["p_stable"] > 0.0) & (systems["p_stable"] < 0.25),
            (systems["p_stable"] >= 0.25) & (systems["p_stable"] < 0.5),
            (systems["p_stable"] >= 0.5) & (systems["p_stable"] < 0.75),
            (systems["p_stable"] >= 0.75) & (systems["p_stable"] < 1.0),
            systems["p_stable"] == 1.0,
        ],
        [0, 1, 2, 3, 4, 5],
        default=-1,
    ).astype(int)

    # Construct all requested N_fail(T) distributions on the event-time axis.
    count_frame = pd.DataFrame({"system_id": complete_ids})
    horizon_rows: list[dict[str, Any]] = []
    total_events = int(cohort["failed"].sum())
    for horizon in fail_horizons:
        hit = cohort["failed"] & (cohort["time_unbound"] <= horizon)
        per_system = hit.groupby(cohort["system_id"]).sum().reindex(complete_ids, fill_value=0).astype(int)
        column = f"n_fail_t{horizon:g}"
        count_frame[column] = per_system.to_numpy(dtype=np.int64)
        failed = int(hit.sum())
        horizon_rows.append(
            {
                "horizon": float(horizon),
                "failed_by_horizon": failed,
                "cohort_trajectories": int(len(cohort)),
                "trajectory_fraction": float(failed / len(cohort)),
                "eventual_failures": total_events,
                "eventual_failures_seen_fraction": float(failed / total_events) if total_events else 0.0,
            }
        )
    by_horizon = pd.DataFrame(horizon_rows)
    systems = systems.merge(count_frame, on="system_id", validate="one_to_one")

    # Initial conditions are summarized at the base-system level by the median
    # over all 32 valid variations, avoiding a privileged phase/variation.
    fields, initial_source, time_steps = _load_initial_conditions(
        dataset_dir, sampling_id, labels, group_size
    )
    max_index = int(labels["index"].max())
    if any(values.shape[0] <= max_index for values in fields.values()):
        raise ValueError("Initial-condition arrays do not cover every completion-table index.")
    physical = _physical_features(fields)
    physical.insert(0, "index", np.arange(len(physical), dtype=np.int64))
    cohort_physical = cohort[["index", "system_id"]].merge(physical, on="index", how="left", validate="one_to_one")
    grouped_physical = cohort_physical.groupby("system_id", sort=True).median(numeric_only=True)
    grouped_physical = grouped_physical.reindex(complete_ids)
    for name in ["a_out_over_a_in", "e_in", "e_out", "i_mut_deg", "q_out", "ma_margin"]:
        systems[name] = grouped_physical[name].to_numpy(dtype=np.float64)
    systems["abs_ma_margin"] = systems["ma_margin"].abs()

    bins = pd.DataFrame(
        {
            "stable_fraction_bin": np.arange(6, dtype=np.int64),
            "bin_label": ["p=0", "0<p<0.25", "0.25<=p<0.5", "0.5<=p<0.75", "0.75<=p<1", "p=1"],
        }
    )
    bin_counts = systems["stable_fraction_bin"].value_counts().reindex(range(6), fill_value=0)
    bins["systems"] = bin_counts.to_numpy(dtype=np.int64)
    bins["fraction"] = bins["systems"] / float(len(systems))

    event_times = cohort.loc[cohort["failed"], "time_unbound"].to_numpy(dtype=np.float64)
    all_times = cohort["time_unbound"].to_numpy(dtype=np.float64)
    km = _km_curve(all_times, cohort["failed"].to_numpy(dtype=bool), stable_horizon)
    fail_summary = _quantiles(event_times)

    features = ["ma_margin", "abs_ma_margin", "a_out_over_a_in", "e_in", "e_out", "i_mut_deg", "q_out"]
    corr_rows = []
    for feature in features:
        selected = systems[[feature, "p_stable", "phase_sensitivity"]].dropna()
        corr_rows.append(
            {
                "feature": feature,
                "n_systems": int(len(selected)),
                "spearman_p_stable": float(selected[feature].corr(selected["p_stable"], method="spearman")),
                "spearman_sensitivity": float(selected[feature].corr(selected["phase_sensitivity"], method="spearman")),
                "pearson_p_stable": float(selected[feature].corr(selected["p_stable"], method="pearson")),
                "pearson_sensitivity": float(selected[feature].corr(selected["phase_sensitivity"], method="pearson")),
            }
        )
    correlations = pd.DataFrame(corr_rows)
    ma_distance = _ma_distance_summary(systems)

    invalid = labels.loc[~labels["valid"], LABEL_COLUMNS + ["system_id", "variation_id"]]
    excluded = catalog.loc[~catalog["complete_valid_system"], :].copy()
    excluded.to_csv(output_dir / "excluded_systems.csv", index=False)
    invalid.to_csv(output_dir / "invalid_trajectories.csv", index=False)
    systems.to_csv(output_dir / "system_level_statistics.csv", index=False)
    bins.to_csv(output_dir / "stable_fraction_bins.csv", index=False)
    by_horizon.to_csv(output_dir / "failure_by_time_horizon.csv", index=False)
    count_frame.to_csv(output_dir / "system_fail_counts_by_time_horizon.csv", index=False)
    km.to_csv(output_dir / "survival_curve.csv", index=False)
    correlations.to_csv(output_dir / "coarse_ic_correlations.csv", index=False)
    ma_distance.to_csv(output_dir / "phase_sensitivity_by_ma_distance.csv", index=False)

    _plot_stable_fraction(systems, output_dir)
    _plot_sensitivity(systems, output_dir)
    _plot_failure_times(event_times, output_dir, stable_horizon)
    _plot_failure_cdf_survival(event_times, km, output_dir)
    _plot_fail_count_histograms(count_frame, fail_horizons, output_dir)
    _plot_physical_relations(systems, output_dir)
    _plot_ma_distance_summary(ma_distance, output_dir)

    p_quantiles = _quantiles(systems["p_stable"].to_numpy())
    s_quantiles = _quantiles(systems["phase_sensitivity"].to_numpy())
    fully_unstable = int((systems["p_stable"] == 0.0).sum())
    fully_stable = int((systems["p_stable"] == 1.0).sum())
    phase_sensitive = int(systems["is_phase_sensitive"].sum())
    high_sensitivity = int((systems["phase_sensitivity"] > 0.5).sum())
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "completion_table": str(label_path),
        "initial_conditions_source": initial_source,
        "sampling_id": int(sampling_id),
        "group_size": int(group_size),
        "stable_horizon": float(stable_horizon),
        "time_steps_in_result_shards": time_steps,
        "time_axis": _load_time_axis_metadata(dataset_dir, sampling_id),
        "sampling_relationship": (
            "Sampling 2 and sampling 3 are paired temporal-resolution versions of the same dataset, with corresponding base-system and variation identities. Sampling 3 is the higher-resolution version, while the local sampling3 files contain only a 1024-step prefix. This prefix limits stored trajectory history; the stable/failure labels in this audit come from the completion tables, so differences in those labels cannot be attributed to prefix truncation alone."
            if sampling_id in {2, 3}
            else None
        ),
        "raw_trajectories": int(len(labels)),
        "raw_systems": int(labels["system_id"].nunique()),
        "invalid_trajectories": int((~labels["valid"]).sum()),
        "valid_trajectories_all_systems": int(labels["valid"].sum()),
        "complete_valid_systems": int(len(systems)),
        "cohort_trajectories": int(len(cohort)),
        "valid_rows_excluded_with_incomplete_systems": int(labels["valid"].sum() - len(cohort)),
        "systems_excluded": int(len(excluded)),
        "stable_trajectories": int(cohort["stable"].sum()),
        "unstable_variations": int(cohort["failed"].sum()),
        "stable_trajectory_rate": float(cohort["stable"].mean()),
        "unstable_variation_fraction": float(cohort["failed"].mean()),
        "fully_unstable_systems": fully_unstable,
        "fully_unstable_fraction": fully_unstable / len(systems),
        "fully_stable_systems": fully_stable,
        "fully_stable_fraction": fully_stable / len(systems),
        "phase_sensitive_systems": phase_sensitive,
        "phase_sensitive_fraction": phase_sensitive / len(systems),
        "p_stable_quantiles": p_quantiles,
        "phase_sensitivity_mean": float(systems["phase_sensitivity"].mean()),
        "phase_sensitivity_quantiles": s_quantiles,
        "sensitivity_gt_0_5_systems": high_sensitivity,
        "sensitivity_gt_0_5_fraction": high_sensitivity / len(systems),
        "failure_time_quantiles": fail_summary,
        "requested_fail_horizons": list(fail_horizons),
        "failures_by_time_horizon": horizon_rows,
        "stable_fraction_bins": bins.to_dict(orient="records"),
        "notes": [
            "Primary system statistics use only systems with all group_size valid variations.",
            "Stable means time_unbound >= stable_horizon; observed failures are time_unbound < stable_horizon.",
            "Requested T=32,64,128,256,512 values are absolute time_unbound horizons, not stored-sample prefix lengths.",
            "The MA margin is log(Rp/Rp_crit) using the project's existing Mardling–Aarseth Equation (2).",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    _write_summary_markdown(output_dir / "summary.md", summary, bins, by_horizon, correlations, ma_distance)
    return summary


def _resolve_dataset_dir(value: str | None, sampling_id: int) -> Path:
    if value:
        return Path(value)
    if sampling_id == 2:
        candidates = [ROOT / "dataset_v3", Path("/data/sampling_2")]
    elif sampling_id == 3:
        candidates = [
            ROOT / "dataset_sampling_3_prefix1024",
            Path("/data/dataset_sampling_3_prefix1024_float64"),
            ROOT / "dataset_sampling_3_prefix512",
        ]
    else:
        candidates = [ROOT / f"dataset_sampling_{sampling_id}", Path(f"/data/sampling_{sampling_id}")]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--stable-horizon", type=float, default=DEFAULT_TIME_HORIZON)
    parser.add_argument(
        "--fail-horizons",
        type=_parse_float_list,
        default=DEFAULT_FAIL_HORIZONS,
        help="Absolute time_unbound horizons, e.g. 32,64,128,256,512.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir, args.sampling_id)
    output_dir = args.output_dir or ROOT / "results" / f"system_level_audit_sampling_{args.sampling_id}"
    summary = run_audit(
        dataset_dir=dataset_dir,
        sampling_id=args.sampling_id,
        output_dir=output_dir,
        group_size=args.group_size,
        stable_horizon=args.stable_horizon,
        fail_horizons=args.fail_horizons,
    )
    print(f"Dataset: {summary['dataset_dir']}")
    print(
        f"Systems: {summary['complete_valid_systems']:,} complete valid / "
        f"{summary['raw_systems']:,} raw"
    )
    print(
        f"Stable variations: {summary['stable_trajectories']:,}/"
        f"{summary['cohort_trajectories']:,} ({summary['stable_trajectory_rate']:.2%})"
    )
    print(
        f"Phase-sensitive systems: {summary['phase_sensitive_systems']:,}/"
        f"{summary['complete_valid_systems']:,} ({summary['phase_sensitive_fraction']:.2%})"
    )
    print(f"Wrote audit artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
