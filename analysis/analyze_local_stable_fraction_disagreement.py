#!/usr/bin/env python3
"""Quantify local stable-fraction disagreement in the original orbital15 space.

This analysis works at the base-system level.  It keeps the repository's
orbital15 sin/cos representation for Euclidean nearest-neighbour distances,
standardizes those 15 channels across base systems, and compares the observed
32-variation stable fractions of nearby systems.

The metric
    L_i(k) = mean_j |f_i - f_j|
is called local stable-fraction disagreement here.  It is not a Lyapunov
exponent, a chaos score, or a formal fractal-dimension estimate.

Run from the repository root, for example::

    conda run -n stellar python analysis/analyze_local_stable_fraction_disagreement.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.analyze_divergence import load_completion_table
from tasks.common.traditional_stability import triple_stability
from tasks.task_b.main import ORBITAL15_RESULT_KEYS

# Reuse the previously validated project-specific loading and grouping logic.
from analysis.plot_ma_boundary import (
    _aggregate_stable_fraction,
    _discover_result_paths,
    _initial_data_for_cases,
    _load_result_initial_conditions,
    _load_static_initial_conditions,
    _locate_static_ics,
)


GROUP_SIZE = 32
DEFAULT_SAMPLING_ID = 3
DEFAULT_DATASET_DIR = ROOT / "dataset_sampling_3_prefix1024"
DEFAULT_FIGURE_DIR = ROOT / "figures"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "analysis"
DEFAULT_K_VALUES = (5, 10, 20, 50)
ZERO_VARIANCE_TOLERANCE = 1.0e-12

ORBITAL15_FEATURE_NAMES = (
    "aratios_0",
    "mratios_0",
    "mratios_1",
    "eccs_0",
    "eccs_1",
    "incs_0",
    "incs_1",
    "APs_sin_0",
    "APs_sin_1",
    "APs_cos_0",
    "APs_cos_1",
    "LANs_sin_0",
    "LANs_sin_1",
    "LANs_cos_0",
    "LANs_cos_1",
)


def parse_k_values(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k values must be positive comma-separated integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sampling-id", type=int, default=DEFAULT_SAMPLING_ID)
    parser.add_argument("--ics-path", type=Path, default=None)
    parser.add_argument("--completion-table", type=Path, default=None)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--representative-variation", type=int, default=0)
    parser.add_argument("--k-values", type=parse_k_values, default=DEFAULT_K_VALUES)
    parser.add_argument("--near-boundary", type=float, default=0.1)
    parser.add_argument("--trend-bins", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--png-dpi", type=int, default=400)
    return parser


def load_base_system_data(args: argparse.Namespace) -> tuple[
    Any, np.ndarray, np.ndarray, Path, str
]:
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if not 0 <= args.representative_variation < args.group_size:
        raise ValueError("--representative-variation must lie in [0, group_size)")
    if args.time_horizon <= 0:
        raise ValueError("--time-horizon must be positive")

    completion_path = args.completion_table or (
        args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    )
    if not completion_path.is_file():
        raise FileNotFoundError(f"Completion table does not exist: {completion_path}")
    completion = load_completion_table(completion_path)

    static_path = _locate_static_ics(args.dataset_dir, args.sampling_id, args.ics_path)
    if static_path is not None:
        initial = _load_static_initial_conditions(static_path)
        label_completion = completion
        source_description = (
            f"static IC file {static_path.resolve()}, "
            f"variation_id={args.representative_variation}"
        )
    else:
        result_paths = _discover_result_paths(
            args.dataset_dir, args.sampling_id, args.max_shards
        )
        initial = _load_result_initial_conditions(result_paths, completion)
        available = np.isin(
            completion["index"].to_numpy(dtype=np.int64), initial.indices
        )
        label_completion = completion.loc[available].copy()
        source_description = (
            f"t=0 of {len(result_paths)} result shard(s) under "
            f"{result_paths[0].parent.resolve()}, "
            f"variation_id={args.representative_variation}"
        )

    grouped = _aggregate_stable_fraction(
        label_completion, args.group_size, args.time_horizon
    )
    case_ids = grouped["raw_case_id"].to_numpy(dtype=np.int64)
    stable_fraction = grouped["stable_fraction"].to_numpy(dtype=np.float64)

    if args.max_systems is not None:
        limit = max(0, int(args.max_systems))
        case_ids = case_ids[:limit]
        stable_fraction = stable_fraction[:limit]
    if case_ids.size == 0:
        raise ValueError("No base systems remain after filtering.")

    base_fields = _initial_data_for_cases(
        initial,
        case_ids,
        args.group_size,
        args.representative_variation,
    )
    return initial, case_ids, stable_fraction, completion_path, source_description, base_fields


def filter_valid_base_systems(
    initial: Any,
    case_ids: np.ndarray,
    stable_fraction: np.ndarray,
    base_fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Drop only invalid base rows, while reporting every validity issue."""
    total = case_ids.size
    required_fields = ("aratios", "mratios", "eccs", "incs", "APs", "LANs", "masses")
    matrix = np.column_stack(
        [base_fields[field].reshape(total, -1) for field in required_fields]
    )
    finite = np.isfinite(matrix).all(axis=1)

    e_in = base_fields["eccs"][:, 0]
    e_out = base_fields["eccs"][:, 1]
    valid_ecc = (
        np.isfinite(e_in)
        & np.isfinite(e_out)
        & (e_in >= 0.0)
        & (e_in < 1.0)
        & (e_out >= 0.0)
        & (e_out < 1.0)
    )
    aratio = base_fields["aratios"][:, 0]
    masses = base_fields["masses"]
    valid_geometry = (
        np.isfinite(aratio)
        & (aratio > 0.0)
        & (masses > 0.0).all(axis=1)
    )
    valid_fraction = np.isfinite(stable_fraction) & (stable_fraction >= 0.0) & (
        stable_fraction <= 1.0
    )
    valid = finite & valid_ecc & valid_geometry & valid_fraction

    issues = {
        "nonfinite_rows": int((~finite).sum()),
        "invalid_eccentricity_rows": int((~valid_ecc).sum()),
        "invalid_geometry_rows": int((~valid_geometry).sum()),
        "invalid_stable_fraction_rows": int((~valid_fraction).sum()),
    }
    if not np.all(valid):
        print(
            f"Dropping {int((~valid).sum())} base systems with invalid/non-finite "
            "initial conditions or stable fractions."
        )
        case_ids = case_ids[valid]
        stable_fraction = stable_fraction[valid]
        base_fields = {
            field: values[valid] for field, values in base_fields.items()
        }
    if case_ids.size == 0:
        raise ValueError("No valid base systems remain.")
    return case_ids, stable_fraction, base_fields, issues


def build_orbital15_features(
    base_fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct the exact Task-B orbital15 channel order.

    APs and LANs remain sin/cos pairs.  Raw angular values are never supplied
    to the Euclidean kNN calculation.
    """
    blocks = (
        base_fields["aratios"],
        base_fields["mratios"],
        base_fields["eccs"],
        base_fields["incs"],
        np.sin(base_fields["APs"]),
        np.cos(base_fields["APs"]),
        np.sin(base_fields["LANs"]),
        np.cos(base_fields["LANs"]),
    )
    features = np.concatenate(blocks, axis=1).astype(np.float64, copy=False)
    if features.shape[1] != len(ORBITAL15_FEATURE_NAMES):
        raise ValueError(
            f"Expected 15 orbital15 features, got shape {features.shape}."
        )
    return features, ORBITAL15_FEATURE_NAMES


def standardize_features(
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    zero_mask = (~np.isfinite(stds)) | (stds <= ZERO_VARIANCE_TOLERANCE)
    safe_stds = stds.copy()
    safe_stds[zero_mask] = 1.0
    standardized = (features - means) / safe_stds
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized orbital15 features contain NaN or inf.")
    zero_names = [
        name for name, is_zero in zip(feature_names, zero_mask) if is_zero
    ]
    return standardized, means, safe_stds, zero_names


def compute_mutual_inclination_and_ma_margin(
    base_fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    inclinations = base_fields["incs"]
    longitudes = base_fields["LANs"]
    omega_in = np.arctan2(
        np.sin(longitudes[:, 0]), np.cos(longitudes[:, 0])
    )
    omega_out = np.arctan2(
        np.sin(longitudes[:, 1]), np.cos(longitudes[:, 1])
    )
    cosine_mutual = (
        np.cos(inclinations[:, 0]) * np.cos(inclinations[:, 1])
        + np.sin(inclinations[:, 0])
        * np.sin(inclinations[:, 1])
        * np.cos(omega_in - omega_out)
    )
    clipped = int(np.count_nonzero((cosine_mutual < -1.0) | (cosine_mutual > 1.0)))
    cosine_mutual = np.clip(cosine_mutual, -1.0, 1.0)
    mutual_inclination = np.arccos(cosine_mutual)

    aratio = base_fields["aratios"][:, 0]
    e_out = base_fields["eccs"][:, 1]
    masses = base_fields["masses"]
    criterion = triple_stability(
        a_in=np.ones_like(aratio),
        a_out=1.0 / aratio,
        e_out=e_out,
        mutual_inclination=mutual_inclination,
        m1=masses[:, 0],
        m2=masses[:, 1],
        m3=masses[:, 2],
    )
    margin = np.asarray(criterion["stability_margin"], dtype=np.float64)
    return mutual_inclination, margin, float(clipped)


def compute_knn_metrics(
    standardized_features: np.ndarray,
    stable_fraction: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    n_systems = standardized_features.shape[0]
    max_k = max(k_values)
    if max_k >= n_systems:
        raise ValueError(
            f"Need more than max(k)={max_k} base systems, got {n_systems}."
        )

    model = NearestNeighbors(
        n_neighbors=max_k + 1, metric="euclidean", algorithm="auto"
    )
    model.fit(standardized_features)
    distances, indices = model.kneighbors(standardized_features)

    neighbor_indices = np.empty((n_systems, max_k), dtype=np.int64)
    neighbor_distances = np.empty((n_systems, max_k), dtype=np.float64)
    for row in range(n_systems):
        other = indices[row] != row
        row_indices = indices[row][other]
        row_distances = distances[row][other]
        if row_indices.size < max_k:
            raise RuntimeError(
                f"Could not find {max_k} other neighbours for base system {row}."
            )
        neighbor_indices[row] = row_indices[:max_k]
        neighbor_distances[row] = row_distances[:max_k]

    fraction_differences = np.abs(
        stable_fraction[:, None] - stable_fraction[neighbor_indices]
    )
    result: dict[str, Any] = {
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
        "fraction_differences": fraction_differences,
        "nearest_neighbor_index": neighbor_indices[:, 0],
        "nearest_neighbor_distance": neighbor_distances[:, 0],
        "nearest_neighbor_stable_fraction": stable_fraction[neighbor_indices[:, 0]],
        "nearest_neighbor_abs_fraction_diff": fraction_differences[:, 0],
    }
    for k in k_values:
        result[f"local_disagreement_k{k}"] = np.mean(
            fraction_differences[:, :k], axis=1
        )
        result[f"mean_neighbor_distance_k{k}"] = np.mean(
            neighbor_distances[:, :k], axis=1
        )
        result[f"max_neighbor_abs_fraction_diff_k{k}"] = np.max(
            fraction_differences[:, :k], axis=1
        )
    return result


def spearman_value(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    result = spearmanr(x, y)
    statistic = getattr(result, "statistic", result[0])
    return float(statistic)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(n_samples, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def binned_trend(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> pd.DataFrame:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        edges = np.linspace(x_min - 0.5, x_max + 0.5, 2)
    else:
        edges = np.linspace(x_min, x_max, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for bin_id, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_id == len(edges) - 2:
            in_bin = (x >= left) & (x <= right)
        else:
            in_bin = (x >= left) & (x < right)
        values = y[in_bin]
        if values.size == 0:
            continue
        lower, upper = bootstrap_mean_ci(values, rng, bootstrap_samples)
        rows.append(
            {
                "bin": bin_id,
                "x_center": float(np.mean(x[in_bin])),
                "x_left": float(left),
                "x_right": float(right),
                "mean_local_disagreement": float(np.mean(values)),
                "median_local_disagreement": float(np.median(values)),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "count": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    png_dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def plot_main_figure(
    delta_ma: np.ndarray,
    local_disagreement: np.ndarray,
    trend: pd.DataFrame,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    set_publication_style()
    figure, axis = plt.subplots(figsize=(4.8, 3.7), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(True, color="0.85", linewidth=0.55, alpha=0.65)
    axis.scatter(
        delta_ma,
        local_disagreement,
        s=10.0,
        color="#377eb8",
        alpha=0.38,
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
        label="Base systems",
    )
    if not trend.empty:
        finite = np.isfinite(
            trend[["x_center", "mean_local_disagreement", "ci95_lower", "ci95_upper"]]
            .to_numpy(dtype=float)
        ).all(axis=1)
        trend_plot = trend.loc[finite]
        if not trend_plot.empty:
            means = trend_plot["mean_local_disagreement"].to_numpy()
            lower = trend_plot["ci95_lower"].to_numpy()
            upper = trend_plot["ci95_upper"].to_numpy()
            axis.errorbar(
                trend_plot["x_center"],
                means,
                yerr=np.vstack((means - lower, upper - means)),
                fmt="o-",
                color="#d95f02",
                markersize=3.2,
                linewidth=1.15,
                elinewidth=0.8,
                capsize=1.8,
                label="Binned mean ± 95% bootstrap CI",
                zorder=3,
            )
    axis.axvline(0.0, color="0.18", linestyle="--", linewidth=1.0, zorder=1)
    axis.set_xlabel(r"Mardling–Aarseth margin, $\delta_{\rm MA}$")
    axis.set_ylabel(r"Local stable-fraction disagreement, $L_i$")
    axis.set_ylim(bottom=0.0)
    x_min = min(float(np.min(delta_ma)), 0.0)
    x_max = max(float(np.max(delta_ma)), 0.0)
    x_span = max(x_max - x_min, 1.0e-6)
    axis.set_xlim(x_min - 0.04 * x_span, x_max + 0.04 * x_span)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.legend(frameon=False, fontsize=7.2, loc="best")
    return save_figure(
        figure, output_dir, "local_disagreement_vs_ma_margin", png_dpi
    )


def plot_histogram(
    local_disagreement: np.ndarray,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    set_publication_style()
    figure, axis = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", color="0.85", linewidth=0.55, alpha=0.65)
    maximum = float(np.max(local_disagreement))
    if np.isclose(maximum, 0.0):
        bins = np.linspace(0.0, 1.0, 21)
    else:
        bins = np.linspace(0.0, min(1.0, maximum * 1.05), 31)
    axis.hist(
        local_disagreement,
        bins=bins,
        color="#377eb8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    median = float(np.median(local_disagreement))
    p90 = float(np.quantile(local_disagreement, 0.9))
    axis.axvline(median, color="#222222", linestyle="-", linewidth=0.9, label=f"Median = {median:.3f}")
    axis.axvline(p90, color="#d95f02", linestyle="--", linewidth=0.9, label=f"90th percentile = {p90:.3f}")
    axis.set_xlabel(r"Local stable-fraction disagreement, $L_i$")
    axis.set_ylabel("Number of base systems")
    axis.set_xlim(left=0.0)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.legend(frameon=False, fontsize=7.3, loc="best")
    return save_figure(
        figure, output_dir, "local_disagreement_histogram", png_dpi
    )


def plot_density_diagnostic(
    mean_distance: np.ndarray,
    local_disagreement: np.ndarray,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    set_publication_style()
    figure, axis = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(True, color="0.85", linewidth=0.55, alpha=0.65)
    axis.scatter(
        mean_distance,
        local_disagreement,
        s=10.0,
        color="#4daf4a",
        alpha=0.38,
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
    )
    axis.set_xlabel(r"Mean $k=20$ neighbour distance (standardized orbital15)")
    axis.set_ylabel(r"Local stable-fraction disagreement, $L_i$")
    axis.set_ylim(bottom=0.0)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    return save_figure(
        figure, output_dir, "local_disagreement_vs_neighbor_distance", png_dpi
    )


def cross_check_existing_labels(
    root: Path,
    grouped: pd.DataFrame,
    sampling_id: int,
) -> None:
    """Cross-check an existing derived raw-label table when available."""
    candidate = root / "results" / "task_b_sampling3_prefix512" / "labels_filtered.csv"
    if sampling_id != 3 or not candidate.is_file():
        print("  stable_fraction cross-check: no existing aggregate/derived label file used")
        return
    existing = pd.read_csv(candidate)
    required = {"raw_case_id", "time_unbound", "diagnosis"}
    if not required.issubset(existing.columns):
        print(f"  stable_fraction cross-check: skipped malformed {candidate}")
        return
    existing = existing[
        (existing["time_unbound"] >= 0.0) & (existing["diagnosis"] == 0)
    ].copy()
    existing["stable"] = existing["time_unbound"] >= 10000.0
    existing_grouped = existing.groupby("raw_case_id")["stable"].mean()
    reference = grouped.set_index("raw_case_id")["stable_fraction"]
    common = reference.index.intersection(existing_grouped.index)
    if common.empty:
        print("  stable_fraction cross-check: no common systems")
        return
    difference = np.abs(
        reference.loc[common].to_numpy() - existing_grouped.loc[common].to_numpy()
    )
    print(
        "  stable_fraction cross-check: "
        f"{len(common)} common systems, first systems checked={common[:5].tolist()}, "
        f"max absolute difference={np.max(difference):.6g}"
    )


def fraction_text(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.4%})"


def print_summary(
    *,
    case_ids: np.ndarray,
    stable_fraction: np.ndarray,
    delta_ma: np.ndarray,
    mutual_inclination: np.ndarray,
    standardized_features: np.ndarray,
    feature_names: tuple[str, ...],
    zero_variance_features: list[str],
    issues: dict[str, int],
    source_description: str,
    completion_path: Path,
    args: argparse.Namespace,
    knn: dict[str, Any],
    initial: Any,
    q_out_difference: float,
    clipped_cosines: float,
) -> None:
    total = case_ids.size
    abs_delta = np.abs(delta_ma)
    near = abs_delta < args.near_boundary
    far = ~near
    print("\nLocal stable-fraction disagreement summary")
    print(f"  number of base systems: {total}")
    print(f"  representative IC convention: {source_description}")
    print(f"  stable label convention: stable=1 when time_unbound >= {args.time_horizon:g} and diagnosis == 0; unstable=0")
    print(f"  completion table: {completion_path.resolve()}")
    print(f"  exact orbital15 raw order: {', '.join(ORBITAL15_RESULT_KEYS)}")
    print(f"  exact standardized feature order: {', '.join(feature_names)}")
    print(f"  k values evaluated: {args.k_values}")
    print(f"  MA implementation: tasks.common.traditional_stability.triple_stability; natural-log margin")
    print(f"  standardized feature matrix shape: {standardized_features.shape}")
    print(f"  zero/effectively-zero variance features ({len(zero_variance_features)}): {zero_variance_features or 'none'}")
    print(
        "  data issues: "
        f"nonfinite={issues['nonfinite_rows']}, invalid_ecc={issues['invalid_eccentricity_rows']}, "
        f"invalid_geometry={issues['invalid_geometry_rows']}, invalid_fraction={issues['invalid_stable_fraction_rows']}"
    )
    print(
        f"  stored individual-inclination range [rad]: "
        f"{np.nanmin(initial.fields['incs']):.6g} .. {np.nanmax(initial.fields['incs']):.6g}"
    )
    print(f"  mutual inclination [deg] min / median / max: {np.degrees(mutual_inclination).min():.6g} / {np.median(np.degrees(mutual_inclination)):.6g} / {np.degrees(mutual_inclination).max():.6g}")
    print(f"  maximum |q_out(masses) - mratios[1]|: {q_out_difference:.6g}")
    print(f"  mutual-inclination cosine values clipped: {int(clipped_cosines)}")

    for k in args.k_values:
        local = knn[f"local_disagreement_k{k}"]
        rho_margin = spearman_value(local, abs_delta)
        near_mean = float(np.mean(local[near])) if np.any(near) else float("nan")
        far_mean = float(np.mean(local[far])) if np.any(far) else float("nan")
        print(
            f"  k={k}: mean={np.mean(local):.6g}, median={np.median(local):.6g}, "
            f"p90={np.quantile(local, 0.9):.6g}, "
            f"Spearman(L, |delta_MA|)={rho_margin:.6g}, "
            f"near |delta_MA|<{args.near_boundary:g}: {fraction_text(int(near.sum()), total)}, "
            f"mean_near={near_mean:.6g}, mean_far={far_mean:.6g}"
        )

    local20 = knn["local_disagreement_k20"]
    distance20 = knn["mean_neighbor_distance_k20"]
    rho_density = spearman_value(local20, distance20)
    print(
        f"  Spearman(L_k20, mean_neighbor_distance_k20): {rho_density:.6g}"
    )
    nearest_diff = knn["nearest_neighbor_abs_fraction_diff"]
    nearest_dist = knn["nearest_neighbor_distance"]
    print(
        f"  nearest-neighbour |stable_fraction difference| > 0.25: "
        f"{fraction_text(int((nearest_diff > 0.25).sum()), total)}"
    )
    print(
        f"  nearest-neighbour |stable_fraction difference| > 0.5: "
        f"{fraction_text(int((nearest_diff > 0.5).sum()), total)}"
    )
    print(
        f"  nearest-neighbour distance min / median / p90 / max: "
        f"{np.min(nearest_dist):.6g} / {np.median(nearest_dist):.6g} / "
        f"{np.quantile(nearest_dist, 0.9):.6g} / {np.max(nearest_dist):.6g}"
    )


def save_per_system_table(
    output_dir: Path,
    case_ids: np.ndarray,
    stable_fraction: np.ndarray,
    delta_ma: np.ndarray,
    mutual_inclination: np.ndarray,
    knn: dict[str, Any],
    k_values: tuple[int, ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, np.ndarray] = {
        "base_system_id": case_ids,
        "stable_fraction": stable_fraction,
        "delta_ma": delta_ma,
        "imut_deg": np.degrees(mutual_inclination),
    }
    for k in k_values:
        table[f"local_disagreement_k{k}"] = knn[f"local_disagreement_k{k}"]
        table[f"mean_neighbor_distance_k{k}"] = knn[f"mean_neighbor_distance_k{k}"]
        table[f"max_neighbor_abs_fraction_diff_k{k}"] = knn[
            f"max_neighbor_abs_fraction_diff_k{k}"
        ]
    table.update(
        {
            "nearest_neighbor_distance": knn["nearest_neighbor_distance"],
            "nearest_neighbor_stable_fraction": knn[
                "nearest_neighbor_stable_fraction"
            ],
            "nearest_neighbor_abs_fraction_diff": knn[
                "nearest_neighbor_abs_fraction_diff"
            ],
        }
    )
    path = output_dir / "local_stable_fraction_disagreement.csv"
    pd.DataFrame(table).to_csv(path, index=False, float_format="%.10g")
    return path


def main() -> None:
    args = build_parser().parse_args()
    if args.png_dpi < 300:
        raise ValueError("--png-dpi must be at least 300")
    if args.trend_bins < 2:
        raise ValueError("--trend-bins must be at least 2")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    if 20 not in args.k_values:
        raise ValueError("--k-values must include 20 because the main and density diagnostics use k=20")

    initial, case_ids, stable_fraction, completion_path, source_description, base_fields = load_base_system_data(args)
    grouped = _aggregate_stable_fraction(
        load_completion_table(completion_path),
        args.group_size,
        args.time_horizon,
    )
    cross_check_existing_labels(ROOT, grouped, args.sampling_id)

    case_ids, stable_fraction, base_fields, issues = filter_valid_base_systems(
        initial, case_ids, stable_fraction, base_fields
    )

    inclinations_all = initial.fields["incs"]
    finite_inclinations = inclinations_all[np.isfinite(inclinations_all)]
    if (
        finite_inclinations.size == 0
        or np.min(finite_inclinations) < -1.0e-10
        or np.max(finite_inclinations) > np.pi + 1.0e-10
    ):
        raise ValueError(
            "Stored inclinations are not consistent with the repository's radian convention 0 <= i <= pi."
        )

    features, feature_names = build_orbital15_features(base_fields)
    standardized, feature_means, feature_stds, zero_variance = standardize_features(
        features, feature_names
    )
    del feature_means, feature_stds

    mutual_inclination, delta_ma, clipped_cosines = (
        compute_mutual_inclination_and_ma_margin(base_fields)
    )
    q_out = base_fields["masses"][:, 2] / (
        base_fields["masses"][:, 0] + base_fields["masses"][:, 1]
    )
    q_out_difference = float(
        np.max(np.abs(q_out - base_fields["mratios"][:, 1]))
    )

    knn = compute_knn_metrics(standardized, stable_fraction, args.k_values)
    rng = np.random.default_rng(args.seed)
    trend = binned_trend(
        delta_ma,
        knn["local_disagreement_k20"],
        args.trend_bins,
        rng,
        args.bootstrap_samples,
    )
    trend_path = args.analysis_dir / "local_disagreement_ma_margin_trend.csv"
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    trend.to_csv(trend_path, index=False, float_format="%.10g")

    main_paths = plot_main_figure(
        delta_ma,
        knn["local_disagreement_k20"],
        trend,
        args.figure_dir,
        args.png_dpi,
    )
    histogram_paths = plot_histogram(
        knn["local_disagreement_k20"],
        args.figure_dir,
        args.png_dpi,
    )
    density_paths = plot_density_diagnostic(
        knn["mean_neighbor_distance_k20"],
        knn["local_disagreement_k20"],
        args.figure_dir,
        args.png_dpi,
    )
    table_path = save_per_system_table(
        args.analysis_dir,
        case_ids,
        stable_fraction,
        delta_ma,
        mutual_inclination,
        knn,
        args.k_values,
    )

    print_summary(
        case_ids=case_ids,
        stable_fraction=stable_fraction,
        delta_ma=delta_ma,
        mutual_inclination=mutual_inclination,
        standardized_features=standardized,
        feature_names=feature_names,
        zero_variance_features=zero_variance,
        issues=issues,
        source_description=source_description,
        completion_path=completion_path,
        args=args,
        knn=knn,
        initial=initial,
        q_out_difference=q_out_difference,
        clipped_cosines=clipped_cosines,
    )
    print(f"  created: {main_paths[0].resolve()}")
    print(f"  created: {main_paths[1].resolve()}")
    print(f"  created: {histogram_paths[0].resolve()}")
    print(f"  created: {histogram_paths[1].resolve()}")
    print(f"  created: {density_paths[0].resolve()}")
    print(f"  created: {density_paths[1].resolve()}")
    print(f"  created: {trend_path.resolve()}")
    print(f"  created: {table_path.resolve()}")


if __name__ == "__main__":
    main()
