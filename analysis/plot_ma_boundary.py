#!/usr/bin/env python3
"""Plot the empirical stability boundary in initial-condition space.

The figure contains one point per complete 32-variation base system.  The
point coordinates are the Mardling--Aarseth margin and the mutual inclination
of a representative initial condition; its colour is the observed fraction
of the 32 trajectories that survive to ``time_horizon``.

The repository has two relevant input layouts:

* ``ICs_sampling_*.hdf5`` stores static initial conditions directly.
* the high-resolution sampling-3 data stores the same orbital quantities at
  ``*_complete[:, 0, ...]`` in each result shard, but has no separate IC file.

When no separate IC file is present, this script uses the first time step of
variation 0.  This is the project convention used by Task A for the static
initial-condition input, and Task D likewise treats the first 16 variations
as the nominal set.  The script deliberately does not average angular
variables across the perturbation ensemble.

Run from the repository root, for example::

    conda run -n stellar python analysis/plot_ma_boundary.py

For the older sampling-2 data, pass ``--dataset-dir dataset_v3
--sampling-id 2`` when that external data mount is available.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.analyze_divergence import load_completion_table, select_valid_rows
from tasks.common.traditional_stability import triple_stability
from tasks.task_b.main import ORBITAL15_RESULT_KEYS


GROUP_SIZE = 32
DEFAULT_SAMPLING_ID = 3
DEFAULT_DATASET_DIR = ROOT / "dataset_sampling_3_prefix1024"
DEFAULT_OUTPUT_DIR = ROOT / "figures"
RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")

# Task B defines orbital15 from these result keys in this order.  Its angle
# transform is concatenate([sin(values), cos(values)]), so the exact 15-column
# transformed order is recorded explicitly here for the diagnostic output.
ORBITAL15_FEATURE_ORDER = (
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
STATIC_ORBITAL15_KEYS = tuple(key.removesuffix("_complete") for key in ORBITAL15_RESULT_KEYS)
PLOT_FIELDS = (*STATIC_ORBITAL15_KEYS, "masses")
FIELD_DIMS = {
    "aratios": 1,
    "mratios": 2,
    "eccs": 2,
    "incs": 2,
    "APs": 2,
    "LANs": 2,
    "masses": 3,
}


@dataclass(frozen=True)
class InitialConditions:
    """Initial-condition arrays keyed by global trajectory index."""

    indices: np.ndarray
    fields: dict[str, np.ndarray]
    source_description: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sampling-id", type=int, default=DEFAULT_SAMPLING_ID)
    parser.add_argument(
        "--ics-path",
        type=Path,
        default=None,
        help="Optional static IC HDF5 path; otherwise ICs_sampling_<id>.hdf5 is detected automatically.",
    )
    parser.add_argument(
        "--completion-table",
        type=Path,
        default=None,
        help="Completion table; defaults to completed_systems_sampling_<id>.txt under --dataset-dir.",
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument(
        "--representative-variation",
        type=int,
        default=0,
        help="Variation used for the base-system coordinates. The project reference is variation 0.",
    )
    parser.add_argument("--max-shards", type=int, default=None, help="Optional debugging limit on result shards.")
    parser.add_argument("--max-systems", type=int, default=None, help="Optional debugging limit on plotted systems.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png-dpi", type=int, default=400)
    return parser


def _result_batch_id(path: Path) -> int:
    match = RESULT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Could not parse sampling result filename: {path}")
    return int(match.group(1))


def _discover_result_paths(dataset_dir: Path, sampling_id: int, max_shards: int | None) -> list[Path]:
    paths = sorted(
        dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"),
        key=_result_batch_id,
    )
    if max_shards is not None:
        paths = paths[: max(0, max_shards)]
    if not paths:
        raise FileNotFoundError(
            f"No result shards matching results_batch_*_sampling_{sampling_id}.hdf5 under {dataset_dir}"
        )
    return paths


def _normalise_field_array(raw: np.ndarray, field: str, path: Path) -> np.ndarray:
    """Flatten the component axis while preserving one row per trajectory."""
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != FIELD_DIMS[field]:
        raise ValueError(
            f"Expected {field!r} to have N x {FIELD_DIMS[field]} initial components in {path}; "
            f"got shape {array.shape}."
        )
    return array


def _read_initial_block(handle: h5py.File, path: Path, suffix: str, result_layout: bool) -> dict[str, np.ndarray]:
    """Read and validate the raw orbital fields needed by this analysis."""
    fields: dict[str, np.ndarray] = {}
    for field in PLOT_FIELDS:
        key = f"{field}{suffix}"
        if key not in handle:
            raise KeyError(f"Missing required field {key!r} in {path}")
        dataset = handle[key]
        if result_layout:
            if dataset.ndim != 3:
                raise ValueError(f"Expected result field {key!r} to be [N, T, d], got {dataset.shape}")
            raw = dataset[:, 0, ...]
        else:
            raw = dataset[:]
        fields[field] = _normalise_field_array(raw, field, path)

    # APs is not needed by the two plotted coordinates, but it is part of the
    # verified orbital15 schema and must be present with two angular channels.
    angle_key = f"APs{suffix}"
    if angle_key not in handle:
        raise KeyError(f"Missing orbital15 schema field {angle_key!r} in {path}")
    ap_dataset = handle[angle_key]
    ap_raw = ap_dataset[:, 0, ...] if result_layout else ap_dataset[:]
    _normalise_field_array(ap_raw, "APs", path)
    return fields


def _load_static_initial_conditions(path: Path) -> InitialConditions:
    with h5py.File(path, "r") as handle:
        fields = _read_initial_block(handle, path, suffix="", result_layout=False)
    row_count = next(iter(fields.values())).shape[0]
    indices = np.arange(row_count, dtype=np.int64)
    return InitialConditions(
        indices=indices,
        fields=fields,
        source_description=f"static initial conditions: {path.resolve()}",
    )


def _load_result_initial_conditions(
    paths: list[Path],
    completion: pd.DataFrame,
) -> InitialConditions:
    """Read t=0 fields and align each result row to the global trajectory index."""
    index_blocks: list[np.ndarray] = []
    field_blocks: dict[str, list[np.ndarray]] = {field: [] for field in PLOT_FIELDS}

    for path in paths:
        batch_id = _result_batch_id(path)
        batch_labels = completion.loc[completion["batch"] == batch_id].sort_values("index")
        with h5py.File(path, "r") as handle:
            if "times_complete" not in handle:
                raise KeyError(f"Missing 'times_complete' in {path}")
            row_count = int(handle["times_complete"].shape[0])
            if len(batch_labels) != row_count:
                raise ValueError(
                    f"Completion/result row mismatch for batch {batch_id}: "
                    f"labels={len(batch_labels)}, result rows={row_count}."
                )
            fields = _read_initial_block(
                handle,
                path,
                suffix="_complete",
                result_layout=True,
            )
        index_blocks.append(batch_labels["index"].to_numpy(dtype=np.int64))
        for field, values in fields.items():
            field_blocks[field].append(values)

    indices = np.concatenate(index_blocks)
    if np.unique(indices).size != indices.size:
        raise ValueError("Result shards contain duplicate global trajectory indices.")
    order = np.argsort(indices)
    sorted_indices = indices[order]
    fields = {
        field: np.concatenate(values, axis=0)[order]
        for field, values in field_blocks.items()
    }
    return InitialConditions(
        indices=sorted_indices,
        fields=fields,
        source_description=(
            "t=0 initial conditions from result shards: "
            + ", ".join(str(path.resolve()) for path in paths)
        ),
    )


def _locate_static_ics(dataset_dir: Path, sampling_id: int, ics_path: Path | None) -> Path | None:
    if ics_path is not None:
        if not ics_path.is_file():
            raise FileNotFoundError(f"Requested static IC file does not exist: {ics_path}")
        return ics_path
    candidate = dataset_dir / f"ICs_sampling_{sampling_id}.hdf5"
    return candidate if candidate.is_file() else None


def _aggregate_stable_fraction(
    completion: pd.DataFrame,
    group_size: int,
    time_horizon: float,
) -> pd.DataFrame:
    """Reuse the repository validity convention and aggregate valid labels."""
    valid = select_valid_rows(completion).copy()
    valid["raw_case_id"] = valid["index"] // group_size
    valid["variation_id"] = valid["index"] % group_size
    valid["survived_horizon"] = valid["time_unbound"] >= float(time_horizon)

    complete_ids: list[int] = []
    for case_id, frame in valid.groupby("raw_case_id", sort=True):
        variation_ids = np.sort(frame["variation_id"].to_numpy(dtype=np.int64))
        if variation_ids.size == group_size and np.array_equal(
            variation_ids, np.arange(group_size, dtype=np.int64)
        ):
            complete_ids.append(int(case_id))
    if not complete_ids:
        raise ValueError("No complete valid 32-variation systems remain after label filtering.")

    valid = valid[valid["raw_case_id"].isin(complete_ids)].copy()
    grouped = (
        valid.groupby("raw_case_id", sort=True)
        .agg(
            stable_count=("survived_horizon", "sum"),
            stable_fraction=("survived_horizon", "mean"),
            valid_variations=("survived_horizon", "size"),
        )
        .reset_index()
    )
    if not np.all(grouped["valid_variations"].to_numpy(dtype=np.int64) == group_size):
        raise ValueError("Stable-fraction aggregation produced an incomplete system.")
    return grouped


def _lookup_positions(indices: np.ndarray, requested: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    positions = np.searchsorted(indices, requested)
    in_bounds = positions < indices.size
    if not np.all(in_bounds) or not np.array_equal(indices[positions[in_bounds]], requested[in_bounds]):
        missing = requested[~in_bounds | (indices[np.minimum(positions, max(indices.size - 1, 0))] != requested)]
        raise ValueError(f"Initial-condition source is missing requested trajectory indices: {missing[:10].tolist()}")
    return positions


def _initial_data_for_cases(
    initial: InitialConditions,
    case_ids: np.ndarray,
    group_size: int,
    representative_variation: int,
) -> dict[str, np.ndarray]:
    requested = case_ids.astype(np.int64) * group_size + int(representative_variation)
    positions = _lookup_positions(initial.indices, requested)
    return {field: values[positions] for field, values in initial.fields.items()}


def _print_reference_spreads(
    initial: InitialConditions,
    case_ids: np.ndarray,
    group_size: int,
) -> None:
    """Show that the representative is not hiding large IC perturbations."""
    requested = case_ids.astype(np.int64)[:, None] * group_size + np.arange(group_size, dtype=np.int64)[None, :]
    positions = _lookup_positions(initial.indices, requested.reshape(-1))
    print("Within-system initial-condition spread across plotted systems (max component range):")
    for field in ("aratios", "eccs", "incs", "LANs"):
        values = initial.fields[field][positions].reshape(len(case_ids), group_size, -1)
        spread = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        print(f"  {field}: {np.nanmax(spread):.6g}")


def _fraction_text(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.4%})"


def _print_summary(
    case_ids: np.ndarray,
    mutual_inclination: np.ndarray,
    margin: np.ndarray,
    stable_fraction: np.ndarray,
    base_fields: dict[str, np.ndarray],
    all_inclinations: np.ndarray,
    source_description: str,
    completion_path: Path,
    time_horizon: float,
) -> None:
    total = int(case_ids.size)
    zero = np.isclose(stable_fraction, 0.0, atol=1e-12)
    one = np.isclose(stable_fraction, 1.0, atol=1e-12)
    mixed = (stable_fraction > 0.0) & (stable_fraction < 1.0)
    side_tolerance = 1e-12
    negative = margin < -side_tolerance
    positive = margin > side_tolerance
    boundary = ~(negative | positive)

    finite_matrix = np.column_stack([values.reshape(total, -1) for values in base_fields.values()])
    nonfinite_rows = int((~np.isfinite(finite_matrix).all(axis=1)).sum())
    e_in = base_fields["eccs"][:, 0]
    e_out = base_fields["eccs"][:, 1]
    invalid_ecc = (~np.isfinite(e_in)) | (~np.isfinite(e_out)) | (e_in < 0.0) | (e_in >= 1.0) | (e_out < 0.0) | (e_out >= 1.0)

    print("\nMA-boundary plot summary")
    print(f"  initial-condition source: {source_description}")
    print(f"  completion table: {completion_path.resolve()}")
    print(f"  survival horizon: {time_horizon:g}")
    print("  orbital15 raw field order: " + ", ".join(ORBITAL15_RESULT_KEYS))
    print("  orbital15 transformed order: " + ", ".join(ORBITAL15_FEATURE_ORDER))
    print("  plotted coordinates use: aratios[0], eccs[1], incs[0:2], LANs[0:2], masses[0:3]")
    print("  stable fraction source: recomputed from valid completion-table trajectory labels; no aggregate stable_fraction file was present")
    print(f"  number of base systems plotted: {total}")
    print(f"  i_mut [deg] min / median / max: {np.min(mutual_inclination):.6g} / {np.median(mutual_inclination):.6g} / {np.max(mutual_inclination):.6g}")
    print(f"  delta_MA min / median / max: {np.min(margin):.6g} / {np.median(margin):.6g} / {np.max(margin):.6g}")
    print(f"  mean stable fraction: {np.mean(stable_fraction):.6g}")
    print(f"  stable_fraction == 0: {_fraction_text(int(zero.sum()), total)}")
    print(f"  0 < stable_fraction < 1: {_fraction_text(int(mixed.sum()), total)}")
    print(f"  stable_fraction == 1: {_fraction_text(int(one.sum()), total)}")
    print(f"  delta_MA < 0: {_fraction_text(int(negative.sum()), total)}")
    print(f"  delta_MA > 0: {_fraction_text(int(positive.sum()), total)}")
    print(f"  delta_MA == 0 (within 1e-12): {_fraction_text(int(boundary.sum()), total)}")
    print(f"  non-finite base IC rows: {nonfinite_rows}/{total}")
    print(f"  invalid base eccentricity rows: {int(invalid_ecc.sum())}/{total}")
    print(f"  stored individual-inclination range [rad]: {np.nanmin(all_inclinations):.6g} .. {np.nanmax(all_inclinations):.6g} (radian convention confirmed by 0 <= i <= pi)")


def _plot(
    margin: np.ndarray,
    mutual_inclination_degrees: np.ndarray,
    stable_fraction: np.ndarray,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "ma_margin_vs_mutual_inclination.pdf"
    png_path = output_dir / "ma_margin_vs_mutual_inclination.png"

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

    figure, axis = plt.subplots(figsize=(4.8, 3.7), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(True, color="0.85", linewidth=0.55, alpha=0.65)
    points = axis.scatter(
        margin,
        mutual_inclination_degrees,
        c=stable_fraction,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=10.0,
        alpha=0.78,
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    axis.axvline(0.0, color="0.18", linestyle="--", linewidth=1.0, zorder=1)
    colorbar = figure.colorbar(points, ax=axis, pad=0.025, fraction=0.05)
    colorbar.set_label("Stable fraction")
    colorbar.ax.tick_params(width=0.7, length=3)

    axis.set_xlabel("Mardling–Aarseth margin, $\\delta_{\\rm MA}$")
    axis.set_ylabel("Mutual inclination, $i_{\\rm mut}$ (deg)")
    axis.set_ylim(0.0, 180.0)
    x_min = min(float(np.min(margin)), 0.0)
    x_max = max(float(np.max(margin)), 0.0)
    x_span = max(x_max - x_min, 1e-6)
    axis.set_xlim(x_min - 0.04 * x_span, x_max + 0.04 * x_span)
    axis.tick_params(direction="out", length=3.5, width=0.8)

    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    args = build_parser().parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if not 0 <= args.representative_variation < args.group_size:
        raise ValueError("--representative-variation must lie in [0, group_size)")
    if args.time_horizon <= 0:
        raise ValueError("--time-horizon must be positive")
    if args.png_dpi < 300:
        raise ValueError("--png-dpi must be at least 300 for the publication PNG")

    dataset_dir = args.dataset_dir
    completion_path = args.completion_table or dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Completion table does not exist: {completion_path}")
    completion = load_completion_table(completion_path)

    static_path = _locate_static_ics(dataset_dir, args.sampling_id, args.ics_path)
    if static_path is not None:
        initial = _load_static_initial_conditions(static_path)
        label_completion = completion
    else:
        result_paths = _discover_result_paths(dataset_dir, args.sampling_id, args.max_shards)
        initial = _load_result_initial_conditions(result_paths, completion)
        available = np.isin(completion["index"].to_numpy(dtype=np.int64), initial.indices)
        label_completion = completion.loc[available].copy()
        if args.max_shards is not None:
            print(f"Using {len(result_paths)} result shard(s); restricting labels to their available trajectories.")

    grouped = _aggregate_stable_fraction(label_completion, args.group_size, args.time_horizon)
    case_ids = grouped["raw_case_id"].to_numpy(dtype=np.int64)
    stable_fraction = grouped["stable_fraction"].to_numpy(dtype=np.float64)
    if args.max_systems is not None:
        case_ids = case_ids[: max(0, args.max_systems)]
        stable_fraction = stable_fraction[: case_ids.size]
    if case_ids.size == 0:
        raise ValueError("No systems remain after --max-systems filtering.")

    _lookup_positions(initial.indices, case_ids * args.group_size + args.representative_variation)
    base_fields = _initial_data_for_cases(
        initial,
        case_ids,
        args.group_size,
        args.representative_variation,
    )
    _print_reference_spreads(initial, case_ids, args.group_size)

    # The existing repository criterion explicitly takes inclinations in
    # radians.  Confirm that convention from the stored initial fields before
    # calculating the mutual inclination.
    all_inclinations = initial.fields["incs"]
    finite_inclinations = all_inclinations[np.isfinite(all_inclinations)]
    if finite_inclinations.size == 0 or np.nanmin(finite_inclinations) < -1e-10 or np.nanmax(finite_inclinations) > np.pi + 1e-10:
        raise ValueError(
            "Stored inclinations are not consistent with the repository's radian convention 0 <= i <= pi."
        )

    inclinations = base_fields["incs"]
    longitudes = base_fields["LANs"]
    omega_in = np.arctan2(np.sin(longitudes[:, 0]), np.cos(longitudes[:, 0]))
    omega_out = np.arctan2(np.sin(longitudes[:, 1]), np.cos(longitudes[:, 1]))
    cosine_mutual = (
        np.cos(inclinations[:, 0]) * np.cos(inclinations[:, 1])
        + np.sin(inclinations[:, 0])
        * np.sin(inclinations[:, 1])
        * np.cos(omega_in - omega_out)
    )
    cosine_mutual = np.clip(cosine_mutual, -1.0, 1.0)
    mutual_inclination = np.arccos(cosine_mutual)

    e_in = base_fields["eccs"][:, 0]
    e_out = base_fields["eccs"][:, 1]
    aratio = base_fields["aratios"][:, 0]
    masses = base_fields["masses"]
    mratios = base_fields["mratios"]
    q_out_from_masses = masses[:, 2] / (masses[:, 0] + masses[:, 1])
    q_ratio_difference = np.max(np.abs(q_out_from_masses - mratios[:, 1]))

    required_base = np.column_stack(
        [base_fields[field].reshape(case_ids.size, -1) for field in ("aratios", "mratios", "eccs", "incs", "LANs", "masses")]
    )
    finite_base = np.isfinite(required_base).all(axis=1)
    valid_ecc = (
        np.isfinite(e_in)
        & np.isfinite(e_out)
        & (e_in >= 0.0)
        & (e_in < 1.0)
        & (e_out >= 0.0)
        & (e_out < 1.0)
    )
    valid_geometry = (
        np.isfinite(aratio)
        & (aratio > 0.0)
        & np.isfinite(masses).all(axis=1)
        & (masses > 0.0).all(axis=1)
    )
    valid = finite_base & valid_ecc & valid_geometry
    if not np.all(valid):
        print(f"Dropping {int((~valid).sum())} base systems with non-finite/invalid initial geometry.")
        case_ids = case_ids[valid]
        stable_fraction = stable_fraction[valid]
        mutual_inclination = mutual_inclination[valid]
        e_in = e_in[valid]
        e_out = e_out[valid]
        aratio = aratio[valid]
        masses = masses[valid]
        mratios = mratios[valid]
        base_fields = {field: values[valid] for field, values in base_fields.items()}

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
    mutual_inclination_degrees = np.degrees(mutual_inclination)

    # Use the same natural-log margin as triple_stability.  q_out is computed
    # from masses, while mratios[:, 1] is checked as a schema consistency test.
    print(f"  maximum |q_out(masses) - mratios[1]|: {q_ratio_difference:.6g}")
    _print_summary(
        case_ids,
        mutual_inclination_degrees,
        margin,
        stable_fraction,
        base_fields,
        all_inclinations,
        initial.source_description,
        completion_path,
        args.time_horizon,
    )
    pdf_path, png_path = _plot(
        margin,
        mutual_inclination_degrees,
        stable_fraction,
        args.output_dir,
        args.png_dpi,
    )
    print(f"  created: {pdf_path.resolve()}")
    print(f"  created: {png_path.resolve()}")


if __name__ == "__main__":
    main()
