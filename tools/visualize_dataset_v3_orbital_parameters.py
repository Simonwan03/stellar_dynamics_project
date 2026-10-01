from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = (
    "aratios_complete",
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
    "periods_complete",
    "tAnos_complete",
    "energy_complete",
    "angMomentum_complete",
)
ANGLE_PARAMETERS = {"incs_complete", "APs_complete", "LANs_complete", "tAnos_complete"}
ORBIT_PARAMETERS = {
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
    "periods_complete",
    "tAnos_complete",
    "mratios_complete",
}
Y_LABELS = {
    "aratios_complete": r"$a_{in}/a_{out}$",
    "smaxes_complete": "semi-major axis",
    "eccs_complete": "eccentricity",
    "incs_complete": "inclination",
    "APs_complete": "argument of periapsis",
    "LANs_complete": "longitude of ascending node",
    "periods_complete": "period",
    "tAnos_complete": "true anomaly",
    "energy_complete": "total energy",
    "angMomentum_complete": "angular momentum",
    "masses_complete": "mass",
    "mratios_complete": "mass ratio",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot orbital parameters versus time for dataset_v3 trajectories."
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--row", type=int, default=0, help="Local row within the selected shard.")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument(
        "--rows",
        default=None,
        help="Comma-separated local rows; overrides --row and --num-examples.",
    )
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-stop", type=int, default=None)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument(
        "--parameters",
        default=",".join(DEFAULT_PARAMETERS),
        help="Comma-separated HDF5 keys to plot.",
    )
    parser.add_argument("--angle-unit", choices=["degree", "radian"], default="degree")
    parser.add_argument(
        "--unwrap-angles",
        action="store_true",
        help="Unwrap AP/LAN/true-anomaly curves across their 2-pi boundary.",
    )
    parser.add_argument("--columns", type=int, default=2, help="Number of subplot columns.")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "shared_analysis" / "sampling_2" / "dataset_v3_orbital_parameter_viz",
    )
    return parser


def parse_parameters(raw: str) -> list[str]:
    parameters = [item.strip() for item in raw.split(",") if item.strip()]
    if not parameters:
        raise ValueError("--parameters must contain at least one HDF5 key.")
    return list(dict.fromkeys(parameters))


def choose_rows(raw_rows: str | None, first_row: int, count: int, row_count: int) -> list[int]:
    if raw_rows is None:
        rows = list(range(first_row, first_row + count))
    else:
        try:
            rows = [int(item.strip()) for item in raw_rows.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("--rows must be a comma-separated list of integers.") from exc
        rows = list(dict.fromkeys(rows))
        if not rows:
            raise ValueError("--rows must contain at least one row.")
    invalid = [row for row in rows if row < 0 or row >= row_count]
    if invalid:
        raise IndexError(f"Rows outside [0, {row_count}): {invalid}")
    return rows


def component_labels(key: str, width: int) -> list[str]:
    if key in ORBIT_PARAMETERS and width == 2:
        return ["inner orbit", "outer orbit"]
    if key == "masses_complete" and width == 3:
        return ["body 0", "body 1", "body 2"]
    if key == "angMomentum_complete" and width == 3:
        return [r"$L_x$", r"$L_y$", r"$L_z$"]
    if width == 1:
        return [Y_LABELS.get(key, key.removesuffix("_complete"))]
    return [f"component {index}" for index in range(width)]


def load_series(
    handle: h5py.File,
    key: str,
    row: int,
    time_slice: slice,
) -> np.ndarray:
    values = np.asarray(handle[key][row, time_slice], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(
            f"{key} has per-trajectory shape {handle[key].shape[1:]}; "
            "this plotter supports scalar or vector time series only."
        )
    return values


def transform_angles(values: np.ndarray, key: str, unit: str, unwrap: bool) -> np.ndarray:
    if key not in ANGLE_PARAMETERS:
        return values
    transformed = values.copy()
    # Inclination is not periodic. The other stored angles are 2-pi periodic.
    if unwrap and key != "incs_complete":
        for component in range(transformed.shape[1]):
            finite = np.isfinite(transformed[:, component])
            if np.any(finite):
                transformed[finite, component] = np.unwrap(transformed[finite, component])
    if unit == "degree":
        transformed = np.rad2deg(transformed)
    return transformed


def plot_parameters(
    times: np.ndarray,
    data: dict[str, np.ndarray],
    row: int,
    shard_index: int,
    angle_unit: str,
    columns: int,
    path: Path,
    dpi: int,
) -> dict[str, float]:
    rows = math.ceil(len(data) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 3.6 * rows), squeeze=False)
    finite_fractions: dict[str, float] = {}

    for ax, (key, values) in zip(axes.flat, data.items()):
        labels = component_labels(key, values.shape[1])
        for component, label in enumerate(labels):
            ax.plot(times, values[:, component], linewidth=1.1, label=label)
        finite_fraction = float(np.isfinite(values).mean())
        finite_fractions[key] = finite_fraction
        title = key.removesuffix("_complete")
        if finite_fraction < 1.0:
            title += f" (finite {finite_fraction:.1%})"
        ax.set_title(title)
        ylabel = Y_LABELS.get(key, title)
        if key in ANGLE_PARAMETERS:
            ylabel += " (deg)" if angle_unit == "degree" else " (rad)"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)
        if values.shape[1] > 1:
            ax.legend(loc="best", fontsize=8)

    for ax in axes.flat[len(data) :]:
        ax.set_visible(False)

    fig.suptitle(f"dataset_v3 orbital evolution — shard {shard_index:02d}, row {row}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return finite_fractions


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")
    if args.time_stride <= 0:
        raise ValueError("--time-stride must be positive.")
    if args.columns <= 0:
        raise ValueError("--columns must be positive.")

    parameters = parse_parameters(args.parameters)
    shard_path = args.dataset_dir / f"results_batch_{args.shard_index:02d}_sampling_{args.sampling_id}.hdf5"
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing result shard: {shard_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    with h5py.File(shard_path, "r") as handle:
        missing = [key for key in parameters if key not in handle]
        if missing:
            raise KeyError(f"Keys not found in {shard_path}: {', '.join(missing)}")
        reference = handle[parameters[0]]
        row_count, total_steps = int(reference.shape[0]), int(reference.shape[1])
        selected_rows = choose_rows(args.rows, args.row, args.num_examples, row_count)
        stop = total_steps if args.time_stop is None else min(args.time_stop, total_steps)
        if args.time_start < 0 or args.time_start >= stop:
            raise ValueError(f"Invalid time range: start={args.time_start}, stop={stop}.")
        time_slice = slice(args.time_start, stop, args.time_stride)

        for row in selected_rows:
            if "times_complete" in handle:
                times = np.asarray(handle["times_complete"][row, time_slice], dtype=np.float64)
            else:
                times = np.arange(args.time_start, stop, args.time_stride, dtype=np.float64)
            data = {
                key: transform_angles(
                    load_series(handle, key, row, time_slice),
                    key=key,
                    unit=args.angle_unit,
                    unwrap=args.unwrap_angles,
                )
                for key in parameters
            }
            stem = f"sampling{args.sampling_id}_shard{args.shard_index:02d}_row{row}_orbital_parameters"
            output_path = args.output_dir / f"{stem}.png"
            fractions = plot_parameters(
                times=times,
                data=data,
                row=row,
                shard_index=args.shard_index,
                angle_unit=args.angle_unit,
                columns=args.columns,
                path=output_path,
                dpi=args.dpi,
            )
            summaries.append(
                {
                    "row": row,
                    "run_id": args.shard_index * row_count + row,
                    "system_id": (args.shard_index * row_count + row) // 32,
                    "variation_id": (args.shard_index * row_count + row) % 32,
                    "source": str(shard_path),
                    "output": str(output_path),
                    "parameters": parameters,
                    "time_slice": [args.time_start, stop, args.time_stride],
                    "angle_unit": args.angle_unit,
                    "unwrap_angles": args.unwrap_angles,
                    "finite_fraction": fractions,
                }
            )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"examples": summaries}, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(summaries)} figure(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
