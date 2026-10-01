from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BODY_COLORS = ("tab:blue", "tab:orange", "tab:green")
BODY_LABELS = ("body 0", "body 1", "body 2")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize dataset_v3 three-body trajectories.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0, help="Result shard index, e.g. 0 for results_batch_00.")
    parser.add_argument("--row", type=int, default=None, help="Local row inside the selected shard. Random if omitted.")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-stop", type=int, default=None)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--center", choices=["none", "body0", "com"], default="com")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "shared_analysis" / "sampling_2" / "dataset_v3_trajectory_viz")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--animation", action="store_true", help="Also save an animated trajectory.")
    parser.add_argument("--animation-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--trail", type=int, default=120, help="Number of previous sampled steps to show in animation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")
    if args.time_stride <= 0:
        raise ValueError("--time-stride must be positive.")

    shard_path = args.dataset_dir / f"results_batch_{args.shard_index:02d}_sampling_{args.sampling_id}.hdf5"
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing result shard: {shard_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(shard_path, "r") as handle:
        if "positions_complete" not in handle:
            raise KeyError(f"positions_complete not found in {shard_path}")
        row_count = int(handle["positions_complete"].shape[0])
        rows = choose_rows(row_count=row_count, row=args.row, num_examples=args.num_examples, seed=args.seed)
        summaries = []
        for row in rows:
            positions, times = load_trajectory(
                handle=handle,
                row=row,
                time_start=args.time_start,
                time_stop=args.time_stop,
                time_stride=args.time_stride,
            )
            positions = center_positions(positions, mode=args.center)
            stem = f"sampling{args.sampling_id}_shard{args.shard_index:02d}_row{row}"
            orbit_path = args.output_dir / f"{stem}_3d.png"
            coords_path = args.output_dir / f"{stem}_coords.png"
            animation_path = args.output_dir / f"{stem}_animation.{args.animation_format}"
            plot_3d_trajectory(positions, row=row, path=orbit_path, dpi=args.dpi)
            plot_coordinate_timeseries(positions, times, row=row, path=coords_path, dpi=args.dpi)
            if args.animation:
                save_trajectory_animation(
                    positions=positions,
                    row=row,
                    path=animation_path,
                    fps=args.fps,
                    trail=args.trail,
                    dpi=args.dpi,
                    animation_format=args.animation_format,
                )
            summaries.append(
                {
                    "row": int(row),
                    "source": str(shard_path),
                    "time_start": args.time_start,
                    "time_stop": args.time_stop,
                    "time_stride": args.time_stride,
                    "center": args.center,
                    "trajectory_3d": str(orbit_path),
                    "coordinate_timeseries": str(coords_path),
                    "animation": str(animation_path) if args.animation else None,
                }
            )

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"examples": summaries}, handle, indent=2)
    print(f"Wrote {len(summaries)} trajectory visualization(s) to {args.output_dir}")


def choose_rows(row_count: int, row: int | None, num_examples: int, seed: int) -> list[int]:
    if row is not None:
        if row < 0 or row >= row_count:
            raise IndexError(f"--row {row} is outside shard row range [0, {row_count}).")
        return [row + offset for offset in range(num_examples) if row + offset < row_count]
    rng = np.random.default_rng(seed)
    count = min(num_examples, row_count)
    return [int(item) for item in rng.choice(row_count, size=count, replace=False)]


def load_trajectory(
    handle: h5py.File,
    row: int,
    time_start: int,
    time_stop: int | None,
    time_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = int(handle["positions_complete"].shape[1])
    stop = total_steps if time_stop is None else min(time_stop, total_steps)
    if time_start < 0 or time_start >= stop:
        raise ValueError(f"Invalid time range: start={time_start}, stop={stop}.")
    time_slice = slice(time_start, stop, time_stride)
    positions = np.asarray(handle["positions_complete"][row, time_slice], dtype=np.float64)
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    if "times_complete" in handle:
        times = np.asarray(handle["times_complete"][row, time_slice], dtype=np.float64)
    else:
        times = np.arange(time_start, stop, time_stride, dtype=np.float64)
    return positions, times


def center_positions(positions: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return positions
    if mode == "body0":
        return positions - positions[:, :1, :]
    if mode == "com":
        return positions - positions.mean(axis=1, keepdims=True)
    raise ValueError(f"Unsupported centering mode: {mode}")


def plot_3d_trajectory(positions: np.ndarray, row: int, path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for body_idx in range(positions.shape[1]):
        xyz = positions[:, body_idx, :]
        color = BODY_COLORS[body_idx % len(BODY_COLORS)]
        label = BODY_LABELS[body_idx] if body_idx < len(BODY_LABELS) else f"body {body_idx}"
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=1.2, label=label)
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, marker="o", s=24)
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="x", s=36)
    set_equal_3d_axes(ax, positions)
    ax.set_title(f"dataset_v3 trajectory row {row}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_coordinate_timeseries(positions: np.ndarray, times: np.ndarray, row: int, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    coord_names = ("x", "y", "z")
    for coord_idx, ax in enumerate(axes):
        for body_idx in range(positions.shape[1]):
            color = BODY_COLORS[body_idx % len(BODY_COLORS)]
            label = BODY_LABELS[body_idx] if body_idx < len(BODY_LABELS) else f"body {body_idx}"
            ax.plot(times, positions[:, body_idx, coord_idx], color=color, linewidth=1.0, label=label)
        ax.set_ylabel(coord_names[coord_idx])
        ax.grid(alpha=0.25)
    axes[0].set_title(f"coordinate time series row {row}")
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def set_equal_3d_axes(ax: Any, positions: np.ndarray) -> None:
    flat = positions.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_trajectory_animation(
    positions: np.ndarray,
    row: int,
    path: Path,
    fps: int,
    trail: int,
    dpi: int,
    animation_format: str,
) -> None:
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if positions.shape[0] < 2:
        raise ValueError("Need at least two sampled time steps to animate.")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    set_equal_3d_axes(ax, positions)
    ax.set_title(f"dataset_v3 trajectory row {row}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    lines = []
    points = []
    for body_idx in range(positions.shape[1]):
        color = BODY_COLORS[body_idx % len(BODY_COLORS)]
        label = BODY_LABELS[body_idx] if body_idx < len(BODY_LABELS) else f"body {body_idx}"
        (line,) = ax.plot([], [], [], color=color, linewidth=1.3, label=label)
        (point,) = ax.plot([], [], [], color=color, marker="o", markersize=5)
        lines.append(line)
        points.append(point)
    ax.legend(loc="upper right")

    trail = max(1, int(trail))

    def update(frame: int) -> list[Any]:
        start = max(0, frame - trail + 1)
        for body_idx, (line, point) in enumerate(zip(lines, points)):
            xyz = positions[start : frame + 1, body_idx, :]
            current = positions[frame, body_idx, :]
            line.set_data(xyz[:, 0], xyz[:, 1])
            line.set_3d_properties(xyz[:, 2])
            point.set_data([current[0]], [current[1]])
            point.set_3d_properties([current[2]])
        ax.view_init(elev=24, azim=35 + 360 * frame / max(positions.shape[0] - 1, 1))
        return [*lines, *points]

    animation = FuncAnimation(fig, update, frames=positions.shape[0], interval=1000 / fps, blit=False)
    if animation_format == "gif":
        animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        animation.save(path, fps=fps, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
