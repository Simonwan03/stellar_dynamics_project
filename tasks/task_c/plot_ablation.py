from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["baseline", "direct", "traj", "dual"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "direct": "Direct",
    "traj": "Trajectory-context",
    "dual": "Dual-attention",
}
MODEL_COLORS = {
    "baseline": "#7f7f7f",
    "direct": "#1f77b4",
    "traj": "#ff7f0e",
    "dual": "#2ca02c",
}


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def lookup(rows: list[dict[str, str]], prefix: int, variations: int, model: str, key: str) -> float:
    for row in rows:
        if (
            int(row["prefix_len"]) == prefix
            and int(row["variation_count"]) == variations
            and row["model"] == model
        ):
            return as_float(row.get(key))
    return float("nan")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_heatmaps(rows: list[dict[str, str]], prefixes: list[int], variations: list[int], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    matrices: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        matrices[model] = np.array(
            [[lookup(rows, prefix, count, model, "overall_mae") for count in variations] for prefix in prefixes]
        )
    vmin = min(np.nanmin(matrix) for matrix in matrices.values())
    vmax = max(np.nanmax(matrix) for matrix in matrices.values())
    image = None
    for axis, model in zip(axes.flat, MODEL_ORDER):
        image = axis.imshow(matrices[model], cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Number of variations")
        axis.set_ylabel("Prefix length")
        axis.set_xticks(range(len(variations)), labels=[str(value) for value in variations])
        axis.set_yticks(range(len(prefixes)), labels=[str(value) for value in prefixes])
        for row_index in range(len(prefixes)):
            for column_index in range(len(variations)):
                value = matrices[model][row_index, column_index]
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.23 else "black")
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Overall MAE (lower is better)", shrink=0.9)
    fig.suptitle("Task C ablation: overall system MAE", fontsize=15)
    fig.savefig(output / "figure_ablation_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def plot_prefix_variation_tradeoff_heatmaps(
    rows: list[dict[str, str]], prefixes: list[int], variations: list[int], output: Path
) -> None:
    """Show MAE and relative input size in the same prefix-by-variation cells."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    matrices: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        matrices[model] = np.array(
            [[lookup(rows, prefix, count, model, "overall_mae") for count in variations] for prefix in prefixes]
        )
    vmin = min(np.nanmin(matrix) for matrix in matrices.values())
    vmax = max(np.nanmax(matrix) for matrix in matrices.values())
    image = None
    for axis, model in zip(axes.flat, MODEL_ORDER):
        display_matrix = matrices[model].T
        image = axis.imshow(display_matrix, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Prefix length (P)")
        axis.set_ylabel("Variation count (V)")
        axis.set_xticks(range(len(prefixes)), labels=[str(value) for value in prefixes])
        axis.set_yticks(range(len(variations)), labels=[str(value) for value in variations])
        for row_index, count in enumerate(variations):
            for column_index, prefix in enumerate(prefixes):
                value = display_matrix[row_index, column_index]
                input_scale = prefix * count
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}\nP×V={input_scale}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.23 else "black",
                    fontsize=8,
                )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Overall MAE (lower is better)", shrink=0.9)
    fig.suptitle("Prefix × variation trade-off: MAE and relative input size", fontsize=15)
    fig.savefig(output / "figure_prefix_variation_tradeoff_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def plot_input_budget_tradeoff(rows: list[dict[str, str]], output: Path) -> None:
    """Plot learned-model error against P×V and highlight the non-dominated frontier."""
    learned_rows = [row for row in rows if row["model"] != "baseline"]
    points = [
        {
            "model": row["model"],
            "prefix": int(row["prefix_len"]),
            "variations": int(row["variation_count"]),
            "input_scale": int(row["prefix_len"]) * int(row["variation_count"]),
            "mae": float(row["overall_mae"]),
        }
        for row in learned_rows
    ]
    pareto = [
        point
        for point in points
        if not any(
            other["input_scale"] <= point["input_scale"]
            and other["mae"] <= point["mae"]
            and (
                other["input_scale"] < point["input_scale"]
                or other["mae"] < point["mae"]
            )
            for other in points
        )
    ]
    pareto.sort(key=lambda point: (point["input_scale"], point["mae"]))

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model in ["direct", "traj", "dual"]:
        model_points = [point for point in points if point["model"] == model]
        axis.scatter(
            [point["input_scale"] for point in model_points],
            [point["mae"] for point in model_points],
            s=65,
            alpha=0.85,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    axis.plot(
        [point["input_scale"] for point in pareto],
        [point["mae"] for point in pareto],
        color="#111111",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Pareto frontier",
    )
    for point in pareto:
        axis.scatter(
            point["input_scale"],
            point["mae"],
            s=170,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.5,
        )
        axis.annotate(
            f"{MODEL_LABELS[point['model']]}\nP{point['prefix']}/V{point['variations']}",
            (point["input_scale"], point["mae"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    budgets = sorted({point["input_scale"] for point in points})
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.set_xlabel("Relative input size P×V (full feature count = P×V×15)")
    axis.set_ylabel("Overall MAE (lower is better)")
    axis.set_title("Input size versus prediction quality")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "figure_input_budget_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_fixed_budget_allocation(rows: list[dict[str, str]], model: str, output: Path) -> None:
    """Compare temporal-prefix versus variation allocation at equal P×V."""
    points = [
        {
            "prefix": int(row["prefix_len"]),
            "variations": int(row["variation_count"]),
            "input_scale": int(row["prefix_len"]) * int(row["variation_count"]),
            "mae": float(row["overall_mae"]),
            "model": row["model"],
        }
        for row in rows
    ]
    if model == "best":
        best_points: dict[tuple[int, int], dict[str, int | float | str]] = {}
        for point in points:
            key = (int(point["prefix"]), int(point["variations"]))
            if key not in best_points or float(point["mae"]) < float(best_points[key]["mae"]):
                best_points[key] = point
        points = list(best_points.values())
    else:
        points = [point for point in points if point["model"] == model]

    by_budget: dict[int, list[dict[str, int | float | str]]] = {}
    for point in points:
        by_budget.setdefault(int(point["input_scale"]), []).append(point)
    pairs = []
    for budget, candidates in sorted(by_budget.items()):
        if len(candidates) < 2:
            continue
        shorter_prefix = min(candidates, key=lambda point: (int(point["prefix"]), -int(point["variations"])))
        longer_prefix = max(candidates, key=lambda point: (int(point["prefix"]), -int(point["variations"])))
        if int(shorter_prefix["prefix"]) == int(longer_prefix["prefix"]):
            continue
        pairs.append(
            {
                "budget": budget,
                "long": longer_prefix,
                "short": shorter_prefix,
                "delta": float(shorter_prefix["mae"]) - float(longer_prefix["mae"]),
            }
        )
    if not pairs:
        raise ValueError(f"No equal-input-size pairs found for model={model}.")

    budgets = [int(pair["budget"]) for pair in pairs]
    heat = np.array(
        [[float(pair["long"]["mae"]), float(pair["short"]["mae"])] for pair in pairs]
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    image = axes[0].imshow(heat, cmap="viridis_r", aspect="auto")
    title = "Best model per configuration" if model == "best" else MODEL_LABELS[model]
    axes[0].set_title(f"Fixed P×V allocation comparison ({title})")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Fixed relative input size P×V")
    axes[0].set_xticks([0, 1], labels=["Longer prefix / fewer variations", "Shorter prefix / more variations"])
    axes[0].set_yticks(range(len(pairs)), labels=[str(budget) for budget in budgets])
    for row_index, pair in enumerate(pairs):
        for column_index, key in enumerate(("long", "short")):
            point = pair[key]
            axes[0].text(
                column_index,
                row_index,
                f"{float(point['mae']):.3f}\nP{int(point['prefix'])}/V{int(point['variations'])}\n{MODEL_LABELS[str(point['model'])]}",
                ha="center",
                va="center",
                color="white" if float(point["mae"]) < 0.19 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=axes[0], label="Overall MAE (lower is better)", shrink=0.9)

    deltas = [float(pair["delta"]) for pair in pairs]
    colors = ["#2ca02c" if delta > 0 else "#d62728" for delta in deltas]
    axes[1].bar(range(len(pairs)), deltas, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(pairs)), labels=[str(budget) for budget in budgets])
    axes[1].set_xlabel("Fixed relative input size P×V")
    axes[1].set_ylabel("MAE(short/more) − MAE(long/fewer)")
    axes[1].set_title("Δ > 0: longer prefix is better; Δ < 0: more variations are better", fontsize=10, pad=16)
    axes[1].grid(axis="y", alpha=0.25)
    for index, delta in enumerate(deltas):
        axes[1].text(index, delta + (0.002 if delta >= 0 else -0.002), f"{delta:+.3f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=9)
    fig.savefig(output / "figure_fixed_budget_allocation.png", bbox_inches="tight")
    plt.close(fig)


def plot_by_variations(rows: list[dict[str, str]], prefixes: list[int], variations: list[int], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for axis, prefix in zip(axes.flat, prefixes):
        for model in MODEL_ORDER:
            values = [lookup(rows, prefix, count, model, "overall_mae") for count in variations]
            axis.plot(
                variations,
                values,
                marker="o",
                linewidth=2,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        axis.set_title(f"Prefix = {prefix}")
        axis.set_xlabel("Number of variations")
        axis.set_xticks(variations)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Overall MAE")
    axes[1, 0].set_ylabel("Overall MAE")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Effect of variation count", fontsize=15)
    fig.savefig(output / "figure_ablation_by_variations.png", bbox_inches="tight")
    plt.close(fig)


def plot_by_prefix(rows: list[dict[str, str]], prefixes: list[int], variations: list[int], output: Path) -> None:
    fig, axes = plt.subplots(1, len(variations), figsize=(17, 4.5), sharey=True, constrained_layout=True)
    for axis, count in zip(axes, variations):
        for model in MODEL_ORDER:
            values = [lookup(rows, prefix, count, model, "overall_mae") for prefix in prefixes]
            axis.plot(
                prefixes,
                values,
                marker="o",
                linewidth=2,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        axis.set_title(f"Variations = {count}")
        axis.set_xlabel("Prefix length")
        axis.set_xticks(prefixes)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Overall MAE")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Effect of prefix length", fontsize=15)
    fig.savefig(output / "figure_ablation_by_prefix.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_config_horizons(output: Path) -> None:
    metrics_path = output / "prefix_100" / "variations_32" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    horizons = [500, 800, 1500, 2500, 4000, 6500, 10000]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model in MODEL_ORDER:
        values = metrics[model]["horizon_mae"]
        axis.plot(
            horizons,
            values,
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    axis.set_xlabel("time_unbound horizon")
    axis.set_ylabel("System MAE")
    axis.set_title("Per-horizon error at prefix=100, variations=32")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "figure_best_config_horizon_mae.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_config_bars(rows: list[dict[str, str]], output: Path) -> None:
    models = ["baseline", "direct", "traj", "dual"]
    overall = [lookup(rows, 100, 32, model, "overall_mae") for model in models]
    final = [lookup(rows, 100, 32, model, "final_mae") for model in models]
    x = np.arange(len(models))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(x - width / 2, overall, width, label="Overall MAE", color="#4c78a8")
    axis.bar(x + width / 2, final, width, label="Final MAE", color="#f58518")
    axis.set_xticks(x, labels=[MODEL_LABELS[model] for model in models], rotation=15, ha="right")
    axis.set_ylabel("MAE (lower is better)")
    axis.set_title("Best ablation cell: prefix=100, variations=32")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, value in enumerate(overall):
        axis.text(index - width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    for index, value in enumerate(final):
        axis.text(index + width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    fig.savefig(output / "figure_best_config_bars.png", bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Task C prefix/variation ablation results.")
    parser.add_argument("--result-dir", type=Path, default=Path("results/task_c_ablation"))
    parser.add_argument("--allocation-model", choices=["best", *MODEL_ORDER], default="best")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_plot_style()
    rows = read_summary(args.result_dir / "summary.csv")
    prefixes = sorted({int(row["prefix_len"]) for row in rows})
    variations = sorted({int(row["variation_count"]) for row in rows})
    args.result_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(rows, prefixes, variations, args.result_dir)
    plot_prefix_variation_tradeoff_heatmaps(rows, prefixes, variations, args.result_dir)
    plot_input_budget_tradeoff(rows, args.result_dir)
    plot_fixed_budget_allocation(rows, args.allocation_model, args.result_dir)
    plot_by_variations(rows, prefixes, variations, args.result_dir)
    plot_by_prefix(rows, prefixes, variations, args.result_dir)
    plot_best_config_horizons(args.result_dir)
    plot_best_config_bars(rows, args.result_dir)
    print("Wrote Task C ablation figures to", args.result_dir)


if __name__ == "__main__":
    main()
