#!/usr/bin/env python3
"""Run a pair-aware K-by-T factorial experiment on sampling3."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from scripts import run_fixed_budget_sampling3 as base
from tasks.task_b.phase_ensemble import (
    BudgetCondition,
    common_preinstability_mask,
    nested_phase_draws,
    prefix_endpoints_in_label_time,
)

PAIR_COUNTS = (0.5, 1, 2, 4, 8)
PREFIXES = (32, 64, 128, 256)
CONDITIONS = tuple(BudgetCondition(k, t, unit_size=2) for k in PAIR_COUNTS for t in PREFIXES)
DEFAULT_OUTPUT = ROOT / "results/factorial_sampling3_Kpairs0p5-8_T32-256"


def condition_key(condition: BudgetCondition) -> tuple[float, int]:
    return float(condition.k), int(condition.t)


def parse_args() -> Any:
    parser = base.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT,
        train_seeds=1,
        epochs=10,
        patience=3,
        batch_systems=16,
        eval_batch_systems=16,
    )
    return parser.parse_args()


def predict_grid(
    model: torch.nn.Module,
    predictor: Any,
    features: np.ndarray,
    system_rows: np.ndarray,
    system_indices: np.ndarray,
    pair_draws: np.ndarray,
    singleton_sides: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    args: Any,
) -> dict[tuple[float, int], np.ndarray]:
    output: dict[tuple[float, int], np.ndarray] = {}
    for condition in CONDITIONS:
        key = condition_key(condition)
        values = np.empty((pair_draws.shape[0], system_indices.size), dtype=np.float32)
        is_singleton = np.isclose(condition.k, 0.5)
        pair_count = 1 if is_singleton else int(condition.k)
        for draw in range(pair_draws.shape[0]):
            member_mask = base.singleton_member_mask(singleton_sides[draw], device) if is_singleton else None
            values[draw] = predictor(
                model,
                features,
                system_rows,
                system_indices,
                pair_draws[draw, :, :pair_count],
                condition.t,
                mean,
                std,
                device,
                args.eval_batch_systems,
                bool(args.amp and device.type == "cuda"),
                member_mask=member_mask,
            )
        output[key] = values
        print(f"  predicted M={condition.k:g}, T={condition.t} over {pair_draws.shape[0]} pair draws", flush=True)
    return output


def _valid_metric_row(
    *, model: str, target_name: str, control: str, condition: BudgetCondition,
    target: np.ndarray, prediction: np.ndarray, seed: int,
) -> dict[str, Any]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    metrics = base.point_metrics(target[valid], prediction[valid])
    intervals = base.bootstrap_metric_intervals(
        target[valid], prediction[valid], _BOOTSTRAPS,
        np.random.default_rng(seed),
    )
    return {
        "model": model, "target": target_name, "control": control,
        "M_pairs": condition.k, "K": condition.k, "trajectories": 2 * condition.k,
        "T": condition.t, "stored_points": condition.budget,
        **metrics,
        **{f"{name}_ci_low": bounds[0] for name, bounds in intervals.items()},
        **{f"{name}_ci_high": bounds[1] for name, bounds in intervals.items()},
    }


def _comparison_row(
    *, model: str, target_name: str, control: str, effect: str,
    fixed_value: float, reference: BudgetCondition, comparison: BudgetCondition,
    target: np.ndarray, reference_prediction: np.ndarray,
    comparison_prediction: np.ndarray, seed: int,
) -> dict[str, Any]:
    delta, low, high, p_value = base.bootstrap_delta_mae(
        target, reference_prediction, comparison_prediction,
        _BOOTSTRAPS, np.random.default_rng(seed),
    )
    valid = np.isfinite(target) & np.isfinite(reference_prediction) & np.isfinite(comparison_prediction)
    return {
        "model": model, "target": target_name, "control": control,
        "effect": effect,
        "fixed_T": fixed_value if effect in ("breadth_at_fixed_T", "pair_completion_at_fixed_T") else np.nan,
        "fixed_M_pairs": fixed_value if effect == "depth_at_fixed_M" else np.nan,
        "reference_M_pairs": reference.k, "reference_T": reference.t,
        "comparison_M_pairs": comparison.k, "comparison_T": comparison.t,
        "n_systems": int(valid.sum()),
        "delta_mae": delta, "ci_low": low, "ci_high": high,
        "p_raw_bootstrap": p_value, "p_holm": np.nan,
    }


def summarize_grid(
    output_dir: Path,
    metric_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    labels = [f"{m:g}" for m in PAIR_COUNTS]
    lines = [
        "# Sampling3 pair-aware K–T factorial grid — preliminary one-seed results",
        "",
        "## Setup",
        "",
        "- `M_pairs=0.5` means one trajectory selected from one matched pair; `M_pairs>=1` means both trajectories from each selected pair.",
        "- Pairing is variation `i` with variation `i+16`; the half-pair side is randomly selected and then held fixed across all T values within a draw.",
        f"- Grid: `M_pairs = {', '.join(labels)}`; `T = {', '.join(map(str, PREFIXES))}`. Stored-point count is `2 × M_pairs × T` (so a half-pair uses `T` points).",
        f"- One training seed: `{config['training']['train_seeds'][0]}`; max epochs `{config['training']['epochs']}`, patience `{config['training']['patience']}`.",
        f"- Test predictions average {config['pair_draws']} nested pair draws per system; intervals resample systems {config['bootstrap_replicates']} times.",
        "- Bootstrap intervals quantify test-system uncertainty, not training-seed variability.",
        "",
        "## Primary result: all-32 target, pair-DeepSets",
        "",
        "### Fixed-T slices: MAE by observed phase allocation",
        "",
        "| T | " + " | ".join(f"M={label}" for label in labels) + " |",
        "|---:|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    primary = metric_frame[(metric_frame.model == "deep_sets") & (metric_frame.target == "all_phase") & (metric_frame.control == "standard")]
    for t in PREFIXES:
        cells = {float(row.M_pairs): row for row in primary[primary["T"] == t].itertuples(index=False)}
        values = [f"{cells[m].mae:.3f} [{cells[m].mae_ci_low:.3f}, {cells[m].mae_ci_high:.3f}]" for m in PAIR_COUNTS]
        lines.append(f"| {t} | " + " | ".join(values) + " |")
    lines += [
        "", "### Fixed-M depth slices: MAE by prefix length", "",
        "| M pairs | T=32 | T=64 | T=128 | T=256 |", "|---:|---:|---:|---:|---:|",
    ]
    for m in PAIR_COUNTS:
        cells = {int(row.T): row for row in primary[primary["M_pairs"] == m].itertuples(index=False)}
        values = [f"{cells[t].mae:.3f} [{cells[t].mae_ci_low:.3f}, {cells[t].mae_ci_high:.3f}]" for t in PREFIXES]
        lines.append(f"| {m:g} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## How to read the contrasts",
        "",
        "`pair_completion_at_fixed_T` compares one member (`M=0.5`) with both members of that same sampled pair (`M=1`). `breadth_at_fixed_T` compares `M=1` with more matched pairs (`M=2,4,8`). `depth_at_fixed_M` compares `T=32` with longer prefixes at the same M. Positive `delta_mae` favors the comparison allocation. `factorial_comparisons.csv` contains paired system-bootstrap intervals and Holm adjustments within each fixed-T or fixed-M slice.",
        "",
        "This is a single-seed preliminary result. The grid separates phase allocation from prefix length, while the bootstrap does not address model-seed uncertainty.",
        "",
        "## Files",
        "",
        "- `config.json`, `pair_partition.csv`, `system_targets.csv`",
        "- `per_system_predictions.csv`, `factorial_metrics.csv`, `factorial_comparisons.csv`",
        "- `preinstability_eligibility.csv`",
        "- `figure1_factorial_heatmap.png`, `figure2_factorial_slices.png`, `figure3_heldout_controls.png`"
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, metrics: pd.DataFrame) -> None:
    primary = metrics[(metrics.model == "deep_sets") & (metrics.target == "all_phase") & (metrics.control == "standard")]
    matrix = np.array([[float(primary[(primary["M_pairs"] == m) & (primary["T"] == t)].mae.iloc[0]) for m in PAIR_COUNTS] for t in PREFIXES])
    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(PAIR_COUNTS)), labels=[f"{m:g}" for m in PAIR_COUNTS])
    ax.set_yticks(range(len(PREFIXES)), labels=[str(t) for t in PREFIXES])
    ax.set_xlabel("Observed allocation M (0.5 = one track; M≥1 = full pairs)")
    ax.set_ylabel("Prefix length T per trajectory")
    ax.set_title("All-32 target: system-level MAE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] > np.median(matrix) else "black")
    fig.colorbar(im, ax=ax, label="MAE")
    base._save_figure(fig, output_dir, "figure1_factorial_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for t in PREFIXES:
        frame = primary[primary["T"] == t].sort_values("M_pairs")
        axes[0].plot(frame["M_pairs"], frame["mae"], marker="o", label=f"T={t}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(PAIR_COUNTS, labels=[f"{m:g}" for m in PAIR_COUNTS])
    axes[0].set_xlabel("Allocation M (0.5 = one track)")
    axes[0].set_ylabel("System-level MAE")
    axes[0].set_title("Breadth at fixed T")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for m in PAIR_COUNTS:
        frame = primary[primary["M_pairs"] == m].sort_values("T")
        axes[1].plot(frame["T"], frame["mae"], marker="o", label=f"M={m:g}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(PREFIXES, labels=[str(t) for t in PREFIXES])
    axes[1].set_xlabel("Prefix length T")
    axes[1].set_title("Depth at fixed M")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Pair-aware K–T factorial grid")
    base._save_figure(fig, output_dir, "figure2_factorial_slices")

    held = metrics[(metrics.model == "deep_sets") & (metrics.target == "heldout_phase")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, control, title in ((axes[0], "standard", "Held-out pairs"), (axes[1], "heldout_preinstability", "Held-out + event-free observed pairs")):
        frame = held[held.control == control]
        values = np.array([[float(frame[(frame["M_pairs"] == m) & (frame["T"] == t)].mae.iloc[0]) for m in PAIR_COUNTS] for t in PREFIXES])
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(PAIR_COUNTS)), labels=[f"{m:g}" for m in PAIR_COUNTS])
        ax.set_yticks(range(len(PREFIXES)), labels=[str(t) for t in PREFIXES])
        ax.set_title(title)
        ax.set_xlabel("Allocation M (0.5 = one track)")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white" if values[i,j] > np.median(values) else "black")
    axes[0].set_ylabel("Prefix length T")
    fig.colorbar(im, ax=axes, label="MAE", shrink=0.9)
    fig.suptitle("Held-out target robustness")
    base._save_figure(fig, output_dir, "figure3_heldout_controls")


def main() -> None:
    global _BOOTSTRAPS
    args = parse_args()
    if args.group_size != base.GROUP_SIZE:
        raise ValueError("Factorial runner expects the 32-variation sampling3 systems.")
    if args.train_seeds <= 0 or args.phase_draws <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("train-seeds, pair-draws, and bootstrap-replicates must be positive.")
    _BOOTSTRAPS = int(args.bootstrap_replicates)
    args.output_dir = base.ensure_dir(args.output_dir.resolve())
    dataset_dir = base.resolve_dataset_dir(args.dataset_dir)
    args.cache_path = base.resolve_cache_path(args.cache_path, dataset_dir, args.output_dir)
    audit = pd.read_csv(args.audit_csv).sort_values("system_id").reset_index(drop=True)
    system_ids = audit.system_id.to_numpy(dtype=np.int64)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    manifest = base.rebuild_split_manifest(audit, args.output_dir, seed=args.seed, ratios=ratios)
    pair_order, _ = base.save_pair_partition(system_ids, args.output_dir, args.seed)
    manifest_sorted = manifest.sort_values("raw_case_id").reset_index(drop=True)
    split_names = manifest_sorted.split.astype(str).to_numpy()
    ma_margin = audit.set_index("system_id").reindex(system_ids).ma_margin.to_numpy(dtype=np.float64)
    labels, label_source = base.load_labels(dataset_dir, args.label_fallback)
    stable, p_all, p_heldout, failure_times = base.make_targets(labels, audit, manifest_sorted, pair_order, system_ids, args.output_dir)
    seed_values = [int(args.seed + 1009 * i) for i in range(args.train_seeds)]
    config: dict[str, Any] = {
        "experiment": "sampling3_pair_aware_KT_factorial",
        "pairing": "variation i with i+16 for i=0,...,15",
        "K_definition": "M_pairs=0.5 observes one randomly selected member of one pair; M_pairs>=1 observes both members of each selected pair",
        "grid": [{"M_pairs": c.k, "trajectories": int(round(2*c.k)), "T": c.t, "stored_points": c.budget} for c in CONDITIONS],
        "total_budget": "varies by grid cell as 2*M_pairs*T; M_pairs=0.5 therefore uses T stored points",
        "target_all_phase": "stable fraction among all 32 variations",
        "target_heldout": "stable fraction among the fixed 8 held-out pairs (16 variations)",
        "dataset_dir": dataset_dir,
        "label_source": label_source,
        "audit_csv": args.audit_csv,
        "output_dir": args.output_dir,
        "cache_path": args.cache_path,
        "group_size": base.GROUP_SIZE,
        "split_version": 1,
        "split_seed": args.seed,
        "split_counts": manifest_sorted.split.value_counts().to_dict(),
        "pair_partition_seed": args.seed,
        "pair_draws": args.phase_draws,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_streams": {"train_seeds": seed_values, "eval_all": args.seed+31031, "eval_candidate": args.seed+41041, "val_all": args.seed+51051, "val_candidate": args.seed+61061, "singleton_side_all": args.seed+71071, "singleton_side_candidate": args.seed+81081, "val_singleton_side_all": args.seed+91091, "val_singleton_side_candidate": args.seed+101101},
        "model": {"architecture": "shared TCN + symmetric pair embedding + DeepSets", "TCN_hidden": 160, "TCN_levels": 5, "pair_features": ["observed-member mean", "absolute difference for full pair", "observed member fraction"], "output": "sigmoid stable fraction"},
        "training": {"uniform_condition_sampling": "all 20 (M_pairs,T) cells", "loss": "MSE for pair regressors; BCE for per-trajectory baseline", "checkpoint": "mean validation MAE across 20 grid cells", "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs, "patience": args.patience, "train_seeds": seed_values, "batch_systems": args.batch_systems},
        "evaluation": {"primary_metric": "system-level MAE", "bootstrap_unit": "base system", "bootstrap_replicates": args.bootstrap_replicates, "holm_adjustment": "within each fixed-T pair-completion/breadth or fixed-M depth slice"},
        "run_scope": "preliminary single-seed grid" if args.train_seeds == 1 else "multi-seed grid",
        "status": "prepared",
    }
    base.save_json(args.output_dir / "config.json", config)
    if args.prepare_only:
        print(f"Prepared factorial grid in {args.output_dir}", flush=True)
        return

    result_paths = base.discover_result_paths(dataset_dir, 3)
    if len(result_paths) != 40:
        raise FileNotFoundError(f"Expected 40 sampling3 HDF5 shards under {dataset_dir}, found {len(result_paths)}.")
    from scripts.run_task_b_highres import make_cache
    shape, finite = make_cache(result_paths, args.cache_path, prefix_len=1024, time_stride=1, rebuild=False)
    feature_array = np.memmap(args.cache_path, mode="r", dtype=np.float32, shape=shape)
    batch_ids = [base.batch_id_from_path(path) for path in result_paths]
    row_counts = []
    import h5py
    for path in result_paths:
        with h5py.File(path, "r") as handle:
            row_counts.append(int(handle["times_complete"].shape[0]))
    aligned_labels = base.load_aligned_labels(dataset_dir / "completed_systems_sampling_3.txt", batch_ids, row_counts)
    system_rows = base.resolve_system_rows(aligned_labels, system_ids, finite, shape[0])
    train_indices = np.flatnonzero(split_names == "train")
    val_indices = np.flatnonzero(split_names == "val")
    test_indices = np.flatnonzero(split_names == "test")
    mean, std = base.calculate_normalization(feature_array, system_rows[train_indices].reshape(-1))
    np.savez(args.output_dir / "orbital15_normalization.npz", mean=mean, std=std)
    grid, source_steps, source_end, time_metadata = base.load_time_axis(result_paths)
    endpoint_values = prefix_endpoints_in_label_time(grid, PREFIXES, source_steps, base.LABEL_HORIZON, source_end)
    endpoints = dict(zip(PREFIXES, endpoint_values.tolist()))
    config["time_axis"] = time_metadata
    config["prefix_endpoint_label_time"] = {str(k): float(v) for k, v in endpoints.items()}
    config["feature_cache"] = {"path": str(args.cache_path), "shape": shape, "dtype": "float32"}

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    all_pool = np.tile(np.arange(base.PAIR_COUNT, dtype=np.int16), (len(system_ids), 1))
    candidate_pool = pair_order[:, :8].copy()
    draws_all = nested_phase_draws(all_pool[test_indices], args.phase_draws, np.random.default_rng(args.seed+31031))
    draws_candidate = nested_phase_draws(candidate_pool[test_indices], args.phase_draws, np.random.default_rng(args.seed+41041))
    val_all = nested_phase_draws(all_pool[val_indices], 1, np.random.default_rng(args.seed+51051))[0]
    val_candidate = nested_phase_draws(candidate_pool[val_indices], 1, np.random.default_rng(args.seed+61061))[0]
    singleton_sides_all = np.random.default_rng(args.seed+71071).integers(0, 2, size=draws_all.shape[:2], dtype=np.int8)
    singleton_sides_candidate = np.random.default_rng(args.seed+81081).integers(0, 2, size=draws_candidate.shape[:2], dtype=np.int8)
    val_singleton_sides_all = np.random.default_rng(args.seed+91091).integers(0, 2, size=val_indices.size, dtype=np.int8)
    val_singleton_sides_candidate = np.random.default_rng(args.seed+101101).integers(0, 2, size=val_indices.size, dtype=np.int8)
    np.savez_compressed(
        args.output_dir / "factorial_phase_draws.npz",
        test_system_ids=system_ids[test_indices],
        eval_all_pair_ids=draws_all,
        eval_all_singleton_sides=singleton_sides_all,
        eval_candidate_pair_ids=draws_candidate,
        eval_candidate_singleton_sides=singleton_sides_candidate,
        val_system_ids=system_ids[val_indices],
        val_all_pair_ids=val_all,
        val_all_singleton_sides=val_singleton_sides_all,
        val_candidate_pair_ids=val_candidate,
        val_candidate_singleton_sides=val_singleton_sides_candidate,
    )
    condition_pre_masks: dict[tuple[float, int], np.ndarray] = {}
    singleton_variations = base.PAIR_MEMBERS[draws_candidate[:, :, 0], singleton_sides_candidate]
    for condition in CONDITIONS:
        condition_pre_masks[condition_key(condition)] = common_preinstability_mask(
            failure_times[test_indices], draws_candidate, (condition,), np.array([endpoints[condition.t]]),
            pair_members=base.PAIR_MEMBERS,
            singleton_variations=singleton_variations if np.isclose(condition.k, 0.5) else None,
        )
    # Every allocation is evaluated on the same draw/system support. This
    # keeps fixed-T and fixed-M contrasts paired after the event-free filter.
    common_pre = np.logical_and.reduce(list(condition_pre_masks.values()))
    pre_masks = {key: common_pre for key in condition_pre_masks}
    pd.DataFrame([
        {"M_pairs": c.k, "trajectories": int(round(2*c.k)), "T": c.t, "stored_points": c.budget,
         "prefix_endpoint_label_time": endpoints[c.t],
         "condition_specific_eligible_draw_systems": int(condition_pre_masks[condition_key(c)].sum()),
         "common_eligible_draw_systems": int(common_pre.sum()),
         "systems_with_any_common_eligible_draw": int(common_pre.any(axis=0).sum())}
        for c in CONDITIONS
    ]).to_csv(args.output_dir / "preinstability_eligibility.csv", index=False)

    prediction_store: dict[tuple[str, str], list[dict[tuple[float, int], np.ndarray]]] = {}
    for i, seed in enumerate(seed_values):
        print(f"Training seed {i+1}/{len(seed_values)} ({seed})", flush=True)
        for mode, pool, target, val_draw in (("all_phase", all_pool, p_all, val_all), ("heldout_phase", candidate_pool, p_heldout, val_candidate)):
            model, history = base.train_regression_model(
                features=feature_array, system_rows=system_rows, stable_labels=stable,
                train_indices=train_indices, val_indices=val_indices, phase_pool=pool,
                val_draws=val_draw, targets=target, conditions=CONDITIONS, mean=mean, std=std,
                device=device, args=args, seed=seed+(0 if mode == "all_phase" else 200000),
                model_name=f"deep_sets_{mode}",
                val_singleton_sides=val_singleton_sides_all if mode == "all_phase" else val_singleton_sides_candidate,
            )
            pd.DataFrame(history).to_csv(args.output_dir / f"train_log_deep_sets_{mode}_seed{seed}.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "seed": seed, "mode": mode}, args.output_dir / f"deep_sets_{mode}_seed{seed}.pt")
            draw_pool = draws_all if mode == "all_phase" else draws_candidate
            prediction_store.setdefault(("deep_sets", mode), []).append(
                predict_grid(model, base.predict_regression_condition, feature_array, system_rows, test_indices, draw_pool,
                             singleton_sides_all if mode == "all_phase" else singleton_sides_candidate,
                             mean, std, device, args)
            )
        baseline, history = base.train_baseline_model(
            features=feature_array, system_rows=system_rows, stable_labels=stable,
            train_indices=train_indices, val_indices=val_indices, val_draws_all=val_all,
            p_all=p_all, conditions=CONDITIONS, mean=mean, std=std, device=device,
            args=args, seed=seed+400000, val_singleton_sides=val_singleton_sides_all,
        )
        pd.DataFrame(history).to_csv(args.output_dir / f"train_log_mean_probability_seed{seed}.csv", index=False)
        torch.save({"state_dict": baseline.state_dict(), "seed": seed}, args.output_dir / f"mean_probability_seed{seed}.pt")
        for mode, draw_pool in (("all_phase", draws_all), ("heldout_phase", draws_candidate)):
            prediction_store.setdefault(("mean_probability", mode), []).append(
                predict_grid(baseline, base.predict_baseline_condition, feature_array, system_rows, test_indices, draw_pool,
                             singleton_sides_all if mode == "all_phase" else singleton_sides_candidate,
                             mean, std, device, args)
            )

    targets = {"all_phase": p_all[test_indices], "heldout_phase": p_heldout[test_indices]}
    standard: dict[tuple[str, str], dict[tuple[float, int], np.ndarray]] = {}
    for model_target, seeds in prediction_store.items():
        standard[model_target] = {key: np.stack([seed[key] for seed in seeds]).mean(axis=(0, 1)) for key in seeds[0]}
    pre_predictions: dict[tuple[str, tuple[float, int]], np.ndarray] = {}
    for model in ("deep_sets", "mean_probability"):
        seed_preds = prediction_store[(model, "heldout_phase")]
        for condition in CONDITIONS:
            key = condition_key(condition)
            stack = np.stack([seed[key] for seed in seed_preds])
            mask = pre_masks[key]
            denominator = mask.sum(axis=0) * len(seed_preds)
            total = (stack * mask[None, :, :]).sum(axis=(0, 1))
            pre_predictions[(model, key)] = np.divide(total, denominator, out=np.full(test_indices.size, np.nan), where=denominator > 0)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    seed_base = args.seed + 707071
    for (model, target_name), cells in standard.items():
        target = targets[target_name]
        for condition in CONDITIONS:
            key = condition_key(condition)
            pred = cells[key]
            metric_rows.append(_valid_metric_row(
                model=model, target_name=target_name, control="standard", condition=condition,
                target=target, prediction=pred, seed=seed_base+int(round(condition.k*1000))+condition.t+(0 if model=="deep_sets" else 100)+(0 if target_name=="all_phase" else 200),
            ))
            for idx, test_idx in enumerate(test_indices):
                pred_rows.append({
                    "system_id": int(system_ids[test_idx]), "model": model, "target": target_name,
                    "control": "standard", "M_pairs": condition.k, "K": condition.k,
                    "trajectories": 2*condition.k, "T": condition.t, "stored_points": condition.budget,
                    "target_value": float(target[idx]), "prediction": float(pred[idx]),
                    "absolute_error": float(abs(target[idx]-pred[idx])), "eligible_draws": args.phase_draws,
                    "ma_margin": float(ma_margin[test_idx]),
                    "ma_group": "near" if abs(ma_margin[test_idx])<0.1 else "far" if abs(ma_margin[test_idx])>=0.3 else "middle",
                })
    for model in ("deep_sets", "mean_probability"):
        target = targets["heldout_phase"]
        for condition in CONDITIONS:
            key = condition_key(condition)
            pred = pre_predictions[(model, key)]
            metric_rows.append(_valid_metric_row(
                model=model, target_name="heldout_phase", control="heldout_preinstability", condition=condition,
                target=target, prediction=pred, seed=seed_base+int(round(condition.k*2000))+condition.t+(300 if model=="deep_sets" else 400),
            ))
            for idx, test_idx in enumerate(test_indices):
                if not np.isfinite(pred[idx]):
                    continue
                pred_rows.append({
                    "system_id": int(system_ids[test_idx]), "model": model, "target": "heldout_phase",
                    "control": "heldout_preinstability", "M_pairs": condition.k, "K": condition.k,
                    "trajectories": 2*condition.k, "T": condition.t, "stored_points": condition.budget,
                    "target_value": float(target[idx]), "prediction": float(pred[idx]),
                    "absolute_error": float(abs(target[idx]-pred[idx])), "eligible_draws": int(pre_masks[key][:,idx].sum()),
                    "ma_margin": float(ma_margin[test_idx]),
                    "ma_group": "near" if abs(ma_margin[test_idx])<0.1 else "far" if abs(ma_margin[test_idx])>=0.3 else "middle",
                })
    metrics = pd.DataFrame(metric_rows)
    comparisons: list[dict[str, Any]] = []
    for (model, target_name), cells in standard.items():
        target = targets[target_name]
        for t in PREFIXES:
            rows = []
            ps = []
            half = BudgetCondition(0.5, t, 2)
            one = BudgetCondition(1, t, 2)
            row = _comparison_row(
                model=model, target_name=target_name, control="standard",
                effect="pair_completion_at_fixed_T", fixed_value=t,
                reference=half, comparison=one, target=target,
                reference_prediction=cells[(0.5, t)], comparison_prediction=cells[(1.0, t)],
                seed=seed_base+50000+t,
            )
            rows.append(row); ps.append(row["p_raw_bootstrap"])
            for m in (2, 4, 8):
                comp = BudgetCondition(m, t, 2)
                row = _comparison_row(
                    model=model, target_name=target_name, control="standard",
                    effect="breadth_at_fixed_T", fixed_value=t,
                    reference=one, comparison=comp, target=target,
                    reference_prediction=cells[(1.0, t)], comparison_prediction=cells[(float(m), t)],
                    seed=seed_base+10000+t*10+m,
                )
                rows.append(row); ps.append(row["p_raw_bootstrap"])
            for row, adj in zip(rows, base.holm_adjust(ps)):
                row["p_holm"] = adj
                comparisons.append(row)
        for m in PAIR_COUNTS:
            rows = []
            ps = []
            ref = BudgetCondition(m, 32, 2)
            for t in (64, 128, 256):
                comp = BudgetCondition(m, t, 2)
                row = _comparison_row(
                    model=model, target_name=target_name, control="standard",
                    effect="depth_at_fixed_M", fixed_value=m, reference=ref,
                    comparison=comp, target=target, reference_prediction=cells[(m, 32)],
                    comparison_prediction=cells[(m, t)], seed=seed_base+20000+int(round(m*100))+t,
                )
                rows.append(row); ps.append(row["p_raw_bootstrap"])
            for row, adj in zip(rows, base.holm_adjust(ps)):
                row["p_holm"] = adj
                comparisons.append(row)
    for model in ("deep_sets", "mean_probability"):
        cells = {condition_key(c): pre_predictions[(model, condition_key(c))] for c in CONDITIONS}
        target = targets["heldout_phase"]
        for t in PREFIXES:
            rows = []
            ps = []
            row = _comparison_row(
                model=model, target_name="heldout_phase", control="heldout_preinstability",
                effect="pair_completion_at_fixed_T", fixed_value=t,
                reference=BudgetCondition(0.5, t, 2), comparison=BudgetCondition(1, t, 2),
                target=target, reference_prediction=cells[(0.5, t)], comparison_prediction=cells[(1.0, t)],
                seed=seed_base+30000+t,
            )
            rows.append(row); ps.append(row["p_raw_bootstrap"])
            for m in (2, 4, 8):
                row = _comparison_row(
                    model=model, target_name="heldout_phase", control="heldout_preinstability",
                    effect="breadth_at_fixed_T", fixed_value=t,
                    reference=BudgetCondition(1, t, 2), comparison=BudgetCondition(m, t, 2),
                    target=target, reference_prediction=cells[(1.0, t)], comparison_prediction=cells[(float(m), t)],
                    seed=seed_base+40000+t*10+m,
                )
                rows.append(row); ps.append(row["p_raw_bootstrap"])
            for row, adj in zip(rows, base.holm_adjust(ps)):
                row["p_holm"] = adj
                comparisons.append(row)
        for m in PAIR_COUNTS:
            rows = []
            ps = []
            for t in (64, 128, 256):
                row = _comparison_row(
                    model=model, target_name="heldout_phase", control="heldout_preinstability",
                    effect="depth_at_fixed_M", fixed_value=m,
                    reference=BudgetCondition(m, 32, 2), comparison=BudgetCondition(m, t, 2),
                    target=target, reference_prediction=cells[(m, 32)], comparison_prediction=cells[(m, t)],
                    seed=seed_base+60000+int(round(m*100))+t,
                )
                rows.append(row); ps.append(row["p_raw_bootstrap"])
            for row, adj in zip(rows, base.holm_adjust(ps)):
                row["p_holm"] = adj
                comparisons.append(row)

    prediction_frame=pd.DataFrame(pred_rows)
    comparison_frame=pd.DataFrame(comparisons)
    metrics.to_csv(args.output_dir/"factorial_metrics.csv",index=False)
    prediction_frame.to_csv(args.output_dir/"per_system_predictions.csv",index=False)
    comparison_frame.to_csv(args.output_dir/"factorial_comparisons.csv",index=False)
    base.save_json(args.output_dir/"metrics.json",{"primary_metric":"system-level MAE","bootstrap_unit":"base system","bootstrap_replicates":args.bootstrap_replicates,"pair_draws":args.phase_draws,"models":sorted(prediction_frame.model.unique().tolist())})
    write_figures(args.output_dir,metrics)
    config["status"]="complete_preliminary_single_seed" if args.train_seeds==1 else "complete"
    config["models_saved"]=True
    base.save_json(args.output_dir/"config.json",config)
    summarize_grid(args.output_dir,metrics,comparison_frame,config)
    print(f"Factorial experiment complete: {args.output_dir}",flush=True)


_BOOTSTRAPS=2000
if __name__ == "__main__":
    main()
