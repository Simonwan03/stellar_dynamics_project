"""Algebraic stability criteria for hierarchical triples.

This is a direct implementation of Equation (2)::

    R_p,crit / a_in = 2.8 [
        (1 + q_out) (1 + e_out) / sqrt(1 - e_out)
    ]^(2/5) (1 - 0.3 i_mut / pi)

where ``R_p = a_out (1 - e_out)``.  This module also implements the updated
Vynatheya et al. (2022) Equation (4), which adds inner-eccentricity and
non-linear mutual-inclination corrections.  Both criteria report a log ratio
whose strict positivity indicates stability.

Run from the repository root::

    conda run -n stellar python -m tasks.common.traditional_stability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from tasks.common.analyze_divergence import load_completion_table, select_valid_rows


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ICS = ROOT / "dataset_v3" / "ICs_sampling_2.hdf5"
DEFAULT_COMPLETION_TABLE = ROOT / "dataset_v3" / "completed_systems_sampling_2.txt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "traditional_stability"
REQUIRED_KEYS = ("aratios", "eccs", "masses", "h_uvecs")


def triple_stability(
    a_in: np.ndarray | float,
    a_out: np.ndarray | float,
    e_out: np.ndarray | float,
    mutual_inclination: np.ndarray | float,
    m1: np.ndarray | float,
    m2: np.ndarray | float,
    m3: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Evaluate paper Equation (2), with NumPy broadcasting support."""
    arrays = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=np.float64)
            for value in (a_in, a_out, e_out, mutual_inclination, m1, m2, m3)
        ]
    )
    a_in_array, a_out_array, e_out_array, inclination, mass1, mass2, mass3 = arrays

    if np.any(~np.isfinite(np.stack(arrays))):
        raise ValueError("Criterion inputs must all be finite.")
    if np.any(a_in_array <= 0.0) or np.any(a_out_array <= 0.0):
        raise ValueError("Semi-major axes must be positive.")
    if np.any((e_out_array < 0.0) | (e_out_array >= 1.0)):
        raise ValueError("Outer eccentricity must satisfy 0 <= e_out < 1.")
    if np.any((inclination < 0.0) | (inclination > np.pi)):
        raise ValueError("Mutual inclination must be in radians and satisfy 0 <= i_mut <= pi.")
    if np.any(mass1 <= 0.0) or np.any(mass2 <= 0.0) or np.any(mass3 <= 0.0):
        raise ValueError("All masses must be positive.")

    q_out = mass3 / (mass1 + mass2)
    a_ratio_actual = a_out_array / a_in_array
    rp_over_ain_actual = a_ratio_actual * (1.0 - e_out_array)
    rp_crit_over_ain = (
        2.8
        * np.power(
            (1.0 + q_out) * (1.0 + e_out_array) / np.sqrt(1.0 - e_out_array),
            2.0 / 5.0,
        )
        * (1.0 - 0.3 * inclination / np.pi)
    )
    a_ratio_crit = rp_crit_over_ain / (1.0 - e_out_array)
    margin = np.log(rp_over_ain_actual / rp_crit_over_ain)
    return {
        "q_out": q_out,
        "a_out_over_a_in": a_ratio_actual,
        "Rp_over_a_in": rp_over_ain_actual,
        "Rp_crit_over_a_in": rp_crit_over_ain,
        "a_out_over_a_in_crit": a_ratio_crit,
        "stability_margin": margin,
        "distance_to_boundary": np.abs(margin),
        "stable": margin > 0.0,
    }


def vynatheya2022_stability(
    a_in: np.ndarray | float,
    a_out: np.ndarray | float,
    e_in: np.ndarray | float,
    e_out: np.ndarray | float,
    mutual_inclination: np.ndarray | float,
    m1: np.ndarray | float,
    m2: np.ndarray | float,
    m3: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Evaluate Vynatheya et al. (2022) Equations (3)--(4).

    The quadrupole LK estimate for ``e_in_max`` is real only inside the
    classical LK inclination window. Outside that window the radicand is
    clipped to zero, corresponding to no LK-driven eccentricity correction.
    """

    arrays = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=np.float64)
            for value in (a_in, a_out, e_in, e_out, mutual_inclination, m1, m2, m3)
        ]
    )
    (
        a_in_array,
        a_out_array,
        e_in_array,
        e_out_array,
        inclination,
        mass1,
        mass2,
        mass3,
    ) = arrays

    if np.any(~np.isfinite(np.stack(arrays))):
        raise ValueError("Criterion inputs must all be finite.")
    if np.any(a_in_array <= 0.0) or np.any(a_out_array <= 0.0):
        raise ValueError("Semi-major axes must be positive.")
    if np.any((e_in_array < 0.0) | (e_in_array >= 1.0)):
        raise ValueError("Inner eccentricity must satisfy 0 <= e_in < 1.")
    if np.any((e_out_array < 0.0) | (e_out_array >= 1.0)):
        raise ValueError("Outer eccentricity must satisfy 0 <= e_out < 1.")
    if np.any((inclination < 0.0) | (inclination > np.pi)):
        raise ValueError("Mutual inclination must be in radians and satisfy 0 <= i_mut <= pi.")
    if np.any(mass1 <= 0.0) or np.any(mass2 <= 0.0) or np.any(mass3 <= 0.0):
        raise ValueError("All masses must be positive.")

    q_out = mass3 / (mass1 + mass2)
    lk_radicand = 1.0 - (5.0 / 3.0) * np.cos(inclination) ** 2
    e_in_max = np.sqrt(np.maximum(lk_radicand, 0.0))
    e_in_avg = 0.5 * e_in_max**2
    e_in_tilde = np.maximum(e_in_array, e_in_avg)

    a_ratio_actual = a_out_array / a_in_array
    y_tilde = a_ratio_actual * (1.0 - e_out_array) / (1.0 + e_in_tilde)
    inclination_factor = (
        ((1.0 - 0.2 * e_in_tilde + e_out_array) / 8.0)
        * (np.cos(inclination) - 1.0)
        + 1.0
    )
    y_crit = (
        2.4
        * np.power(
            (1.0 + q_out) / ((1.0 + e_in_tilde) * np.sqrt(1.0 - e_out_array)),
            2.0 / 5.0,
        )
        * inclination_factor
    )
    a_ratio_crit = y_crit * (1.0 + e_in_tilde) / (1.0 - e_out_array)
    margin = np.log(y_tilde / y_crit)
    return {
        "q_out": q_out,
        "a_out_over_a_in": a_ratio_actual,
        "e_in_max": e_in_max,
        "e_in_avg": e_in_avg,
        "e_in_tilde": e_in_tilde,
        "Y_tilde": y_tilde,
        "Y_tilde_crit": y_crit,
        "inclination_factor": inclination_factor,
        "a_out_over_a_in_crit": a_ratio_crit,
        "stability_margin": margin,
        "distance_to_boundary": np.abs(margin),
        "stable": margin > 0.0,
    }


def mutual_inclination_from_unit_vectors(h_uvecs: np.ndarray) -> np.ndarray:
    vectors = np.asarray(h_uvecs, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[1:] != (2, 3):
        raise ValueError(f"Expected h_uvecs with shape (N, 2, 3), got {vectors.shape}.")
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(~np.isfinite(vectors)) or np.any(norms <= 0.0):
        raise ValueError("h_uvecs must contain finite, non-zero vectors.")
    unit_vectors = vectors / norms[..., None]
    cosine = np.sum(unit_vectors[:, 0] * unit_vectors[:, 1], axis=-1)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def load_initial_conditions(path: Path, group_size: int) -> pd.DataFrame:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle]
        if missing:
            raise KeyError(f"Missing required HDF5 keys in {path}: {missing}")
        row_counts = {key: int(handle[key].shape[0]) for key in REQUIRED_KEYS}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Required HDF5 arrays have inconsistent row counts: {row_counts}")

        stored_aratio = np.asarray(handle["aratios"][:, 0], dtype=np.float64)
        eccentricities = np.asarray(handle["eccs"][:], dtype=np.float64)
        masses = np.asarray(handle["masses"][:], dtype=np.float64)
        mutual_inclination = mutual_inclination_from_unit_vectors(handle["h_uvecs"][:])

    if eccentricities.shape[1:] != (2,):
        raise ValueError(f"Expected eccs with shape (N, 2), got {eccentricities.shape}.")
    if masses.shape[1:] != (3,):
        raise ValueError(f"Expected masses with shape (N, 3), got {masses.shape}.")
    if np.any(stored_aratio <= 0.0):
        raise ValueError("Stored a_in/a_out ratios must be positive.")

    # Only the ratio matters.  Set a_in=1 and a_out=1/stored_aratio.
    criterion = triple_stability(
        a_in=np.ones_like(stored_aratio),
        a_out=1.0 / stored_aratio,
        e_out=eccentricities[:, 1],
        mutual_inclination=mutual_inclination,
        m1=masses[:, 0],
        m2=masses[:, 1],
        m3=masses[:, 2],
    )
    indices = np.arange(stored_aratio.size, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "index": indices,
            "group_id": indices // group_size,
            "variation_id": indices % group_size,
            "e_in": eccentricities[:, 0],
            "e_out": eccentricities[:, 1],
            "mutual_inclination": mutual_inclination,
            **criterion,
        }
    )
    frame["stable"] = frame["stable"].astype(np.int8)
    return frame


def load_vynatheya2022_initial_conditions(path: Path, group_size: int) -> pd.DataFrame:
    """Load initial conditions and evaluate Equation (4) per variation."""

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle]
        if missing:
            raise KeyError(f"Missing required HDF5 keys in {path}: {missing}")
        row_counts = {key: int(handle[key].shape[0]) for key in REQUIRED_KEYS}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Required HDF5 arrays have inconsistent row counts: {row_counts}")

        stored_aratio = np.asarray(handle["aratios"][:, 0], dtype=np.float64)
        eccentricities = np.asarray(handle["eccs"][:], dtype=np.float64)
        masses = np.asarray(handle["masses"][:], dtype=np.float64)
        mutual_inclination = mutual_inclination_from_unit_vectors(handle["h_uvecs"][:])

    if eccentricities.shape[1:] != (2,):
        raise ValueError(f"Expected eccs with shape (N, 2), got {eccentricities.shape}.")
    if masses.shape[1:] != (3,):
        raise ValueError(f"Expected masses with shape (N, 3), got {masses.shape}.")
    if np.any(stored_aratio <= 0.0):
        raise ValueError("Stored a_in/a_out ratios must be positive.")

    criterion = vynatheya2022_stability(
        a_in=np.ones_like(stored_aratio),
        a_out=1.0 / stored_aratio,
        e_in=eccentricities[:, 0],
        e_out=eccentricities[:, 1],
        mutual_inclination=mutual_inclination,
        m1=masses[:, 0],
        m2=masses[:, 1],
        m3=masses[:, 2],
    )
    indices = np.arange(stored_aratio.size, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "index": indices,
            "group_id": indices // group_size,
            "variation_id": indices % group_size,
            "e_in": eccentricities[:, 0],
            "e_out": eccentricities[:, 1],
            "mutual_inclination": mutual_inclination,
            **criterion,
        }
    )
    frame["stable"] = frame["stable"].astype(np.int8)
    return frame


def classification_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(observed, dtype=bool)
    estimate = np.asarray(predicted, dtype=bool)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("Observed and predicted labels must be non-empty arrays with matching shapes.")
    tp = int(np.sum(truth & estimate))
    tn = int(np.sum(~truth & ~estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "samples": int(truth.size),
        "accuracy": float(np.mean(truth == estimate)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def attach_observed_labels(
    predictions: pd.DataFrame,
    completion_table: Path,
    censor_time: float | None,
    group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, Any]]:
    raw_labels = load_completion_table(completion_table)
    valid_labels = select_valid_rows(raw_labels)
    if censor_time is None:
        censor_time = float(valid_labels["time_unbound"].max())

    labels = raw_labels[["index", "time_unbound", "diagnosis"]].copy()
    labels["valid_observation"] = (labels["time_unbound"] >= 0.0) & (labels["diagnosis"] == 0)
    labels["observed_stable"] = np.where(
        labels["valid_observation"],
        (labels["time_unbound"] >= censor_time).astype(np.int8),
        np.nan,
    )
    labels["observed_time_unbound"] = np.where(
        labels["valid_observation"], labels["time_unbound"], np.nan
    )
    merged = predictions.merge(labels, on="index", how="left", validate="one_to_one")
    valid_rows = merged[merged["valid_observation"].fillna(False)].copy()
    individual_metrics = classification_metrics(
        valid_rows["observed_stable"].to_numpy(dtype=bool),
        valid_rows["stable"].to_numpy(dtype=bool),
    )

    prediction_groups = (
        predictions.groupby("group_id", sort=True)
        .agg(
            variations=("variation_id", "size"),
            unique_variations=("variation_id", "nunique"),
            stability_margin=("stability_margin", "min"),
        )
        .reset_index()
    )
    prediction_groups["stable"] = (prediction_groups["stability_margin"] > 0.0).astype(np.int8)
    prediction_groups["distance_to_boundary"] = prediction_groups["stability_margin"].abs()

    label_groups = (
        merged.groupby("group_id", sort=True)
        .agg(
            valid_variations=("valid_observation", "sum"),
            observed_stable=("observed_stable", "min"),
            earliest_time_unbound=("observed_time_unbound", "min"),
        )
        .reset_index()
    )
    groups = prediction_groups.merge(label_groups, on="group_id", how="left", validate="one_to_one")
    groups["complete_observation"] = groups["valid_variations"] == group_size
    valid_groups = groups[groups["complete_observation"]].copy()
    group_metrics = classification_metrics(
        valid_groups["observed_stable"].to_numpy(dtype=bool),
        valid_groups["stable"].to_numpy(dtype=bool),
    )
    return merged, groups, censor_time, {"individual": individual_metrics, "group": group_metrics}


def margin_summary(frame: pd.DataFrame) -> dict[str, Any]:
    margin = frame["stability_margin"].to_numpy(dtype=np.float64)
    return {
        "total": int(len(frame)),
        "predicted_stable": int(frame["stable"].sum()),
        "predicted_unstable": int((frame["stable"] == 0).sum()),
        "predicted_stable_fraction": float(frame["stable"].mean()),
        "margin_percentiles": {
            "p01": float(np.percentile(margin, 1)),
            "p10": float(np.percentile(margin, 10)),
            "p50": float(np.percentile(margin, 50)),
            "p90": float(np.percentile(margin, 90)),
            "p99": float(np.percentile(margin, 99)),
        },
    }


def run(
    input_path: Path,
    output_dir: Path,
    completion_table: Path | None,
    group_size: int,
    censor_time: float | None,
) -> dict[str, Any]:
    predictions = load_initial_conditions(input_path, group_size)
    groups = (
        predictions.groupby("group_id", sort=True)
        .agg(
            variations=("variation_id", "size"),
            stability_margin=("stability_margin", "min"),
        )
        .reset_index()
    )
    groups["stable"] = (groups["stability_margin"] > 0.0).astype(np.int8)
    groups["distance_to_boundary"] = groups["stability_margin"].abs()
    evaluation = None
    resolved_censor_time = censor_time
    if completion_table is not None:
        predictions, groups, resolved_censor_time, evaluation = attach_observed_labels(
            predictions, completion_table, censor_time, group_size
        )

    summary: dict[str, Any] = {
        "input_hdf5": str(input_path.resolve()),
        "completion_table": None if completion_table is None else str(completion_table.resolve()),
        "criterion": "paper Equation (2)",
        "formula": (
            "Rp_crit/a_in = 2.8 * ((1+q_out)*(1+e_out)/sqrt(1-e_out))^(2/5) "
            "* (1-0.3*i_mut/pi)"
        ),
        "decision": "stable iff Rp/a_in > Rp_crit/a_in",
        "inner_eccentricity_correction": False,
        "group_size": int(group_size),
        "group_prediction": "minimum stability margin across variations",
        "censor_time": resolved_censor_time,
        "individual": margin_summary(predictions),
        "group": margin_summary(groups),
        "evaluation": evaluation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "traditional_stability_predictions.csv", index=False)
    groups.to_csv(output_dir / "traditional_stability_group_predictions.csv", index=False)
    with (output_dir / "traditional_stability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate paper Equation (2) triple-stability criterion.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ICS, help="Initial-condition HDF5 file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--completion-table", type=Path, default=DEFAULT_COMPLETION_TABLE)
    parser.add_argument("--no-evaluate", action="store_true", help="Do not compare against completion labels.")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--censor-time",
        type=float,
        default=None,
        help="Observed stability horizon; defaults to the largest valid time_unbound.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        input_path=args.input,
        output_dir=args.output_dir,
        completion_table=None if args.no_evaluate else args.completion_table,
        group_size=args.group_size,
        censor_time=args.censor_time,
    )
    individual = summary["individual"]
    group = summary["group"]
    print(
        f"Individual predictions: {individual['predicted_stable']:,}/{individual['total']:,} stable "
        f"({individual['predicted_stable_fraction']:.2%})"
    )
    print(
        f"Group predictions: {group['predicted_stable']:,}/{group['total']:,} stable "
        f"({group['predicted_stable_fraction']:.2%})"
    )
    if summary["evaluation"] is not None:
        print(f"Individual accuracy: {summary['evaluation']['individual']['accuracy']:.2%}")
        print(f"Group accuracy: {summary['evaluation']['group']['accuracy']:.2%}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
