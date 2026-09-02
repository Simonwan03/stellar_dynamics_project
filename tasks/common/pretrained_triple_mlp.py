"""Adapter for the official Vynatheya et al. pretrained triple MLP.

The upstream artifact is a scikit-learn pickle. Loading pickle can execute
arbitrary code, so this adapter refuses to deserialize it unless its SHA-256
matches one of the known official scikit-learn 1.2.2 artifacts.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from tasks.common.traditional_stability import mutual_inclination_from_unit_vectors


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED_MLP_PATH = (
    ROOT / "external_checkpoints" / "triple_stability" / "mlp_model_trip_v1.2.2.pkl"
)
DEFAULT_PRETRAINED_GHOST_MLP_PATH = (
    ROOT
    / "external_checkpoints"
    / "triple_stability"
    / "mlp_model_trip_ghost_v1.2.2.pkl"
)
PRETRAINED_MLP_SOURCE_URL = (
    "https://raw.githubusercontent.com/pavanvyn/triple-stability/"
    "main/mlp_model_trip_v1.2.2.pkl"
)
PRETRAINED_MLP_SHA256 = "f8bc2f192813145f42683669adaef6448a287523875895343453055b3acf1436"
PRETRAINED_GHOST_MLP_SOURCE_URL = (
    "https://raw.githubusercontent.com/pavanvyn/triple-stability/"
    "main/mlp_model_trip_ghost_v1.2.2.pkl"
)
PRETRAINED_GHOST_MLP_SHA256 = (
    "a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866"
)
PRETRAINED_MLP_TRAINED_SKLEARN_VERSION = "1.2.2"

PRETRAINED_MLP_ARTIFACTS: dict[str, dict[str, str]] = {
    PRETRAINED_MLP_SHA256: {
        "variant": "semimajor_axis_change",
        "source_url": PRETRAINED_MLP_SOURCE_URL,
        "label_definition": (
            "unstable if unbound or either semimajor axis changes by more than 10 percent"
        ),
    },
    PRETRAINED_GHOST_MLP_SHA256: {
        "variant": "ghost_orbit_divergence",
        "source_url": PRETRAINED_GHOST_MLP_SOURCE_URL,
        "label_definition": (
            "unstable if unbound or a nearby ghost orbit diverges beyond the upstream threshold"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pretrained_mlp_features(
    q_in: np.ndarray,
    q_out: np.ndarray,
    alpha: np.ndarray,
    e_in: np.ndarray,
    e_out: np.ndarray,
    mutual_inclination: np.ndarray,
) -> np.ndarray:
    """Build the exact six inputs used by the upstream inference script."""

    arrays = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=np.float64)
            for value in (q_in, q_out, alpha, e_in, e_out, mutual_inclination)
        ]
    )
    if np.any(~np.isfinite(np.stack(arrays))):
        raise ValueError("Pretrained MLP inputs must all be finite.")
    q_in_array, q_out_array, alpha_array, e_in_array, e_out_array, inclination = arrays
    if np.any(q_in_array <= 0.0) or np.any(q_in_array > 1.0):
        raise ValueError("q_in must satisfy 0 < q_in <= 1.")
    if np.any(q_out_array <= 0.0):
        raise ValueError("q_out must be positive.")
    if np.any(alpha_array <= 0.0) or np.any(alpha_array > 1.0):
        raise ValueError("alpha=a_in/a_out must satisfy 0 < alpha <= 1.")
    if np.any((e_in_array < 0.0) | (e_in_array >= 1.0)):
        raise ValueError("e_in must satisfy 0 <= e_in < 1.")
    if np.any((e_out_array < 0.0) | (e_out_array >= 1.0)):
        raise ValueError("e_out must satisfy 0 <= e_out < 1.")
    if np.any((inclination < 0.0) | (inclination > np.pi)):
        raise ValueError("Mutual inclination must satisfy 0 <= i_mut <= pi.")
    return np.column_stack(
        [
            q_in_array.reshape(-1),
            q_out_array.reshape(-1),
            alpha_array.reshape(-1),
            e_in_array.reshape(-1),
            e_out_array.reshape(-1),
            (inclination / np.pi).reshape(-1),
        ]
    )


def load_pretrained_mlp_initial_conditions(
    input_path: Path,
    model_path: Path,
    group_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the verified official pretrained MLP on every IC variation."""

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if not model_path.exists():
        source_url = (
            PRETRAINED_GHOST_MLP_SOURCE_URL
            if "ghost" in model_path.name.lower()
            else PRETRAINED_MLP_SOURCE_URL
        )
        raise FileNotFoundError(
            f"Missing pretrained MLP artifact: {model_path}. Download the official file with:\n"
            f"curl -L {source_url} -o {model_path}"
        )
    artifact_sha256 = sha256_file(model_path)
    artifact = PRETRAINED_MLP_ARTIFACTS.get(artifact_sha256)
    if artifact is None:
        raise ValueError(
            "Refusing to load unverified pretrained MLP pickle: "
            f"expected one of {sorted(PRETRAINED_MLP_ARTIFACTS)}, got {artifact_sha256}."
        )

    required_keys = ("aratios", "eccs", "masses", "h_uvecs")
    with h5py.File(input_path, "r") as handle:
        missing = [key for key in required_keys if key not in handle]
        if missing:
            raise KeyError(f"Missing required HDF5 keys in {input_path}: {missing}")
        row_counts = {key: int(handle[key].shape[0]) for key in required_keys}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Required HDF5 arrays have inconsistent row counts: {row_counts}")
        alpha = np.asarray(handle["aratios"][:, 0], dtype=np.float64)
        eccentricities = np.asarray(handle["eccs"][:], dtype=np.float64)
        masses = np.asarray(handle["masses"][:], dtype=np.float64)
        mutual_inclination = mutual_inclination_from_unit_vectors(handle["h_uvecs"][:])

    if eccentricities.shape[1:] != (2,):
        raise ValueError(f"Expected eccs with shape (N, 2), got {eccentricities.shape}.")
    if masses.shape[1:] != (3,):
        raise ValueError(f"Expected masses with shape (N, 3), got {masses.shape}.")
    inner_min = np.minimum(masses[:, 0], masses[:, 1])
    inner_max = np.maximum(masses[:, 0], masses[:, 1])
    q_in = inner_min / inner_max
    q_out = masses[:, 2] / (masses[:, 0] + masses[:, 1])
    features = build_pretrained_mlp_features(
        q_in,
        q_out,
        alpha,
        eccentricities[:, 0],
        eccentricities[:, 1],
        mutual_inclination,
    )

    # The checksum verification above limits deserialization to the official artifact.
    with model_path.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = pickle.load(handle)
    if int(getattr(model, "n_features_in_", -1)) != 6:
        raise ValueError("Official pretrained MLP must accept exactly six input features.")
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    unstable_columns = np.flatnonzero(classes == 1)
    if probabilities.shape != (features.shape[0], classes.size) or unstable_columns.size != 1:
        raise ValueError(
            f"Unexpected pretrained MLP output shape/classes: {probabilities.shape}, {classes}."
        )
    probability_unstable = probabilities[:, int(unstable_columns[0])]
    probability_stable = 1.0 - probability_unstable
    stable = probability_unstable < 0.5

    in_training_domain = (
        (q_in >= 1e-2)
        & (q_in <= 1.0)
        & (q_out >= 1e-2)
        & (q_out <= 1e2)
        & (alpha >= 1e-4)
        & (alpha <= 1.0)
        & (eccentricities[:, 0] >= 0.0)
        & (eccentricities[:, 0] <= 1.0)
        & (eccentricities[:, 1] >= 0.0)
        & (eccentricities[:, 1] <= 1.0)
    )
    indices = np.arange(features.shape[0], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "index": indices,
            "group_id": indices // group_size,
            "variation_id": indices % group_size,
            "q_in": q_in,
            "q_out": q_out,
            "alpha": alpha,
            "e_in": eccentricities[:, 0],
            "e_out": eccentricities[:, 1],
            "mutual_inclination": mutual_inclination,
            "probability_stable": probability_stable,
            "probability_unstable": probability_unstable,
            "stable": stable.astype(np.int8),
            "in_training_domain": in_training_domain,
        }
    )
    metadata: dict[str, Any] = {
        "artifact_path": str(model_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "variant": artifact["variant"],
        "source_url": artifact["source_url"],
        "upstream_label_definition": artifact["label_definition"],
        "trained_sklearn_version": PRETRAINED_MLP_TRAINED_SKLEARN_VERSION,
        "runtime_sklearn_version": sklearn.__version__,
        "input_order": ["q_in", "q_out", "alpha", "e_in", "e_out", "i_mut/pi"],
        "class_1_semantics": "unstable",
        "variation_decision": "stable iff probability_unstable < 0.5",
        "rows": int(len(frame)),
        "rows_inside_training_domain": int(in_training_domain.sum()),
        "rows_outside_training_domain": int((~in_training_domain).sum()),
    }
    return frame, metadata
