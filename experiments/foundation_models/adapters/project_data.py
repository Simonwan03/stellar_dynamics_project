from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

RESULT_RE = re.compile(r"results_batch_(\d+)_sampling_(\d+)\.hdf5$")
ICS_RE = re.compile(r"ICs_sampling_(\d+)\.hdf5$")
LABEL_RE = re.compile(r"completed_systems_sampling_(\d+)\.txt$")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_SIZE = 8


@dataclass
class DatasetChoice:
    root: Path
    sampling_id: int
    result_paths: list[Path]
    label_path: Path | None
    ics_path: Path | None
    local_batch_ids: list[int]
    label_batch_ids: list[int]
    label_row_count: int
    local_result_row_count: int
    time_unbound_max: float | None
    invalid_label_count: int
    time_length: int | None
    feature_shapes: dict[str, list[int]]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def experiment_path(*parts: str) -> Path:
    return EXPERIMENT_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(to_jsonable(data), f, indent=2)


def scan_local_datasets() -> list[DatasetChoice]:
    result_groups: dict[tuple[Path, int], list[tuple[int, Path]]] = {}
    ics_paths: dict[tuple[Path, int], Path] = {}
    label_paths: dict[tuple[Path, int], Path] = {}

    for path in REPO_ROOT.rglob("*.hdf5"):
        if ".git" in path.parts or "experiments" in path.parts:
            continue
        result_match = RESULT_RE.fullmatch(path.name)
        if result_match:
            batch_id = int(result_match.group(1))
            sampling_id = int(result_match.group(2))
            result_groups.setdefault((path.parent, sampling_id), []).append((batch_id, path))
            continue
        ics_match = ICS_RE.fullmatch(path.name)
        if ics_match:
            sampling_id = int(ics_match.group(1))
            ics_paths[(path.parent, sampling_id)] = path

    for path in REPO_ROOT.rglob("*.txt"):
        if ".git" in path.parts or "experiments" in path.parts:
            continue
        label_match = LABEL_RE.fullmatch(path.name)
        if label_match:
            sampling_id = int(label_match.group(1))
            label_paths[(path.parent, sampling_id)] = path

    choices: list[DatasetChoice] = []
    for key, batch_and_paths in sorted(result_groups.items(), key=lambda item: (item[0][0].name, item[0][1])):
        root, sampling_id = key
        batch_and_paths = sorted(batch_and_paths, key=lambda x: x[0])
        result_paths = [p for _, p in batch_and_paths]
        local_batch_ids = [b for b, _ in batch_and_paths]
        label_path = label_paths.get(key)
        ics_path = ics_paths.get(key)
        label_df = _read_label_file(label_path) if label_path else None
        first_info = inspect_result_file(result_paths[0]) if result_paths else {}
        local_result_rows = 0
        if result_paths:
            first_target = first_info.get("positions_complete") or first_info.get("aratios_complete")
            if first_target:
                local_result_rows = len(result_paths) * int(first_target[0])
        choices.append(
            DatasetChoice(
                root=root,
                sampling_id=sampling_id,
                result_paths=result_paths,
                label_path=label_path,
                ics_path=ics_path,
                local_batch_ids=local_batch_ids,
                label_batch_ids=sorted(label_df["num_batch"].unique().tolist()) if label_df is not None else [],
                label_row_count=0 if label_df is None else int(len(label_df)),
                local_result_row_count=local_result_rows,
                time_unbound_max=None if label_df is None else float(label_df["time_unbound"].max()),
                invalid_label_count=0
                if label_df is None
                else int(((label_df["time_unbound"] < 0) | (label_df["diagnosis"] != 0)).sum()),
                time_length=None
                if "times_complete" not in first_info
                else int(first_info["times_complete"][1]),
                feature_shapes={k: v for k, v in first_info.items()},
            )
        )
    return choices


def choose_default_dataset(preferred_sampling_ids: list[int] | None = None) -> DatasetChoice:
    choices = scan_local_datasets()
    if not choices:
        raise FileNotFoundError("No local result shards were found in the repository.")
    preferred_sampling_ids = preferred_sampling_ids or [1, 0]
    for sampling_id in preferred_sampling_ids:
        candidates = [c for c in choices if c.sampling_id == sampling_id]
        if candidates:
            return sorted(candidates, key=lambda c: (-len(c.result_paths), c.root.name))[0]
    return sorted(choices, key=lambda c: (-len(c.result_paths), -c.sampling_id))[0]


def choose_dataset_from_root(root: str | Path, sampling_id: int) -> DatasetChoice:
    """Resolve one dataset explicitly, including when ``root`` is a symlink."""
    root = Path(root).resolve()
    result_paths = sorted(root.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if not result_paths:
        raise FileNotFoundError(f"No result shards for sampling_id={sampling_id} under {root}.")
    batch_ids = []
    for path in result_paths:
        match = RESULT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected result shard name: {path.name}")
        batch_ids.append(int(match.group(1)))
    label_path = root / f"completed_systems_sampling_{sampling_id}.txt"
    ics_path = root / f"ICs_sampling_{sampling_id}.hdf5"
    label_df = _read_label_file(label_path if label_path.exists() else None)
    first_info = inspect_result_file(result_paths[0])
    first_target = first_info.get("positions_complete") or first_info.get("aratios_complete")
    return DatasetChoice(
        root=root,
        sampling_id=int(sampling_id),
        result_paths=result_paths,
        label_path=label_path if label_path.exists() else None,
        ics_path=ics_path if ics_path.exists() else None,
        local_batch_ids=batch_ids,
        label_batch_ids=[] if label_df is None else sorted(label_df["num_batch"].unique().tolist()),
        label_row_count=0 if label_df is None else int(len(label_df)),
        local_result_row_count=0 if first_target is None else len(result_paths) * int(first_target[0]),
        time_unbound_max=None if label_df is None else float(label_df["time_unbound"].max()),
        invalid_label_count=0
        if label_df is None
        else int(((label_df["time_unbound"] < 0) | (label_df["diagnosis"] != 0)).sum()),
        time_length=None if "times_complete" not in first_info else int(first_info["times_complete"][1]),
        feature_shapes=first_info,
    )


def inspect_result_file(path: Path) -> dict[str, list[int]]:
    info: dict[str, list[int]] = {}
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if isinstance(f[key], h5py.Dataset):
                info[key] = list(f[key].shape)
    return info


def _read_label_file(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "num_batch", "time_unbound", "aux_value", "diagnosis"],
    )
    return df.astype(
        {
            "index": "int64",
            "num_batch": "int64",
            "time_unbound": "float32",
            "aux_value": "float32",
            "diagnosis": "int64",
        }
    )


def load_labels(
    choice: DatasetChoice,
    max_rows: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> pd.DataFrame:
    if choice.label_path is None:
        raise FileNotFoundError(f"No label file found for sampling_id={choice.sampling_id} under {choice.root}")
    labels = _read_label_file(choice.label_path)
    assert labels is not None
    labels = labels[labels["num_batch"].isin(choice.local_batch_ids)].copy()
    labels = labels.sort_values(["num_batch", "index"], ascending=True).reset_index(drop=True)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    labels["raw_case_id"] = labels["index"] // group_size
    if max_rows is not None:
        labels = labels.iloc[:max_rows].copy().reset_index(drop=True)
    return labels


def required_keys_for_representation(representation: str) -> list[str]:
    representation = representation.lower()
    if representation == "orbital15":
        return [
            "aratios_complete",
            "mratios_complete",
            "eccs_complete",
            "incs_complete",
            "APs_complete",
            "LANs_complete",
        ]
    if representation == "raw_state18":
        return ["positions_complete", "velocities_complete"]
    if representation == "geom9":
        return ["positions_complete", "velocities_complete"]
    if representation in {"full38", "full36"}:
        return [
            "aratios_complete",
            "mratios_complete",
            "masses_complete",
            "eccs_complete",
            "smaxes_complete",
            "energy_complete",
            "angMomentum_complete",
            "APs_complete",
            "LANs_complete",
            "incs_complete",
            "positions_complete",
            "velocities_complete",
        ]
    raise ValueError(f"Unsupported representation: {representation}")


def load_result_arrays(
    choice: DatasetChoice,
    keys: list[str],
    time_steps: int | None = None,
    max_rows: int | None = None,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    shards: list[dict[str, np.ndarray]] = []
    remaining = None if max_rows is None else int(max_rows)
    for result_path in choice.result_paths:
        with h5py.File(result_path, "r") as f:
            row_count = int(f[keys[0]].shape[0])
            row_sl = slice(None) if remaining is None else slice(0, min(remaining, row_count))
            time_sl = slice(None) if time_steps is None else slice(0, time_steps)
            shard = {}
            for key in keys:
                if key not in f:
                    raise KeyError(f"{key} missing from {result_path}")
                if key == "times_complete":
                    shard[key] = f[key][row_sl, time_sl].astype(dtype)
                else:
                    shard[key] = f[key][row_sl, time_sl, ...].astype(dtype)
            shards.append(shard)
        if remaining is not None:
            remaining -= int(shard[keys[0]].shape[0])
            if remaining <= 0:
                break
    if not shards:
        raise ValueError("No result shards were loaded.")
    return {key: np.concatenate([shard[key] for shard in shards], axis=0) for key in keys}


def sincos(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


def pairwise_geom(pos: np.ndarray, vel: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x1, x2, x3 = pos[:, :, 0, :], pos[:, :, 1, :], pos[:, :, 2, :]
    v1, v2, v3 = vel[:, :, 0, :], vel[:, :, 1, :], vel[:, :, 2, :]

    d12 = x1 - x2
    d13 = x1 - x3
    d23 = x2 - x3
    u12 = v1 - v2
    u13 = v1 - v3
    u23 = v2 - v3

    r12 = np.linalg.norm(d12, axis=-1)
    r13 = np.linalg.norm(d13, axis=-1)
    r23 = np.linalg.norm(d23, axis=-1)
    v12 = np.linalg.norm(u12, axis=-1)
    v13 = np.linalg.norm(u13, axis=-1)
    v23 = np.linalg.norm(u23, axis=-1)
    rdot12 = np.sum(d12 * u12, axis=-1) / (r12 + eps)
    rdot13 = np.sum(d13 * u13, axis=-1) / (r13 + eps)
    rdot23 = np.sum(d23 * u23, axis=-1) / (r23 + eps)

    return np.stack([r12, r13, r23, v12, v13, v23, rdot12, rdot13, rdot23], axis=-1).astype(np.float32)


def build_representation(data: dict[str, np.ndarray], representation: str, time_steps: int | None = None) -> np.ndarray:
    representation = representation.lower()
    if time_steps is not None:
        data = {k: v[:, :time_steps].copy() for k, v in data.items()}

    if representation == "orbital15":
        ar = data["aratios_complete"].astype(np.float32)
        mr = data["mratios_complete"].astype(np.float32)
        ec = data["eccs_complete"].astype(np.float32)
        inc = data["incs_complete"].astype(np.float32)
        ap = sincos(data["APs_complete"])
        lan = sincos(data["LANs_complete"])
        x = np.concatenate([ar, mr, ec, inc, ap, lan], axis=-1)
        if x.shape[-1] != 15:
            raise ValueError(f"orbital15 expected 15 channels, got {x.shape[-1]}")
        return x.astype(np.float32)

    if representation == "raw_state18":
        pos = data["positions_complete"].astype(np.float32).reshape(data["positions_complete"].shape[0], data["positions_complete"].shape[1], -1)
        vel = data["velocities_complete"].astype(np.float32).reshape(data["velocities_complete"].shape[0], data["velocities_complete"].shape[1], -1)
        x = np.concatenate([pos, vel], axis=-1)
        if x.shape[-1] != 18:
            raise ValueError(f"raw_state18 expected 18 channels, got {x.shape[-1]}")
        return x.astype(np.float32)

    if representation == "geom9":
        return pairwise_geom(data["positions_complete"].astype(np.float32), data["velocities_complete"].astype(np.float32))

    if representation in {"full38", "full36"}:
        n = data["aratios_complete"].shape[0]
        t = data["aratios_complete"].shape[1]
        feats = [
            data["aratios_complete"].astype(np.float32),
            data["mratios_complete"].astype(np.float32),
            data["masses_complete"].astype(np.float32),
            data["eccs_complete"].astype(np.float32),
            data["smaxes_complete"].astype(np.float32),
            data["energy_complete"].astype(np.float32),
            data["angMomentum_complete"].astype(np.float32),
            sincos(data["APs_complete"]),
            sincos(data["LANs_complete"]),
            sincos(data["incs_complete"]),
            pairwise_geom(data["positions_complete"].astype(np.float32), data["velocities_complete"].astype(np.float32)),
            np.repeat(np.linspace(0.0, 1.0, t, dtype=np.float32)[None, :, None], n, axis=0),
        ]
        return np.concatenate(feats, axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported representation: {representation}")


def clean_and_group_trajectories(
    X: np.ndarray,
    labels: pd.DataFrame,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(X) != len(labels):
        raise ValueError(f"Feature rows and label rows differ: {len(X)} vs {len(labels)}")
    labels = labels.copy()
    valid_label_mask = (labels["time_unbound"] >= 0) & (labels["diagnosis"] == 0)
    finite_mask = np.isfinite(X).all(axis=(1, 2))
    keep_mask = valid_label_mask.to_numpy() & finite_mask
    X = X[keep_mask]
    labels = labels.loc[keep_mask].copy().reset_index(drop=True)

    case_counts = labels["raw_case_id"].value_counts().sort_index()
    complete_case_ids = case_counts[case_counts == group_size].index.to_numpy()
    complete_mask = labels["raw_case_id"].isin(complete_case_ids).to_numpy()
    X = X[complete_mask]
    labels = labels.loc[complete_mask].copy().reset_index(drop=True)

    labels["case_id"], _ = pd.factorize(labels["raw_case_id"], sort=True)
    labels["perturb_id"] = labels.groupby("case_id").cumcount().astype(int)
    return X.astype(np.float32), labels


def case_summary(labels: pd.DataFrame, group_size: int = DEFAULT_GROUP_SIZE) -> pd.DataFrame:
    max_time = float(labels["time_unbound"].max())
    grouped = labels.groupby("case_id", sort=True)
    summary = grouped.agg(
        batch_id=("num_batch", "first"),
        raw_case_id=("raw_case_id", "first"),
        num_traj=("case_id", "size"),
        stable_count=("time_unbound", lambda s: int(np.sum(np.isclose(s.to_numpy(), max_time)))),
        max_time_unbound=("time_unbound", "max"),
    )
    summary["stable_fraction"] = summary["stable_count"] / float(group_size)
    summary["tail_label"] = summary["stable_count"].map(
        lambda c: "allstable" if c == group_size else ("allunstable" if c == 0 else f"mixed{c}of{group_size}")
    )
    return summary.reset_index()


def stratified_case_split(
    case_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    rng = np.random.default_rng(seed)
    case_df = case_df.copy()
    case_df["split"] = ""
    for _, sub_df in case_df.groupby("tail_label", sort=True):
        case_ids = sub_df["case_id"].to_numpy().copy()
        rng.shuffle(case_ids)
        n = len(case_ids)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_test = min(n_test, n)
        n_val = min(n_val, max(0, n - n_test - 1))
        test_ids = case_ids[:n_test]
        val_ids = case_ids[n_test : n_test + n_val]
        train_ids = case_ids[n_test + n_val :]
        case_df.loc[case_df["case_id"].isin(train_ids), "split"] = "train"
        case_df.loc[case_df["case_id"].isin(val_ids), "split"] = "val"
        case_df.loc[case_df["case_id"].isin(test_ids), "split"] = "test"
    if (case_df["split"] == "").any():
        raise RuntimeError("Some cases were not assigned to a split.")
    return case_df


def build_export_table(
    labels: pd.DataFrame,
    case_df: pd.DataFrame,
    choice: DatasetChoice,
    representation: str,
) -> pd.DataFrame:
    merged = labels.merge(case_df[["case_id", "split", "tail_label", "stable_count"]], on="case_id", how="left")
    merged["sampling_id"] = choice.sampling_id
    merged["representation"] = representation
    merged["system_name"] = merged.apply(
        lambda row: f"s{choice.sampling_id}_b{int(row['num_batch']):02d}_{row['tail_label']}",
        axis=1,
    )
    return merged


def get_export_root(config: dict[str, Any], representation: str | None = None) -> Path:
    root = experiment_path(config["export_root"])
    representation = representation or config.get("representation", "orbital15")
    if representation != "orbital15":
        root = root.parent / f"{root.name}_{representation}"
    return root


def convert_to_arrow_file(path: Path, trajectory_channels_first: np.ndarray) -> None:
    from gluonts.dataset.arrow import ArrowWriter

    if trajectory_channels_first.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, time), got {trajectory_channels_first.shape}")
    dataset = [{"start": datetime.now().strftime("%Y-%m-%d %H:%M"), "target": trajectory_channels_first}]
    ensure_dir(path.parent)
    ArrowWriter(compression="lz4").write_to_file(dataset, path=path)


def export_panda_arrow_dataset(
    X: np.ndarray,
    export_df: pd.DataFrame,
    export_root: Path,
) -> dict[str, Path]:
    if len(X) != len(export_df):
        raise ValueError(f"Trajectory rows and manifest rows differ: {len(X)} vs {len(export_df)}")
    manifest_dir = ensure_dir(export_root / "manifests")
    sample_counters: dict[tuple[str, str], int] = {}
    relpaths: list[str] = []
    absolute_paths: list[str] = []

    for idx, row in export_df.reset_index(drop=True).iterrows():
        split = str(row["split"])
        system_name = str(row["system_name"])
        key = (split, system_name)
        sample_idx = sample_counters.get(key, 0)
        sample_counters[key] = sample_idx + 1
        file_path = export_root / split / system_name / f"{sample_idx}_T-{X.shape[1]}.arrow"
        convert_to_arrow_file(file_path, X[idx].T)
        relpaths.append(str(file_path.relative_to(export_root)))
        absolute_paths.append(str(file_path))

    export_df = export_df.copy()
    export_df["arrow_relpath"] = relpaths
    export_df["arrow_abspath"] = absolute_paths

    paths = {}
    for split, sub_df in export_df.groupby("split", sort=True):
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        sub_df.to_csv(manifest_path, index=False)
        paths[str(split)] = manifest_path

    summary_path = manifest_dir / "dataset_summary.json"
    summary = {
        "export_root": str(export_root),
        "n_trajectories": int(len(export_df)),
        "n_cases": int(export_df["case_id"].nunique()),
        "splits": export_df["split"].value_counts().sort_index().to_dict(),
        "systems_per_split": {
            split: sorted(sub_df["system_name"].unique().tolist()) for split, sub_df in export_df.groupby("split", sort=True)
        },
        "channels": int(X.shape[-1]),
        "time_steps": int(X.shape[1]),
    }
    write_json(summary_path, summary)
    paths["summary"] = summary_path
    return paths


def summarize_audit(choice: DatasetChoice) -> dict[str, Any]:
    return {
        "root": str(choice.root),
        "sampling_id": choice.sampling_id,
        "label_path": None if choice.label_path is None else str(choice.label_path),
        "ics_path": None if choice.ics_path is None else str(choice.ics_path),
        "result_paths": [str(p) for p in choice.result_paths],
        "local_batch_ids": choice.local_batch_ids,
        "label_row_count": choice.label_row_count,
        "local_result_row_count": choice.local_result_row_count,
        "time_unbound_max": choice.time_unbound_max,
        "invalid_label_count": choice.invalid_label_count,
        "time_length": choice.time_length,
        "feature_shapes": choice.feature_shapes,
    }


def write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)
