"""Check the paper datasets before launching expensive A-F reproduction runs.

This checks file presence, the sampling-3 shard layout, the completion-table
fingerprint, and the shared B/C/F system manifest. It does not download model
weights or claim that training results will match on different hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLING3_LABEL_SHA256 = "a3ae2b34c0305d8c230401bc15029fdc9db1aa2e1c85303e1c5bf7bd94671216"
SPLIT = ROOT / "configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(task: str, dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    sampling_id = 2 if task == "A" else 3
    if not dataset_dir.is_dir():
        return [f"Data directory is missing or its symlink is broken: {dataset_dir}"]
    completion = dataset_dir / f"completed_systems_sampling_{sampling_id}.txt"
    if not completion.is_file():
        errors.append(f"Missing completion table: {completion}")
    if task == "A":
        ics = dataset_dir / "ICs_sampling_2.hdf5"
        if not ics.is_file():
            errors.append(f"Missing initial conditions: {ics}")
        return errors

    shards = sorted(dataset_dir.glob("results_batch_*_sampling_3.hdf5"))
    expected_names = {f"results_batch_{number:02d}_sampling_3.hdf5" for number in range(40)}
    actual_names = {path.name for path in shards}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        errors.append(f"Expected the paper's 40 sampling-3 shards; missing={missing}, extra={extra}")
    if completion.is_file() and sha256(completion) != SAMPLING3_LABEL_SHA256:
        errors.append(f"Completion table differs from the saved paper data: {completion}")

    if shards:
        try:
            import h5py
        except ImportError:
            errors.append("h5py is unavailable; install the project environment")
        else:
            for path in shards:
                try:
                    with h5py.File(path, "r") as handle:
                        shape = tuple(handle["times_complete"].shape)
                    if shape != (4000, 1024):
                        errors.append(f"Unexpected times_complete shape {shape}: {path}")
                except (OSError, KeyError) as exc:
                    errors.append(f"Cannot read times_complete in {path}: {exc}")

    if task in {"B", "C", "F"}:
        if not SPLIT.is_file() or not SPLIT.with_suffix(".json").is_file():
            errors.append(f"Missing committed system split or metadata: {SPLIT}")
        elif completion.is_file():
            try:
                from tasks.common.case_split import ensure_case_split_manifest
                manifest = ensure_case_split_manifest(completion, SPLIT)
                counts = manifest["split"].value_counts().to_dict()
                if counts != {"train": 3498, "val": 750, "test": 750}:
                    errors.append(f"Unexpected system split counts: {counts}")
            except (ImportError, OSError, ValueError) as exc:
                errors.append(f"System split does not match completion table: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list("ABCDEF"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir or Path("dataset_v3" if args.task == "A" else "dataset_sampling_3_prefix1024")
    errors = check(args.task, dataset_dir)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: Task {args.task} inputs checked under {dataset_dir}")
    if args.task in {"B", "C", "F"}:
        print(f"PASS: system split validated against {SPLIT.relative_to(ROOT)}")
    print("Pretrained weights and full training/evaluation remain unchecked.")


if __name__ == "__main__":
    main()
