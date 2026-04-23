from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.project_data import (
    build_export_table,
    build_representation,
    case_summary,
    choose_default_dataset,
    clean_and_group_trajectories,
    export_panda_arrow_dataset,
    get_export_root,
    load_labels,
    load_result_arrays,
    load_yaml,
    required_keys_for_representation,
    stratified_case_split,
    summarize_audit,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--representation", default=None)
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    config = load_yaml(exp_root / args.config)
    representation = args.representation or config.get("representation", "orbital15")
    choice = choose_default_dataset(config.get("sampling_id_preference", [1, 0]))

    max_cases = config.get("max_cases")
    max_rows = None if max_cases is None else int(max_cases) * 8
    time_steps = config.get("raw_prefix_steps")
    keys = required_keys_for_representation(representation)

    print("Using dataset choice:", summarize_audit(choice))
    print("Representation:", representation)
    print("Required keys:", keys)
    print("Requested time steps:", time_steps)
    print("Requested max rows:", max_rows)

    labels = load_labels(choice, max_rows=max_rows)
    data = load_result_arrays(choice, keys=keys, time_steps=time_steps, max_rows=max_rows)
    X = build_representation(data, representation=representation)
    X, labels = clean_and_group_trajectories(X, labels)
    case_df = case_summary(labels)
    case_df = stratified_case_split(
        case_df,
        train_ratio=float(config["train_ratio"]),
        val_ratio=float(config["val_ratio"]),
        test_ratio=float(config["test_ratio"]),
        seed=int(config["seed"]),
    )
    export_df = build_export_table(labels, case_df, choice, representation=representation)
    export_root = get_export_root(config, representation=representation)
    manifest_paths = export_panda_arrow_dataset(X, export_df, export_root=export_root)

    write_json(
        export_root / "manifests" / "export_metadata.json",
        {
            "representation": representation,
            "channels": int(X.shape[-1]),
            "time_steps": int(X.shape[1]),
            "n_trajectories": int(len(X)),
            "n_cases": int(export_df["case_id"].nunique()),
            "split_counts": export_df["split"].value_counts().sort_index().to_dict(),
            "system_counts": export_df["system_name"].value_counts().sort_index().to_dict(),
            "manifests": {k: str(v) for k, v in manifest_paths.items()},
            "source_dataset": summarize_audit(choice),
        },
    )

    print("Export complete")
    print("Export root:", export_root)
    print("Trajectory tensor shape:", X.shape)
    print("Split counts:")
    print(export_df["split"].value_counts().sort_index().to_string())
    print("Wrote manifests:")
    for key, path in manifest_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
