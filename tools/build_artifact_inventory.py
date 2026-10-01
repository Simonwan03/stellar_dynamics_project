"""Index the task-grouped results tree by task and source dataset."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "docs/artifact_inventory.csv"
OUT_MD = ROOT / "docs/artifact_inventory.md"
DATA_KEYS = {"sampling_id", "dataset_dir", "data_dir", "data_glob", "completion_table", "completion_table_name", "input_hdf5", "source_paths", "label_source", "source_dataset", "dataset", "manifest", "input_table"}
META_NAMES = ("config.json", "run_config.json", "manifest.json", "metadata.json", "protocol.json", "metrics.json", "summary.json", "feature_metadata.json", "divergence_summary.json", "traditional_stability_summary.json")


def task_for(name: str) -> str:
    if name.startswith("abc_"):
        return "A+B+C"
    if name.startswith("task_ab_") or name.startswith(("fixed_budget_", "factorial_")):
        return "A+B"
    match = re.match(r"task_([a-g])(?:_|$)", name)
    if match:
        task = match.group(1).upper()
        return "E+F" if task == "E" and "masked" in name else task
    if name.startswith("panda_temporal_adapter"):
        return "E"
    if name.startswith("divergence_"):
        return "D"
    return "shared/analysis"


def metadata_files(directory: Path) -> list[Path]:
    files = [directory / name for name in META_NAMES if (directory / name).is_file()]
    if not files:
        for pattern in ("*/config.json", "*/*/config.json", "*/metadata.json", "*/manifest.json", "*/metrics.json", "*/*/metrics.json", "*/labels_filtered.json"):
            files.extend(directory.glob(pattern))
    return [path for path in sorted(set(files))[:80] if path.stat().st_size < 2_000_000]


def data_fields(value: object, prefix: str = "") -> list[tuple[str, object]]:
    fields = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{prefix}.{key}" if prefix else key
            if key in DATA_KEYS and isinstance(item, (str, int, float, list)):
                fields.append((current, item))
            elif isinstance(item, dict):
                fields.extend(data_fields(item, current))
    return fields


def sampling_from(value: object) -> set[int]:
    if isinstance(value, (int, float)) and value in (0, 1, 2, 3):
        return {int(value)}
    if isinstance(value, list):
        return set().union(*(sampling_from(item) for item in value)) if value else set()
    if not isinstance(value, str):
        return set()
    found = {int(n) for n in re.findall(r"sampling[_/-]?([0-3])(?:\b|[_/.-])", value, re.I)}
    if "dataset_v3" in value:
        found.add(2)
    return found


def classify_data(directory: Path) -> tuple[str, str, str, str]:
    sources = set()
    evidence = []
    configs = metadata_files(directory)
    for path in configs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for key, value in data_fields(data):
            ids = sampling_from(value)
            if ids:
                sources.update(ids)
                evidence.append(f"{path.relative_to(ROOT)}:{key}")
    config = str(configs[0].relative_to(ROOT)) if configs else ""
    if sources:
        return "+".join(f"sampling_{n}" for n in sorted(sources)), "saved configuration", "; ".join(dict.fromkeys(evidence[:3])), config
    if directory.name == "time_unbound_distribution_sampling_2_3":
        return "sampling_2+sampling_3", "directory name and summary", "summary.json:sampling[2,3]", config
    if directory.name == "task_e_e1e2_summary":
        linked = ROOT / "results/task_e/sampling_2/task_e_chronos2_orbital15_ctx512_pred128/config.json"
        if linked.is_file() and sampling_from(linked.read_text(encoding="utf-8")) == {2}:
            return "sampling_2", "linked run configuration", str(linked.relative_to(ROOT)), config
    ids = sampling_from(directory.name)
    if ids:
        return "+".join(f"sampling_{n}" for n in sorted(ids)), "directory name", directory.name, config
    return "unknown", "unverified", "no dataset field in inspected metadata", config


def kind_for(name: str, directory: Path) -> str:
    if "smoke" in name or "prelim" in name or "example" in name:
        return "exploratory"
    if "cache" in name or "prepared" in name or "checkpoint" in name:
        return "cache/checkpoint"
    if name in {"analysis", "traditional_stability"} or any(x in name for x in ("audit", "statistics", "distribution")):
        return "analysis"
    if any((directory / n).is_file() for n in ("metrics.json", "test_metrics.json", "e2_metrics.json", "run_config.json")):
        return "metrics/run"
    return "other artifact"


def build_rows() -> list[dict[str, str]]:
    with (ROOT / "docs/artifact_path_map.csv").open(newline="", encoding="utf-8") as handle:
        known = {row["new_path"]: row for row in csv.DictReader(handle)}
    task_names = {"task_a": "A", "task_b": "B", "task_c": "C",
                  "task_d": "D", "task_e": "E", "task_f": "F",
                  "shared_analysis": "shared/analysis"}
    rows = []
    items = []
    for path in sorted((ROOT / "results").glob("*/*/*")):
        if path.is_dir() and path.name == "masked_forecasting":
            items.extend(sorted(path.iterdir()))
        else:
            items.append(path)
    for path in items:
        if not (path.is_dir() or path.is_file()):
            continue
        relative = str(path.relative_to(ROOT))
        saved = known.get(relative, {})
        parts = path.relative_to(ROOT).parts
        section = parts[1]
        task = saved.get("task") or (task_for(path.name) if section == "comparisons" else task_names[section])
        dataset = saved.get("dataset", parts[2])
        if dataset == "dataset_unknown":
            dataset = "unknown"
        if path.is_dir():
            observed, basis, evidence, config = classify_data(path)
            if observed != "unknown" and observed != dataset:
                raise ValueError(f"Dataset mismatch for {relative}: {observed} vs {dataset}")
            rows.append({"path": relative, "task": task, "dataset": dataset,
                         "basis": basis if observed != "unknown" else saved.get("basis", "directory path"),
                         "kind": kind_for(path.name, path), "metadata": config,
                         "evidence": evidence if observed != "unknown" else saved.get("old_path", relative)})
        else:
            rows.append({"path": relative, "task": task, "dataset": dataset,
                         "basis": saved.get("basis", "directory path"),
                         "kind": "standalone file", "metadata": "",
                         "evidence": saved.get("old_path", relative)})
    return rows


def write_inventory(rows: list[dict[str, str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "task", "dataset", "basis", "kind", "metadata", "evidence"))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["dataset"])].append(row)
    lines = [
        "# Artifact inventory", "",
        "All experiment outputs are under `results/`, grouped by task and source dataset. Task E forecasting runs are in `results/task_e/`; Task F masked forecasting runs are in `results/task_f/sampling_3/masked_forecasting/`. Cross-task comparisons are in `results/comparisons/`, and shared figures and audits are in `results/shared_analysis/`. `dataset_unknown` marks artifacts without a verified source dataset. The [path map](artifact_path_map.csv) records original and current locations; the [merge map](artifact_merge_map.csv) records the `results/`/`runs/` consolidation. Artifact type is inferred from names and saved files; see the manuscript reproduction map for final model selection.", "",
        "Regenerate with: python tools/build_artifact_inventory.py. The script reads saved JSON metadata and directory names. It does not open HDF5 shards or rewrite results. Here dataset_v3 means sampling_2.", "",
        "| Task | Dataset | Items |", "| --- | --- | ---: |",
    ]
    for (task, dataset), items in sorted(grouped.items()):
        lines.append(f"| {task} | {dataset} | {len(items)} |")
    for (task, dataset), items in sorted(grouped.items()):
        lines += ["", f"## {task} / {dataset}", ""]
        for row in items:
            suffix = f" - {row['kind']}"
            if row["basis"] != "saved configuration":
                suffix += f"; source: {row['basis']}"
            lines.append(f"- [{row['path']}](../{row['path']}){suffix}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_inventory(build_rows())
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
