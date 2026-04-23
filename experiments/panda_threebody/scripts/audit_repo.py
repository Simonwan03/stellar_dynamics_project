from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.project_data import (
    choose_default_dataset,
    inspect_result_file,
    scan_local_datasets,
    summarize_audit,
    write_json,
    write_markdown,
)


def make_markdown() -> str:
    choices = scan_local_datasets()
    default = choose_default_dataset([1, 0])
    lines = [
        "# Generated Audit",
        "",
        f"Default dataset choice: `{default.root.name}` / sampling `{default.sampling_id}`",
        "",
        "## Discovered dataset candidates",
        "",
    ]
    for choice in choices:
        lines.extend(
            [
                f"### {choice.root} (sampling {choice.sampling_id})",
                "",
                f"- result shards: {len(choice.result_paths)}",
                f"- local batch ids: {choice.local_batch_ids}",
                f"- local result rows: {choice.local_result_row_count}",
                f"- label path: `{choice.label_path}`",
                f"- label rows: {choice.label_row_count}",
                f"- invalid labels: {choice.invalid_label_count}",
                f"- time horizon max: {choice.time_unbound_max}",
                f"- first shard time length: {choice.time_length}",
                "- first shard shapes:",
            ]
        )
        for key, shape in choice.feature_shapes.items():
            lines.append(f"  - `{key}`: `{tuple(shape)}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    default = choose_default_dataset([1, 0])
    print("Default dataset choice:")
    print(summarize_audit(default))
    print()
    print("First shard shapes:")
    print(inspect_result_file(default.result_paths[0]))

    if args.write:
        report_path = Path(__file__).resolve().parents[1] / "reports" / "data_audit_generated.md"
        write_markdown(report_path, make_markdown())
        write_json(report_path.with_suffix(".json"), summarize_audit(default))
        print(f"Wrote generated audit to {report_path}")


if __name__ == "__main__":
    main()
