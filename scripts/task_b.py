"""Task B entry point for the saved low- and high-resolution pipelines.

The two pipelines have different memory strategies and CLI defaults. Select the
intended experiment explicitly with --resolution when reproducing a result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: Sequence[str] | None = None) -> None:
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--resolution", choices=("low", "high"), default="high")
    selector.add_argument("--sampling-id", type=int)
    selected, forwarded = selector.parse_known_args(argv)

    expected_sampling_id = 2 if selected.resolution == "low" else 3
    if selected.sampling_id is not None:
        if selected.sampling_id != expected_sampling_id:
            selector.error(
                f"--resolution {selected.resolution} requires "
                f"--sampling-id {expected_sampling_id}; got {selected.sampling_id}"
            )
        forwarded.extend(("--sampling-id", str(selected.sampling_id)))

    if selected.resolution == "low":
        from tasks.task_b.main import main as run_selected
    else:
        from tasks.task_b.highres import main as run_selected

    if "--help" in forwarded or "-h" in forwarded:
        print(
            "Task B pipeline: --resolution low (sampling 2) or high "
            "(sampling 3; default). The options below belong to the selected pipeline.\n"
        )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *forwarded]
        run_selected()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
