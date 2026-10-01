"""Select a task's recorded dataset profile without changing its implementation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def run_entry(
    *,
    task: str,
    selected_main: Callable[[], None],
    default_resolution: str,
    supported: tuple[str, ...],
    defaults: dict[str, tuple[str, ...]],
    argv: Sequence[str] | None = None,
    forward_sampling_id: bool = True,
) -> None:
    """Apply dataset-specific defaults, then let explicit task options win."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--resolution", choices=("low", "high"), default=default_resolution)
    selector.add_argument("--sampling-id", type=int)
    selected, forwarded = selector.parse_known_args(argv)
    if selected.resolution not in supported:
        selector.error(
            f"Task {task} has no supported {selected.resolution}-resolution pipeline. "
            f"Available: {', '.join(supported)}."
        )
    expected_id = 2 if selected.resolution == "low" else 3
    if selected.sampling_id is not None and selected.sampling_id != expected_id:
        selector.error(
            f"--resolution {selected.resolution} requires --sampling-id {expected_id}; "
            f"got {selected.sampling_id}"
        )
    options = [*defaults[selected.resolution], *forwarded]
    if forward_sampling_id:
        options.extend(("--sampling-id", str(expected_id)))
    if "--help" in forwarded or "-h" in forwarded:
        print(
            f"Task {task}: --resolution {', '.join(supported)} "
            f"(default: {default_resolution}; low=sampling 2, high=sampling 3).\n"
        )
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *options]
        selected_main()
    finally:
        sys.argv = original_argv
