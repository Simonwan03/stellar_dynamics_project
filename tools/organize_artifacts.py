"""Apply the reviewed artifact path map and update textual path references.

Run once from the repository root after reviewing docs/artifact_path_map.csv.
The map is kept as a permanent provenance record. This script does not alter data
values or experiment settings; it changes directory names and path strings only.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs/artifact_path_map.csv"
SKIP = {MAP, ROOT / "docs/artifact_merge_map.csv", ROOT / "docs/artifact_inventory.csv", ROOT / "docs/artifact_inventory.md"}


def entries() -> list[tuple[str, str]]:
    with MAP.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(row["old_path"], row["new_path"]) for row in rows]
    if len(pairs) != len(set(pairs)) or len({new for _, new in pairs}) != len(pairs):
        raise ValueError("Duplicate path mapping")
    for old, new in pairs:
        if not old.startswith(("results/", "runs/")) or not new.startswith(("results/", "runs/")):
            raise ValueError(f"Unexpected mapping: {old} -> {new}")
    return pairs


def move(pairs: list[tuple[str, str]]) -> None:
    for old, new in pairs:
        source, target = ROOT / old, ROOT / new
        if source.exists() and target.exists():
            raise FileExistsError(f"Both paths exist: {old} and {new}")
        if not source.exists() and not target.exists():
            raise FileNotFoundError(f"Neither path exists: {old} or {new}")
    for old, new in pairs:
        source, target = ROOT / old, ROOT / new
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)
            print(f"{old} -> {new}")


def rewrite_references(pairs: list[tuple[str, str]]) -> int:
    # A trailing boundary prevents a shorter directory name from matching a
    # longer sibling name. Longest names are handled first for clarity.
    rules = [(re.compile(re.escape(old) + r"(?=$|[/\s\"'`,;:)\]}<])"), new)
             for old, new in sorted(pairs, key=lambda p: len(p[0]), reverse=True)]
    found = subprocess.run(
        ["rg", "--no-ignore", "-l", "-0", "results/|runs/", "."],
        cwd=ROOT, check=False, stdout=subprocess.PIPE,
    )
    if found.returncode not in (0, 1):
        raise RuntimeError("Could not search path references")
    count = 0
    for raw in filter(None, found.stdout.split(b"\0")):
        path = ROOT / os.fsdecode(raw)
        if path in SKIP or not path.is_file() or path.stat().st_size > 10_000_000:
            continue
        try:
            before = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        after = before
        for pattern, new in rules:
            after = pattern.sub(lambda _: new, after)
        if after != before:
            path.write_text(after, encoding="utf-8")
            count += 1
    return count


if __name__ == "__main__":
    paths = entries()
    move(paths)
    changed = rewrite_references(paths)
    print(f"Organized {len(paths)} artifact items; updated {changed} text files")
