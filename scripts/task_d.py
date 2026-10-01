"""Task D high-resolution median orbital-decorrelation entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_d.main import main as task_main


def main() -> None:
    run_entry(task="D", selected_main=task_main, default_resolution="high",
              supported=("high",), defaults={"high": ()})


if __name__ == "__main__":
    main()
