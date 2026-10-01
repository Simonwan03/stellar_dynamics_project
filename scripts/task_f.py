"""Task F high-resolution masked-prefix entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_f.main import main as task_main


def main() -> None:
    run_entry(
        task="F", selected_main=task_main, default_resolution="high",
        supported=("high",),
        defaults={
            "high": (
                "--split-manifest", "configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv",
            )
        },
    )


if __name__ == "__main__":
    main()
