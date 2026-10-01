"""Task C survival-curve entry point for both dataset resolutions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_c.survival import main as task_main


def main() -> None:
    run_entry(
        task="C", selected_main=task_main, default_resolution="low",
        supported=("low", "high"),
        defaults={
            "low": ("--dataset-dir", "dataset_v3", "--output-dir", "results/task_c/sampling_2/task_c_survival20"),
            "high": (
                "--dataset-dir", "dataset_sampling_3_prefix1024",
                "--output-dir", "results/task_c/sampling_3/task_c_survival20",
                "--prefix-len", "128", "--hazard-t-min", "1",
                "--split-manifest", "configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv",
            ),
        },
    )


if __name__ == "__main__":
    main()
