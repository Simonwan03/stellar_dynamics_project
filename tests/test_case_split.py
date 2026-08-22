from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.common.case_split import (
    build_case_catalog_from_labels,
    case_split_indices,
    case_split_membership,
    create_case_split_manifest,
    ensure_case_split_manifest,
)


class CommonCaseSplitTest(unittest.TestCase):
    def _labels(self, case_count: int = 20, group_size: int = 4) -> pd.DataFrame:
        rows: list[dict[str, int | float]] = []
        for case_id in range(case_count):
            for variation_id in range(group_size):
                rows.append(
                    {
                        "index": case_id * group_size + variation_id,
                        "num_batch": case_id // 2,
                        "time_unbound": 100.0 if variation_id <= case_id % (group_size + 1) else 0.0,
                        "wall_time": 0.0,
                        "diagnosis": 0,
                    }
                )
        # This case has only three valid variations and must be excluded from
        # the canonical universe rather than assigned independently by a task.
        rows = [row for row in rows if row["index"] // group_size != case_count - 1 or row["index"] % group_size != group_size - 1]
        return pd.DataFrame(rows)

    def test_catalog_keeps_only_complete_valid_systems(self) -> None:
        labels = self._labels()
        catalog = build_case_catalog_from_labels(labels, group_size=4, time_horizon=100.0)
        self.assertEqual(len(catalog), 19)
        self.assertNotIn(19, catalog["raw_case_id"].to_numpy())
        self.assertTrue(np.all(catalog["stable_fraction"].between(0.0, 1.0)))

    def test_split_is_deterministic_and_group_safe(self) -> None:
        catalog = build_case_catalog_from_labels(self._labels(), group_size=4, time_horizon=100.0)
        first = create_case_split_manifest(catalog, seed=42)
        second = create_case_split_manifest(catalog, seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["split"].value_counts().sum(), len(catalog))
        self.assertEqual(first["split"].value_counts().to_dict(), {"train": 13, "val": 3, "test": 3})

        sample_case_ids = np.repeat(catalog["raw_case_id"].to_numpy(dtype=np.int64), 4)
        membership = case_split_membership(sample_case_ids, first)
        for case_id in catalog["raw_case_id"]:
            values = np.unique(membership[sample_case_ids == case_id])
            self.assertEqual(values.size, 1)
        indices = case_split_indices(sample_case_ids, first)
        self.assertEqual(sum(len(rows) for rows in indices.values()), len(sample_case_ids))
        self.assertEqual({name: len(rows) for name, rows in indices.items()}, {"train": 52, "val": 12, "test": 12})

    def test_dataset_v3_count_rounding(self) -> None:
        stable_count = np.arange(4998, dtype=np.int64) % 33
        catalog = pd.DataFrame(
            {
                "raw_case_id": np.arange(4998, dtype=np.int64),
                "stable_count": stable_count,
                "stable_fraction": stable_count / 32.0,
                "stable_fraction_bin": [
                    0 if count == 0 else 1 if count <= 8 else 2 if count <= 16 else 3 if count < 32 else 4
                    for count in stable_count
                ],
            }
        )
        manifest = create_case_split_manifest(catalog, seed=42)
        self.assertEqual(
            manifest["split"].value_counts().reindex(["train", "val", "test"]).to_dict(),
            {"train": 3498, "val": 750, "test": 750},
        )

    def test_manifest_round_trip_validates_catalog(self) -> None:
        labels = self._labels(case_count=8, group_size=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "completed.txt"
            labels.to_csv(labels_path, sep=" ", header=False, index=False)
            manifest_path = root / "common_split.csv"
            first = ensure_case_split_manifest(
                labels_path,
                manifest_path,
                group_size=4,
                time_horizon=100.0,
                seed=7,
            )
            second = ensure_case_split_manifest(
                labels_path,
                manifest_path,
                group_size=4,
                time_horizon=100.0,
                seed=7,
            )
            reused_with_different_training_seed = ensure_case_split_manifest(
                labels_path,
                manifest_path,
                group_size=4,
                time_horizon=100.0,
                seed=99,
            )
            pd.testing.assert_frame_equal(first, second)
            pd.testing.assert_frame_equal(first, reused_with_different_training_seed)
            self.assertTrue(manifest_path.exists())
            self.assertTrue(manifest_path.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
