from __future__ import annotations

import unittest

import numpy as np

from tasks.task_e.data import cap_train_variations_per_system


class TaskESamplingTest(unittest.TestCase):
    def test_one_variation_per_system_is_reproducible(self) -> None:
        splits = {
            "train": np.array([0, 1, 2, 3, 32, 33, 34, 64, 65], dtype=np.int64),
            "val": np.array([96, 97, 128, 129, 130], dtype=np.int64),
            "test": np.array([160, 161, 162], dtype=np.int64),
        }

        sampled = cap_train_variations_per_system(
            splits,
            max_train_variations_per_system=1,
            group_size=32,
            seed=42,
        )
        repeated = cap_train_variations_per_system(
            splits,
            max_train_variations_per_system=1,
            group_size=32,
            seed=42,
        )

        train = sampled["train"]
        self.assertEqual(len(train), len(np.unique(train // 32)))
        np.testing.assert_array_equal(train, repeated["train"])
        self.assertTrue(set(train).issubset(set(splits["train"])))
        np.testing.assert_array_equal(sampled["val"], splits["val"])
        np.testing.assert_array_equal(sampled["test"], splits["test"])

    def test_limit_preserves_systems_and_allows_partial_groups(self) -> None:
        splits = {"train": np.array([0, 1, 2, 32, 33], dtype=np.int64)}

        sampled = cap_train_variations_per_system(
            splits,
            max_train_variations_per_system=2,
            group_size=32,
            seed=7,
        )

        self.assertEqual(len(sampled["train"]), 4)
        self.assertEqual(set(sampled["train"] // 32), {0, 1})

    def test_zero_keeps_existing_splits_unchanged(self) -> None:
        splits = {"train": np.array([0, 1, 32], dtype=np.int64)}

        sampled = cap_train_variations_per_system(
            splits,
            max_train_variations_per_system=0,
            group_size=32,
            seed=42,
        )

        np.testing.assert_array_equal(sampled["train"], splits["train"])


if __name__ == "__main__":
    unittest.main()
