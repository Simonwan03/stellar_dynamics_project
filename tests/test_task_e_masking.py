from __future__ import annotations

import unittest

import torch
from torch.utils.data import Dataset

from tasks.task_e.masking import MaskedForecastWindowDataset


class _SyntheticWindows(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "context": torch.arange(24, dtype=torch.float32).view(8, 3),
            "target": torch.zeros(4, 3),
            "trajectory_index": torch.tensor(index),
            "start": torch.tensor(0),
        }


class TaskEMaskingTest(unittest.TestCase):
    def test_fixed_mask_appends_indicator_and_preserves_target(self) -> None:
        dataset = MaskedForecastWindowDataset(
            _SyntheticWindows(),
            regime="fixed",
            seed=42,
            mask_ratio=0.5,
            mask_kind="random",
        )
        item = dataset[0]
        self.assertEqual(tuple(item["context"].shape), (8, 4))
        self.assertEqual(tuple(item["target"].shape), (4, 3))
        hidden = item["context"][:, -1] == 0
        self.assertEqual(int(hidden.sum()), 4)
        self.assertTrue(torch.all(item["context"][hidden, :-1] == 0))
        self.assertTrue(torch.all(item["context"][~hidden, -1] == 1))

    def test_dynamic_mask_changes_with_epoch_but_is_reproducible(self) -> None:
        first = MaskedForecastWindowDataset(
            _SyntheticWindows(), regime="mixed", seed=42, mask_ratio=0.5
        )
        second = MaskedForecastWindowDataset(
            _SyntheticWindows(), regime="mixed", seed=42, mask_ratio=0.5
        )
        first.set_epoch(3)
        second.set_epoch(3)
        torch.testing.assert_close(first[0]["context"], second[0]["context"])
        first.set_epoch(4)
        self.assertFalse(torch.equal(first[0]["context"], second[0]["context"]))


if __name__ == "__main__":
    unittest.main()
