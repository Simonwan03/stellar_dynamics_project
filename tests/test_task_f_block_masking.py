import unittest

import numpy as np

from tasks.task_f.block_masking import contiguous_observation_mask


class TaskFBlockMaskingTests(unittest.TestCase):
    def test_one_block_has_exact_hidden_count(self) -> None:
        observed = contiguous_observation_mask(16, 0.5, seed=42, sample_id=7, num_blocks=1)
        self.assertEqual(int((~observed).sum()), 8)
        hidden = np.flatnonzero(~observed)
        np.testing.assert_array_equal(hidden, np.arange(hidden[0], hidden[0] + 8))

    def test_two_blocks_have_exact_count_and_observed_gap(self) -> None:
        observed = contiguous_observation_mask(32, 0.75, seed=42, sample_id=7, num_blocks=2)
        self.assertEqual(int((~observed).sum()), 24)
        hidden = np.flatnonzero(~observed)
        transitions = np.diff(hidden)
        self.assertEqual(int((transitions > 1).sum()), 1)

    def test_masks_are_reproducible_and_epoch_dependent(self) -> None:
        first = contiguous_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=2, num_blocks=2)
        repeat = contiguous_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=2, num_blocks=2)
        next_epoch = contiguous_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=3, num_blocks=2)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, next_epoch))


if __name__ == "__main__":
    unittest.main()
