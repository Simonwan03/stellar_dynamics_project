import unittest

import numpy as np
import torch

from tasks.task_f.masking import apply_observation_mask, random_observation_mask
from tasks.task_f.models import MaskedAttentionClassifier, MaskedTCNClassifier


class TaskFMaskingTests(unittest.TestCase):
    def test_requested_ratios_hide_exact_random_positions(self) -> None:
        for ratio, expected in [(0.25, 4), (0.50, 8), (0.75, 12)]:
            mask = random_observation_mask(16, ratio, seed=42, sample_id=7)
            self.assertEqual(int((~mask).sum()), expected)
            hidden = np.flatnonzero(~mask)
            self.assertFalse(np.array_equal(hidden, np.arange(16 - expected, 16)))

    def test_zero_ratio_keeps_all_positions_observed(self) -> None:
        mask = random_observation_mask(16, 0.0, seed=42, sample_id=7)
        self.assertTrue(np.all(mask))


    def test_masks_are_reproducible_but_change_by_epoch(self) -> None:
        first = random_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=2)
        repeat = random_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=2)
        next_epoch = random_observation_mask(64, 0.5, seed=3, sample_id=9, epoch=3)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, next_epoch))

    def test_masking_hides_all_channels_and_appends_flag(self) -> None:
        values = np.arange(24, dtype=np.float32).reshape(6, 4)
        observed = np.array([True, False, True, False, False, True])
        result = apply_observation_mask(values, observed)
        self.assertEqual(result.shape, (6, 5))
        np.testing.assert_array_equal(result[~observed, :-1], 0.0)
        np.testing.assert_array_equal(result[observed, :-1], values[observed])
        np.testing.assert_array_equal(result[:, -1], observed.astype(np.float32))

    def test_masking_can_omit_observed_flag_channel(self) -> None:
        values = np.arange(24, dtype=np.float32).reshape(6, 4)
        observed = np.array([True, False, True, False, False, True])
        result = apply_observation_mask(values, observed, append_mask_channel=False)
        self.assertEqual(result.shape, values.shape)
        np.testing.assert_array_equal(result[~observed], 0.0)
        np.testing.assert_array_equal(result[observed], values[observed])

    def test_mask_is_independent_of_labels(self) -> None:
        labels = [0, 1]
        masks = [
            random_observation_mask(32, 0.25, seed=5, sample_id=11)
            for _label in labels
        ]
        np.testing.assert_array_equal(masks[0], masks[1])

    def test_models_accept_masked_sequences(self) -> None:
        x = torch.randn(3, 32, 16)
        x[:, ::3, :-1] = 0.0
        x[..., -1] = 1.0
        x[:, ::3, -1] = 0.0
        models = [
            MaskedTCNClassifier(15, hidden=16, levels=2, dropout=0.0),
            MaskedAttentionClassifier(
                15, hidden=16, heads=4, layers=1, dropout=0.0, max_len=32
            ),
        ]
        for model in models:
            output = model(x)
            self.assertEqual(tuple(output.shape), (3, 1))
            self.assertTrue(torch.isfinite(output).all())

    def test_models_accept_sequences_without_mask_channel(self) -> None:
        x = torch.randn(3, 32, 15)
        x[:, ::3] = 0.0
        models = [
            MaskedTCNClassifier(15, hidden=16, levels=2, dropout=0.0, append_mask_channel=False),
            MaskedAttentionClassifier(
                15, hidden=16, heads=4, layers=1, dropout=0.0, max_len=32, append_mask_channel=False
            ),
        ]
        for model in models:
            output = model(x)
            self.assertEqual(tuple(output.shape), (3, 1))
            self.assertTrue(torch.isfinite(output).all())

if __name__ == "__main__":
    unittest.main()
