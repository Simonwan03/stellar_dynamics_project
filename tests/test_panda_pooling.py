import unittest

import torch

from experiments.foundation_models.scripts.train_panda_unbound_classifier import pool_encoder_features


class PandaPoolingTest(unittest.TestCase):
    def test_patch_mean_keeps_channel_blocks(self) -> None:
        encoded = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        pooled = pool_encoder_features(encoded, "patch_mean_keep_channels")
        expected = encoded.mean(dim=2).flatten(start_dim=1)
        self.assertEqual(tuple(pooled.shape), (2, 15))
        torch.testing.assert_close(pooled, expected)

    def test_global_mean_remains_available(self) -> None:
        encoded = torch.randn(2, 3, 4, 5)
        pooled = pool_encoder_features(encoded, "global_mean")
        self.assertEqual(tuple(pooled.shape), (2, 5))
        torch.testing.assert_close(pooled, encoded.mean(dim=(1, 2)))


if __name__ == "__main__":
    unittest.main()
