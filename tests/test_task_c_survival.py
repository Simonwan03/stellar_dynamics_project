from __future__ import annotations

import unittest

import numpy as np
import torch

from tasks.task_c.survival import (
    FrozenEmbeddingHazardModel,
    MaskedHazardNLL,
    StatsTrajectoryEncoder,
    build_discrete_hazard_targets,
    hazard_logits_to_survival,
    make_log_bin_edges,
    task_b_spock_summary,
)


class EventSurvivalTargetTest(unittest.TestCase):
    def test_event_censoring_and_mask(self) -> None:
        edges = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        times = np.asarray([[1.0, 3.0, 8.0, 9.0]], dtype=np.float32)
        target, mask, metadata = build_discrete_hazard_targets(times, edges, return_metadata=True)

        np.testing.assert_array_equal(target[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(target[0, 1], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(mask[0, 1], [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(target[0, 2], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[0, 2], [1.0, 1.0, 1.0])
        self.assertTrue(bool(metadata["censored"][0, 2]))
        self.assertTrue(bool(metadata["censored"][0, 3]))

    def test_hazard_survival_is_monotone(self) -> None:
        logits = torch.tensor([[0.0, -1.0, 1.0]], requires_grad=True)
        survival = hazard_logits_to_survival(logits)
        self.assertTrue(torch.all(survival[..., 1:] <= survival[..., :-1]))
        loss = MaskedHazardNLL()(logits, torch.zeros_like(logits), torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_log_edges(self) -> None:
        edges = make_log_bin_edges(5.0, 10000.0, 20)
        self.assertEqual(edges.size, 21)
        self.assertTrue(np.all(np.diff(edges) > 0.0))
        self.assertAlmostEqual(float(edges[0]), 5.0)
        self.assertAlmostEqual(float(edges[-1]), 10000.0)

    def test_frozen_embedding_heads(self) -> None:
        x = torch.randn(2, 32, 15 * 768)
        direct = FrozenEmbeddingHazardModel(15 * 768, 20, 32, 0.1, pooling="direct")
        meanstd = FrozenEmbeddingHazardModel(15 * 768, 20, 32, 0.1, pooling="meanstd")
        attention = FrozenEmbeddingHazardModel(15 * 768, 20, 32, 0.1, pooling="attention")
        self.assertEqual(tuple(direct(x).shape), (2, 32, 20))
        self.assertEqual(tuple(meanstd(x).shape), (2, 32, 20))
        self.assertEqual(tuple(attention(x).shape), (2, 32, 20))

    def test_stats_encoder_uses_task_b_eight_statistics(self) -> None:
        x = torch.tensor(
            [[[0.0, 1.0], [1.0, 3.0], [3.0, 5.0]]], dtype=torch.float32
        )
        summary = task_b_spock_summary(x)
        values = x.numpy()
        time = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
        centered = time - time.mean()
        expected = np.stack(
            [
                values[:, 0],
                values[:, -1],
                values.mean(axis=1),
                values.std(axis=1),
                np.einsum("ntc,t->nc", values, centered) / np.dot(centered, centered),
                values.min(axis=1),
                values.max(axis=1),
                values[:, -1] - values[:, 0],
            ],
            axis=2,
        ).reshape(1, -1)
        np.testing.assert_allclose(summary.numpy(), expected, rtol=1e-6, atol=1e-6)

        encoder = StatsTrajectoryEncoder(input_dim=15, hidden_dim=32, dropout=0.0)
        self.assertEqual(encoder.net[0].normalized_shape, (15 * 8,))
        self.assertEqual(encoder.net[1].in_features, 15 * 8)
        self.assertEqual(tuple(encoder(torch.randn(4, 6, 15)).shape), (4, 32))


if __name__ == "__main__":
    unittest.main()
