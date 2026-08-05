from __future__ import annotations

import unittest

import numpy as np
import torch

from tasks.task_c.main import InitialConditionMLP, hazard_logits_to_survival, standardize_case_train


class TaskCInitialConditionTest(unittest.TestCase):
    def test_initial_condition_mlp_outputs_requested_curve(self) -> None:
        model = InitialConditionMLP(input_dim=15, horizon=7, hidden_dim=128, dropout=0.0)
        logits = model(torch.randn(4, 15))
        survival = hazard_logits_to_survival(logits)

        self.assertEqual(logits.shape, (4, 7))
        self.assertEqual(survival.shape, (4, 7))
        self.assertTrue(torch.all((survival >= 0.0) & (survival <= 1.0)))
        self.assertTrue(torch.all(survival[:, 1:] <= survival[:, :-1]))

    def test_static_case_normalization_reduces_over_systems(self) -> None:
        rng = np.random.default_rng(42)
        train = rng.normal(size=(8, 15)).astype(np.float32)
        val = rng.normal(size=(3, 15)).astype(np.float32)
        test = rng.normal(size=(3, 15)).astype(np.float32)

        train_norm, [val_norm, test_norm], stats = standardize_case_train(train, val, test)

        self.assertEqual(train_norm.shape, train.shape)
        self.assertEqual(val_norm.shape, val.shape)
        self.assertEqual(test_norm.shape, test.shape)
        self.assertEqual(stats["mean"].shape, (1, 15))
        self.assertEqual(stats["std"].shape, (1, 15))
        np.testing.assert_allclose(train_norm.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(train_norm.std(axis=0), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
