from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from tasks.task_c.main import (
    build_case_tensors,
    DualAttentionSurvivalModel,
    DualHeadSurvivalLoss,
    InitialConditionMLP,
    TrajectoryContextSurvivalModel,
    build_variation_subset_indices,
    hazard_logits_to_survival,
    standardize_case_train,
)


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

    def test_trajectory_context_has_variation_and_system_heads(self) -> None:
        model = TrajectoryContextSurvivalModel(
            input_dim=5,
            hidden_dim=8,
            horizon=7,
            encoder_type="cnn",
            dropout=0.0,
        )
        variation_logits, system_logits = model(torch.randn(2, 32, 12, 5))
        self.assertEqual(variation_logits.shape, (2, 32, 7))
        self.assertEqual(system_logits.shape, (2, 7))

        variation_survival = hazard_logits_to_survival(variation_logits)
        system_survival = hazard_logits_to_survival(system_logits)
        self.assertTrue(torch.all(variation_survival[:, :, 1:] <= variation_survival[:, :, :-1]))
        self.assertTrue(torch.all(system_survival[:, 1:] <= system_survival[:, :-1]))

    def test_dual_attention_joint_loss_has_gradients(self) -> None:
        model = DualAttentionSurvivalModel(input_dim=5, hidden_dim=8, horizon=4, dropout=0.0)
        x = torch.randn(2, 32, 6, 5)
        y = torch.ones(2, 32, 4)
        y[:, :, 2:] = 0.0
        q = y.mean(dim=1)
        variation_logits, system_logits = model(x)
        criterion = DualHeadSurvivalLoss(system_weight=1.0, consistency_weight=0.1)
        components = criterion.components(variation_logits, system_logits, y, q)
        components["loss"].backward()

        self.assertTrue(torch.isfinite(components["loss"]))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.parameters() if parameter.grad is not None), 0.0)

    def test_balanced_variation_subsets_are_nested(self) -> None:
        subsets = [build_variation_subset_indices(32, count) for count in (1, 2, 4, 8, 16, 32)]
        for smaller, larger in zip(subsets, subsets[1:]):
            self.assertTrue(set(smaller).issubset(set(larger)))
        np.testing.assert_array_equal(subsets[0], np.asarray([0]))
        np.testing.assert_array_equal(subsets[1], np.asarray([0, 16]))
        np.testing.assert_array_equal(subsets[-1], np.arange(32))

    def test_complete_case_tensorization_preserves_variation_order(self) -> None:
        labels = pd.DataFrame(
            {
                "index": [*range(32, 64), *range(0, 32), 64],
                "time_unbound": [float(i) for i in range(32, 64)] + [float(i) for i in range(32)] + [0.0],
            }
        )
        X = np.arange(len(labels) * 3, dtype=np.float32).reshape(len(labels), 1, 3)
        X_cases, y_cases, case_ids = build_case_tensors(
            X,
            labels,
            group_size=32,
            future_times=np.asarray([10.0, 40.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(case_ids, np.asarray([0, 1]))
        self.assertEqual(X_cases.shape, (2, 32, 1, 3))
        np.testing.assert_array_equal(X_cases[0, :, 0, 0], X[32:64, 0, 0])
        np.testing.assert_array_equal(y_cases[0, :, 0], np.asarray([0.0] * 10 + [1.0] * 22))


if __name__ == "__main__":
    unittest.main()
