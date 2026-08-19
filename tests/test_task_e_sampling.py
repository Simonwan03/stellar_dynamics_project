from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from tasks.task_e.data import (
    DynamicTrajectoryWindowDataset,
    ForecastWindowDataset,
    RollingWindowDataset,
    StandardScaler,
    select_deterministic_windows,
)
from tasks.task_e.metrics import (
    bootstrap_block_ci,
    compute_forecast_metrics,
)


class TaskESamplingTest(unittest.TestCase):
    def _base(self) -> ForecastWindowDataset:
        base = ForecastWindowDataset.__new__(ForecastWindowDataset)
        base.record_trajectories = np.asarray([0] * 10 + [32] * 3, dtype=np.int64)
        base.record_starts = np.asarray(list(range(0, 100, 10)) + [0, 10, 20], dtype=np.int64)
        base.valid_lengths = np.full(33, 64, dtype=np.int64)
        base.sources = [SimpleNamespace(time_steps=64)]
        base.feature_metadata = {"num_features": 1, "groups": []}
        base.scaler = StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        base.stride = 2
        base._handles = {}
        return base

    def test_deterministic_selection_covers_trajectory_positions(self) -> None:
        base = self._base()
        selected = select_deterministic_windows(base, 4, (0.10, 0.35, 0.65, 0.90))
        self.assertEqual(len(selected), 7)
        starts = base.record_starts[np.asarray(selected.indices, dtype=np.int64)].tolist()
        self.assertEqual(starts[:4], [10, 30, 60, 80])

    def test_dynamic_sampling_changes_epoch_but_balances_trajectories(self) -> None:
        base = self._base()
        sampled = DynamicTrajectoryWindowDataset(base, windows_per_trajectory=2, seed=123)
        first = sampled.selected_indices.copy()
        self.assertEqual(len(sampled), 4)
        sampled.set_epoch(1)
        second = sampled.selected_indices.copy()
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(np.unique(base.record_trajectories[second]).size, 2)

    def test_rolling_selection_uses_two_dispersed_starts(self) -> None:
        base = self._base()

        def load_raw_window(index: int, total_len: int | None = None) -> np.ndarray:
            del index
            length = int(total_len or 4)
            return np.arange(length, dtype=np.float32).reshape(length, 1)

        base.load_raw_window = load_raw_window  # type: ignore[method-assign]
        rolling = RollingWindowDataset(
            base,
            context_len=4,
            block_len=2,
            num_blocks=4,
            windows_per_trajectory=2,
            target_positions=(0.25, 0.75),
        )
        self.assertEqual(rolling.candidate_count, 4)
        self.assertEqual(len(rolling), 4)
        self.assertNotEqual(rolling.records[0][2], rolling.records[1][2])


class TaskEHierarchicalMetricTest(unittest.TestCase):
    def test_system_balanced_metric_differs_from_window_weighted_metric(self) -> None:
        prediction = np.asarray([[[1.0]], [[1.0]], [[1.0]], [[3.0]]], dtype=np.float32)
        target = np.zeros_like(prediction)
        metrics, _, _ = compute_forecast_metrics(
            prediction,
            target,
            StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)),
            ["x"],
            trajectory_ids=np.asarray([0, 0, 0, 2], dtype=np.int64),
            group_size=2,
            bootstrap_samples=50,
            bootstrap_seed=77,
        )
        self.assertAlmostEqual(metrics["mae_z_window_weighted"], 1.5)
        self.assertAlmostEqual(metrics["mae_z"], 2.0)
        self.assertAlmostEqual(metrics["horizon_mae_system_balanced"][0], 2.0)
        self.assertEqual(metrics["bootstrap_ci95"]["mae_z"]["num_systems"], 2)
        repeated, _, _ = compute_forecast_metrics(
            prediction,
            target,
            StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)),
            ["x"],
            trajectory_ids=np.asarray([0, 0, 0, 2], dtype=np.int64),
            group_size=2,
            bootstrap_samples=50,
            bootstrap_seed=77,
        )
        self.assertEqual(metrics["bootstrap_ci95"], repeated["bootstrap_ci95"])

    def test_block_bootstrap_reports_requested_metrics(self) -> None:
        values = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        result = bootstrap_block_ci(values, bootstrap_samples=20, bootstrap_seed=3)
        self.assertIn("block_1_mae", result)
        self.assertIn("block_4_mae", result)
        self.assertIn("error_amplification_ratio", result)


if __name__ == "__main__":
    unittest.main()
