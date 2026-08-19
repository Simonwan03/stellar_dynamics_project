from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from tasks.task_e import main as task_e_main
from tasks.task_e.data import ForecastWindowDataset, RollingWindowDataset, StandardScaler
from tasks.task_e.metrics import compute_forecast_metrics


class RollingEvaluationTest(unittest.TestCase):
    def _synthetic_base(self) -> ForecastWindowDataset:
        base = ForecastWindowDataset.__new__(ForecastWindowDataset)
        base.record_trajectories = np.asarray([0], dtype=np.int64)
        base.record_starts = np.asarray([0], dtype=np.int64)
        base.valid_lengths = np.asarray([12], dtype=np.int64)
        base.sources = [SimpleNamespace(time_steps=12)]
        base.feature_metadata = {"num_features": 1}
        base.scaler = StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        base._handles = {}

        def load_raw_window(index: int, total_len: int | None = None) -> np.ndarray:
            del index
            length = int(total_len or 4)
            return np.arange(length, dtype=np.float32).reshape(length, 1)

        base.load_raw_window = load_raw_window  # type: ignore[method-assign]
        base.load_transformed_window = load_raw_window  # type: ignore[method-assign]
        return base

    def test_rolling_dataset_requires_complete_future(self) -> None:
        base = self._synthetic_base()
        rolling = RollingWindowDataset(base, context_len=4, block_len=2, num_blocks=4)
        self.assertEqual(len(rolling), 1)
        self.assertEqual(tuple(rolling[0]["context"].shape), (4, 1))
        self.assertEqual(tuple(rolling[0]["target"].shape), (8, 1))

    def test_recursive_rollout_does_not_use_target_context(self) -> None:
        base = self._synthetic_base()

        def forecast(context: np.ndarray) -> np.ndarray:
            return np.repeat(context[:, -1:, :], 2, axis=1)

        metrics = task_e_main.run_rolling_evaluation(
            base,
            forecast,
            context_len=4,
            block_len=2,
            num_blocks=4,
            batch_size=1,
            num_workers=0,
            log_every_batches=0,
        )
        self.assertEqual(metrics["num_windows"], 1)
        self.assertEqual(len(metrics["rolling_horizon_mae"]), 8)
        self.assertAlmostEqual(metrics["block_1_mae"], 1.5)
        self.assertAlmostEqual(metrics["block_4_mae"], 7.5)
        self.assertAlmostEqual(metrics["error_amplification_ratio"], 7.5 / 1.5)


class TaskEMetricExtensionTest(unittest.TestCase):
    def test_current_protocol_defaults(self) -> None:
        args = task_e_main.build_arg_parser().parse_args(["--output_dir", "/tmp/task_e_test"])
        self.assertEqual(args.context_len, 512)
        self.assertEqual(args.pred_len, 128)
        self.assertEqual(args.rolling_blocks, 4)
        self.assertEqual(args.train_windows_per_trajectory, 8)
        self.assertEqual(args.val_windows_per_trajectory, 4)
        self.assertEqual(args.test_windows_per_trajectory, 4)
        self.assertEqual(args.rolling_windows_per_trajectory, 2)
        self.assertEqual(args.bootstrap_samples, 1000)
        self.assertFalse(args.run_rolling)
        self.assertNotIn("panda_adapter", task_e_main.ALL_MODELS)

    def test_pearson_and_unit_circle_error_are_reported(self) -> None:
        target = np.zeros((2, 3, 15), dtype=np.float32)
        prediction = target.copy()
        feature_names = [
            "aratios_0",
            "mratios_0",
            "mratios_1",
            "eccs_0",
            "eccs_1",
            "incs_0",
            "incs_1",
            "APs_sin_0",
            "APs_sin_1",
            "APs_cos_0",
            "APs_cos_1",
            "LANs_sin_0",
            "LANs_sin_1",
            "LANs_cos_0",
            "LANs_cos_1",
        ]
        prediction[:, :, 7] = 0.6
        prediction[:, :, 9] = 0.8
        prediction[:, :, 8] = 0.0
        prediction[:, :, 10] = 1.0
        prediction[:, :, 11] = 0.0
        prediction[:, :, 13] = 0.0
        prediction[:, :, 12] = 0.0
        prediction[:, :, 14] = 2.0
        metrics, horizon_rows, _ = compute_forecast_metrics(
            prediction,
            target,
            StandardScaler(np.zeros(15, dtype=np.float32), np.ones(15, dtype=np.float32)),
            feature_names,
        )
        self.assertIn("pearson", metrics)
        self.assertEqual(len(metrics["horizon_mae"]), 3)
        self.assertAlmostEqual(metrics["unit_circle_error"], 1.0)
        self.assertEqual(len(horizon_rows), 3)


if __name__ == "__main__":
    unittest.main()
