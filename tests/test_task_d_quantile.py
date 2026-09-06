from __future__ import annotations

import unittest

import numpy as np

from tasks.task_d.quantile import (
    QUANTILES,
    aggregate_pair_times,
    evaluate_quantiles,
    interpolate_first_crossing,
    select_tabular_features,
    summarize_set_features,
)


class TaskDQuantileTest(unittest.TestCase):
    def test_log_interpolation_recovers_exponential_crossing(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        growth = np.exp(2.0 * times)[None, :]
        crossing, observed, index = interpolate_first_crossing(
            growth, times, threshold=np.e, interpolate=True
        )
        self.assertTrue(observed[0])
        self.assertEqual(index[0], 1)
        self.assertAlmostEqual(crossing[0], 0.5)

    def test_system_label_uses_median_pair_lambda(self) -> None:
        # For threshold e^5, pair T_L values are crossing_time / 5.
        pair_tl = np.arange(1.0, 17.0)
        result = aggregate_pair_times(pair_tl * 5.0, threshold_power=5.0, min_valid_pairs=8)
        expected = 1.0 / np.median(1.0 / pair_tl)
        self.assertAlmostEqual(result["lyapunov_time_system_periods"], expected)
        self.assertEqual(result["num_valid_pairs"], 16)

    def test_insufficient_pair_crossings_has_no_system_label(self) -> None:
        crossing = np.array([5.0] * 7 + [np.nan] * 9)
        result = aggregate_pair_times(crossing, threshold_power=5.0, min_valid_pairs=8)
        self.assertTrue(np.isnan(result["lyapunov_time_system_periods"]))

    def test_set_summary_is_permutation_invariant(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.normal(size=(3, 16, 5))
        permuted = values[:, rng.permutation(16)]
        names = [f"x{i}" for i in range(5)]
        summary, output_names = summarize_set_features(values, names)
        permuted_summary, _ = summarize_set_features(permuted, names)
        np.testing.assert_allclose(summary, permuted_summary)
        self.assertEqual(len(output_names), 20)

    def test_first_feature_mode_uses_only_first_variation(self) -> None:
        values = np.arange(2 * 16 * 3, dtype=np.float64).reshape(2, 16, 3)
        selected, output_names = select_tabular_features(
            values, ["a", "b", "c"], mode="first"
        )
        np.testing.assert_array_equal(selected, values[:, 0, :])
        self.assertEqual(output_names, ["a", "b", "c"])

    def test_quantile_metrics_report_calibration_and_no_crossing(self) -> None:
        y = np.arange(8, dtype=np.float64)
        prediction = np.column_stack((y - 1.0, y, y + 1.0))
        metrics = evaluate_quantiles(y, prediction)
        self.assertEqual(tuple(QUANTILES), (0.25, 0.5, 0.75))
        self.assertEqual(metrics["quantile_crossing_rate"], 0.0)
        self.assertEqual(metrics["median_mae_log"], 0.0)


if __name__ == "__main__":
    unittest.main()
