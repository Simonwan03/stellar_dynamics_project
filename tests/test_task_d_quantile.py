from __future__ import annotations

import unittest

import numpy as np

from tasks.task_d.quantile import (
    aggregate_pair_times,
    evaluate_regression,
    interpolate_first_crossing,
    select_tabular_features,
    summarize_set_features,
)


class TaskDRegressionTest(unittest.TestCase):
    def test_log_interpolation_recovers_exponential_crossing(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        growth = np.exp(2.0 * times)[None, :]
        crossing, observed, index = interpolate_first_crossing(
            growth, times, threshold=np.e, interpolate=True
        )
        self.assertTrue(observed[0])
        self.assertEqual(index[0], 1)
        self.assertAlmostEqual(crossing[0], 0.5)

    def test_system_label_is_direct_median_crossing_time(self) -> None:
        crossings = np.arange(1.0, 17.0)
        result = aggregate_pair_times(
            crossings, threshold_power=5.0, min_valid_pairs=8
        )
        self.assertAlmostEqual(
            result["median_decorrelation_time_periods"], np.median(crossings)
        )
        self.assertEqual(result["num_valid_pairs"], 16)

    def test_insufficient_pair_crossings_has_no_system_label(self) -> None:
        crossing = np.array([5.0] * 7 + [np.nan] * 9)
        result = aggregate_pair_times(crossing, min_valid_pairs=8)
        self.assertTrue(np.isnan(result["median_decorrelation_time_periods"]))

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

    def test_regression_metrics_and_bootstrap_ci(self) -> None:
        y = np.arange(8, dtype=np.float64)
        metrics = evaluate_regression(y, y, bootstrap_samples=50, seed=4)
        self.assertEqual(metrics["mae_log"], 0.0)
        self.assertEqual(metrics["rmse_log"], 0.0)
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["factor_2_accuracy"], 1.0)
        self.assertEqual(metrics["bootstrap_95_ci"]["mae_log"]["low"], 0.0)


if __name__ == "__main__":
    unittest.main()
