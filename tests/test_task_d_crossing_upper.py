from __future__ import annotations

import unittest

import numpy as np

from tasks.task_d.main import build_arg_parser
from tasks.task_d.pipeline import (
    DEFAULT_VALID_PAIRS,
    _aggregate_tau,
    _compute_group_divergence_label,
    compute_pair_aratio_divergence,
    compute_pair_divergence_time,
)


class TaskDCrossingUpperTest(unittest.TestCase):
    def test_min_aggregation_returns_earliest_finite_crossing(self) -> None:
        value = _aggregate_tau([12.0, 3.0, np.nan, 8.0], aggregation="min")
        self.assertEqual(value, 3.0)

    def test_max_lambda_gives_minimum_instability_time(self) -> None:
        times_1d = np.arange(30, dtype=np.float64)
        rates = 0.10 + 0.01 * np.arange(16, dtype=np.float64)
        aratios = np.zeros((32, times_1d.size, 1), dtype=np.float64)
        for pair_idx, rate in enumerate(rates):
            aratios[pair_idx + 16, :, 0] = 1.0e-9 * np.exp(rate * times_1d)
        times = np.broadcast_to(times_1d[None, :], (32, times_1d.size)).copy()
        label = _compute_group_divergence_label(
            positions_group=None,
            velocities_group=None,
            masses_group=None,
            times_group=times,
            system_id=0,
            valid_pairs=DEFAULT_VALID_PAIRS,
            threshold=1000.0,
            growth_lower=1.0,
            growth_upper=100.0,
            min_growth_points=5,
            aggregation="median",
            censoring_rule="median_pair",
            lambda_r2_min=0.0,
            min_valid_lambda_pairs=8,
            max_pair_log_mad=2.0,
            aratios_group=aratios,
            label_observable="aratio",
            lyapunov_aggregation="max",
        )
        expected_lambda = float(rates.max())
        self.assertEqual(label["valid_lambda_pair_count"], 16)
        self.assertAlmostEqual(label["lambda_observable"], expected_lambda, places=6)
        self.assertAlmostEqual(label["lyapunov_time_label"], 1.0 / expected_lambda, places=6)

    def test_min_instability_does_not_require_pair_consistency(self) -> None:
        times_1d = np.arange(30, dtype=np.float64)
        aratios = np.zeros((32, times_1d.size, 1), dtype=np.float64)
        aratios[16, :, 0] = 1.0e-9 * np.exp(0.1 * times_1d)
        for pair_idx in range(1, 16):
            aratios[pair_idx + 16, :, 0] = 1.0e-9 * np.exp(-0.1 * times_1d)
        times = np.broadcast_to(times_1d[None, :], (32, times_1d.size)).copy()
        label = _compute_group_divergence_label(
            positions_group=None,
            velocities_group=None,
            masses_group=None,
            times_group=times,
            system_id=0,
            valid_pairs=DEFAULT_VALID_PAIRS,
            threshold=1000.0,
            growth_lower=1.0,
            growth_upper=100.0,
            min_growth_points=5,
            aggregation="median",
            censoring_rule="median_pair",
            lambda_r2_min=0.0,
            min_valid_lambda_pairs=1,
            max_pair_log_mad=2.0,
            aratios_group=aratios,
            label_observable="aratio",
            lyapunov_aggregation="max",
            require_pair_consistency=False,
        )
        self.assertEqual(label["valid_lambda_pair_count"], 1)
        self.assertTrue(label["lyapunov_label_valid"])

    def test_crossing_upper_alias_is_available(self) -> None:
        args = build_arg_parser().parse_args(
            ["--target-mode", "crossing_upper", "--aggregation", "min"]
        )
        self.assertEqual(args.target_mode, "crossing_upper")
        self.assertEqual(args.aggregation, "min")

    def test_aratio_pair_fits_observable_growth_rate(self) -> None:
        times = np.arange(12, dtype=np.float64)
        growth = np.exp(0.2 * times)
        aratios = np.stack(
            [np.zeros_like(times), 1.0e-9 * growth],
            axis=0,
        )[..., None]
        result = compute_pair_aratio_divergence(
            aratios=aratios,
            times=np.stack([times, times]),
            idx_i=0,
            idx_j=1,
            threshold=8.0,
            growth_lower=1.0,
            growth_upper=10.0,
            min_growth_points=5,
        )
        self.assertAlmostEqual(result["lambda_aratio"], 0.2, places=6)
        self.assertFalse(result["censored"])
        self.assertEqual(result["tau"], times[12 - 1])

    def test_threshold_equality_counts_as_proxy_crossing(self) -> None:
        times = np.arange(3, dtype=np.float64)
        aratios = np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[1.0], [1000.0], [2000.0]],
            ],
            dtype=np.float64,
        )
        result = compute_pair_aratio_divergence(
            aratios=aratios,
            times=np.stack([times, times]),
            idx_i=0,
            idx_j=1,
            threshold=1000.0,
            min_growth_points=2,
        )
        self.assertFalse(result["censored"])
        self.assertEqual(result["tau"], 1.0)

    def test_smax_proxy_uses_inner_semimajor_axis(self) -> None:
        times = np.stack([np.arange(3, dtype=np.float64)] * 2)
        smaxes = np.ones((2, 3, 2), dtype=np.float64)
        smaxes[1, :, 0] = [2.0, 1001.0, 2001.0]
        # The outer orbit differs strongly but must not affect inner_index=0.
        smaxes[1, :, 1] = [10.0, 10.0, 10.0]
        tau, censored, ratio, _ = compute_pair_divergence_time(
            smaxes=smaxes,
            times=times,
            idx_i=0,
            idx_j=1,
            threshold=1000.0,
        )
        self.assertFalse(censored)
        self.assertEqual(tau, 1.0)
        self.assertEqual(ratio[1], 1000.0)

    def test_proxy_is_the_default_task(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.label_observable, "smax")
        self.assertEqual(args.target_mode, "lyapunov_proxy")
        self.assertEqual(args.aggregation, "min")
        self.assertEqual(args.censoring_rule, "all_pairs")

    def test_fitted_lyapunov_time_compatibility_mode_is_available(self) -> None:
        args = build_arg_parser().parse_args(["--target-mode", "fitted_lyapunov_time"])
        self.assertEqual(args.target_mode, "fitted_lyapunov_time")

    def test_lambda_fast_target_is_available(self) -> None:
        args = build_arg_parser().parse_args(["--target-mode", "lambda_fast"])
        self.assertEqual(args.target_mode, "lambda_fast")


if __name__ == "__main__":
    unittest.main()
