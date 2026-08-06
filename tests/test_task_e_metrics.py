from __future__ import annotations

import unittest

from tasks.task_e.metrics import summarize_horizon_trends


class HorizonTrendTest(unittest.TestCase):
    def test_detects_nondecreasing_horizon_error(self) -> None:
        rows = [
            {
                "horizon_step": step,
                "mse_z": float(step),
                "mae_z": float(step) * 0.5,
                "rmse_z": float(step) ** 0.5,
                "mse_raw": float(step) * 2.0,
                "mae_raw": float(step),
                "rmse_raw": float(step) ** 0.5,
            }
            for step in range(1, 33)
        ]
        summary = summarize_horizon_trends(rows)

        self.assertEqual(summary["num_horizons"], 32)
        for metric in ("mse_z", "mae_z", "mse_raw", "mae_raw"):
            trend = summary["metrics"][metric]
            self.assertTrue(trend["is_nondecreasing"])
            self.assertTrue(trend["overall_increased"])
            self.assertEqual(trend["nondecreasing_adjacent_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
