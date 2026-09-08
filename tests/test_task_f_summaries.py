import unittest

import numpy as np

from tasks.task_c.survival import task_b_spock_summary
from tasks.task_f.summaries import summarize_observed_prefix


class TaskFSummaryTests(unittest.TestCase):
    def test_unmasked_summary_matches_task_b_summary(self) -> None:
        values = np.asarray(
            [[[0.0, 1.0], [1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]], dtype=np.float32
        )
        observed = np.ones(values.shape[:2], dtype=bool)
        expected = task_b_spock_summary(__import__("torch").from_numpy(values)).numpy()
        np.testing.assert_allclose(summarize_observed_prefix(values, observed), expected, rtol=1e-6, atol=1e-6)

    def test_masked_summary_ignores_hidden_values(self) -> None:
        values = np.asarray(
            [[[1.0], [100.0], [3.0], [200.0], [5.0]]], dtype=np.float32
        )
        observed = np.asarray([[True, False, True, False, True]])
        summary = summarize_observed_prefix(values, observed)
        self.assertAlmostEqual(float(summary[0, 0]), 1.0)
        self.assertAlmostEqual(float(summary[0, 1]), 5.0)
        self.assertAlmostEqual(float(summary[0, 2]), 3.0)
        self.assertAlmostEqual(float(summary[0, 5]), 1.0)
        self.assertAlmostEqual(float(summary[0, 6]), 5.0)


if __name__ == "__main__":
    unittest.main()
