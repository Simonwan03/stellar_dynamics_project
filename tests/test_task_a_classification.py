from __future__ import annotations

import unittest

import numpy as np

from tasks.common.experiment_utils import select_binary_threshold_by_f1


class TaskAClassificationTest(unittest.TestCase):
    def test_threshold_is_selected_from_validation_f1(self) -> None:
        y_val = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        probability = np.asarray([0.1, 0.2, 0.3, 0.4, 0.8], dtype=np.float64)
        self.assertAlmostEqual(select_binary_threshold_by_f1(y_val, probability), 0.4)

    def test_threshold_falls_back_for_single_class_validation(self) -> None:
        y_val = np.zeros(4, dtype=np.int64)
        probability = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.assertAlmostEqual(select_binary_threshold_by_f1(y_val, probability), 0.5)


if __name__ == "__main__":
    unittest.main()
