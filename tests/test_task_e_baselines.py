from __future__ import annotations

import unittest

import numpy as np

from tasks.task_e.baselines import forecast_linear_extrapolation_batch
from tasks.task_e.data import StandardScaler


class LinearExtrapolationBaselineTest(unittest.TestCase):
    def test_recovers_linear_trend(self) -> None:
        context = np.arange(6, dtype=np.float32).reshape(1, 6, 1)
        scaler = StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        prediction = forecast_linear_extrapolation_batch(
            context,
            pred_len=3,
            scaler=scaler,
            fit_window=4,
            project_angle_pairs=False,
        )
        np.testing.assert_allclose(prediction[0, :, 0], [6.0, 7.0, 8.0], atol=1e-6)

    def test_one_step_fit_is_persistence(self) -> None:
        context = np.asarray([[[2.0], [4.0], [8.0]]], dtype=np.float32)
        scaler = StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        prediction = forecast_linear_extrapolation_batch(
            context,
            pred_len=2,
            scaler=scaler,
            fit_window=1,
            project_angle_pairs=False,
        )
        np.testing.assert_allclose(prediction[0, :, 0], [8.0, 8.0], atol=1e-6)

    def test_angle_projection_uses_transformed_coordinates(self) -> None:
        names = ["APs_sin_0", "APs_cos_0"]
        # Nontrivial channel scaling catches accidental projection in z-space.
        scaler = StandardScaler(
            np.asarray([1.0, -2.0], dtype=np.float32),
            np.asarray([2.0, 0.5], dtype=np.float32),
        )
        context_raw = np.asarray(
            [[[1.0, -1.0], [2.0, -1.5], [3.0, -2.0], [4.0, -2.5]]],
            dtype=np.float32,
        )
        context_z = scaler.transform(context_raw)
        prediction_z = forecast_linear_extrapolation_batch(
            context_z,
            pred_len=2,
            scaler=scaler,
            feature_names=names,
            fit_window=3,
            project_angle_pairs=True,
        )
        prediction_raw = scaler.inverse_transform(prediction_z)
        norms = np.linalg.norm(prediction_raw[0], axis=-1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_rejects_nonfinite_context(self) -> None:
        context = np.zeros((1, 4, 1), dtype=np.float32)
        context[0, 0, 0] = np.nan
        scaler = StandardScaler(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            forecast_linear_extrapolation_batch(context, 2, scaler=scaler)


if __name__ == "__main__":
    unittest.main()
