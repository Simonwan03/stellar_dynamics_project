from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.common.pretrained_triple_mlp import (
    PRETRAINED_GHOST_MLP_SHA256,
    PRETRAINED_MLP_ARTIFACTS,
    PRETRAINED_MLP_SHA256,
    build_pretrained_mlp_features,
    load_pretrained_mlp_initial_conditions,
)
from tasks.task_a.main import aggregate_pretrained_mlp_predictions


class PretrainedTripleMlpFeatureTest(unittest.TestCase):
    def test_both_official_1_2_2_artifacts_are_registered(self) -> None:
        self.assertEqual(
            PRETRAINED_MLP_ARTIFACTS[PRETRAINED_MLP_SHA256]["variant"],
            "semimajor_axis_change",
        )
        self.assertEqual(
            PRETRAINED_MLP_ARTIFACTS[PRETRAINED_GHOST_MLP_SHA256]["variant"],
            "ghost_orbit_divergence",
        )

    def test_exact_upstream_feature_order_and_inclination_normalization(self) -> None:
        features = build_pretrained_mlp_features(
            q_in=np.asarray([0.25, 0.5]),
            q_out=np.asarray([2.0, 3.0]),
            alpha=np.asarray([0.1, 0.2]),
            e_in=np.asarray([0.3, 0.4]),
            e_out=np.asarray([0.5, 0.6]),
            mutual_inclination=np.asarray([0.0, np.pi / 2.0]),
        )
        np.testing.assert_allclose(
            features,
            np.asarray(
                [
                    [0.25, 2.0, 0.1, 0.3, 0.5, 0.0],
                    [0.5, 3.0, 0.2, 0.4, 0.6, 0.5],
                ]
            ),
        )

    def test_unverified_pickle_is_rejected_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pkl"
            model_path.write_bytes(b"not an official pickle")
            with self.assertRaisesRegex(ValueError, "Refusing to load unverified"):
                load_pretrained_mlp_initial_conditions(
                    Path(directory) / "unused.hdf5", model_path, group_size=32
                )


class PretrainedTripleMlpAggregationTest(unittest.TestCase):
    def test_probability_aggregation_matches_task_targets(self) -> None:
        predictions = pd.DataFrame(
            {
                "group_id": [0, 0, 1, 1],
                "variation_id": [0, 1, 0, 1],
                "stable": [1, 0, 1, 1],
                "probability_stable": [0.9, 0.4, 0.8, 0.7],
                "probability_unstable": [0.1, 0.6, 0.2, 0.3],
                "in_training_domain": [True, True, True, False],
            }
        )
        groups = aggregate_pretrained_mlp_predictions(predictions, group_size=2)
        self.assertAlmostEqual(float(groups.loc[0, "predicted_stable_fraction"]), 0.65)
        self.assertAlmostEqual(float(groups.loc[0, "probability_stable"]), 0.4)
        self.assertEqual(int(groups.loc[0, "prediction_stable"]), 0)
        self.assertAlmostEqual(float(groups.loc[1, "predicted_stable_fraction"]), 0.75)
        self.assertAlmostEqual(float(groups.loc[1, "probability_stable"]), 0.7)
        self.assertEqual(int(groups.loc[1, "prediction_stable"]), 1)
        self.assertEqual(int(groups.loc[1, "variations_inside_training_domain"]), 1)


if __name__ == "__main__":
    unittest.main()
