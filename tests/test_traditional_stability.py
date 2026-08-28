import unittest

import numpy as np
import pandas as pd

from tasks.common.traditional_stability import triple_stability, vynatheya2022_stability
from tasks.task_a.main import aggregate_variation_stability


class EquationTwoStabilityTest(unittest.TestCase):
    def test_matches_equation_two_directly(self) -> None:
        a_in = 2.0
        a_out = 30.0
        e_out = 0.4
        inclination = 0.7
        m1, m2, m3 = 1.0, 2.0, 1.5

        result = triple_stability(a_in, a_out, e_out, inclination, m1, m2, m3)

        q_out = m3 / (m1 + m2)
        expected_actual = a_out * (1.0 - e_out) / a_in
        expected_critical = (
            2.8
            * ((1.0 + q_out) * (1.0 + e_out) / np.sqrt(1.0 - e_out)) ** (2.0 / 5.0)
            * (1.0 - 0.3 * inclination / np.pi)
        )
        self.assertAlmostEqual(float(result["Rp_over_a_in"]), expected_actual)
        self.assertAlmostEqual(float(result["Rp_crit_over_a_in"]), expected_critical)
        self.assertAlmostEqual(
            float(result["stability_margin"]),
            float(np.log(expected_actual / expected_critical)),
        )
        self.assertEqual(bool(result["stable"]), expected_actual > expected_critical)

    def test_semimajor_axis_form_is_algebraically_equivalent(self) -> None:
        result = triple_stability(1.0, 10.0, 0.5, 0.0, 1.0, 1.0, 1.0)
        pericentre_margin = np.log(result["Rp_over_a_in"] / result["Rp_crit_over_a_in"])
        semimajor_margin = np.log(result["a_out_over_a_in"] / result["a_out_over_a_in_crit"])
        self.assertAlmostEqual(float(pericentre_margin), float(semimajor_margin))


class Vynatheya2022StabilityTest(unittest.TestCase):
    def test_matches_equations_three_and_four_directly(self) -> None:
        a_in = 2.0
        a_out = 30.0
        e_in = 0.1
        e_out = 0.4
        inclination = np.pi / 2.0
        m1, m2, m3 = 1.0, 2.0, 1.5

        result = vynatheya2022_stability(
            a_in, a_out, e_in, e_out, inclination, m1, m2, m3
        )

        q_out = m3 / (m1 + m2)
        expected_e_max = 1.0
        expected_e_avg = 0.5
        expected_e_tilde = 0.5
        expected_y = (a_out / a_in) * (1.0 - e_out) / (1.0 + expected_e_tilde)
        expected_inclination_factor = (
            ((1.0 - 0.2 * expected_e_tilde + e_out) / 8.0)
            * (np.cos(inclination) - 1.0)
            + 1.0
        )
        expected_y_crit = (
            2.4
            * (
                (1.0 + q_out)
                / ((1.0 + expected_e_tilde) * np.sqrt(1.0 - e_out))
            )
            ** (2.0 / 5.0)
            * expected_inclination_factor
        )
        self.assertAlmostEqual(float(result["e_in_max"]), expected_e_max)
        self.assertAlmostEqual(float(result["e_in_avg"]), expected_e_avg)
        self.assertAlmostEqual(float(result["e_in_tilde"]), expected_e_tilde)
        self.assertAlmostEqual(float(result["Y_tilde"]), expected_y)
        self.assertAlmostEqual(float(result["Y_tilde_crit"]), expected_y_crit)
        self.assertAlmostEqual(
            float(result["stability_margin"]), np.log(expected_y / expected_y_crit)
        )

    def test_lk_radicand_is_clipped_outside_lk_window(self) -> None:
        result = vynatheya2022_stability(1.0, 10.0, 0.3, 0.2, 0.0, 1.0, 1.0, 1.0)
        self.assertEqual(float(result["e_in_max"]), 0.0)
        self.assertEqual(float(result["e_in_avg"]), 0.0)
        self.assertAlmostEqual(float(result["e_in_tilde"]), 0.3)
        self.assertTrue(np.isfinite(float(result["stability_margin"])))

    def test_rejects_invalid_inner_eccentricity(self) -> None:
        with self.assertRaisesRegex(ValueError, "Inner eccentricity"):
            vynatheya2022_stability(1.0, 10.0, 1.0, 0.2, 0.0, 1.0, 1.0, 1.0)

    def test_boundary_uses_strict_stability_inequality(self) -> None:
        probe = vynatheya2022_stability(
            1.0, 10.0, 0.2, 0.3, np.pi / 3.0, 1.0, 1.0, 1.0
        )
        boundary_ratio = float(probe["a_out_over_a_in_crit"])
        boundary = vynatheya2022_stability(
            1.0, boundary_ratio, 0.2, 0.3, np.pi / 3.0, 1.0, 1.0, 1.0
        )
        self.assertAlmostEqual(float(boundary["stability_margin"]), 0.0)
        self.assertFalse(bool(boundary["stable"]))


class TaskAEquationAggregationTest(unittest.TestCase):
    def test_fraction_and_all_stable_aggregation(self) -> None:
        predictions = pd.DataFrame(
            {
                "group_id": [0, 0, 1, 1],
                "variation_id": [0, 1, 0, 1],
                "stable": [1, 0, 1, 1],
                "stability_margin": [0.4, -0.2, 0.2, 0.3],
            }
        )
        groups = aggregate_variation_stability(predictions, group_size=2)
        self.assertAlmostEqual(float(groups.loc[0, "predicted_stable_fraction"]), 0.5)
        self.assertEqual(int(groups.loc[0, "prediction_stable"]), 0)
        self.assertAlmostEqual(float(groups.loc[0, "stability_margin"]), -0.2)
        self.assertAlmostEqual(float(groups.loc[1, "predicted_stable_fraction"]), 1.0)
        self.assertEqual(int(groups.loc[1, "prediction_stable"]), 1)

    def test_rejects_incomplete_groups(self) -> None:
        predictions = pd.DataFrame(
            {
                "group_id": [0],
                "variation_id": [0],
                "stable": [1],
                "stability_margin": [0.4],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly 2 unique variations"):
            aggregate_variation_stability(predictions, group_size=2)


if __name__ == "__main__":
    unittest.main()
