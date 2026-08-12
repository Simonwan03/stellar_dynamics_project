import unittest

import numpy as np

from tasks.common.traditional_stability import triple_stability


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


if __name__ == "__main__":
    unittest.main()
