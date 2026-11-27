"""
Tests for Age Resolution, User-Provided Age, and Unavailable Fallback.
"""

import unittest
from modules.nationality_detection.age import AgeEstimator, AgeResult


class TestAgeModule(unittest.TestCase):
    """Verifies user-provided age declaration, demographic brackets, and unavailable fallback."""

    def test_01_undeclared_age_returns_unavailable(self):
        # When no age is provided by the user
        result = AgeEstimator.resolve_age(declared_age=None, person_id="Person 1")
        self.assertIsInstance(result, AgeResult)
        self.assertFalse(result.is_available)
        self.assertFalse(result.is_user_provided)
        self.assertEqual(result.bracket, "Age estimation unavailable")
        self.assertEqual(result.source, "None")
        self.assertIn("disabled pending a validated", result.reason)

    def test_02_zero_or_negative_age_returns_unavailable(self):
        result_zero = AgeEstimator.resolve_age(declared_age=0)
        self.assertFalse(result_zero.is_available)
        self.assertEqual(result_zero.bracket, "Age estimation unavailable")

        result_neg = AgeEstimator.resolve_age(declared_age=-5)
        self.assertFalse(result_neg.is_available)
        self.assertEqual(result_neg.bracket, "Age estimation unavailable")

    def test_03_user_provided_young_adult_age(self):
        # Test 19-year-old case
        result = AgeEstimator.resolve_age(declared_age=19, person_id="Person 1")
        self.assertTrue(result.is_available)
        self.assertTrue(result.is_user_provided)
        self.assertEqual(result.declared_age, 19)
        self.assertEqual(result.source, "User-provided age")
        self.assertEqual(result.bracket, "19 years (Young Adult)")

    def test_04_user_provided_demographic_brackets(self):
        # Child (< 18)
        res_child = AgeEstimator.resolve_age(declared_age=12)
        self.assertEqual(res_child.bracket, "12 years (Child / Adolescent)")

        # Middle-aged (36-55)
        res_mid = AgeEstimator.resolve_age(declared_age=42)
        self.assertEqual(res_mid.bracket, "42 years (Middle-aged Adult)")

        # Senior (55+)
        res_senior = AgeEstimator.resolve_age(declared_age=68)
        self.assertEqual(res_senior.bracket, "68 years (Senior Adult)")

    def test_05_multi_person_age_isolation(self):
        # Person 1 has age 19 declared
        res_p1 = AgeEstimator.resolve_age(declared_age=19, person_id="Person 1")
        # Person 2 has no age declared
        res_p2 = AgeEstimator.resolve_age(declared_age=None, person_id="Person 2")

        self.assertTrue(res_p1.is_available)
        self.assertEqual(res_p1.declared_age, 19)
        self.assertEqual(res_p1.person_id, "Person 1")

        self.assertFalse(res_p2.is_available)
        self.assertIsNone(res_p2.declared_age)
        self.assertEqual(res_p2.person_id, "Person 2")
        self.assertEqual(res_p2.bracket, "Age estimation unavailable")


if __name__ == "__main__":
    unittest.main()
