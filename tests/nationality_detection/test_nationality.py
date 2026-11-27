"""
Tests for Nationality Resolution, Provenance Tracking, and MRZ Disambiguation.
"""

import unittest
from modules.nationality_detection.constants import (
    classify_nationality_category,
    is_african_nation,
    AFRICAN_COUNTRY_CODES,
)
from modules.nationality_detection.nationality import (
    NationalityInfo,
    NationalityResolver,
)


class TestNationalityModule(unittest.TestCase):
    """Verifies non-appearance nationality logic, catalog lookups, and MRZ parsing."""

    def test_01_classify_category_indian(self):
        self.assertEqual(classify_nationality_category("IND"), "INDIAN")
        self.assertEqual(classify_nationality_category("in"), "INDIAN")
        self.assertEqual(classify_nationality_category("India"), "INDIAN")
        self.assertEqual(classify_nationality_category("Republic of India"), "INDIAN")

    def test_02_classify_category_us(self):
        self.assertEqual(classify_nationality_category("USA"), "US")
        self.assertEqual(classify_nationality_category("us"), "US")
        self.assertEqual(classify_nationality_category("United States"), "US")
        self.assertEqual(classify_nationality_category("United States of America"), "US")

    def test_03_classify_category_african_region(self):
        # Test various African sovereign nations
        self.assertTrue(is_african_nation("NGA"))  # Nigeria
        self.assertTrue(is_african_nation("KEN"))  # Kenya
        self.assertTrue(is_african_nation("EGY"))  # Egypt
        self.assertTrue(is_african_nation("ZAF"))  # South Africa
        self.assertTrue(is_african_nation("Ghana"))
        self.assertTrue(is_african_nation("Ethiopia"))

        self.assertEqual(classify_nationality_category("Nigeria"), "AFRICAN_REGION")
        self.assertEqual(classify_nationality_category("NGA"), "AFRICAN_REGION")
        self.assertEqual(classify_nationality_category("Kenya"), "AFRICAN_REGION")
        self.assertEqual(classify_nationality_category("ZAF"), "AFRICAN_REGION")

        # Non-African nations must not be in African category
        self.assertFalse(is_african_nation("FRA"))
        self.assertFalse(is_african_nation("JPN"))
        self.assertFalse(is_african_nation("DEU"))

    def test_04_classify_category_other(self):
        self.assertEqual(classify_nationality_category("FRA"), "OTHER")
        self.assertEqual(classify_nationality_category("France"), "OTHER")
        self.assertEqual(classify_nationality_category("Japan"), "OTHER")
        self.assertEqual(classify_nationality_category("Germany"), "OTHER")
        self.assertEqual(classify_nationality_category("Brazil"), "OTHER")
        self.assertEqual(classify_nationality_category(""), "OTHER")

    def test_05_resolve_declaration_iso_lookups(self):
        # Lookup by Alpha-3
        res_ind = NationalityResolver.resolve_declaration("IND")
        self.assertEqual(res_ind.country_name, "India")
        self.assertEqual(res_ind.alpha_2, "IN")
        self.assertEqual(res_ind.category, "INDIAN")
        self.assertEqual(res_ind.source, "User Declared")
        self.assertFalse(res_ind.is_authentic_verified)

        # Lookup by Alpha-2
        res_us = NationalityResolver.resolve_declaration("US")
        self.assertEqual(res_us.country_name, "United States")
        self.assertEqual(res_us.alpha_3, "USA")
        self.assertEqual(res_us.category, "US")

        # Lookup by Name
        res_nga = NationalityResolver.resolve_declaration("Nigeria")
        self.assertEqual(res_nga.alpha_3, "NGA")
        self.assertEqual(res_nga.category, "AFRICAN_REGION")
        self.assertTrue(res_nga.is_african_region)

        # Empty declaration
        res_empty = NationalityResolver.resolve_declaration("")
        self.assertEqual(res_empty.category, "OTHER")
        self.assertIn("Empty", res_empty.source)

    def test_06_mrz_parsing_strict_disambiguation(self):
        # Standard TD3 Passport MRZ:
        # Line 1: Issuing state at chars 3..5: 'UTO' (Utopia)
        # Line 2: Holder nationality at chars 11..13: 'IND' (India)
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36IND7408122F1204159ZE184226B<<<<<10"

        parsed = NationalityResolver.parse_mrz_td3(line1, line2)
        self.assertTrue(parsed["success"])
        
        # Strictly verify that issuing authority != holder nationality
        self.assertEqual(parsed["document_issuing_country"], "UTO")
        self.assertEqual(parsed["holder_nationality_code"], "IND")
        self.assertEqual(parsed["holder_nationality_name"], "India")

        nat_info = parsed["nationality_info"]
        self.assertEqual(nat_info.country_name, "India")
        self.assertEqual(nat_info.category, "INDIAN")
        self.assertEqual(nat_info.document_issuing_country, "UTO")
        self.assertIn("Unverified Document", nat_info.source)
        # Ensure authenticity verification is explicitly false
        self.assertFalse(nat_info.is_authentic_verified)

    def test_07_mrz_malformed_lines(self):
        parsed = NationalityResolver.parse_mrz_td3("SHORT", "TOO_SHORT")
        self.assertFalse(parsed["success"])
        self.assertIn("malformed", parsed["error"])


if __name__ == "__main__":
    unittest.main()
