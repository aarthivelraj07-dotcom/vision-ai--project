"""
Nationality Resolution and MRZ Document Extraction Module for Task 4.

Ethical Invariance:
Nationality represents a legal citizenship status, not a biological phenotype.
This module strictly extracts nationality from:
1. Explicit user declaration (primary route), OR
2. Standard ICAO Doc 9303 machine-readable travel document text fields.

It NEVER infers, guesses, or estimates nationality, ethnicity, or race
from facial imagery, skin color, or clothing.

Document Authenticity Notice:
Extracting text fields from an MRZ does NOT establish document authenticity.
Authenticity requires cryptographic CSCA signature verification or RFID inspection.
Extracted data is explicitly flagged as unverified document data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import re

from .constants import (
    ISO_COUNTRIES,
    _BY_ALPHA_3,
    _BY_ALPHA_2,
    _BY_NAME_NORM,
    AFRICAN_COUNTRY_CODES,
    classify_nationality_category,
    is_african_nation,
)


@dataclass
class NationalityInfo:
    """Represents declared or extracted nationality and provenance metadata."""
    country_name: str
    alpha_2: str
    alpha_3: str
    category: str  # "INDIAN", "US", "AFRICAN_REGION", "OTHER"
    is_african_region: bool
    source: str  # "User Declared" or "Extracted from Document MRZ (Unverified Document - Authenticity Not Established)"
    is_authentic_verified: bool = False  # Strictly False unless cryptographic verification is performed
    document_issuing_country: Optional[str] = None  # Separate from person nationality
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country_name": self.country_name,
            "alpha_2": self.alpha_2,
            "alpha_3": self.alpha_3,
            "category": self.category,
            "is_african_region": self.is_african_region,
            "source": self.source,
            "is_authentic_verified": self.is_authentic_verified,
            "document_issuing_country": self.document_issuing_country,
            "notes": self.notes,
        }


class NationalityResolver:
    """
    Resolves nationality strictly from declaration or document MRZ data.
    """

    @staticmethod
    def resolve_declaration(declaration: str) -> NationalityInfo:
        """
        Resolves an explicit user declaration against the ISO 3166-1 catalog.
        """
        clean = (declaration or "").strip()
        if not clean:
            return NationalityInfo(
                country_name="Unspecified / Undeclared",
                alpha_2="XX",
                alpha_3="XXX",
                category="OTHER",
                is_african_region=False,
                source="User Declared (Empty)",
                is_authentic_verified=False,
                notes="Nationality was not declared. Falling back to universal profile.",
            )

        code_upper = clean.upper()
        name_lower = clean.lower()

        # 1. Match by Alpha-3
        if code_upper in _BY_ALPHA_3:
            entry = _BY_ALPHA_3[code_upper]
            cat = classify_nationality_category(entry["alpha_3"])
            is_afr = is_african_nation(entry["alpha_3"])
            return NationalityInfo(
                country_name=entry["name"],
                alpha_2=entry["alpha_2"],
                alpha_3=entry["alpha_3"],
                category=cat,
                is_african_region=is_afr,
                source="User Declared",
                is_authentic_verified=False,
            )

        # 2. Match by Alpha-2
        if code_upper in _BY_ALPHA_2:
            entry = _BY_ALPHA_2[code_upper]
            cat = classify_nationality_category(entry["alpha_3"])
            is_afr = is_african_nation(entry["alpha_3"])
            return NationalityInfo(
                country_name=entry["name"],
                alpha_2=entry["alpha_2"],
                alpha_3=entry["alpha_3"],
                category=cat,
                is_african_region=is_afr,
                source="User Declared",
                is_authentic_verified=False,
            )

        # 3. Match by Name
        if name_lower in _BY_NAME_NORM:
            entry = _BY_NAME_NORM[name_lower]
            cat = classify_nationality_category(entry["alpha_3"])
            is_afr = is_african_nation(entry["alpha_3"])
            return NationalityInfo(
                country_name=entry["name"],
                alpha_2=entry["alpha_2"],
                alpha_3=entry["alpha_3"],
                category=cat,
                is_african_region=is_afr,
                source="User Declared",
                is_authentic_verified=False,
            )

        # 4. Fallback for custom or unlisted sovereign entry
        cat = classify_nationality_category(clean)
        is_afr = is_african_nation(clean)
        return NationalityInfo(
            country_name=clean.title(),
            alpha_2=code_upper[:2] if len(code_upper) >= 2 else "XX",
            alpha_3=code_upper[:3] if len(code_upper) >= 3 else "XXX",
            category=cat,
            is_african_region=is_afr,
            source="User Declared",
            is_authentic_verified=False,
            notes="Custom or non-standard ISO name provided by user.",
        )

    @staticmethod
    def parse_mrz_td3(line1: str, line2: str) -> Dict[str, Any]:
        """
        Parses standard ICAO Doc 9303 TD3 (Passport, 2 lines of 44 characters).
        
        Strict Disambiguation:
        - Line 1 chars 3..5: Document Issuing State / Authority (e.g. UTO, GBR, FRA)
        - Line 2 chars 11..13: Holder Nationality / Citizenship (e.g. UTO, IND, NGA)
        
        Notice: Extracted MRZ data does NOT verify authenticity.
        """
        l1 = (line1 or "").strip().upper()
        l2 = (line2 or "").strip().upper()

        if len(l1) < 5 or len(l2) < 13:
            return {
                "success": False,
                "error": "MRZ lines are too short or malformed. TD3 requires 44 characters per line.",
                "nationality_info": None,
                "issuing_country": None,
            }

        # Extract Issuing State: Line 1, index 2..5 (3 characters)
        issuing_code = l1[2:5].replace("<", "")

        # Extract Holder Nationality: Line 2, index 10..13 (3 characters)
        nationality_code = l2[10:13].replace("<", "")

        # Document Issuing Country metadata
        issuing_name = _BY_ALPHA_3.get(issuing_code, {}).get("name", issuing_code)

        # Resolve Holder Nationality
        if nationality_code in _BY_ALPHA_3:
            nat_entry = _BY_ALPHA_3[nationality_code]
            country_name = nat_entry["name"]
            alpha_2 = nat_entry["alpha_2"]
            alpha_3 = nat_entry["alpha_3"]
        else:
            country_name = f"Country Code ({nationality_code})"
            alpha_2 = "XX"
            alpha_3 = nationality_code

        cat = classify_nationality_category(alpha_3)
        is_afr = is_african_nation(alpha_3)

        notes = (
            f"Document Issuing State: {issuing_name} ({issuing_code}); "
            f"Holder Nationality: {country_name} ({alpha_3}). "
            "Extracted via optical MRZ parser. Document authenticity has NOT been cryptographically verified."
        )

        nat_info = NationalityInfo(
            country_name=country_name,
            alpha_2=alpha_2,
            alpha_3=alpha_3,
            category=cat,
            is_african_region=is_afr,
            source="Extracted from Document MRZ (Unverified Document - Authenticity Not Established)",
            is_authentic_verified=False,
            document_issuing_country=issuing_code,
            notes=notes,
        )

        return {
            "success": True,
            "document_issuing_country": issuing_code,
            "document_issuing_country_name": issuing_name,
            "holder_nationality_code": alpha_3,
            "holder_nationality_name": country_name,
            "nationality_info": nat_info,
        }
