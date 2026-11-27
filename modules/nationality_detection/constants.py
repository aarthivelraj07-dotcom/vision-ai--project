"""
Constants, Country Catalogs, and Category Mapping for Task 4.

Defines ISO 3166-1 sovereign nation entries, African regional country groupings,
verified FER emotion labels, and perceptual CIELAB dress color palettes.
"""

from typing import Dict, Any, List, Optional, Set

# Universal 7 facial emotion classes from FER-2013 / model_creation.ipynb (exact order verified)
EMOTION_CLASSES: List[str] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Standard dress color categories and representative CIELAB / Hex anchors
COLOR_PALETTE_CIELAB: Dict[str, Dict[str, Any]] = {
    "Black": {"L": 15.0, "a": 0.0, "b": 0.0, "hex": "#1A1A1A"},
    "White": {"L": 95.0, "a": 0.0, "b": 0.0, "hex": "#F5F5F5"},
    "Gray": {"L": 55.0, "a": 0.0, "b": 0.0, "hex": "#808080"},
    "Red": {"L": 53.0, "a": 65.0, "b": 45.0, "hex": "#D32F2F"},
    "Navy Blue": {"L": 25.0, "a": 12.0, "b": -35.0, "hex": "#1B2A4A"},
    "Blue": {"L": 50.0, "a": -5.0, "b": -45.0, "hex": "#1976D2"},
    "Green": {"L": 50.0, "a": -55.0, "b": 35.0, "hex": "#388E3C"},
    "Yellow": {"L": 85.0, "a": -10.0, "b": 75.0, "hex": "#FBC02D"},
    "Orange": {"L": 65.0, "a": 40.0, "b": 60.0, "hex": "#F57C00"},
    "Purple": {"L": 40.0, "a": 45.0, "b": -30.0, "hex": "#7B1FA2"},
    "Pink": {"L": 75.0, "a": 45.0, "b": 5.0, "hex": "#EC407A"},
    "Brown": {"L": 38.0, "a": 20.0, "b": 30.0, "hex": "#5D4037"},
    "Cyan": {"L": 70.0, "a": -35.0, "b": -15.0, "hex": "#0097A7"},
}

# 54 Sovereign African Nations (ISO 3166-1 alpha-3 codes)
# Treated as a regional geographic category grouping sovereign states, not a country or nationality.
AFRICAN_COUNTRY_CODES: Set[str] = {
    "DZA",  # Algeria
    "AGO",  # Angola
    "BEN",  # Benin
    "BWA",  # Botswana
    "BFA",  # Burkina Faso
    "BDI",  # Burundi
    "CPV",  # Cabo Verde
    "CMR",  # Cameroon
    "CAF",  # Central African Republic
    "TCD",  # Chad
    "COM",  # Comoros
    "COG",  # Republic of the Congo
    "COD",  # Democratic Republic of the Congo
    "DJI",  # Djibouti
    "EGY",  # Egypt
    "GNQ",  # Equatorial Guinea
    "ERI",  # Eritrea
    "SWZ",  # Eswatini
    "ETH",  # Ethiopia
    "GAB",  # Gabon
    "GMB",  # Gambia
    "GHA",  # Ghana
    "GIN",  # Guinea
    "GNB",  # Guinea-Bissau
    "CIV",  # Ivory Coast (Côte d'Ivoire)
    "KEN",  # Kenya
    "LSO",  # Lesotho
    "LBR",  # Liberia
    "LBY",  # Libya
    "MDG",  # Madagascar
    "MWI",  # Malawi
    "MLI",  # Mali
    "MRT",  # Mauritania
    "MUS",  # Mauritius
    "MAR",  # Morocco
    "MOZ",  # Mozambique
    "NAM",  # Namibia
    "NER",  # Niger
    "NGA",  # Nigeria
    "RWA",  # Rwanda
    "STP",  # Sao Tome and Principe
    "SEN",  # Senegal
    "SYC",  # Seychelles
    "SLE",  # Sierra Leone
    "SOM",  # Somalia
    "ZAF",  # South Africa
    "SSD",  # South Sudan
    "SDN",  # Sudan
    "TZA",  # Tanzania
    "TGO",  # Togo
    "TUN",  # Tunisia
    "UGA",  # Uganda
    "ZMB",  # Zambia
    "ZWE",  # Zimbabwe
}

# Common ISO 3166-1 Sovereign Nations Catalog (Name, Alpha-2, Alpha-3)
ISO_COUNTRIES: List[Dict[str, str]] = [
    # Top target nations
    {"name": "India", "alpha_2": "IN", "alpha_3": "IND"},
    {"name": "United States", "alpha_2": "US", "alpha_3": "USA"},
    # African nations
    {"name": "Nigeria", "alpha_2": "NG", "alpha_3": "NGA"},
    {"name": "Kenya", "alpha_2": "KE", "alpha_3": "KEN"},
    {"name": "Egypt", "alpha_2": "EG", "alpha_3": "EGY"},
    {"name": "South Africa", "alpha_2": "ZA", "alpha_3": "ZAF"},
    {"name": "Ghana", "alpha_2": "GH", "alpha_3": "GHA"},
    {"name": "Ethiopia", "alpha_2": "ET", "alpha_3": "ETH"},
    {"name": "Morocco", "alpha_2": "MA", "alpha_3": "MAR"},
    {"name": "Tanzania", "alpha_2": "TZ", "alpha_3": "TZA"},
    {"name": "Algeria", "alpha_2": "DZ", "alpha_3": "DZA"},
    {"name": "Angola", "alpha_2": "AO", "alpha_3": "AGO"},
    {"name": "Cameroon", "alpha_2": "CM", "alpha_3": "CMR"},
    {"name": "Congo", "alpha_2": "CG", "alpha_3": "COG"},
    {"name": "Cote d'Ivoire", "alpha_2": "CI", "alpha_3": "CIV"},
    {"name": "DR Congo", "alpha_2": "CD", "alpha_3": "COD"},
    {"name": "Madagascar", "alpha_2": "MG", "alpha_3": "MDG"},
    {"name": "Mozambique", "alpha_2": "MZ", "alpha_3": "MOZ"},
    {"name": "Rwanda", "alpha_2": "RW", "alpha_3": "RWA"},
    {"name": "Senegal", "alpha_2": "SN", "alpha_3": "SEN"},
    {"name": "Sudan", "alpha_2": "SD", "alpha_3": "SDN"},
    {"name": "Tunisia", "alpha_2": "TN", "alpha_3": "TUN"},
    {"name": "Uganda", "alpha_2": "UG", "alpha_3": "UGA"},
    {"name": "Zambia", "alpha_2": "ZM", "alpha_3": "ZMB"},
    {"name": "Zimbabwe", "alpha_2": "ZW", "alpha_3": "ZWE"},
    # Other world nations
    {"name": "United Kingdom", "alpha_2": "GB", "alpha_3": "GBR"},
    {"name": "Canada", "alpha_2": "CA", "alpha_3": "CAN"},
    {"name": "Australia", "alpha_2": "AU", "alpha_3": "AUS"},
    {"name": "Germany", "alpha_2": "DE", "alpha_3": "DEU"},
    {"name": "France", "alpha_2": "FR", "alpha_3": "FRA"},
    {"name": "Japan", "alpha_2": "JP", "alpha_3": "JPN"},
    {"name": "China", "alpha_2": "CN", "alpha_3": "CHN"},
    {"name": "Brazil", "alpha_2": "BR", "alpha_3": "BRA"},
    {"name": "Singapore", "alpha_2": "SG", "alpha_3": "SGP"},
    {"name": "United Arab Emirates", "alpha_2": "AE", "alpha_3": "ARE"},
    {"name": "Italy", "alpha_2": "IT", "alpha_3": "ITA"},
    {"name": "Spain", "alpha_2": "ES", "alpha_3": "ESP"},
    {"name": "Mexico", "alpha_2": "MX", "alpha_3": "MEX"},
    {"name": "South Korea", "alpha_2": "KR", "alpha_3": "KOR"},
    {"name": "Netherlands", "alpha_2": "NL", "alpha_3": "NLD"},
    {"name": "Sweden", "alpha_2": "SE", "alpha_3": "SWE"},
    {"name": "Switzerland", "alpha_2": "CH", "alpha_3": "CHE"},
    {"name": "New Zealand", "alpha_2": "NZ", "alpha_3": "NZL"},
    {"name": "Russian Federation", "alpha_2": "RU", "alpha_3": "RUS"},
    {"name": "Malaysia", "alpha_2": "MY", "alpha_3": "MYS"},
    {"name": "Indonesia", "alpha_2": "ID", "alpha_3": "IDN"},
    {"name": "Saudi Arabia", "alpha_2": "SA", "alpha_3": "SAU"},
    {"name": "Argentina", "alpha_2": "AR", "alpha_3": "ARG"},
]

# Quick lookup tables
_BY_ALPHA_3: Dict[str, Dict[str, str]] = {c["alpha_3"].upper(): c for c in ISO_COUNTRIES}
_BY_ALPHA_2: Dict[str, Dict[str, str]] = {c["alpha_2"].upper(): c for c in ISO_COUNTRIES}
_BY_NAME_NORM: Dict[str, Dict[str, str]] = {c["name"].strip().lower(): c for c in ISO_COUNTRIES}

# Specific alias additions
_BY_NAME_NORM["usa"] = _BY_ALPHA_3["USA"]
_BY_NAME_NORM["us"] = _BY_ALPHA_3["USA"]
_BY_NAME_NORM["united states of america"] = _BY_ALPHA_3["USA"]
_BY_NAME_NORM["uk"] = _BY_ALPHA_3["GBR"]
_BY_NAME_NORM["britain"] = _BY_ALPHA_3["GBR"]
_BY_NAME_NORM["ivory coast"] = _BY_ALPHA_3["CIV"]
_BY_NAME_NORM["congo"] = _BY_ALPHA_3["COG"]
_BY_NAME_NORM["russia"] = _BY_ALPHA_3["RUS"]
_BY_NAME_NORM["uae"] = _BY_ALPHA_3["ARE"]


def is_african_nation(country_code_or_name: str) -> bool:
    """Check if the provided country code or name belongs to the African regional category."""
    if not country_code_or_name:
        return False
    norm = country_code_or_name.strip().upper()
    if norm in AFRICAN_COUNTRY_CODES:
        return True
    
    # Try looking up in catalog
    if norm in _BY_ALPHA_2:
        return _BY_ALPHA_2[norm]["alpha_3"] in AFRICAN_COUNTRY_CODES
    
    lower = country_code_or_name.strip().lower()
    if lower in _BY_NAME_NORM:
        return _BY_NAME_NORM[lower]["alpha_3"] in AFRICAN_COUNTRY_CODES
    
    return False


def classify_nationality_category(country_code_or_name: str) -> str:
    """
    Classify declared or extracted nationality into one of the four assignment categories:
    1. 'INDIAN': India (IND / IN)
    2. 'US': United States (USA / US)
    3. 'AFRICAN_REGION': Any sovereign nation in the African regional group
    4. 'OTHER': Any other sovereign nationality
    """
    if not country_code_or_name:
        return "OTHER"
    
    code = country_code_or_name.strip().upper()
    lower = country_code_or_name.strip().lower()

    if code in ("IND", "IN") or lower in ("india", "republic of india"):
        return "INDIAN"
    
    if code in ("USA", "US") or lower in ("united states", "united states of america", "usa"):
        return "US"
    
    if is_african_nation(country_code_or_name):
        return "AFRICAN_REGION"
    
    return "OTHER"
