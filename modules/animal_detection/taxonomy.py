"""
Animal Dietary Taxonomy and Visual Styling Configuration.
Implements the 3-category dietary classification system:
- Carnivore: Red bounding boxes (BGR: (0, 0, 255), Hex: #EF4444)
- Herbivore: Green bounding boxes (BGR: (0, 255, 0), Hex: #10B981)
- Omnivore / Other: Amber/Gold bounding boxes (BGR: (0, 180, 255), Hex: #F59E0B)

Taxonomically and physiologically verified classification for the 10 MS COCO animal classes.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Any


class DietCategory(str, Enum):
    """Three mutually exclusive dietary categories."""
    CARNIVORE = "Carnivore"
    HERBIVORE = "Herbivore"
    OMNIVORE_OTHER = "Omnivore / Other"


# 10 Supported Animal classes from MS COCO (indices 14 to 23)
COCO_ANIMAL_CLASSES: Dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

COCO_ANIMAL_INDICES: List[int] = list(COCO_ANIMAL_CLASSES.keys())

# Explicit Biological / Ecological Dietary Mapping
SPECIES_DIET_MAPPING: Dict[str, DietCategory] = {
    # Confirmed Carnivores (Obligate hypercarnivores requiring animal tissue)
    "cat": DietCategory.CARNIVORE,

    # Confirmed Herbivores (Strict plant/foliage/grass grazers & browsers)
    "horse": DietCategory.HERBIVORE,
    "sheep": DietCategory.HERBIVORE,
    "cow": DietCategory.HERBIVORE,
    "elephant": DietCategory.HERBIVORE,
    "zebra": DietCategory.HERBIVORE,
    "giraffe": DietCategory.HERBIVORE,

    # Omnivores / Other (Physiological or ecological omnivores and generalist classes)
    "dog": DietCategory.OMNIVORE_OTHER,
    "bear": DietCategory.OMNIVORE_OTHER,
    "bird": DietCategory.OMNIVORE_OTHER,
}

# Extensible Wild Species Aliases (for future specialized fine-tuned weights)
EXTENDED_SPECIES_ALIASES: Dict[str, DietCategory] = {
    "lion": DietCategory.CARNIVORE,
    "tiger": DietCategory.CARNIVORE,
    "leopard": DietCategory.CARNIVORE,
    "cheetah": DietCategory.CARNIVORE,
    "jaguar": DietCategory.CARNIVORE,
    "wolf": DietCategory.CARNIVORE,
    "fox": DietCategory.OMNIVORE_OTHER,
    "hyena": DietCategory.CARNIVORE,
    "crocodile": DietCategory.CARNIVORE,
    "alligator": DietCategory.CARNIVORE,
    "shark": DietCategory.CARNIVORE,
    "polar bear": DietCategory.CARNIVORE,
    "panda": DietCategory.HERBIVORE,
    "deer": DietCategory.HERBIVORE,
    "antelope": DietCategory.HERBIVORE,
    "rabbit": DietCategory.HERBIVORE,
    "pig": DietCategory.OMNIVORE_OTHER,
    "monkey": DietCategory.OMNIVORE_OTHER,
}

# Color Codes for Visual Bounding Boxes and UI Indicators
CATEGORY_COLORS_BGR: Dict[DietCategory, Tuple[int, int, int]] = {
    DietCategory.CARNIVORE: (0, 0, 255),          # Vivid Red (BGR)
    DietCategory.HERBIVORE: (0, 255, 0),          # Emerald Green (BGR)
    DietCategory.OMNIVORE_OTHER: (0, 180, 255),   # Amber / Gold (BGR: B=0, G=180, R=255)
}

CATEGORY_COLORS_RGB: Dict[DietCategory, Tuple[int, int, int]] = {
    DietCategory.CARNIVORE: (239, 68, 68),        # Vivid Red (RGB: #EF4444)
    DietCategory.HERBIVORE: (16, 185, 129),       # Emerald Green (RGB: #10B981)
    DietCategory.OMNIVORE_OTHER: (245, 158, 11),  # Amber / Gold (RGB: #F59E0B)
}

CATEGORY_COLORS_HEX: Dict[DietCategory, str] = {
    DietCategory.CARNIVORE: "#EF4444",
    DietCategory.HERBIVORE: "#10B981",
    DietCategory.OMNIVORE_OTHER: "#F59E0B",
}

# Scientific & Ecological Rationale for Each Classification
DIETARY_RATIONALE: Dict[str, str] = {
    "cat": (
        "Obligate Hypercarnivore: Domestic cats (Felis catus) have a physiological requirement "
        "for nutrients found exclusively in animal tissue, including preformed taurine, arginine, "
        "arachidonic acid, and vitamin A. Their natural diet requires >70% meat protein."
    ),
    "horse": (
        "Hindgut Fermenter Herbivore: Horses possess a large cecum and colon specialized for "
        "microbial fermentation of plant cellulose and grasses. They do not consume meat."
    ),
    "sheep": (
        "Ruminant Herbivore: Sheep have a four-chambered stomach specialized for regurgitation, "
        "remastication, and microbial digestion of grasses and forbs."
    ),
    "cow": (
        "Ruminant Herbivore: Cattle are obligate bovine grazers with a complex rumination digestive "
        "system dedicated to breaking down cellulose and fibrous plant vegetation."
    ),
    "elephant": (
        "Megaherbivore: Elephants consume vast quantities of foliage, bark, grasses, roots, "
        "and fruits exclusively. Their digestive system is strictly adapted for plant matter."
    ),
    "zebra": (
        "Hindgut Fermenter Herbivore: Zebras are monogastric equids adapted to grazing coarse, "
        "high-fiber savanna grasses without consuming animal protein."
    ),
    "giraffe": (
        "Browsing Ruminant Herbivore: Giraffes are specialized high-canopy browsers feeding "
        "primarily on acacia leaves, seed pods, and shrubs rich in calcium and tannins."
    ),
    "dog": (
        "Adapted Omnivore: While taxonomically in the order Carnivora, domestic dogs (Canis familiaris) "
        "possess amylase gene duplications (AMY2B) that allow efficient digestion of starches, grains, "
        "and plant matter alongside animal proteins."
    ),
    "bear": (
        "Ecological Omnivore: Family Ursidae members (such as Brown and Black bears) possess an "
        "opportunistic omnivorous diet dominated by berries, roots, nuts, vegetation, and insects, "
        "complemented seasonally by salmon, ungulates, or carrion."
    ),
    "bird": (
        "Generalist Avian Class: In MS COCO, 'bird' is a broad polyphyletic category comprising "
        "granivores (seed eaters), insectivores, frugivores, piscivores, and raptors. Classifying "
        "the generic class as Omnivore / Other is the most scientifically sound approach."
    ),
}


class AnimalTaxonomy:
    """Utility class for querying species dietary classifications and styling."""

    @classmethod
    def get_category(cls, class_or_category: Any) -> DietCategory:
        """
        Lookup the dietary category for a given animal class name, category enum, or category string.
        Defaults to DietCategory.OMNIVORE_OTHER if unknown.
        """
        if isinstance(class_or_category, DietCategory):
            return class_or_category

        cleaned = str(class_or_category).strip().lower()

        # Check if directly matches a DietCategory value or name
        for cat in DietCategory:
            if cleaned == cat.value.lower() or cleaned == cat.name.lower():
                return cat

        if cleaned in SPECIES_DIET_MAPPING:
            return SPECIES_DIET_MAPPING[cleaned]
        if cleaned in EXTENDED_SPECIES_ALIASES:
            return EXTENDED_SPECIES_ALIASES[cleaned]
        return DietCategory.OMNIVORE_OTHER

    @classmethod
    def get_color_bgr(cls, class_or_category: str) -> Tuple[int, int, int]:
        """Return OpenCV BGR color tuple for a class name or category."""
        if isinstance(class_or_category, DietCategory):
            category = class_or_category
        else:
            category = cls.get_category(class_or_category)
        return CATEGORY_COLORS_BGR.get(category, (0, 180, 255))

    @classmethod
    def get_color_rgb(cls, class_or_category: str) -> Tuple[int, int, int]:
        """Return RGB color tuple for a class name or category."""
        if isinstance(class_or_category, DietCategory):
            category = class_or_category
        else:
            category = cls.get_category(class_or_category)
        return CATEGORY_COLORS_RGB.get(category, (245, 158, 11))

    @classmethod
    def get_color_hex(cls, class_or_category: str) -> str:
        """Return hex color string for a class name or category."""
        if isinstance(class_or_category, DietCategory):
            category = class_or_category
        else:
            category = cls.get_category(class_or_category)
        return CATEGORY_COLORS_HEX.get(category, "#F59E0B")

    @classmethod
    def is_carnivore(cls, class_name: str) -> bool:
        """Check if an animal is a confirmed carnivore."""
        return cls.get_category(class_name) == DietCategory.CARNIVORE

    @classmethod
    def is_herbivore(cls, class_name: str) -> bool:
        """Check if an animal is an herbivore."""
        return cls.get_category(class_name) == DietCategory.HERBIVORE

    @classmethod
    def is_omnivore_other(cls, class_name: str) -> bool:
        """Check if an animal is an omnivore or generalist other."""
        return cls.get_category(class_name) == DietCategory.OMNIVORE_OTHER

    @classmethod
    def format_label(
        cls,
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None
    ) -> str:
        """
        Format a professional detection label including track ID, species, confidence, and category tag.
        Example: '#1 Zebra: 92.4% [HERBIVORE]' or 'Cat: 88.0% [CARNIVORE]'
        """
        cat = cls.get_category(class_name)
        species_title = class_name.capitalize()
        cat_tag = f"[{cat.value.upper()}]"
        if track_id is not None:
            return f"#{track_id} {species_title} {confidence:.0%} {cat_tag}"
        return f"{species_title} {confidence:.0%} {cat_tag}"

    @classmethod
    def get_supported_species(cls) -> List[str]:
        """Return sorted list of supported COCO animal classes."""
        return sorted(list(SPECIES_DIET_MAPPING.keys()))

    @classmethod
    def get_dietary_rationale(cls, class_name: str) -> str:
        """Return the biological and ecological rationale for the species classification."""
        cleaned = class_name.strip().lower()
        return DIETARY_RATIONALE.get(
            cleaned,
            f"Species '{class_name}' is classified under standard dietary taxonomy guidelines."
        )
