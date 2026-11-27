"""
Task 4: Nationality Detection & Conditional Profile Analysis.
Modular, independent package preserving Tasks 1, 2, and 3 freeze.

Nationality is determined strictly via explicit user declaration or standard
document data fields, never from visual appearance, skin tone, or attire.
"""

from .constants import (
    ISO_COUNTRIES,
    AFRICAN_COUNTRY_CODES,
    EMOTION_CLASSES,
    classify_nationality_category,
    is_african_nation,
)
from .nationality import NationalityInfo, NationalityResolver
from .dress_color import DressColorResult, DressColorAnalyzer
from .emotion import EmotionResult, EmotionDetector
from .age import AgeResult, AgeEstimator
from .pipeline import Task4Pipeline, Task4Result

__all__ = [
    "ISO_COUNTRIES",
    "AFRICAN_COUNTRY_CODES",
    "EMOTION_CLASSES",
    "classify_nationality_category",
    "is_african_nation",
    "NationalityInfo",
    "NationalityResolver",
    "DressColorResult",
    "DressColorAnalyzer",
    "EmotionResult",
    "EmotionDetector",
    "AgeResult",
    "AgeEstimator",
    "Task4Pipeline",
    "Task4Result",
]
