"""
Task 3: Drowsiness & Fatigue Detection Module.
Provides real-time facial landmark localization, Eye Aspect Ratio (EAR),
Mouth Aspect Ratio (MAR), and drowsiness detection.
"""

from modules.drowsiness.detector import (
    DrowsinessDetector,
    get_drowsiness_detector,
    calculate_ear,
    calculate_mar,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
)

__all__ = [
    "DrowsinessDetector",
    "get_drowsiness_detector",
    "calculate_ear",
    "calculate_mar",
    "DEFAULT_EAR_THRESHOLD",
    "DEFAULT_MAR_THRESHOLD",
]
