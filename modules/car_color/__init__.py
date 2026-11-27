"""
Task 6: Car Colour & Pedestrian Detection Module.
Detects cars and pedestrians in traffic scenes, classifies car colour (Blue vs. Other vs. Uncertain),
and monitors pedestrians within an adjustable traffic signal crossing zone.
"""

from .detector import (
    CarColorDetector,
    CarColorResult,
    DetectedObject,
    TrafficSceneResult,
    SignalZone,
)

__all__ = [
    "CarColorDetector",
    "CarColorResult",
    "DetectedObject",
    "TrafficSceneResult",
    "SignalZone",
]
