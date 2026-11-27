"""
Task 5: Indian Sign Language (ISL) Static Sign Recognition Module.
Focuses on 12 static handshape classes: 1, 2, 3, 4, 5, A, B, C, L, V, W, Y.
"""

from .detector import ISLHandDetector, ISLRecognitionResult
from .detector import is_within_operating_window, get_next_operating_window, get_current_ist_time

__all__ = [
    "ISLHandDetector",
    "ISLRecognitionResult",
    "is_within_operating_window",
    "get_next_operating_window",
    "get_current_ist_time",
]
