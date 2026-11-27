"""
Age Resolution Module for Task 4.

Ethical & Scientific Transparency Notice:
Automated biometric facial age estimation is fundamentally uncalibrated and produces
severe demographic errors (e.g., misclassifying young adults as seniors due to camera
sharpness, sensor noise, or high-contrast facial features).

To adhere to rigorous scientific integrity:
1. Automated heuristic age prediction is DISABLED.
2. In accordance with user requirements, age is obtained via an optional self-declaration
   field clearly labelled 'User-provided age'.
3. When age is not declared by the user, the module returns 'Age estimation unavailable'
   until a validated, debiased demographic model is available.
4. When multiple occupants are present, declared age is strictly mapped to the designated
   subject and is NEVER cross-assigned to other persons in the frame.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class AgeResult:
    """Represents age resolution status and provenance."""
    bracket: str
    declared_age: Optional[int]
    source: str  # "User-provided age" or "None"
    is_available: bool
    is_user_provided: bool = False
    is_experimental: bool = False
    person_id: str = "Person 1"
    reason: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    disclaimer: str = (
        "Notice: Automated visual age estimation is disabled to prevent demographic misclassification. "
        "Displayed age is strictly user-provided or reported as unavailable."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bracket": self.bracket,
            "declared_age": self.declared_age,
            "source": self.source,
            "is_available": self.is_available,
            "is_user_provided": self.is_user_provided,
            "is_experimental": self.is_experimental,
            "person_id": self.person_id,
            "reason": self.reason,
            "resolution": self.resolution,
            "disclaimer": self.disclaimer,
        }


class AgeEstimator:
    """
    Resolves age for Task 4 occupants.
    Replaces unvalidated facial texture heuristics with an explicit user-declared age
    or reports 'Age estimation unavailable' until a validated model is available.
    """

    @classmethod
    def resolve_age(
        cls,
        declared_age: Optional[int] = None,
        person_id: str = "Person 1",
        face_resolution: Optional[Tuple[int, int]] = None,
        custom_reason: Optional[str] = None
    ) -> AgeResult:
        """
        Resolves occupant age:
        - If declared_age is provided (> 0): Formats as 'User-provided age' with broad demographic bracket.
        - If declared_age is None or 0: Explicitly reports 'Age estimation unavailable'.
        """
        if declared_age is not None and declared_age > 0:
            if declared_age < 18:
                cat = "Child / Adolescent"
            elif declared_age <= 35:
                cat = "Young Adult"
            elif declared_age <= 55:
                cat = "Middle-aged Adult"
            else:
                cat = "Senior Adult"

            bracket_str = f"{declared_age} years ({cat})"
            return AgeResult(
                bracket=bracket_str,
                declared_age=declared_age,
                source="User-provided age",
                is_available=True,
                is_user_provided=True,
                is_experimental=False,
                person_id=person_id,
                reason="Explicitly provided via user declaration.",
                resolution=face_resolution,
            )

        # When no declaration is provided
        fallback_reason = custom_reason or (
            f"No user age declared for {person_id}. Automated biometric age estimation is disabled pending a validated, debiased demographic model."
        )

        return AgeResult(
            bracket="Age estimation unavailable",
            declared_age=None,
            source="None",
            is_available=False,
            is_user_provided=False,
            is_experimental=False,
            person_id=person_id,
            reason=fallback_reason,
            resolution=face_resolution,
        )

    @classmethod
    def estimate(
        cls,
        face_bgr_or_gray: Optional[np.ndarray] = None,
        declared_age: Optional[int] = None,
        person_id: str = "Person 1"
    ) -> AgeResult:
        """
        Backwards-compatible wrapper.
        If declared_age is provided, resolves it.
        Otherwise, returns 'Age estimation unavailable'.
        """
        res = (face_bgr_or_gray.shape[1], face_bgr_or_gray.shape[0]) if face_bgr_or_gray is not None and face_bgr_or_gray.size > 0 else (0, 0)
        return cls.resolve_age(declared_age=declared_age, person_id=person_id, face_resolution=res)
