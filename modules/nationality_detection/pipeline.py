"""
Pipeline Orchestrator for Task 4: Nationality Detection & Conditional Profile Analysis.

Orchestrates:
1. Multi-face localization (YuNet / Haar Cascade) supporting single or multiple occupants.
2. Independent biometric extraction per detected occupant (upper-torso dress colour, FER CNN emotion).
3. Non-appearance nationality resolution (User declaration or document MRZ).
4. Strict multi-occupant age isolation:
   - Automated texture/heuristic age estimation is disabled.
   - Age is obtained exclusively via user declaration.
   - Declared age is strictly bound to the designated subject and is NEVER cross-assigned to other persons in the frame.
   - Undeclared occupants explicitly report 'Age estimation unavailable'.
5. Strict enforcement of the 4-category conditional output display rules:
   - Indian: Age and Dress Colour
   - US: Age and Emotion
   - African category: Emotion and Dress Colour
   - Other nationalities: Nationality and Emotion
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .constants import classify_nationality_category, is_african_nation
from .nationality import NationalityInfo, NationalityResolver
from .dress_color import DressColorResult, DressColorAnalyzer
from .emotion import EmotionResult, EmotionDetector
from .age import AgeResult, AgeEstimator

_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PersonProfile:
    """Represents biometric analysis and conditional fields for a specific occupant."""
    person_id: str  # "Person 1", "Person 2", etc.
    face_box: Tuple[int, int, int, int]
    age: AgeResult
    dress_color: Optional[DressColorResult]
    emotion: Optional[EmotionResult]
    displayed_fields: Dict[str, Any]
    suppressed_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "face_box": self.face_box,
            "age": self.age.to_dict(),
            "dress_color": self.dress_color.to_dict() if self.dress_color else None,
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "displayed_fields": self.displayed_fields,
            "suppressed_fields": self.suppressed_fields,
        }


@dataclass
class Task4Result:
    """Consolidated result of Task 4 analysis supporting single and multi-occupant frames."""
    has_face: bool
    nationality_info: NationalityInfo
    active_condition: str  # "INDIAN", "US", "AFRICAN_REGION", "OTHER"
    displayed_fields: Dict[str, Any]
    suppressed_fields: List[str]
    dress_color: Optional[DressColorResult]
    emotion: Optional[EmotionResult]
    age: Optional[AgeResult]
    face_box: Optional[Tuple[int, int, int, int]]
    num_faces: int = 0
    selected_subject: str = "Person 1"
    people: List[PersonProfile] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_face": self.has_face,
            "num_faces": self.num_faces,
            "selected_subject": self.selected_subject,
            "nationality_info": self.nationality_info.to_dict(),
            "active_condition": self.active_condition,
            "displayed_fields": self.displayed_fields,
            "suppressed_fields": self.suppressed_fields,
            "dress_color": self.dress_color.to_dict() if self.dress_color else None,
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "age": self.age.to_dict() if self.age else None,
            "face_box": self.face_box,
            "people": [p.to_dict() for p in self.people],
            "error_message": self.error_message,
        }


class Task4Pipeline:
    """
    Unified static image analyzer applying ethical nationality resolution,
    multi-face isolation, and conditional profile presentation.
    """

    def __init__(self, cascade_path: Optional[str] = None, yunet_path: Optional[str] = None):
        self.cascade_path = self._resolve_cascade_path(cascade_path)
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.yunet_path = self._resolve_yunet_path(yunet_path)
        self.emotion_detector: Optional[EmotionDetector] = None

    def _resolve_yunet_path(self, user_path: Optional[str]) -> Optional[str]:
        if user_path and os.path.exists(user_path):
            return os.path.abspath(user_path)
        default_yunet = os.path.join(_root_dir, "models", "attendance", "face_detection_yunet_2023mar.onnx")
        if os.path.exists(default_yunet):
            return default_yunet
        return None

    def _resolve_cascade_path(self, user_path: Optional[str]) -> str:
        if user_path and os.path.exists(user_path):
            return os.path.abspath(user_path)
        root_cascade = os.path.join(_root_dir, "haarcascade_frontalface_default.xml")
        if os.path.exists(root_cascade):
            return root_cascade
        cv2_data_cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cv2_data_cascade):
            return cv2_data_cascade
        return "haarcascade_frontalface_default.xml"

    def detect_all_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects all face bounding boxes (x, y, w, h) in the image.
        Sorts faces from left to right (by x-coordinate) for consistent identification.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detected_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Try YuNet primary deep detector
        if self.yunet_path and os.path.exists(self.yunet_path):
            try:
                detector = cv2.FaceDetectorYN_create(
                    self.yunet_path,
                    "",
                    (w, h),
                    score_threshold=0.45,
                    nms_threshold=0.3,
                    top_k=100
                )
                _, faces = detector.detect(image)
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        fx = max(0, fx)
                        fy = max(0, fy)
                        fw = min(w - fx, fw)
                        fh = min(h - fy, fh)
                        if fw >= 20 and fh >= 20:
                            detected_boxes.append((fx, fy, fw, fh))
            except Exception:
                pass

        # 2. Fallback to Haar Cascade if YuNet returned nothing
        if len(detected_boxes) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(30, 30)
            )
            if len(faces) > 0:
                for b in faces:
                    detected_boxes.append(tuple(int(v) for v in b))

        # Sort left-to-right (x-coordinate, then y-coordinate)
        detected_boxes.sort(key=lambda b: (b[0], b[1]))
        return detected_boxes

    @staticmethod
    def normalize_subject_id(subject: Optional[str]) -> str:
        """
        Normalizes subject identifiers (e.g., 'Person 2 (Face 2 from left)', '2', 'person 2')
        into canonical 'Person K' format. Leaves unmatched/invalid strings as-is for safe rejection.
        """
        if not subject:
            return "Person 1"
        s = str(subject).strip()
        import re
        m = re.match(r"^Person\s*(\d+)", s, re.IGNORECASE)
        if m:
            return f"Person {m.group(1)}"
        if s.isdigit():
            return f"Person {s}"
        return s

    def detect_primary_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the largest face bounding box (x, y, w, h) in the image."""
        faces = self.detect_all_faces(image)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: b[2] * b[3])

    def process_image(
        self,
        image: np.ndarray,
        nationality_declaration: Optional[str] = None,
        mrz_data: Optional[Tuple[str, str]] = None,
        declared_age: Optional[int] = None,
        declared_ages: Optional[Dict[str, int]] = None,
        selected_subject: str = "Person 1"
    ) -> Task4Result:
        """
        Processes a static image with declared or document-extracted nationality.
        Supports single or multiple occupants with strict person-to-age isolation.
        
        Applies conditional output rules strictly:
        - Indian: Age and Dress Colour
        - US: Age and Emotion
        - African category: Emotion and Dress Colour
        - Other nationalities: Nationality and Emotion
        """
        # 1. Resolve Nationality
        if mrz_data is not None and len(mrz_data) == 2:
            mrz_res = NationalityResolver.parse_mrz_td3(mrz_data[0], mrz_data[1])
            if mrz_res["success"]:
                nat_info = mrz_res["nationality_info"]
            else:
                nat_info = NationalityResolver.resolve_declaration(nationality_declaration or "")
        else:
            nat_info = NationalityResolver.resolve_declaration(nationality_declaration or "")

        condition = nat_info.category  # "INDIAN", "US", "AFRICAN_REGION", "OTHER"
        canonical_subject = self.normalize_subject_id(selected_subject)

        if image is None or image.size == 0:
            return Task4Result(
                has_face=False,
                nationality_info=nat_info,
                active_condition=condition,
                displayed_fields={},
                suppressed_fields=[],
                dress_color=None,
                emotion=None,
                age=None,
                face_box=None,
                num_faces=0,
                selected_subject=canonical_subject,
                people=[],
                annotated_image=None,
                error_message="Image is empty or could not be decoded.",
            )

        # 2. Detect all faces
        faces = self.detect_all_faces(image)
        if len(faces) == 0:
            return Task4Result(
                has_face=False,
                nationality_info=nat_info,
                active_condition=condition,
                displayed_fields={"nationality": nat_info.to_dict()},
                suppressed_fields=["face_biometrics"],
                dress_color=None,
                emotion=None,
                age=None,
                face_box=None,
                num_faces=0,
                selected_subject=canonical_subject,
                people=[],
                annotated_image=image.copy() if image is not None else None,
                error_message="No face detected in the uploaded image. Please ensure the portrait is well-lit and faces are clearly visible.",
            )

        # 3. Analyze each occupant independently
        people: List[PersonProfile] = []
        annotated = image.copy()
        box_colors = [
            (0, 255, 0),      # Bright Green
            (0, 165, 255),    # Orange
            (255, 105, 180),  # Hot Pink
            (255, 255, 0),    # Cyan/Yellow
            (0, 255, 255),    # Yellow
            (255, 191, 0),    # Deep Sky
            (147, 112, 219),  # Medium Purple
            (50, 205, 50),    # Lime Green
            (255, 99, 71),    # Tomato Red
            (0, 191, 255),    # Deep Sky Blue
        ]

        for idx, face_box in enumerate(faces):
            p_id = f"Person {idx + 1}"
            fx, fy, fw, fh = face_box
            face_crop = image[fy : fy + fh, fx : fx + fw]

            # Biometrics
            dress_res = DressColorAnalyzer.analyze(image, face_box)
            if self.emotion_detector is None:
                self.emotion_detector = EmotionDetector.get_instance()
            emotion_res = self.emotion_detector.predict(face_crop)

            # Age resolution: Strict person-level isolation
            # Check if per-person age was declared, or default declared_age applies to canonical_subject
            p_age_val: Optional[int] = None
            if declared_ages and p_id in declared_ages:
                p_age_val = declared_ages[p_id]
            elif declared_age is not None and declared_age > 0 and (p_id == canonical_subject):
                # User-provided age applies strictly to the selected subject
                p_age_val = declared_age
            else:
                # Occupants without explicit declaration report unavailable
                p_age_val = None

            age_res = AgeEstimator.resolve_age(
                declared_age=p_age_val,
                person_id=p_id,
                face_resolution=(fw, fh)
            )

            # Conditional fields per person
            p_displayed: Dict[str, Any] = {}
            p_suppressed: List[str] = []

            if condition == "INDIAN":
                p_displayed["age"] = age_res.to_dict()
                p_displayed["dress_colour"] = dress_res.to_dict()
                p_suppressed = ["emotion"]
            elif condition == "US":
                p_displayed["age"] = age_res.to_dict()
                p_displayed["emotion"] = emotion_res.to_dict()
                p_suppressed = ["dress_colour"]
            elif condition == "AFRICAN_REGION":
                p_displayed["emotion"] = emotion_res.to_dict()
                p_displayed["dress_colour"] = dress_res.to_dict()
                p_suppressed = ["age"]
            else:
                p_displayed["nationality"] = nat_info.to_dict()
                p_displayed["emotion"] = emotion_res.to_dict()
                p_suppressed = ["age", "dress_colour"]

            person_prof = PersonProfile(
                person_id=p_id,
                face_box=face_box,
                age=age_res,
                dress_color=dress_res,
                emotion=emotion_res,
                displayed_fields=p_displayed,
                suppressed_fields=p_suppressed,
            )
            people.append(person_prof)

            # Visual Annotation
            color = box_colors[idx % len(box_colors)]
            is_target = (p_id == canonical_subject)
            thickness = 3 if (is_target and len(faces) > 1) else 2
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, thickness)

            label_prefix = f"{p_id} [Selected Subject]" if (is_target and len(faces) > 1) else p_id
            lines = [f"{label_prefix} ({condition})"]
            if "age" in p_displayed:
                lines.append(f"Age: {age_res.bracket}")
            if "dress_colour" in p_displayed:
                lines.append(f"Dress: {dress_res.color_name}")
            if "emotion" in p_displayed:
                lines.append(f"Emotion: {emotion_res.dominant_emotion} ({emotion_res.confidence*100:.1f}%)")
            if "nationality" in p_displayed:
                lines.append(f"Nat: {nat_info.country_name}")

            # Adaptive font scale and line spacing for crowded scenes
            font_scale = 0.40 if len(faces) > 5 else 0.50
            line_step = 16 if len(faces) > 5 else 20
            y_offset = max(20, fy - 8)
            for l_idx, text in enumerate(lines):
                y_pos = y_offset - (len(lines) - 1 - l_idx) * line_step
                if y_pos < 16:
                    y_pos = fy + fh + 16 + l_idx * line_step
                cv2.putText(annotated, text, (fx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
                cv2.putText(annotated, text, (fx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

            # Draw Torso Box if dress colour is displayed and determined
            if "dress_colour" in p_displayed and dress_res.torso_box:
                tx, ty, tw, th = dress_res.torso_box
                cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), color, 1)

        # Target occupant matching selected_subject (defaults safely to Person 1)
        target_person = next((p for p in people if p.person_id == canonical_subject), people[0])

        return Task4Result(
            has_face=True,
            num_faces=len(people),
            selected_subject=canonical_subject,
            nationality_info=nat_info,
            active_condition=condition,
            displayed_fields=target_person.displayed_fields,
            suppressed_fields=target_person.suppressed_fields,
            dress_color=target_person.dress_color,
            emotion=target_person.emotion,
            age=target_person.age,
            face_box=target_person.face_box,
            people=people,
            annotated_image=annotated,
            error_message=None,
        )
