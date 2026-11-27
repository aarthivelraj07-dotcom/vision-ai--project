"""
Facial Landmark and Biometric Detection Engine for Task 3: Drowsiness Detection.
Configured strictly for CPU inference using OpenCV DNN and existing verified models.

Features:
- Fast, accurate face localization and anchor alignment via OpenCV YuNet.
- Fallback to OpenCV Haar Cascades when needed.
- 6-point physical landmark localization per eye for Soukupova-Cech Eye Aspect Ratio (EAR).
- 4/8-point physical landmark localization for Mouth Aspect Ratio (MAR) yawn detection.
- Aspect ratio computation and real-time biometric thresholding.
- Graceful error handling for corrupted buffers, grayscale, and 4-channel inputs.
"""

import os
import threading
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from modules.drowsiness.age_estimator import get_age_estimator
from modules.drowsiness.classifier import get_drowsiness_classifier

# Default physiological thresholds
DEFAULT_EAR_THRESHOLD = 0.22      # Below 0.22 indicates eye closure
DEFAULT_MAR_THRESHOLD = 0.55      # Above 0.55 indicates yawning mouth opening

# Path to YuNet ONNX model from attendance module or root models
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YUNET_PATH = os.path.join(PROJECT_ROOT, "models", "attendance", "face_detection_yunet_2023mar.onnx")
HAAR_FACE_PATH = os.path.join(PROJECT_ROOT, "haarcascade_frontalface_default.xml")


def calculate_ear(eye_landmarks: List[np.ndarray]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) using the Soukupova-Cech equation:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Args:
        eye_landmarks: List of 6 2D points [p1, p2, p3, p4, p5, p6]
    Returns:
        float: Computed EAR value (typically 0.28-0.38 for open eye, < 0.20 for closed)
    """
    if eye_landmarks is None or len(eye_landmarks) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = [np.array(pt, dtype=np.float32) for pt in eye_landmarks]

    # Vertical eyelid distances
    dist_v1 = float(np.linalg.norm(p2 - p6))
    dist_v2 = float(np.linalg.norm(p3 - p5))

    # Horizontal eye corner distance
    dist_h = float(np.linalg.norm(p1 - p4))

    if dist_h <= 0.001:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(round(ear, 4))


def calculate_mar(mouth_landmarks: List[np.ndarray]) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for yawn detection.
    MAR = (||m2 - m8|| + ||m3 - m7|| + ||m4 - m6||) / (2 * ||m1 - m5||)
    or simplified 4-point formulation:
    MAR = ||m_top - m_bot|| / ||m_left - m_right||
    """
    if mouth_landmarks is None or len(mouth_landmarks) < 4:
        return 0.0

    if len(mouth_landmarks) == 4:
        m_left, m_top, m_right, m_bot = [np.array(pt, dtype=np.float32) for pt in mouth_landmarks]
        dist_h = float(np.linalg.norm(m_left - m_right))
        dist_v = float(np.linalg.norm(m_top - m_bot))
        if dist_h <= 0.001:
            return 0.0
        return float(round(dist_v / dist_h, 4))

    # 6 or 8-point mouth formula
    m1 = np.array(mouth_landmarks[0], dtype=np.float32)
    m5 = np.array(mouth_landmarks[3] if len(mouth_landmarks) == 6 else mouth_landmarks[4], dtype=np.float32)
    dist_h = float(np.linalg.norm(m1 - m5))
    if dist_h <= 0.001:
        return 0.0

    # Average vertical span
    v_spans = []
    mid = len(mouth_landmarks) // 2
    for i in range(1, mid):
        top_pt = np.array(mouth_landmarks[i], dtype=np.float32)
        bot_pt = np.array(mouth_landmarks[-i], dtype=np.float32)
        v_spans.append(float(np.linalg.norm(top_pt - bot_pt)))

    mar = sum(v_spans) / (len(v_spans) * dist_h) if v_spans else 0.0
    return float(round(mar, 4))


class DrowsinessDetector:
    """
    Facial Landmark & Drowsiness Detection Engine.
    Executes on CPU using OpenCV DNN and localized gradient contour analysis.
    """
    _instance: Optional["DrowsinessDetector"] = None
    _lock = threading.RLock()

    def __init__(
        self,
        yunet_path: Optional[str] = None,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        mar_threshold: float = DEFAULT_MAR_THRESHOLD
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yunet_path = self._resolve_model_path(yunet_path)
        self.detector = None
        self.haar_detector = None
        self.is_loaded = False

        self._load_network()

    @classmethod
    def get_instance(
        cls,
        yunet_path: Optional[str] = None,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        mar_threshold: float = DEFAULT_MAR_THRESHOLD
    ) -> "DrowsinessDetector":
        """Thread-safe singleton accessor."""
        with cls._lock:
            if cls._instance is None or not cls._instance.is_loaded:
                cls._instance = cls(
                    yunet_path=yunet_path,
                    ear_threshold=ear_threshold,
                    mar_threshold=mar_threshold
                )
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _resolve_model_path(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            return os.path.abspath(path)
        if os.path.exists(YUNET_PATH):
            return YUNET_PATH
        return ""

    def _load_network(self):
        """Initialize YuNet and Haar detectors on CPU."""
        if self.yunet_path and os.path.exists(self.yunet_path):
            try:
                self.detector = cv2.FaceDetectorYN_create(
                    self.yunet_path,
                    "",
                    (320, 320),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception as e:
                print(f"Warning: Could not initialize YuNet: {e}")

        # Fallback Haar cascade
        haar_path = HAAR_FACE_PATH if os.path.exists(HAAR_FACE_PATH) else (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if os.path.exists(haar_path):
            try:
                self.haar_detector = cv2.CascadeClassifier(haar_path)
            except Exception:
                pass

        self.is_loaded = (self.detector is not None or self.haar_detector is not None)

    def _normalize_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Convert input to 3-channel BGR and validate dimensions."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None

        try:
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] == 1:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 3:
                    return image
            return None
        except Exception:
            return None

    def detect_all_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all human faces and their 5 anchor points in the frame.
        Returns a list of face metadata dictionaries sorted left to right.
        """
        norm_img = self._normalize_image(image)
        if norm_img is None:
            return []

        h, w = norm_img.shape[:2]
        if h <= 0 or w <= 0:
            return []

        all_faces = []

        # 1. Try YuNet deep learning detector
        if self.detector is not None:
            try:
                self.detector.setInputSize((w, h))
                ret, faces = self.detector.detect(norm_img)
                if ret > 0 and faces is not None and len(faces) > 0:
                    for face in faces:
                        conf = float(face[14])
                        if conf < 0.50:
                            continue

                        fx, fy, fw, fh = [int(v) for v in face[:4]]
                        rx, ry = float(face[4]), float(face[5])
                        lx, ly = float(face[6]), float(face[7])
                        nx, ny = float(face[8]), float(face[9])
                        rmx, rmy = float(face[10]), float(face[11])
                        lmx, lmy = float(face[12]), float(face[13])

                        all_faces.append({
                            "bbox": (max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh)),
                            "confidence": conf,
                            "right_eye_center": (rx, ry),
                            "left_eye_center": (lx, ly),
                            "nose_tip": (nx, ny),
                            "right_mouth": (rmx, rmy),
                            "left_mouth": (lmx, lmy),
                            "face_raw": face,
                            "method": "YuNet"
                        })
            except Exception:
                pass

        # 2. Fallback to Haar cascade if no faces detected
        if len(all_faces) == 0 and self.haar_detector is not None:
            try:
                gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
                faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                for (fx, fy, fw, fh) in faces:
                    rx, ry = fx + fw * 0.32, fy + fh * 0.38
                    lx, ly = fx + fw * 0.68, fy + fh * 0.38
                    nx, ny = fx + fw * 0.50, fy + fh * 0.55
                    rmx, rmy = fx + fw * 0.35, fy + fh * 0.78
                    lmx, lmy = fx + fw * 0.65, fy + fh * 0.78
                    all_faces.append({
                        "bbox": (fx, fy, fw, fh),
                        "confidence": 0.80,
                        "right_eye_center": (rx, ry),
                        "left_eye_center": (lx, ly),
                        "nose_tip": (nx, ny),
                        "right_mouth": (rmx, rmy),
                        "left_mouth": (lmx, lmy),
                        "face_raw": None,
                        "method": "Haar"
                    })
            except Exception:
                pass

        # Sort spatially left to right
        all_faces.sort(key=lambda f: f["bbox"][0])
        return all_faces

    def detect_face_and_anchors(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect the primary face and anchor points (eyes, nose, mouth).
        Maintains backward compatibility with single-face callers.
        """
        faces = self.detect_all_faces(image)
        if faces and len(faces) > 0:
            return max(faces, key=lambda f: f["confidence"])
        return None

    def extract_eye_6_landmarks(
        self,
        image: np.ndarray,
        eye_center: Tuple[float, float],
        eye_w: int,
        eye_h: int
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract the 6 physical eyelid contour landmarks from an eye region:
        p1: Outer corner
        p2: Upper eyelid at 1/3 width
        p3: Upper eyelid at 2/3 width
        p4: Inner corner
        p5: Lower eyelid at 2/3 width
        p6: Lower eyelid at 1/3 width
        """
        cx, cy = int(eye_center[0]), int(eye_center[1])
        h_img, w_img = image.shape[:2]

        x1 = max(0, cx - eye_w // 2)
        y1 = max(0, cy - eye_h // 2)
        x2 = min(w_img, x1 + eye_w)
        y2 = min(h_img, y1 + eye_h)

        actual_w = x2 - x1
        actual_h = y2 - y1
        if actual_w <= 4 or actual_h <= 4:
            # Degenerate eye box
            pts = [np.array([cx, cy], dtype=np.float32) for _ in range(6)]
            return pts, 0.0

        eye_roi = image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        roi_std = float(np.std(gray_roi))
        roi_range = float(np.max(gray_roi) - np.min(gray_roi))

        # Check contrast: closed eyelid skin has low contrast, while open iris/sclera has high contrast
        if roi_std < 18.0 or roi_range < 35.0:
            aperture = 1.5
        else:
            _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_spans = []
            for c in range(1, actual_w - 1):
                dark_pixels = np.where(binarized[:, c] == 255)[0]
                if len(dark_pixels) >= 3:
                    span = float(dark_pixels[-1] - dark_pixels[0] + 1)
                    col_spans.append(span)
            if len(col_spans) >= 2:
                aperture = float(np.percentile(col_spans, 75))
            else:
                aperture = 1.5

        mid_y = actual_h // 2
        col_c1 = max(1, actual_w // 3)
        col_c2 = min(actual_w - 2, 2 * actual_w // 3)

        # 6 physical landmark points in absolute image coordinates
        p1 = np.array([x1, cy], dtype=np.float32)
        p2 = np.array([x1 + col_c1, cy - aperture / 2.0], dtype=np.float32)
        p3 = np.array([x1 + col_c2, cy - aperture / 2.0], dtype=np.float32)
        p4 = np.array([x2 - 1, cy], dtype=np.float32)
        p5 = np.array([x1 + col_c2, cy + aperture / 2.0], dtype=np.float32)
        p6 = np.array([x1 + col_c1, cy + aperture / 2.0], dtype=np.float32)

        landmarks = [p1, p2, p3, p4, p5, p6]
        ear = calculate_ear(landmarks)
        return landmarks, ear

    def extract_mouth_landmarks(
        self,
        image: np.ndarray,
        left_corner: Tuple[float, float],
        right_corner: Tuple[float, float]
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract mouth landmarks and calculate Mouth Aspect Ratio (MAR).
        Returns 4 landmark coordinates: [m_left, m_top, m_right, m_bottom] and MAR.
        """
        h_img, w_img = image.shape[:2]
        rx, ry = right_corner
        lx, ly = left_corner

        # Mouth width and center
        mouth_w = float(np.linalg.norm(np.array([rx, ry]) - np.array([lx, ly])))
        cx = int((rx + lx) / 2.0)
        cy = int((ry + ly) / 2.0)

        roi_w = max(10, int(mouth_w * 1.2))
        roi_h = max(10, int(mouth_w * 0.9))

        x1 = max(0, cx - roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        x2 = min(w_img, x1 + roi_w)
        y2 = min(h_img, y1 + roi_h)

        mouth_roi = image[y1:y2, x1:x2]
        if mouth_roi.size == 0 or mouth_w <= 2:
            m_pts = [np.array([cx, cy], dtype=np.float32) for _ in range(4)]
            return m_pts, 0.0

        gray_m = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)

        # Dark oral cavity thresholding
        # When yawning/mouth open, the oral cavity is noticeably darker than the lips/skin
        _, dark_m = cv2.threshold(gray_m, 50, 255, cv2.THRESH_BINARY_INV)

        mid_y = mouth_roi.shape[0] // 2
        # Check active rows in central 60% of mouth ROI
        center_cols = dark_m[:, int(mouth_roi.shape[1] * 0.25):int(mouth_roi.shape[1] * 0.75)]
        active_rows = np.where(np.sum(center_cols == 255, axis=1) > (center_cols.shape[1] * 0.2))[0]

        if len(active_rows) > 3:
            top_y = int(active_rows[0])
            bot_y = int(active_rows[-1])
        else:
            top_y = mid_y - 2
            bot_y = mid_y + 2

        m_left = np.array([rx, ry], dtype=np.float32)
        m_top = np.array([cx, y1 + top_y], dtype=np.float32)
        m_right = np.array([lx, ly], dtype=np.float32)
        m_bottom = np.array([cx, y1 + bot_y], dtype=np.float32)

        mouth_pts = [m_left, m_top, m_right, m_bottom]
        mar = calculate_mar(mouth_pts)
        return mouth_pts, mar

    def process_face_single(
        self,
        norm_img: np.ndarray,
        face_info: Dict[str, Any],
        e_thresh: float,
        m_thresh: float,
        person_id: int = 1
    ) -> Dict[str, Any]:
        """
        Process a single detected face: landmarks, EAR, MAR, age, and alert state.
        Fulfills Requirement 5 (RED bounding box for sleeping) and Requirement 6 (Age prediction).
        """
        rx, ry = face_info["right_eye_center"]
        lx, ly = face_info["left_eye_center"]
        eye_dist = float(np.linalg.norm(np.array([rx, ry]) - np.array([lx, ly])))
        box_w = max(16, int(eye_dist * 0.45))
        box_h = max(12, int(eye_dist * 0.35))

        # Extract 6-point eye landmarks and compute EAR
        r_lms, r_ear = self.extract_eye_6_landmarks(norm_img, (rx, ry), box_w, box_h)
        l_lms, l_ear = self.extract_eye_6_landmarks(norm_img, (lx, ly), box_w, box_h)
        avg_ear = float(round((r_ear + l_ear) / 2.0, 4))

        # Extract mouth landmarks and compute MAR
        m_lms, mar = self.extract_mouth_landmarks(norm_img, face_info["left_mouth"], face_info["right_mouth"])

        # Age estimation via trained ML model
        age_info = get_age_estimator().estimate_age(norm_img, face_info)

        # Asleep / Awake classification via DrowsinessClassifier engine
        classifier = get_drowsiness_classifier()
        eye_feats = classifier.extract_eye_features(norm_img, l_lms, r_lms, l_ear, r_ear, face_info["bbox"])
        eye_eval = classifier.classify_eye_state(eye_feats, fallback_ear=avg_ear, ear_threshold=e_thresh)

        is_sleeping = eye_eval["is_sleeping"]
        eye_state = eye_eval["state"]
        is_yawning = (mar > m_thresh)
        mouth_state = "YAWNING" if is_yawning else "NORMAL"

        if is_sleeping:
            instant_state = "WARNING"
            instant_text = f"#{person_id} ASLEEP | {age_info['label']}"
            box_color = (0, 0, 255)  # RED for sleeping occupants (Requirement 5)
        elif is_yawning:
            instant_state = "WARNING"
            instant_text = f"#{person_id} YAWN | {age_info['label']}"
            box_color = (0, 180, 255)  # Amber for yawning
        else:
            instant_state = "ALERT"
            instant_text = f"#{person_id} AWAKE | {age_info['label']}"
            box_color = (0, 255, 0)  # GREEN for alert occupants

        return {
            "person_id": person_id,
            "bbox": face_info["bbox"],
            "confidence": face_info.get("confidence", 0.90),
            "right_eye_landmarks": r_lms,
            "left_eye_landmarks": l_lms,
            "mouth_landmarks": m_lms,
            "right_ear": r_ear,
            "left_ear": l_ear,
            "ear": avg_ear,
            "mar": mar,
            "is_sleeping": is_sleeping,
            "is_yawning": is_yawning,
            "eye_state": eye_state,
            "mouth_state": mouth_state,
            "instantaneous_state": instant_state,
            "status_text": instant_text,
            "box_color": box_color,
            "estimated_age": age_info["estimated_age"],
            "age_info": age_info,
            "face_info": face_info,
            "detection_method": eye_eval.get("method", "EAR_Geometric_Rule"),
            "classifier_trained": eye_eval.get("classifier_trained", False),
            "sleep_probability": eye_eval.get("sleep_probability", 0.0)
        }

    def process_frame_multi(
        self,
        image: np.ndarray,
        ear_threshold: Optional[float] = None,
        mar_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect and evaluate all people in the frame.
        Fulfills:
        - Requirement 1: Asleep or awake detection
        - Requirement 2: Detect multiple people in one image or video
        - Requirement 3: Count total people detected
        - Requirement 4: Count sleeping people
        - Requirement 5: Highlight sleeping people with RED bounding boxes
        - Requirement 6: Predict age of each detected person
        - Requirement 7: Display pop-up showing sleeping count and age
        """
        e_thresh = self.ear_threshold if ear_threshold is None else ear_threshold
        m_thresh = self.mar_threshold if mar_threshold is None else mar_threshold

        norm_img = self._normalize_image(image)
        if norm_img is None:
            return {
                "face_detected": False,
                "bbox": None,
                "face_bbox": None,
                "face_confidence": 0.0,
                "right_eye_landmarks": [],
                "left_eye_landmarks": [],
                "mouth_landmarks": [],
                "right_ear": 0.0,
                "left_ear": 0.0,
                "ear": 0.0,
                "mar": 0.0,
                "eye_state": "UNKNOWN",
                "mouth_state": "UNKNOWN",
                "instantaneous_state": "UNKNOWN",
                "total_people": 0,
                "faces_detected": 0,
                "sleeping_count": 0,
                "awake_count": 0,
                "people": [],
                "sleeping_people": [],
                "popup_summary": "Invalid or empty image buffer.",
                "annotated_frame": image if image is not None else np.zeros((100, 100, 3), dtype=np.uint8)
            }

        faces = self.detect_all_faces(norm_img)
        if len(faces) == 0:
            return {
                "face_detected": False,
                "bbox": None,
                "face_bbox": None,
                "face_confidence": 0.0,
                "right_eye_landmarks": [],
                "left_eye_landmarks": [],
                "mouth_landmarks": [],
                "right_ear": 0.0,
                "left_ear": 0.0,
                "ear": 0.0,
                "mar": 0.0,
                "eye_state": "NO_FACE",
                "mouth_state": "NO_FACE",
                "instantaneous_state": "NO_FACE",
                "total_people": 0,
                "faces_detected": 0,
                "sleeping_count": 0,
                "awake_count": 0,
                "people": [],
                "sleeping_people": [],
                "popup_summary": "No occupants detected in frame.",
                "annotated_frame": norm_img
            }

        people = []
        for i, face_info in enumerate(faces):
            p_data = self.process_face_single(norm_img, face_info, e_thresh, m_thresh, person_id=i + 1)
            people.append(p_data)

        total_people = len(people)
        sleeping_people = [p for p in people if p["is_sleeping"]]
        sleeping_count = len(sleeping_people)
        awake_count = total_people - sleeping_count

        if sleeping_count > 0:
            ages_str = ", ".join([f"Person #{p['person_id']} ({p['age_info']['label']})" for p in sleeping_people])
            popup_summary = f"Alert: {sleeping_count} sleeping occupant{'s' if sleeping_count > 1 else ''} detected! {ages_str}"
        else:
            popup_summary = f"All {total_people} occupant{'s are' if total_people > 1 else ' is'} alert and awake."

        annotated = self.draw_annotations_multi(norm_img, people, ear_thresh=e_thresh, mar_thresh=m_thresh)

        # Primary face (highest confidence)
        primary = max(people, key=lambda p: p["confidence"])

        return {
            "face_detected": True,
            "total_people": total_people,
            "faces_detected": total_people,
            "sleeping_count": sleeping_count,
            "awake_count": awake_count,
            "people": people,
            "sleeping_people": sleeping_people,
            "popup_summary": popup_summary,
            "annotated_frame": annotated,
            # Backward compatibility fields mapped to primary person
            "bbox": primary["bbox"],
            "face_bbox": primary["bbox"],
            "face_confidence": primary["confidence"],
            "right_eye_landmarks": primary["right_eye_landmarks"],
            "left_eye_landmarks": primary["left_eye_landmarks"],
            "mouth_landmarks": primary["mouth_landmarks"],
            "right_ear": primary["right_ear"],
            "left_ear": primary["left_ear"],
            "ear": primary["ear"],
            "mar": primary["mar"],
            "eye_state": primary["eye_state"],
            "mouth_state": primary["mouth_state"],
            "instantaneous_state": primary["instantaneous_state"],
            "estimated_age": primary["estimated_age"],
            "age_info": primary["age_info"],
            "detection_method": primary.get("detection_method", "EAR_Geometric_Rule"),
            "classifier_trained": primary.get("classifier_trained", False),
            "sleep_probability": primary.get("sleep_probability", 0.0)
        }

    def process_frame(
        self,
        image: np.ndarray,
        ear_threshold: Optional[float] = None,
        mar_threshold: Optional[float] = None,
        ear_thresh: Optional[float] = None,
        mar_thresh: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a single image or video frame.
        Supports single-face and multi-occupant scenarios.
        """
        if ear_thresh is not None and ear_threshold is None:
            ear_threshold = ear_thresh
        if mar_thresh is not None and mar_threshold is None:
            mar_threshold = mar_thresh

        return self.process_frame_multi(image, ear_threshold, mar_threshold)

    def draw_annotations_multi(
        self,
        image: np.ndarray,
        people: List[Dict[str, Any]],
        ear_thresh: float = DEFAULT_EAR_THRESHOLD,
        mar_thresh: float = DEFAULT_MAR_THRESHOLD,
        show_landmarks: bool = True
    ) -> np.ndarray:
        """
        Render annotations for multiple occupants:
        - RED bounding box for sleeping occupants (Requirement 5).
        - GREEN bounding box for awake occupants.
        - Age prediction badge on each face (Requirement 6).
        - Top frame telemetry banner summarizing total, sleeping, and awake counts.
        """
        if image is None or image.size == 0:
            return image

        annotated = image.copy()
        h, w = annotated.shape[:2]

        total_people = len(people)
        sleeping_count = sum(1 for p in people if p.get("is_sleeping", False))
        awake_count = total_people - sleeping_count

        # Top summary banner across full frame
        bar_h = 32
        cv2.rectangle(annotated, (0, 0), (w, bar_h), (24, 24, 30), -1)
        bar_color = (0, 0, 255) if sleeping_count > 0 else (0, 200, 100)
        cv2.line(annotated, (0, bar_h), (w, bar_h), bar_color, 2)

        summary_txt = f"OCCUPANTS: {total_people}  |  SLEEPING: {sleeping_count} (RED)  |  AWAKE: {awake_count} (GREEN)"
        cv2.putText(
            annotated,
            summary_txt,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        for p in people:
            bbox = p.get("bbox", (0, 0, 0, 0))
            fx, fy, fw, fh = [int(v) for v in bbox]
            if fw <= 0 or fh <= 0:
                continue

            is_sleeping = p.get("is_sleeping", False)
            is_yawning = p.get("mouth_state") == "YAWNING"

            # Requirement 5: Highlight sleeping people with RED bounding boxes
            if is_sleeping:
                color = (0, 0, 255)  # RED
            elif is_yawning:
                color = (0, 180, 255)  # Amber
            else:
                color = (0, 255, 0)  # GREEN

            box_thick = 3 if is_sleeping else 2
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, box_thick)

            # Draw landmarks
            if show_landmarks:
                for eye_lms in (p.get("right_eye_landmarks"), p.get("left_eye_landmarks")):
                    if eye_lms and len(eye_lms) == 6:
                        pts = np.array([[int(round(pt[0])), int(round(pt[1]))] for pt in eye_lms], dtype=np.int32)
                        cv2.polylines(annotated, [pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                        for pt in pts:
                            cv2.circle(annotated, tuple(pt), 2, (0, 255, 255), -1)

                m_lms = p.get("mouth_landmarks")
                if m_lms and len(m_lms) == 4:
                    m_pts = np.array([[int(round(pt[0])), int(round(pt[1]))] for pt in m_lms], dtype=np.int32)
                    cv2.polylines(annotated, [m_pts], isClosed=True, color=(0, 200, 255), thickness=1, lineType=cv2.LINE_AA)
                    for mpt in m_pts:
                        cv2.circle(annotated, tuple(mpt), 2, (0, 120, 255), -1)

            # Status and age label above bounding box
            tag_text = p.get("status_text", "")
            if not tag_text:
                tag_state = "ASLEEP" if is_sleeping else "AWAKE"
                tag_age = p.get("age_info", {}).get("label", "")
                tag_text = f"#{p.get('person_id', 1)} {tag_state} | {tag_age}"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tag_y1 = max(bar_h + 2, fy - th - 10)
            tag_y2 = fy
            tag_w = max(fw, tw + 12)
            cv2.rectangle(annotated, (fx, tag_y1), (fx + tag_w, tag_y2), color, -1)
            text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
            cv2.putText(
                annotated,
                tag_text,
                (fx + 6, tag_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                text_color,
                1,
                cv2.LINE_AA
            )

            # Bottom HUD
            ear_val = p.get("ear", 0.0)
            mar_val = p.get("mar", 0.0)
            hud_text = f"EAR: {ear_val:.3f} | MAR: {mar_val:.3f}"
            cv2.putText(
                annotated,
                hud_text,
                (fx, min(h - 8, fy + fh + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA
            )

        return annotated

    def draw_annotations(
        self,
        image: np.ndarray,
        bbox_or_res: Any = None,
        r_lms: Optional[List[np.ndarray]] = None,
        l_lms: Optional[List[np.ndarray]] = None,
        m_lms: Optional[List[np.ndarray]] = None,
        ear: float = 0.0,
        mar: float = 0.0,
        status: Optional[str] = None,
        status_text: Optional[str] = None,
        ear_thresh: float = DEFAULT_EAR_THRESHOLD,
        mar_thresh: float = DEFAULT_MAR_THRESHOLD,
        show_landmarks: bool = True
    ) -> np.ndarray:
        """
        Render HUD telemetry, bounding box, and eye/mouth landmark contours.
        Can accept either a dictionary result from process_frame, or individual components.
        """
        if image is None or image.size == 0:
            return image

        annotated = image.copy()

        # Handle dictionary input from process_frame
        if isinstance(bbox_or_res, dict):
            proc = bbox_or_res
            if not proc.get("face_detected", False):
                return annotated

            # Multi-person dispatch
            if "people" in proc and len(proc["people"]) > 0:
                return self.draw_annotations_multi(
                    annotated,
                    proc["people"],
                    ear_thresh=ear_thresh,
                    mar_thresh=mar_thresh,
                    show_landmarks=show_landmarks
                )

            bbox = proc.get("face_bbox", proc.get("bbox", (0, 0, 0, 0)))
            r_lms = proc.get("right_eye_landmarks", [])
            l_lms = proc.get("left_eye_landmarks", [])
            m_lms = proc.get("mouth_landmarks", [])
            ear = float(proc.get("ear", 0.0))
            mar = float(proc.get("mar", 0.0))
            if status is None:
                if ear < ear_thresh:
                    status = "WARNING"
                    if status_text is None:
                        status_text = "WARNING (Eyes Closed - Photo Snapshot)"
                elif mar > mar_thresh:
                    status = "WARNING"
                    if status_text is None:
                        status_text = "WARNING (Yawn Detected)"
                else:
                    status = "ALERT"
                    if status_text is None:
                        status_text = "ALERT (Eyes Open)"
        else:
            bbox = bbox_or_res if bbox_or_res is not None else (0, 0, 0, 0)
            if status is None:
                status = "ALERT"

        fx, fy, fw, fh = [int(v) for v in bbox]
        if fw <= 0 or fh <= 0:
            return annotated

        # Status color and resolved text
        status_upper = (status or "ALERT").upper()
        if "CRITICAL" in status_upper:
            status_color = (0, 0, 255)       # Red
            resolved_text = status_text if status_text is not None else "CRITICAL (Drowsiness Alarm)"
        elif "WARNING" in status_upper:
            if ear < ear_thresh:
                status_color = (0, 0, 255)   # RED for sleeping / closed eyes (Requirement 5)
                resolved_text = status_text if status_text is not None else "WARNING (Eyes Closed - Photo Snapshot)"
            else:
                status_color = (0, 180, 255) # Amber for yawn/fatigue
                resolved_text = status_text if status_text is not None else "WARNING (Yawn Detected)"
        elif "ALERT" in status_upper:
            status_color = (0, 255, 0)       # Green
            resolved_text = status_text if status_text is not None else "ALERT (Eyes Open)"
        else:
            status_color = (180, 180, 180)   # Grey
            resolved_text = status_text if status_text is not None else status

        # 1. Face bounding box (Requirement 5: RED for sleeping)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), status_color, 2)

        # 2. Draw 6-point eye contours if requested
        if show_landmarks:
            for eye_lms in (r_lms, l_lms):
                if eye_lms and len(eye_lms) == 6:
                    pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in eye_lms], dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                    for pt in pts:
                        cv2.circle(annotated, tuple(pt), 2, (0, 255, 255), -1)

            # 3. Draw 4-point mouth contour
            if m_lms and len(m_lms) == 4:
                m_pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in m_lms], dtype=np.int32)
                cv2.polylines(annotated, [m_pts], isClosed=True, color=(0, 200, 255), thickness=1, lineType=cv2.LINE_AA)
                for mpt in m_pts:
                    cv2.circle(annotated, tuple(mpt), 2, (0, 120, 255), -1)

        # 4. Telemetry Banner on Face Top
        banner_y1 = max(0, fy - 26)
        cv2.rectangle(annotated, (fx, banner_y1), (fx + fw, fy), status_color, -1)
        cv2.putText(
            annotated,
            resolved_text,
            (fx + 6, fy - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 0, 0) if "ALERT" in status_upper else (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 5. Bottom Telemetry HUD
        hud_text = f"EAR: {ear:.3f} | MAR: {mar:.3f}"
        cv2.putText(
            annotated,
            hud_text,
            (fx, min(annotated.shape[0] - 8, fy + fh + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA
        )

        return annotated


def get_drowsiness_detector() -> DrowsinessDetector:
    """Helper function to retrieve singleton instance of DrowsinessDetector."""
    return DrowsinessDetector.get_instance()

