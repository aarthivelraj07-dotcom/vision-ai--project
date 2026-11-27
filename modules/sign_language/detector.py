"""
Task 5: Indian Sign Language (ISL) Hand Detector and Static Sign Recognition Pipeline.

Architecture:
- Palm Detection: MediaPipe Palm Detection (OpenCV DNN ONNX, 192x192)
- Hand Pose Estimation: MediaPipe Handpose Estimation (OpenCV DNN ONNX, 224x224)
- 21 3D Landmarks: Wrist-centered (L0 origin) and palm-scale normalized (||L9 - L0||_2)
- Static Sign Classifier: Scikit-learn Pipeline (StandardScaler + RBF Support Vector Machine)
- Recognized Classes (12): '1', '2', '3', '4', '5', 'A', 'B', 'C', 'L', 'V', 'W', 'Y'
- Operating Window: 6:00 PM to 10:00 PM IST (Asia/Kolkata server time enforcement)
"""

import os
import hmac
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import cv2
import joblib
import numpy as np

# Operating Time Restrictions (6:00 PM - 10:00 PM IST)
IST = ZoneInfo("Asia/Kolkata")
OPERATING_START = dtime(18, 0, 0)   # 18:00:00 IST (6:00 PM)
OPERATING_END = dtime(22, 0, 0)     # 22:00:00 IST (10:00 PM)

SUPPORTED_CLASSES = ("1", "2", "3", "4", "5", "A", "B", "C", "L", "V", "W", "Y")

# Hand Landmark Skeleton Connection Pairs (21 MediaPipe hand keypoints)
HAND_CONNECTIONS = (
    # Palm base & Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles interconnect
    (5, 9), (9, 13), (13, 17)
)


def get_current_ist_time() -> datetime:
    """Retrieve current time strictly localized to Indian Standard Time (Asia/Kolkata)."""
    return datetime.now(IST)


def is_within_operating_window(now: Optional[datetime] = None) -> bool:
    """
    Check if the server time falls within the authorized operating window (18:00 - 22:00 IST).
    Server-side authoritative check; never trusts browser client clocks.
    """
    if now is None:
        now = get_current_ist_time()
    current_time = now.time()
    return OPERATING_START <= current_time <= OPERATING_END


def get_next_operating_window(now: Optional[datetime] = None) -> str:
    """Return human-readable notification for the next permitted operating window."""
    if now is None:
        now = get_current_ist_time()
    
    if now.time() < OPERATING_START:
        date_str = now.strftime("%Y-%m-%d")
        return f"Today ({date_str}) from 06:00 PM to 10:00 PM IST"
    else:
        tomorrow = now + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        return f"Tomorrow ({date_str}) from 06:00 PM to 10:00 PM IST"


def is_admin_override_configured() -> bool:
    """Check if an administrator secret is configured in the environment."""
    pin = os.environ.get("ISL_ADMIN_PIN") or os.environ.get("ATTENDANCE_ADMIN_PIN", "")
    return len(pin.strip()) > 0


def verify_admin_override(input_pin: str) -> bool:
    """
    Verify administrator override credential in constant time.
    Zero hardcoded passwords; strictly requires server environment variable configuration.
    """
    if not input_pin or not is_admin_override_configured():
        return False
    configured_pin = (os.environ.get("ISL_ADMIN_PIN") or os.environ.get("ATTENDANCE_ADMIN_PIN", "")).strip()
    return hmac.compare_digest(input_pin.strip(), configured_pin)


def generate_anchors() -> np.ndarray:
    """Generate 2,016 anchors for MediaPipe palm detector (192x192 grid)."""
    anchors = []
    # 24x24 grid (stride 8, 2 anchors per cell = 1152)
    for y in range(24):
        for x in range(24):
            for _ in range(2):
                anchors.append([(x + 0.5) * 8 / 192.0, (y + 0.5) * 8 / 192.0, 1.0, 1.0])
    # 12x12 grid (stride 16, 6 anchors per cell = 864)
    for y in range(12):
        for x in range(12):
            for _ in range(6):
                anchors.append([(x + 0.5) * 16 / 192.0, (y + 0.5) * 16 / 192.0, 1.0, 1.0])
    return np.array(anchors, dtype=np.float32)


_CACHED_ANCHORS = generate_anchors()


@dataclass
class ISLRecognitionResult:
    """Structured result returned by ISLHandDetector."""
    detected: bool
    predicted_class: Optional[str] = None
    relative_confidence: float = 0.0
    probabilities: Optional[Dict[str, float]] = None
    landmarks_orig: Optional[np.ndarray] = None  # (21, 3) in image pixel coordinates
    landmarks_norm: Optional[np.ndarray] = None  # (63,) normalized feature vector
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    palm_score: float = 0.0
    hand_score: float = 0.0
    num_hands_detected: int = 0
    message: str = ""


class ISLHandDetector:
    """
    Production detector for static Indian Sign Language (ISL) hand gestures.
    Encapsulates OpenCV DNN ONNX landmark inference and serialized SVM classification.
    """

    def __init__(
        self,
        palm_model_path: Optional[str] = None,
        hand_model_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        root_dir: Optional[str] = None
    ):
        if root_dir is None:
            # Current file is in modules/sign_language/ -> root is 2 levels up
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.root_dir = root_dir

        self.palm_model_path = palm_model_path or os.path.join(
            self.root_dir, "models", "sign_language", "palm_detection_mediapipe_2023feb.onnx"
        )
        self.hand_model_path = hand_model_path or os.path.join(
            self.root_dir, "models", "sign_language", "handpose_estimation_mediapipe_2023feb.onnx"
        )
        self.classifier_path = classifier_path or os.path.join(
            self.root_dir, "models", "sign_language", "isl_classifier_pipeline.joblib"
        )

        self._palm_net: Optional[cv2.dnn.Net] = None
        self._hand_net: Optional[cv2.dnn.Net] = None
        self._classifier = None
        self._anchors = _CACHED_ANCHORS

        # Lazy load networks upon instantiation
        self._load_models()

    def _load_models(self) -> None:
        """Load and verify ONNX vision networks and serialized scikit-learn pipeline."""
        if not os.path.exists(self.palm_model_path):
            raise FileNotFoundError(f"Palm detection ONNX model not found: {self.palm_model_path}")
        if not os.path.exists(self.hand_model_path):
            raise FileNotFoundError(f"Hand pose ONNX model not found: {self.hand_model_path}")
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"ISL classifier pipeline not found: {self.classifier_path}")

        self._palm_net = cv2.dnn.readNetFromONNX(self.palm_model_path)
        self._hand_net = cv2.dnn.readNetFromONNX(self.hand_model_path)
        self._classifier = joblib.load(self.classifier_path)

    @property
    def is_loaded(self) -> bool:
        """Check whether all underlying models are active and memory-resident."""
        return (self._palm_net is not None and 
                self._hand_net is not None and 
                self._classifier is not None)

    @property
    def classes(self) -> List[str]:
        """Return the list of recognized static sign classes."""
        if self._classifier is not None and hasattr(self._classifier, "classes_"):
            return list(self._classifier.classes_)
        return list(SUPPORTED_CLASSES)

    def process_image(self, image_input: Union[str, np.ndarray]) -> ISLRecognitionResult:
        """
        Extract hand landmarks and predict static ISL sign from an image path or BGR array.
        
        Args:
            image_input: File path (str) or BGR image numpy array.
            
        Returns:
            ISLRecognitionResult containing detection status, predicted class,
            relative confidence, landmark coordinates, and bounding box.
        """
        # 1. Load image safely
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return ISLRecognitionResult(detected=False, message=f"File not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                return ISLRecognitionResult(detected=False, message="Failed to decode image file (corrupted or unsupported format).")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) < 2:
                return ISLRecognitionResult(detected=False, message="Empty or invalid image array provided.")
            if len(image_input.shape) == 2:
                # Grayscale to BGR
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                # BGRA to BGR
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image_input.copy()
        else:
            return ISLRecognitionResult(detected=False, message="Invalid input type. Expected image file path or numpy array.")

        h_img, w_img = img_bgr.shape[:2]
        if h_img < 16 or w_img < 16:
            return ISLRecognitionResult(detected=False, message="Image resolution too low for landmark localization.")

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Stage A: Palm Detection (192x192)
        resized_palm = cv2.resize(rgb, (192, 192)).astype(np.float32) / 255.0
        blob_palm = np.expand_dims(resized_palm, axis=0)
        self._palm_net.setInput(blob_palm)
        raw_boxes, raw_scores = self._palm_net.forward(["Identity", "Identity_1"])
        
        scores = 1.0 / (1.0 + np.exp(-raw_scores[0, :, 0]))
        boxes = raw_boxes[0, :, :]

        cand_idx = np.where(scores >= 0.5)[0]
        num_detected = len(cand_idx)

        if num_detected == 0:
            max_s = float(np.max(scores)) if len(scores) > 0 else 0.0
            return ISLRecognitionResult(
                detected=False,
                palm_score=max_s,
                num_hands_detected=0,
                message="No hand detected. Please position your hand clearly in front of the camera."
            )

        # Pick candidate with highest confidence score
        best_idx = int(cand_idx[np.argmax(scores[cand_idx])])
        palm_score = float(scores[best_idx])
        anc = self._anchors[best_idx]

        dx, dy, bw_raw, bh_raw = boxes[best_idx][:4]
        cx = (dx / 192.0 + anc[0]) * w_img
        cy = (dy / 192.0 + anc[1]) * h_img
        bw = (bw_raw / 192.0) * w_img
        bh = (bh_raw / 192.0) * h_img

        # Keypoints for palm rotation alignment (wrist kp0 and middle base kp2)
        kp0_x = (boxes[best_idx][4] / 192.0 + anc[0]) * w_img
        kp0_y = (boxes[best_idx][5] / 192.0 + anc[1]) * h_img
        kp2_x = (boxes[best_idx][8] / 192.0 + anc[0]) * w_img
        kp2_y = (boxes[best_idx][9] / 192.0 + anc[1]) * h_img

        rotation = 0.5 * np.pi - np.arctan2(-(kp2_y - kp0_y), kp2_x - kp0_x)
        box_size = max(bw, bh) * 2.6
        shift_x = -0.5 * (kp2_x - kp0_x)
        shift_y = -0.5 * (kp2_y - kp0_y)
        hand_cx = cx - shift_x * 0.5
        hand_cy = cy - shift_y * 0.5

        # 3. Stage B: Handpose Estimation (224x224 warped crop)
        M = cv2.getRotationMatrix2D((hand_cx, hand_cy), float(np.degrees(rotation)), 224.0 / box_size)
        M[0, 2] += (112.0 - hand_cx)
        M[1, 2] += (112.0 - hand_cy)

        warped = cv2.warpAffine(rgb, M, (224, 224), borderMode=cv2.BORDER_REPLICATE)
        blob_hand = np.expand_dims(warped.astype(np.float32) / 255.0, axis=0)

        self._hand_net.setInput(blob_hand)
        lms_raw, conf_raw, _, _ = self._hand_net.forward(
            ["Identity", "Identity_1", "Identity_2", "Identity_3"]
        )

        hand_score = float(conf_raw[0, 0])
        if hand_score < 0.5:
            return ISLRecognitionResult(
                detected=False,
                palm_score=palm_score,
                hand_score=hand_score,
                num_hands_detected=num_detected,
                message=f"Low hand pose estimation confidence ({hand_score:.2f}). Please hold hand steady."
            )

        lms_21 = lms_raw.reshape((21, 3))

        # Invert affine warp back to original image coordinate frame
        inv_M = cv2.invertAffineTransform(M)
        pts_2d = lms_21[:, :2]
        pts_homo = np.hstack([pts_2d, np.ones((21, 1), dtype=np.float32)])
        orig_pts_2d = pts_homo @ inv_M.T

        orig_lms = np.zeros((21, 3), dtype=np.float32)
        orig_lms[:, :2] = orig_pts_2d
        orig_lms[:, 2] = lms_21[:, 2] * (box_size / 224.0)

        # 4. Feature Normalization (Identical to training protocol)
        # Wrist (L0) at origin, scaled by Euclidean palm length ||L9 - L0||
        wrist = orig_lms[0]
        middle_mcp = orig_lms[9]
        palm_scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
        if palm_scale < 1e-4:
            palm_scale = float(box_size) if box_size > 1e-4 else 1.0

        norm_lms = (orig_lms - wrist) / palm_scale
        feat_vector = norm_lms.flatten().astype(np.float32)  # Shape (63,)

        # 5. Classifier Inference
        X_in = feat_vector.reshape(1, -1)
        pred_class = str(self._classifier.predict(X_in)[0])

        # Relative confidence (Platt scaling posterior estimate)
        probabilities = {}
        if hasattr(self._classifier, "predict_proba"):
            raw_probas = self._classifier.predict_proba(X_in)[0]
            cls_list = list(self._classifier.classes_)
            for c_name, p_val in zip(cls_list, raw_probas):
                probabilities[str(c_name)] = float(p_val)
            pred_conf = probabilities.get(pred_class, 1.0)
        else:
            pred_conf = 1.0
            probabilities[pred_class] = 1.0

        # Compute bounding box
        x_min = int(max(0, np.min(orig_lms[:, 0])))
        y_min = int(max(0, np.min(orig_lms[:, 1])))
        x_max = int(min(w_img - 1, np.max(orig_lms[:, 0])))
        y_max = int(min(h_img - 1, np.max(orig_lms[:, 1])))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return ISLRecognitionResult(
            detected=True,
            predicted_class=pred_class,
            relative_confidence=pred_conf,
            probabilities=probabilities,
            landmarks_orig=orig_lms,
            landmarks_norm=feat_vector,
            bbox=bbox,
            palm_score=palm_score,
            hand_score=hand_score,
            num_hands_detected=num_detected,
            message="Hand detected and static sign recognized successfully."
        )

    def draw_landmarks(
        self,
        image_bgr: np.ndarray,
        result: ISLRecognitionResult,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        draw_label: bool = True
    ) -> np.ndarray:
        """
        Overlay hand landmark skeleton, bounding box, and sign prediction tag onto frame.
        """
        annotated = image_bgr.copy()
        if not result.detected or result.landmarks_orig is None:
            return annotated

        pts = result.landmarks_orig[:, :2].astype(int)

        # Draw skeleton connections
        if draw_skeleton:
            for p1, p2 in HAND_CONNECTIONS:
                pt1 = (int(pts[p1, 0]), int(pts[p1, 1]))
                pt2 = (int(pts[p2, 0]), int(pts[p2, 1]))
                cv2.line(annotated, pt1, pt2, (40, 180, 255), 2, cv2.LINE_AA)

            # Draw keypoint landmarks
            for idx, pt in enumerate(pts):
                # Highlight wrist and fingertips
                color = (0, 0, 255) if idx in (0, 4, 8, 12, 16, 20) else (0, 255, 128)
                radius = 4 if idx in (0, 4, 8, 12, 16, 20) else 3
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), radius, color, -1, cv2.LINE_AA)

        # Draw bounding box
        if draw_bbox and result.bbox is not None:
            bx, by, bw, bh = result.bbox
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 200, 100), 2)

        # Draw label tag
        if draw_label and result.predicted_class is not None:
            label_text = f"Sign: {result.predicted_class} ({result.relative_confidence * 100:.1f}%)"
            bx = result.bbox[0] if result.bbox else 20
            by = max(25, (result.bbox[1] - 10) if result.bbox else 30)

            # Draw background rectangle for readability
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (bx, by - th - 6), (bx + tw + 8, by + 4), (0, 0, 0), -1)
            cv2.putText(annotated, label_text, (bx + 4, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)

        return annotated
