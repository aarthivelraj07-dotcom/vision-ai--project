"""
Machine Learning Drowsiness (Asleep vs Awake) Classifier Engine.
Provides binary classification (Awake = 0, Asleep = 1) using trained ML models
(RandomForestClassifier / LogisticRegression) on eye-region biometric features.
"""

import os
import threading
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "drowsiness", "drowsiness_classifier_model.pkl")


class DrowsinessClassifier:
    """
    Dedicated Binary ML Classifier for Asleep vs. Awake Detection.
    Predicts:
        0 = AWAKE (Eyes open, alert, active gaze)
        1 = ASLEEP (Eyes closed / micro-sleep)
    """
    _instance: Optional["DrowsinessClassifier"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = {}
        self.is_trained = False
        self._load_pipeline()

    @classmethod
    def get_instance(cls) -> "DrowsinessClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            cls._instance = None

    def _load_pipeline(self):
        """Load trained scikit-learn pipeline if available."""
        if os.path.exists(CLASSIFIER_PATH):
            try:
                pipeline = joblib.load(CLASSIFIER_PATH)
                self.model = pipeline.get("model")
                self.scaler = pipeline.get("scaler")
                self.is_trained = (self.model is not None)
            except Exception as e:
                self.is_trained = False
        else:
            self.is_trained = False

    def extract_eye_features(
        self,
        image: np.ndarray,
        left_eye_lms: List[np.ndarray],
        right_eye_lms: List[np.ndarray],
        left_ear: float,
        right_ear: float,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract the 8-dimensional biometric eye-feature vector from a face frame
        matching the EXACT same feature space, definition, and ordering as training:
        1. EAR (mean EAR of left and right eyes)
        2. Normalized vertical aperture
        3. Log Laplacian texture variance
        4. Sobel horizontal edge energy
        5. Sobel vertical edge energy
        6. Contrast ratio
        7. Intensity standard deviation
        8. Minimum-to-mean luminance ratio
        """
        if image is None or not left_eye_lms or not right_eye_lms:
            return None

        h_img, w_img = image.shape[:2]

        def _crop_patch(lms):
            xs = [float(p[0]) for p in lms]
            ys = [float(p[1]) for p in lms]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pw = max(8.0, max_x - min_x)
            ph = max(8.0, max_y - min_y)
            x1 = max(0, int(min_x - pw * 0.25))
            y1 = max(0, int(min_y - ph * 0.25))
            x2 = min(w_img, int(max_x + pw * 0.25))
            y2 = min(h_img, int(max_y + ph * 0.25))
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]
            return None

        left_patch = _crop_patch(left_eye_lms)
        right_patch = _crop_patch(right_eye_lms)

        f_left = self.extract_single_eye_features(left_patch) if left_patch is not None else None
        f_right = self.extract_single_eye_features(right_patch) if right_patch is not None else None

        if f_left is not None and f_right is not None:
            mean_feats = (f_left + f_right) / 2.0
            mean_feats[0] = float((left_ear + right_ear) / 2.0)
            return mean_feats
        elif f_left is not None:
            f_left[0] = float(left_ear)
            return f_left
        elif f_right is not None:
            f_right[0] = float(right_ear)
            return f_right
        else:
            mean_ear = float((left_ear + right_ear) / 2.0)
            return np.array([mean_ear, 0.2, 3.0, 15.0, 15.0, 0.1, 20.0, 0.2], dtype=np.float32)

    def extract_single_eye_features(self, eye_patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 8-dimensional biometric feature vector from a single eye patch
        (compatible with MRL Eye Dataset crops, CEW, and face detector ROIs):
        1. EAR (aperture / width)
        2. Normalized vertical aperture (aperture / height)
        3. Log Laplacian texture variance
        4. Sobel horizontal edge energy
        5. Sobel vertical edge energy
        6. Contrast ratio (central iris vs periphery)
        7. Intensity standard deviation
        8. Minimum-to-mean luminance ratio
        """
        if eye_patch is None or eye_patch.size == 0:
            return None

        gray = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY) if len(eye_patch.shape) == 3 else eye_patch
        h, w = gray.shape[:2]
        if h <= 4 or w <= 4:
            return None

        # 1. Aperture via adaptive thresholding on central 60% palpebral fissure band
        y_start = int(h * 0.20)
        y_end = int(h * 0.80)
        mid_roi = gray[y_start:y_end, :]

        blurred = cv2.GaussianBlur(mid_roi, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

        col_spans = []
        for c in range(int(w * 0.15), int(w * 0.85)):
            dark_pixels = np.where(thresh[:, c] == 255)[0]
            if len(dark_pixels) >= 2:
                col_spans.append(float(dark_pixels[-1] - dark_pixels[0] + 1))

        if len(col_spans) >= 2:
            aperture = float(np.median(col_spans))
        else:
            aperture = 1.0

        ear = float(aperture / max(1.0, float(w)))
        norm_aperture = float(aperture / max(1.0, float(h)))

        # 2. Laplacian texture variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        log_lap_var = float(np.log1p(lap_var))

        # 3. Sobel edge energies
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h_energy = float(np.mean(np.abs(sobel_y)))
        sobel_v_energy = float(np.mean(np.abs(sobel_x)))

        # 4. Central vs peripheral contrast
        ch, cw = max(1, h // 4), max(1, w // 4)
        center_roi = gray[ch:h - ch, cw:w - cw]
        total_mean = float(np.mean(gray))
        center_mean = float(np.mean(center_roi)) if center_roi.size > 0 else total_mean
        contrast_ratio = float(abs(center_mean - total_mean) / (total_mean + 1e-5))

        # 5. Intensity dispersion
        intensity_std = float(np.std(gray))
        min_mean_ratio = float(np.min(gray) / (total_mean + 1e-5))

        return np.array([
            ear,
            norm_aperture,
            log_lap_var,
            sobel_h_energy,
            sobel_v_energy,
            contrast_ratio,
            intensity_std,
            min_mean_ratio
        ], dtype=np.float32)

    def classify_eye_state(
        self,
        features: Optional[np.ndarray],
        fallback_ear: float,
        ear_threshold: float = 0.22
    ) -> Dict[str, Any]:
        """
        Predict whether occupant is Awake (0) or Asleep (1).
        If an ML model is trained, predicts using the trained classifier.
        If no ML model is available, uses the verified Soukupová-Čech EAR threshold.
        """
        if self.is_trained and features is not None and self.model is not None:
            try:
                X = features.reshape(1, -1)
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                pred_label = int(self.model.predict(X)[0])
                prob = self.model.predict_proba(X)[0] if hasattr(self.model, "predict_proba") else None
                sleep_prob = float(prob[1]) if prob is not None else float(pred_label)

                if fallback_ear >= ear_threshold and sleep_prob < 0.70:
                    is_sleeping = False
                elif fallback_ear < ear_threshold and sleep_prob > 0.30:
                    is_sleeping = True
                else:
                    is_sleeping = (pred_label == 1)

                return {
                    "is_sleeping": is_sleeping,
                    "state": "CLOSED" if is_sleeping else "OPEN",
                    "sleep_probability": round(sleep_prob, 3),
                    "method": "ML_Classifier",
                    "classifier_trained": True
                }
            except Exception:
                pass

        # Calibrated geometric rule baseline (Soukupová-Čech EAR)
        is_sleeping = (fallback_ear < ear_threshold)
        return {
            "is_sleeping": is_sleeping,
            "state": "CLOSED" if is_sleeping else "OPEN",
            "sleep_probability": 0.95 if is_sleeping else 0.05,
            "method": "EAR_Geometric_Rule",
            "classifier_trained": self.is_trained
        }


def get_drowsiness_classifier() -> DrowsinessClassifier:
    """Helper function to retrieve singleton instance of DrowsinessClassifier."""
    return DrowsinessClassifier.get_instance()
