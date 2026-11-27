"""
Age Estimation Engine for Task 3: Drowsiness & Occupant Monitoring.
Uses OpenCV SFace deep facial embeddings + craniofacial geometric ratios
with a Random Forest Regressor trained on student and adult facial datasets.
Fulfills Requirement 6: "Predict the age of each detected person."
"""

import os
import threading
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "drowsiness", "age_estimator_model.pkl")
SFACE_PATH = os.path.join(PROJECT_ROOT, "models", "attendance", "face_recognition_sface_2021dec.onnx")


class AgeEstimator:
    """
    Biometric age estimation model with uncertainty quantification.
    Evaluates deep facial representations and craniofacial geometry.
    """
    _instance: Optional["AgeEstimator"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.scaler = None
        self.sface = None
        self._load_pipeline()

    @classmethod
    def get_instance(cls) -> "AgeEstimator":
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
        """Load trained pipeline and OpenCV SFace feature extractor."""
        if os.path.exists(MODEL_PATH):
            try:
                pipeline = joblib.load(MODEL_PATH)
                self.model = pipeline.get("model")
                self.scaler = pipeline.get("scaler")
            except Exception as e:
                print(f"[AgeEstimator] Warning loading pipeline: {e}")

        if os.path.exists(SFACE_PATH):
            try:
                self.sface = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            except Exception as e:
                print(f"[AgeEstimator] Warning loading SFace: {e}")

    def extract_features(self, image: np.ndarray, face_meta_or_array: Any) -> Optional[np.ndarray]:
        """Extract 136-dimensional feature vector."""
        if self.sface is None or image is None:
            return None

        # Convert dict or array to face vector
        if isinstance(face_meta_or_array, dict):
            # Construct YuNet-compatible 15-element array
            bbox = face_meta_or_array.get("bbox", (0, 0, 0, 0))
            rx, ry = face_meta_or_array.get("right_eye_center", (0, 0))
            lx, ly = face_meta_or_array.get("left_eye_center", (0, 0))
            nx, ny = face_meta_or_array.get("nose_tip", (0, 0))
            rmx, rmy = face_meta_or_array.get("right_mouth", (0, 0))
            lmx, lmy = face_meta_or_array.get("left_mouth", (0, 0))
            conf = face_meta_or_array.get("confidence", 0.90)
            face = np.array([
                bbox[0], bbox[1], bbox[2], bbox[3],
                rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy, conf
            ], dtype=np.float32)
        else:
            face = np.array(face_meta_or_array, dtype=np.float32)

        try:
            # 1. SFace Embedding (128-D)
            aligned = self.sface.alignCrop(image, face)
            embedding = self.sface.feature(aligned).flatten()

            # 2. Geometric landmarks from face array
            fx, fy, fw, fh = face[0:4]
            rx, ry = face[4:6]
            lx, ly = face[6:8]
            nx, ny = face[8:10]
            rmx, rmy = face[10:12]
            lmx, lmy = face[12:14]

            fw = max(1.0, float(fw))
            fh = max(1.0, float(fh))

            eye_mid_x = (rx + lx) / 2.0
            eye_mid_y = (ry + ly) / 2.0
            mouth_mid_x = (rmx + lmx) / 2.0
            mouth_mid_y = (rmy + lmy) / 2.0

            craniofacial_ratio = abs(mouth_mid_y - eye_mid_y) / fh
            ipd = np.linalg.norm(np.array([rx, ry]) - np.array([lx, ly]))
            interocular_ratio = ipd / fw
            aspect_ratio = fh / fw
            mouth_w = np.linalg.norm(np.array([rmx, rmy]) - np.array([lmx, lmy]))
            lip_thickness_ratio = abs(mouth_mid_y - ny) / max(1.0, mouth_w)
            lower_face_ratio = (fy + fh - mouth_mid_y) / fh

            # 3. High-frequency skin texture variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h_img, w_img = gray.shape

            f_y1 = max(0, int(fy))
            f_y2 = max(f_y1 + 1, int(fy + fh * 0.25))
            f_x1 = max(0, int(fx + fw * 0.25))
            f_x2 = min(w_img, int(fx + fw * 0.75))
            forehead_roi = gray[f_y1:f_y2, f_x1:f_x2]
            forehead_tex = float(cv2.Laplacian(forehead_roi, cv2.CV_64F).var()) if forehead_roi.size > 10 else 10.0

            c_y1 = max(0, int(ny))
            c_y2 = min(h_img, int(ny + fh * 0.25))
            c_x1 = max(0, int(fx + fw * 0.15))
            c_x2 = min(w_img, int(fx + fw * 0.40))
            cheek_roi = gray[c_y1:c_y2, c_x1:c_x2]
            cheek_tex = float(cv2.Laplacian(cheek_roi, cv2.CV_64F).var()) if cheek_roi.size > 10 else 10.0

            p_y1 = max(0, int(ry - fh * 0.08))
            p_y2 = min(h_img, int(ry + fh * 0.08))
            p_x1 = max(0, int(rx - fw * 0.20))
            p_x2 = min(w_img, int(rx - fw * 0.05))
            periorbital_roi = gray[p_y1:p_y2, p_x1:p_x2]
            peri_tex = float(cv2.Laplacian(periorbital_roi, cv2.CV_64F).var()) if periorbital_roi.size > 10 else 10.0

            extra_feats = np.array([
                craniofacial_ratio,
                interocular_ratio,
                aspect_ratio,
                np.log1p(forehead_tex),
                np.log1p(cheek_tex),
                np.log1p(peri_tex),
                lip_thickness_ratio,
                lower_face_ratio
            ], dtype=np.float32)

            return np.concatenate([embedding, extra_feats])
        except Exception as e:
            return None

    def estimate_age(self, image: np.ndarray, face_data: Any) -> Dict[str, Any]:
        """
        Predict age with calibrated uncertainty brackets.
        Returns:
            Dict containing estimated_age, uncertainty, bracket, label, and confidence.
        """
        feats = self.extract_features(image, face_data)

        if feats is None or self.model is None or self.scaler is None:
            # Fallback baseline when model is uninitialized or face is degenerate
            return {
                "estimated_age": 25,
                "uncertainty": 5,
                "bracket": "20–30 yrs",
                "label": "~25 ± 5 yrs",
                "confidence": 0.50,
                "note": "Baseline demographic approximation (biometrics unresolvable)"
            }

        X = feats.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Ensemble predictions to calculate uncertainty
        tree_preds = [tree.predict(X_scaled)[0] for tree in self.model.estimators_]
        pred_age = float(np.mean(tree_preds))
        std_age = float(np.std(tree_preds))

        # Constrain to plausible human age range
        clamped_age = max(18.0, min(75.0, pred_age))
        uncertainty = max(4.0, min(8.0, round(std_age * 1.5, 0)))

        bracket_low = max(18, int(clamped_age - uncertainty))
        bracket_high = min(80, int(clamped_age + uncertainty))

        confidence = max(0.60, min(0.95, round(1.0 - (std_age / max(1.0, clamped_age)), 2)))

        return {
            "estimated_age": int(round(clamped_age)),
            "uncertainty": int(uncertainty),
            "bracket": f"{bracket_low}–{bracket_high} yrs",
            "label": f"~{int(round(clamped_age))} ± {int(uncertainty)} yrs",
            "confidence": confidence,
            "raw_pred": round(pred_age, 2),
            "note": "Machine learning estimate based on SFace embeddings and craniofacial geometry"
        }


def get_age_estimator() -> AgeEstimator:
    """Helper function to retrieve singleton instance of AgeEstimator."""
    return AgeEstimator.get_instance()
