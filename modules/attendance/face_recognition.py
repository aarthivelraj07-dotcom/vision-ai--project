"""
Face Detection and Face Recognition Module.
Uses OpenCV YuNet (FaceDetectorYN) and SFace (FaceRecognizerSF) deep learning models.
Includes Haar Cascade fallback, biometric face quality validation, embedding extraction,
and open-set classification with configurable confidence thresholds.
"""

import os
import pickle
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional

YUNET_PATH = os.path.join("models", "attendance", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join("models", "attendance", "face_recognition_sface_2021dec.onnx")
HAAR_PATH = os.path.join("haarcascade_frontalface_default.xml")
CLASSIFIER_PATH = os.path.join("models", "attendance", "face_classifier.pkl")

# Default SFace cosine similarity threshold for real human faces (calibrated to 0.50)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50


class FaceRecognitionSystem:
    def __init__(
        self,
        yunet_model: str = YUNET_PATH,
        sface_model: str = SFACE_PATH,
        haar_path: str = HAAR_PATH,
        classifier_path: str = CLASSIFIER_PATH
    ):
        self.yunet_model = yunet_model
        self.sface_model = sface_model
        self.haar_path = haar_path
        self.classifier_path = classifier_path

        self.detector = None
        self.haar_detector = None
        self.recognizer = None
        
        self.registered_embeddings: Optional[np.ndarray] = None
        self.registered_ids: List[str] = []
        self.student_names: Dict[str, str] = {}
        
        self._load_models()
        self.load_classifier()

    def _load_models(self):
        """Initialize YuNet, SFace, and Haar Cascade detectors."""
        if os.path.exists(self.yunet_model):
            try:
                self.detector = cv2.FaceDetectorYN_create(
                    self.yunet_model,
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception as e:
                print(f"Warning: Could not initialize YuNet detector: {e}")

        if os.path.exists(self.sface_model):
            try:
                self.recognizer = cv2.FaceRecognizerSF_create(self.sface_model, "")
            except Exception as e:
                print(f"Warning: Could not initialize SFace recognizer: {e}")

        if os.path.exists(self.haar_path):
            try:
                self.haar_detector = cv2.CascadeClassifier(self.haar_path)
            except Exception as e:
                print(f"Warning: Could not initialize Haar cascade: {e}")

    def detect_faces(self, image: np.ndarray, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        Returns a list of dicts with keys:
          - 'bbox': (x, y, w, h)
          - 'landmarks': 5 facial landmark pairs (10 elements)
          - 'score': float confidence
          - 'raw_face': the 15-dim parameter array formatted for SFace alignCrop
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detected_faces = []

        # 1. Primary: YuNet CNN Detector
        if self.detector is not None:
            self.detector.setInputSize((w, h))
            self.detector.setScoreThreshold(score_threshold)
            try:
                retval, faces = self.detector.detect(image)
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        # Keep coordinates within image boundaries
                        fx = max(0, fx)
                        fy = max(0, fy)
                        fw = min(w - fx, fw)
                        fh = min(h - fy, fh)
                        if fw > 10 and fh > 10:
                            detected_faces.append({
                                'bbox': (fx, fy, fw, fh),
                                'score': float(f[14]),
                                'landmarks': f[4:14],
                                'raw_face': f.astype(np.float32)
                            })
                    return detected_faces
            except Exception as e:
                print(f"YuNet detection exception: {e}")

        # 2. Fallback: Haar Cascade
        if self.haar_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            haar_faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            for (hx, hy, hw, hh) in haar_faces:
                # Approximate 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
                raw = np.array([
                    hx, hy, hw, hh,
                    hx + 0.3 * hw, hy + 0.35 * hh,
                    hx + 0.7 * hw, hy + 0.35 * hh,
                    hx + 0.5 * hw, hy + 0.55 * hh,
                    hx + 0.35 * hw, hy + 0.75 * hh,
                    hx + 0.65 * hw, hy + 0.75 * hh,
                    0.85
                ], dtype=np.float32)
                detected_faces.append({
                    'bbox': (int(hx), int(hy), int(hw), int(hh)),
                    'score': 0.85,
                    'landmarks': raw[4:14],
                    'raw_face': raw
                })

        return detected_faces

    def validate_face_quality(self, image: np.ndarray, face_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate face image quality for biometric registration.
        Checks resolution, sharpness (blur), and lighting/contrast.
        """
        x, y, w, h = face_info['bbox']
        
        # 1. Resolution Check
        if w < 50 or h < 50:
            return False, f"Face resolution too small ({w}x{h} px). Minimum required is 50x50 px."

        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return False, "Invalid face crop."

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # 2. Blur / Sharpness Check (Laplacian variance)
        blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        if blur_score < 8.0:
            return False, f"Face is too blurry (sharpness score {blur_score:.1f} < 8.0). Please hold steady."

        # 3. Brightness & Contrast Check
        mean_brightness = np.mean(gray_crop)
        if mean_brightness < 25.0:
            return False, f"Face is too dark (mean brightness {mean_brightness:.1f} < 25.0). Improve lighting."
        if mean_brightness > 240.0:
            return False, f"Face is overexposed (mean brightness {mean_brightness:.1f} > 240.0). Reduce glare."

        return True, "Quality checks passed."

    def extract_embedding(self, image: np.ndarray, face_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Crop, align, and extract 128-dimensional embedding using SFace.
        """
        if self.recognizer is None:
            raise RuntimeError("FaceRecognizerSF model is not loaded.")

        raw_face = face_info.get('raw_face')
        if raw_face is None:
            return None

        # Align and crop face to standard 112x112 input
        aligned_face = self.recognizer.alignCrop(image, raw_face)
        if aligned_face is None or aligned_face.size == 0:
            return None

        # Extract 128-D embedding vector
        feature = self.recognizer.feature(aligned_face)
        # Normalize to unit length for cosine distance
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm
        return feature.flatten().astype(np.float32)

    def train_classifier(self, embeddings: np.ndarray, labels: List[str], names_map: Dict[str, str]):
        """
        Train/fit the identification model on registered student embeddings.
        Supports continuous updates when new students are registered.
        """
        if embeddings is None or len(embeddings) == 0:
            self.registered_embeddings = None
            self.registered_ids = []
            self.student_names = {}
            if os.path.exists(self.classifier_path):
                os.remove(self.classifier_path)
            return

        self.registered_embeddings = np.asarray(embeddings, dtype=np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(self.registered_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.registered_embeddings = self.registered_embeddings / norms
        
        self.registered_ids = list(labels)
        self.student_names = dict(names_map)

        # Save classifier metadata to disk
        data = {
            'embeddings': self.registered_embeddings,
            'labels': self.registered_ids,
            'names_map': self.student_names
        }
        os.makedirs(os.path.dirname(self.classifier_path), exist_ok=True)
        with open(self.classifier_path, 'wb') as f:
            pickle.dump(data, f)

    def load_classifier(self) -> bool:
        """Load stored classifier data if available."""
        if os.path.exists(self.classifier_path):
            try:
                with open(self.classifier_path, 'rb') as f:
                    data = pickle.load(f)
                self.registered_embeddings = data.get('embeddings')
                self.registered_ids = data.get('labels', [])
                self.student_names = data.get('names_map', {})
                return True
            except Exception as e:
                print(f"Error loading classifier: {e}")
        return False

    def predict_faces(
        self,
        image: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Detect and identify all faces in an image or webcam frame.
        For each face:
          - If cosine similarity >= confidence_threshold: classified as known registered student.
          - If cosine similarity < confidence_threshold: classified as 'Unknown Student'.
        Returns list of prediction dictionaries.
        """
        detected_faces = self.detect_faces(image)
        predictions = []

        if not detected_faces:
            return predictions

        for face in detected_faces:
            bbox = face['bbox']
            det_score = face['score']
            raw_face = face.get('raw_face')

            # Extract embedding
            embedding = self.extract_embedding(image, face)
            if embedding is None:
                continue

            # Check if any students are registered
            if self.registered_embeddings is None or len(self.registered_embeddings) == 0:
                predictions.append({
                    'student_id': 'UNKNOWN',
                    'name': 'Unknown Student',
                    'confidence': 0.0,
                    'is_known': False,
                    'bbox': bbox,
                    'det_score': det_score,
                    'raw_face': raw_face
                })
                continue

            # Compute Cosine Similarities with all registered embeddings
            # (since both query and registered embeddings are unit-normalized, dot product == cosine similarity)
            similarities = np.dot(self.registered_embeddings, embedding)
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            # Also verify via SFace match function if recognizer is present
            if self.recognizer is not None:
                feat1 = embedding.reshape(1, -1)
                feat2 = self.registered_embeddings[best_idx].reshape(1, -1)
                sface_score = float(self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
                # Use max of dot product and sface match
                best_similarity = max(best_similarity, sface_score)

            if best_similarity >= confidence_threshold:
                matched_id = self.registered_ids[best_idx]
                matched_name = self.student_names.get(matched_id, "Registered Student")
                predictions.append({
                    'student_id': matched_id,
                    'name': matched_name,
                    'confidence': best_similarity,
                    'is_known': True,
                    'bbox': bbox,
                    'det_score': det_score,
                    'raw_face': raw_face
                })
            else:
                predictions.append({
                    'student_id': 'UNKNOWN',
                    'name': 'Unknown Student',
                    'confidence': max(0.0, best_similarity),
                    'is_known': False,
                    'bbox': bbox,
                    'det_score': det_score,
                    'raw_face': raw_face
                })

        return predictions

    def draw_annotations(
        self,
        image: np.ndarray,
        predictions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and identifier labels around recognized and unknown faces.
        """
        annotated = image.copy()
        for p in predictions:
            x, y, w, h = p['bbox']
            is_known = p['is_known']
            name = p['name']
            sid = p['student_id']
            conf = p['confidence']

            # Format label with optional emotion classification
            emo = p.get('emotion')
            emo_str = f" | {emo['predicted_class']} ({emo['confidence']:.0%})" if emo and isinstance(emo, dict) and 'predicted_class' in emo else ""

            # Styling colors (BGR format)
            # Known: Lavender-Purple (RGB: 124, 58, 237 -> BGR: 237, 58, 124)
            # Unknown: Coral-Red (RGB: 244, 63, 94 -> BGR: 94, 63, 244)
            if is_known:
                box_color = (237, 58, 124)  # Lavender
                text_label = f"{name} ({sid}) | {conf:.1%}{emo_str}"
            else:
                box_color = (94, 63, 244)   # Coral
                text_label = f"Unknown Student | {conf:.1%}{emo_str}"

            # Draw rounded/styled bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text_label, font, font_scale, thickness)

            # Header position
            header_y1 = max(0, y - th - 10)
            header_y2 = y
            cv2.rectangle(annotated, (x, header_y1), (x + tw + 10, header_y2), box_color, -1)
            cv2.putText(
                annotated,
                text_label,
                (x + 5, header_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated
