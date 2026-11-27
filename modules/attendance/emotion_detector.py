"""
Emotion Detection Module for Task 1: Attendance & Emotion Detection.
Integrates the original trained FER model:
- Architecture: 4 Conv2D blocks (64, 128, 512, 512) with BatchNormalization, ReLU,
  MaxPooling2D, and Dropout(0.25), followed by Flatten, 2 Dense layers (256, 512) with
  BatchNormalization, ReLU, and Dropout(0.25), and a 7-unit Softmax Dense output.
- Input Shape: (None, 48, 48, 1) - 48x48 Grayscale, raw [0.0, 255.0] float32 scale.
- Classes: ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] (exact order).
- Thread-safe singleton model loader to prevent re-instantiation across frames and Streamlit reruns.
- Independent per-face prediction without altering attendance or biometric identification logic.
"""

import os
import json
import threading
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

# Ensure workspace-local KERAS_HOME to prevent sandbox permission errors on Windows
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KERAS_HOME", os.path.join(_root_dir, ".keras"))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
INPUT_SHAPE = (48, 48, 1)
DEFAULT_CONFIDENCE_THRESHOLD = 0.30


class EmotionDetector:
    """
    Emotion detection engine using the original pre-trained FER CNN model.
    Loads architecture and weights from model_a1.json and model_weights1.h5.
    Thread-safe singleton pattern ensures model is loaded into memory only once.
    """
    _instance: Optional["EmotionDetector"] = None
    _lock = threading.RLock()

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.weights_path = self._resolve_path(weights_path, "model_weights1.h5")
        self.config_path = self._resolve_path(config_path, "model_a1.json")
        self.model = None
        self.is_loaded = False

        self._load_model()

    @classmethod
    def get_instance(
        cls,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> "EmotionDetector":
        """
        Thread-safe singleton accessor. Ensures only one copy of the model resides in memory.
        """
        with cls._lock:
            if cls._instance is None or not cls._instance.is_loaded:
                cls._instance = cls(
                    weights_path=weights_path,
                    config_path=config_path,
                    confidence_threshold=confidence_threshold
                )
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (primarily for isolated test environments)."""
        with cls._lock:
            cls._instance = None

    def _resolve_path(self, user_path: Optional[str], default_filename: str) -> str:
        """Resolve model artifact paths across project root and subdirectories."""
        if user_path is not None and os.path.exists(user_path):
            return os.path.abspath(user_path)

        # 1. Check current working directory
        if os.path.exists(default_filename):
            return os.path.abspath(default_filename)

        # 2. Check project root (three levels up from modules/attendance/emotion_detector.py)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_path = os.path.join(root_dir, default_filename)
        if os.path.exists(root_path):
            return root_path

        # 3. Check models/attendance subdirectory
        models_path = os.path.join(root_dir, "models", "attendance", default_filename)
        if os.path.exists(models_path):
            return models_path

        return os.path.abspath(default_filename)

    def _build_model_architecture(self):
        """
        Construct the exact 31-layer CNN architecture matching model_creation.ipynb and model_a1.json:
        4 Conv blocks + Flatten + 2 Dense blocks + 7-unit Softmax Dense output.
        """
        from keras import layers, models

        inputs = layers.Input(shape=INPUT_SHAPE, name='input_1')

        # 1st Conv Block: 64 filters, 3x3, same padding
        x = layers.Conv2D(64, (3, 3), padding='same', name='conv2d')(inputs)
        x = layers.BatchNormalization(name='batch_normalization')(x)
        x = layers.Activation('relu', name='activation')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d')(x)
        x = layers.Dropout(0.25, name='dropout')(x)

        # 2nd Conv Block: 128 filters, 5x5, same padding
        x = layers.Conv2D(128, (5, 5), padding='same', name='conv2d_1')(x)
        x = layers.BatchNormalization(name='batch_normalization_1')(x)
        x = layers.Activation('relu', name='activation_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(x)
        x = layers.Dropout(0.25, name='dropout_1')(x)

        # 3rd Conv Block: 512 filters, 3x3, same padding
        x = layers.Conv2D(512, (3, 3), padding='same', name='conv2d_2')(x)
        x = layers.BatchNormalization(name='batch_normalization_2')(x)
        x = layers.Activation('relu', name='activation_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(x)
        x = layers.Dropout(0.25, name='dropout_2')(x)

        # 4th Conv Block: 512 filters, 3x3, same padding
        x = layers.Conv2D(512, (3, 3), padding='same', name='conv2d_3')(x)
        x = layers.BatchNormalization(name='batch_normalization_3')(x)
        x = layers.Activation('relu', name='activation_3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(x)
        x = layers.Dropout(0.25, name='dropout_3')(x)

        # Flatten layer
        x = layers.Flatten(name='flatten')(x)

        # 1st Fully Connected Block: 256 units
        x = layers.Dense(256, name='dense')(x)
        x = layers.BatchNormalization(name='batch_normalization_4')(x)
        x = layers.Activation('relu', name='activation_4')(x)
        x = layers.Dropout(0.25, name='dropout_4')(x)

        # 2nd Fully Connected Block: 512 units
        x = layers.Dense(512, name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_normalization_5')(x)
        x = layers.Activation('relu', name='activation_5')(x)
        x = layers.Dropout(0.25, name='dropout_5')(x)

        # 7-Class Emotion Softmax Classification Layer
        outputs = layers.Dense(len(EMOTION_CLASSES), activation='softmax', name='dense_2')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='emotion_model')
        return model

    def _load_model(self):
        """
        Load model architecture and weights.
        Attempts model_from_json first; falls back to programmatic architecture construction.
        """
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Original emotion model weights not found at: {self.weights_path}. "
                "Ensure model_weights1.h5 is preserved in the project directory."
            )

        import keras

        # Attempt 1: Deserialization from JSON
        model = None
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    json_str = f.read()
                model = keras.models.model_from_json(json_str)
            except Exception:
                # Keras 3 may reject Keras 2 'Functional' config; proceed to programmatic build
                model = None

        # Attempt 2: Programmatic construction matching model_creation.ipynb exactly
        if model is None:
            model = self._build_model_architecture()

        # Load weights from model_weights1.h5
        model.load_weights(self.weights_path)

        # Validate input and output shapes
        actual_input_shape = model.input_shape
        actual_output_shape = model.output_shape

        if actual_input_shape[1:] != INPUT_SHAPE:
            raise ValueError(
                f"Model input shape mismatch. Expected (None, 48, 48, 1), got {actual_input_shape}"
            )

        if actual_output_shape[-1] != len(EMOTION_CLASSES):
            raise ValueError(
                f"Model output classes mismatch. Expected 7 classes, got {actual_output_shape[-1]}"
            )

        self.model = model
        self.is_loaded = True

    def preprocess_face(self, face_image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Preprocess face image to match model_creation.ipynb training specifications:
        1. Validates face crop validity (non-empty, valid dimensions).
        2. Converts to Grayscale if input is 3-channel (BGR/RGB) or 4-channel (BGRA).
        3. Resizes to (48, 48) pixels.
        4. Casts to float32.
        5. Preserves raw pixel value range [0.0, 255.0] (no rescale=1./255, matching original training).
        6. Expands dimensions to (1, 48, 48, 1).

        Returns:
            np.ndarray of shape (1, 48, 48, 1) with float32 values in [0.0, 255.0], or None if invalid.
        """
        if face_image is None or not isinstance(face_image, np.ndarray):
            return None

        if face_image.size == 0 or face_image.shape[0] < 2 or face_image.shape[1] < 2:
            return None

        try:
            # Grayscale conversion
            if len(face_image.shape) == 3:
                if face_image.shape[2] == 4:
                    gray = cv2.cvtColor(face_image, cv2.COLOR_BGRA2GRAY)
                elif face_image.shape[2] == 3:
                    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                elif face_image.shape[2] == 1:
                    gray = face_image.squeeze(axis=2)
                else:
                    return None
            elif len(face_image.shape) == 2:
                gray = face_image
            else:
                return None

            # Resize to target (48, 48)
            resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Cast to float32 with raw [0.0, 255.0] range (DO NOT divide by 255.0)
            tensor = resized.astype(np.float32)

            # Reshape to (1, 48, 48, 1)
            tensor = np.expand_dims(tensor, axis=(0, -1))

            return tensor
        except Exception:
            return None

    def predict_emotion(
        self,
        face_image: Optional[np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Predict emotion for a single cropped face image.

        Args:
            face_image: BGR, RGB or Grayscale numpy array of face region.
            confidence_threshold: Threshold below which prediction is marked not confident.
                                  Defaults to self.confidence_threshold.

        Returns:
            Dictionary containing:
            - 'predicted_class': Str emotion name (e.g., 'Happy', 'Neutral')
            - 'confidence': Float confidence score in [0.0, 1.0]
            - 'confidence_pct': Float percentage (e.g., 88.45)
            - 'probabilities': Dict mapping all 7 emotion classes to their probability scores
            - 'is_confident': Bool indicating whether confidence >= confidence_threshold
            - 'label': Human-readable formatted string e.g. "Happy (88.5%)"
            Or None if face_image is invalid or unprocessable.
        """
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold

        preprocessed = self.preprocess_face(face_image)
        if preprocessed is None or self.model is None:
            return None

        try:
            # Run model forward pass
            raw_predictions = self.model.predict(preprocessed, verbose=0)
            probabilities_arr = raw_predictions[0]

            top_idx = int(np.argmax(probabilities_arr))
            predicted_class = EMOTION_CLASSES[top_idx]
            confidence = float(probabilities_arr[top_idx])

            probabilities = {
                cls_name: float(probabilities_arr[i])
                for i, cls_name in enumerate(EMOTION_CLASSES)
            }

            is_confident = confidence >= threshold

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'confidence_pct': round(confidence * 100, 2),
                'probabilities': probabilities,
                'is_confident': is_confident,
                'label': f"{predicted_class} ({confidence:.1%})"
            }
        except Exception:
            return None

    def predict_faces_emotions(
        self,
        frame: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]],
        confidence_threshold: Optional[float] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Predict emotions for multiple face bounding boxes within a camera frame independently.

        Args:
            frame: Complete camera frame (H, W, C).
            face_bboxes: List of (x, y, w, h) bounding boxes.
            confidence_threshold: Optional confidence threshold.

        Returns:
            List of emotion prediction dictionaries (one per bounding box), or None for invalid crops.
        """
        results = []
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return [None] * len(face_bboxes)

        frame_h, frame_w = frame.shape[:2]

        for bbox in face_bboxes:
            if not bbox or len(bbox) < 4:
                results.append(None)
                continue

            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(frame_w, int(x + w))
            y2 = min(frame_h, int(y + h))

            if x2 <= x1 or y2 <= y1:
                results.append(None)
                continue

            crop = frame[y1:y2, x1:x2]
            pred = self.predict_emotion(crop, confidence_threshold=confidence_threshold)
            results.append(pred)

        return results

    def get_supported_emotions(self) -> List[str]:
        """Return the list of 7 supported emotion classes in order."""
        return list(EMOTION_CLASSES)
