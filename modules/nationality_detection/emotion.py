"""
Facial Emotion Recognition Module for Task 4.

Integrates the original trained FER CNN:
- Architecture: 31-layer CNN (4 Conv2D blocks + 2 Dense blocks + 7-unit Softmax).
- Pre-trained weights: model_weights1.h5 (trained on FER-2013).
- Input: Grayscale face crop resized to (48, 48, 1), float32 in raw [0.0, 255.0] scale.
- Classes: ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] (exact order).
- Thread-safe singleton pattern ensures model is loaded only once.
"""

import os
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

# Ensure workspace-local KERAS_HOME to prevent sandbox permission errors on Windows
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KERAS_HOME", os.path.join(_root_dir, ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from .constants import EMOTION_CLASSES

DEFAULT_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class EmotionResult:
    """Result of facial expression emotion prediction."""
    dominant_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_confident: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_emotion": self.dominant_emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_confident": self.is_confident,
        }


class EmotionDetector:
    """
    Thread-safe facial expression classifier using the verified 31-layer CNN.
    """
    _instance: Optional["EmotionDetector"] = None
    _lock = threading.RLock()

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.weights_path = self._resolve_weights_path(weights_path)
        self.model = None
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(
        cls,
        weights_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> "EmotionDetector":
        """Thread-safe singleton accessor."""
        with cls._lock:
            if cls._instance is None or not cls._instance.is_loaded:
                cls._instance = cls(weights_path=weights_path, confidence_threshold=confidence_threshold)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (used in isolated unit tests)."""
        with cls._lock:
            cls._instance = None

    def _resolve_weights_path(self, user_path: Optional[str]) -> str:
        """Resolves model_weights1.h5 path from user input or project root."""
        if user_path is not None and os.path.exists(user_path):
            return os.path.abspath(user_path)

        # Check project root
        root_path = os.path.join(_root_dir, "model_weights1.h5")
        if os.path.exists(root_path):
            return root_path

        # Check current working directory
        if os.path.exists("model_weights1.h5"):
            return os.path.abspath("model_weights1.h5")

        return root_path

    def _build_model(self):
        """
        Constructs the exact 31-layer functional CNN architecture matching model_creation.ipynb.
        """
        from keras import layers, models

        inputs = layers.Input(shape=(48, 48, 1), name="input_1")

        # 1st Conv Block: 64 filters, 3x3, same padding
        x = layers.Conv2D(64, (3, 3), padding="same", name="conv2d")(inputs)
        x = layers.BatchNormalization(name="batch_normalization")(x)
        x = layers.Activation("relu", name="activation")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d")(x)
        x = layers.Dropout(0.25, name="dropout")(x)

        # 2nd Conv Block: 128 filters, 5x5, same padding
        x = layers.Conv2D(128, (5, 5), padding="same", name="conv2d_1")(x)
        x = layers.BatchNormalization(name="batch_normalization_1")(x)
        x = layers.Activation("relu", name="activation_1")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1")(x)
        x = layers.Dropout(0.25, name="dropout_1")(x)

        # 3rd Conv Block: 512 filters, 3x3, same padding
        x = layers.Conv2D(512, (3, 3), padding="same", name="conv2d_2")(x)
        x = layers.BatchNormalization(name="batch_normalization_2")(x)
        x = layers.Activation("relu", name="activation_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2")(x)
        x = layers.Dropout(0.25, name="dropout_2")(x)

        # 4th Conv Block: 512 filters, 3x3, same padding
        x = layers.Conv2D(512, (3, 3), padding="same", name="conv2d_3")(x)
        x = layers.BatchNormalization(name="batch_normalization_3")(x)
        x = layers.Activation("relu", name="activation_3")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3")(x)
        x = layers.Dropout(0.25, name="dropout_3")(x)

        # Flatten layer
        x = layers.Flatten(name="flatten")(x)

        # 1st Dense Block: 256 units
        x = layers.Dense(256, name="dense")(x)
        x = layers.BatchNormalization(name="batch_normalization_4")(x)
        x = layers.Activation("relu", name="activation_4")(x)
        x = layers.Dropout(0.25, name="dropout_4")(x)

        # 2nd Dense Block: 512 units
        x = layers.Dense(512, name="dense_1")(x)
        x = layers.BatchNormalization(name="batch_normalization_5")(x)
        x = layers.Activation("relu", name="activation_5")(x)
        x = layers.Dropout(0.25, name="dropout_5")(x)

        # 7-unit Softmax Output
        outputs = layers.Dense(7, activation="softmax", name="dense_2")(x)

        return models.Model(inputs=inputs, outputs=outputs, name="model")

    def _load_model(self):
        """Builds architecture and loads pre-trained weights."""
        if not os.path.exists(self.weights_path):
            self.model = None
            self.is_loaded = False
            return

        try:
            model = self._build_model()
            model.load_weights(self.weights_path)
            self.model = model
            self.is_loaded = True
        except Exception as e:
            self.model = None
            self.is_loaded = False

    def predict(self, face_bgr_or_gray: np.ndarray) -> EmotionResult:
        """
        Predicts facial emotion from face crop.
        Preprocessing: Grayscale -> Resize (48, 48) -> float32 in [0.0, 255.0] range -> (1, 48, 48, 1).
        """
        if face_bgr_or_gray is None or face_bgr_or_gray.size == 0 or not self.is_loaded:
            probs = {c: 0.0 for c in EMOTION_CLASSES}
            probs["Neutral"] = 1.0
            return EmotionResult(
                dominant_emotion="Neutral",
                confidence=0.0,
                probabilities=probs,
                is_confident=False,
            )

        # Convert to grayscale if needed
        if len(face_bgr_or_gray.shape) == 3 and face_bgr_or_gray.shape[2] == 3:
            gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        elif len(face_bgr_or_gray.shape) == 2:
            gray = face_bgr_or_gray
        else:
            gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGRA2GRAY)

        # Resize to 48x48
        roi = cv2.resize(gray, (48, 48))

        # Raw float32 pixel scale [0.0, 255.0], shape (1, 48, 48, 1)
        blob = roi.astype(np.float32)[np.newaxis, :, :, np.newaxis]

        # Predict
        raw_preds = self.model.predict(blob, verbose=0)[0]
        prob_dict = {EMOTION_CLASSES[i]: float(raw_preds[i]) for i in range(len(EMOTION_CLASSES))}

        top_idx = int(np.argmax(raw_preds))
        top_emotion = EMOTION_CLASSES[top_idx]
        top_conf = float(raw_preds[top_idx])

        return EmotionResult(
            dominant_emotion=top_emotion,
            confidence=top_conf,
            probabilities=prob_dict,
            is_confident=(top_conf >= self.confidence_threshold),
        )
