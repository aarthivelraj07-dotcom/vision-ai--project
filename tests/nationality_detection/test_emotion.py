"""
Tests for Facial Emotion Recognition using the pre-trained 31-layer FER CNN.
"""

import os
import unittest
import numpy as np
import cv2

from modules.nationality_detection.constants import EMOTION_CLASSES
from modules.nationality_detection.emotion import EmotionDetector, EmotionResult

_sample_path = os.path.join("datasets", "attendance", "registered_faces", "STU001", "sample_1.jpg")


class TestEmotionModule(unittest.TestCase):
    """Verifies FER CNN model loading, input shape, preprocessing, and probability outputs."""

    @classmethod
    def setUpClass(cls):
        cls.detector = EmotionDetector.get_instance()

    def test_01_model_is_loaded(self):
        self.assertTrue(self.detector.is_loaded)
        self.assertIsNotNone(self.detector.model)

    def test_02_emotion_classes_order_and_count(self):
        self.assertEqual(len(EMOTION_CLASSES), 7)
        expected = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.assertEqual(EMOTION_CLASSES, expected)

    def test_03_inference_on_synthetic_face_crop(self):
        # Create 48x48 dummy face crop
        dummy_face = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = self.detector.predict(dummy_face)
        
        self.assertIsInstance(result, EmotionResult)
        self.assertIn(result.dominant_emotion, EMOTION_CLASSES)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Check all 7 probabilities exist and sum to ~1.0
        self.assertEqual(len(result.probabilities), 7)
        prob_sum = sum(result.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_04_inference_on_genuine_sample_image(self):
        if not os.path.exists(_sample_path):
            self.skipTest(f"Sample image {_sample_path} not found.")

        img = cv2.imread(_sample_path)
        self.assertIsNotNone(img)
        
        result = self.detector.predict(img)
        self.assertIsInstance(result, EmotionResult)
        self.assertIn(result.dominant_emotion, EMOTION_CLASSES)
        self.assertEqual(result.dominant_emotion, "Neutral")
        self.assertGreaterEqual(result.confidence, 0.50)

    def test_05_empty_input_handling(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.detector.predict(empty_img)
        self.assertEqual(result.dominant_emotion, "Neutral")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.is_confident)


if __name__ == "__main__":
    unittest.main()
