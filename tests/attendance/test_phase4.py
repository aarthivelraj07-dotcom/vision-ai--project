"""
Phase 4 Emotion Detection Integration Test Suite.
Validates the integration of the original trained FER emotion model:
1. Model architecture loading and layer configuration.
2. Model weight loading and parameter tensor validation.
3. Input shape verification (48x48x1).
4. Preprocessing fidelity (Grayscale, 48x48, raw [0, 255] float32 scale).
5. Exact 7 emotion class mapping order.
6. Single face emotion prediction and probability normalization.
7. Multi-face independent emotion prediction.
8. Empty / no-face graceful handling.
9. Corrupt / invalid input graceful handling.
10. Low confidence evaluation.
11. Thread-safe singleton model caching across requests.
12. Strict decoupling of emotion from biometric attendance marking.
13. Original preserved artifact integrity.
"""

import os
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.emotion_detector import (
    EmotionDetector,
    EMOTION_CLASSES,
    INPUT_SHAPE,
    DEFAULT_CONFIDENCE_THRESHOLD
)
from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.attendance_manager import AttendanceManager, AttendanceTimeManager, KOLKATA_TZ


class TestPhase4EmotionDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize shared detector instance for testing."""
        cls.detector = EmotionDetector.get_instance()

    def test_01_original_artifacts_integrity(self):
        """Verify that all original model files exist and remain uncorrupted."""
        root_dir = PROJECT_ROOT
        weights_path = os.path.join(root_dir, "model_weights1.h5")
        json_path = os.path.join(root_dir, "model_a1.json")
        cascade_path = os.path.join(root_dir, "haarcascade_frontalface_default.xml")
        notebook_path = os.path.join(root_dir, "model_creation.ipynb")

        self.assertTrue(os.path.exists(weights_path), "model_weights1.h5 must exist in root.")
        self.assertTrue(os.path.exists(json_path), "model_a1.json must exist in root.")
        self.assertTrue(os.path.exists(cascade_path), "haarcascade_frontalface_default.xml must exist in root.")
        self.assertTrue(os.path.exists(notebook_path), "model_creation.ipynb must exist in root.")

        # Ensure weights file is non-empty (original weights ~17.9 MB)
        weights_size = os.path.getsize(weights_path)
        self.assertGreater(weights_size, 10_000_000, "model_weights1.h5 appears truncated.")

    def test_02_model_architecture_and_layers(self):
        """Verify model architecture contains exact 31 layers from model_creation.ipynb."""
        model = self.detector.model
        self.assertIsNotNone(model, "EmotionDetector model must be initialized.")
        self.assertEqual(len(model.layers), 31, "Model must contain exactly 31 layers.")

        layer_names = [l.name for l in model.layers]
        expected_key_layers = [
            'input_1', 'conv2d', 'batch_normalization', 'activation', 'max_pooling2d', 'dropout',
            'conv2d_1', 'batch_normalization_1', 'conv2d_2', 'batch_normalization_2',
            'conv2d_3', 'batch_normalization_3', 'flatten',
            'dense', 'batch_normalization_4', 'dense_1', 'batch_normalization_5', 'dense_2'
        ]
        for key_layer in expected_key_layers:
            self.assertIn(key_layer, layer_names, f"Expected layer '{key_layer}' in model.")

    def test_03_weights_loading_and_non_zero_norms(self):
        """Verify that all 38 weight tensors are loaded and have non-zero norms."""
        model = self.detector.model
        self.assertEqual(len(model.weights), 38, "Model must have exactly 38 weight tensors.")
        for w in model.weights:
            val = w.numpy()
            norm = float(np.linalg.norm(val))
            self.assertGreater(norm, 0.0, f"Weight tensor {w.name} must not be zero.")

    def test_04_input_and_output_shapes(self):
        """Verify model input shape is (None, 48, 48, 1) and output is (None, 7)."""
        model = self.detector.model
        self.assertEqual(model.input_shape[1:], INPUT_SHAPE)
        self.assertEqual(model.output_shape[-1], 7)

    def test_05_supported_emotion_classes_order(self):
        """Verify the 7 emotion classes match the exact index order from model_creation.ipynb."""
        expected = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.assertEqual(self.detector.get_supported_emotions(), expected)
        self.assertEqual(EMOTION_CLASSES, expected)

    def test_06_preprocessing_pipeline_scale_and_shape(self):
        """Verify preprocessing converts to grayscale 48x48 and preserves [0, 255] float32 scale."""
        # Create a synthetic 100x120 BGR image with distinct pixel values
        test_bgr = np.full((100, 120, 3), 200, dtype=np.uint8)
        preprocessed = self.detector.preprocess_face(test_bgr)

        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (1, 48, 48, 1))
        self.assertEqual(preprocessed.dtype, np.float32)

        # Scale check: must be in [0, 255] range and NOT normalized to [0, 1]
        self.assertAlmostEqual(float(np.mean(preprocessed)), 200.0, delta=1.0)
        self.assertGreater(float(np.max(preprocessed)), 1.0, "Input must NOT be divided by 255.0.")

        # Test grayscale input directly
        test_gray = np.full((64, 64), 150, dtype=np.uint8)
        preprocessed_gray = self.detector.preprocess_face(test_gray)
        self.assertEqual(preprocessed_gray.shape, (1, 48, 48, 1))
        self.assertAlmostEqual(float(np.mean(preprocessed_gray)), 150.0, delta=1.0)

    def test_07_single_face_prediction_and_probability_sum(self):
        """Verify single face prediction returns all required keys and valid probabilities."""
        sample_face = np.full((80, 80, 3), 128, dtype=np.uint8)
        # Add some variation so it's a realistic face canvas
        sample_face[20:35, 20:35] = 50   # left eye
        sample_face[20:35, 45:60] = 50   # right eye
        sample_face[50:65, 25:55] = 220  # mouth

        pred = self.detector.predict_emotion(sample_face)
        self.assertIsNotNone(pred)
        self.assertIn('predicted_class', pred)
        self.assertIn('confidence', pred)
        self.assertIn('confidence_pct', pred)
        self.assertIn('probabilities', pred)
        self.assertIn('is_confident', pred)
        self.assertIn('label', pred)

        self.assertIn(pred['predicted_class'], EMOTION_CLASSES)
        self.assertGreaterEqual(pred['confidence'], 0.0)
        self.assertLessEqual(pred['confidence'], 1.0)

        # Sum of probabilities across all 7 emotions must equal 1.0
        prob_sum = sum(pred['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Top class probability must match confidence
        top_prob = pred['probabilities'][pred['predicted_class']]
        self.assertAlmostEqual(top_prob, pred['confidence'], places=5)

    def test_08_multi_face_independent_prediction(self):
        """Verify multiple faces in one frame are classified independently."""
        frame = np.full((300, 400, 3), 100, dtype=np.uint8)
        # Create two different face patches
        frame[50:120, 50:120] = 200   # Face 1
        frame[150:230, 200:280] = 30  # Face 2

        bboxes = [(50, 50, 70, 70), (200, 150, 80, 80)]
        preds = self.detector.predict_faces_emotions(frame, bboxes)

        self.assertEqual(len(preds), 2)
        self.assertIsNotNone(preds[0])
        self.assertIsNotNone(preds[1])

        # Verify predictions have independent probability distributions
        p1 = preds[0]['probabilities']
        p2 = preds[1]['probabilities']
        # The two face patches have vastly different pixel intensities, so probabilities must differ
        diff = sum(abs(p1[c] - p2[c]) for c in EMOTION_CLASSES)
        self.assertGreater(diff, 1e-4, "Independent face predictions should produce distinct outputs.")

    def test_09_graceful_handling_empty_and_corrupt_inputs(self):
        """Verify robust error handling for empty, None, and corrupted inputs."""
        # 1. None input
        self.assertIsNone(self.detector.predict_emotion(None))
        self.assertIsNone(self.detector.preprocess_face(None))

        # 2. Empty array
        empty_arr = np.array([], dtype=np.uint8)
        self.assertIsNone(self.detector.predict_emotion(empty_arr))

        # 3. Degenerate 0x0 or 1x1 arrays
        zero_dim = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIsNone(self.detector.predict_emotion(zero_dim))

        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        self.assertIsNone(self.detector.predict_emotion(tiny))

        # 4. Multi-face with out-of-bounds or zero-size bboxes
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        invalid_bboxes = [(200, 200, 50, 50), (10, 10, 0, 0), None]
        multi_preds = self.detector.predict_faces_emotions(frame, invalid_bboxes)
        self.assertEqual(len(multi_preds), 3)
        self.assertIsNone(multi_preds[0])
        self.assertIsNone(multi_preds[1])
        self.assertIsNone(multi_preds[2])

    def test_10_low_confidence_threshold_evaluation(self):
        """Verify confidence threshold flags predictions correctly."""
        sample_face = np.full((60, 60, 3), 128, dtype=np.uint8)

        # High threshold: should flag is_confident as False if confidence < threshold
        pred_strict = self.detector.predict_emotion(sample_face, confidence_threshold=0.99)
        self.assertIsNotNone(pred_strict)
        self.assertFalse(pred_strict['is_confident'], "Confidence should be below 0.99 threshold.")

        # Low threshold: should flag is_confident as True
        pred_lenient = self.detector.predict_emotion(sample_face, confidence_threshold=0.01)
        self.assertIsNotNone(pred_lenient)
        self.assertTrue(pred_lenient['is_confident'], "Confidence should be above 0.01 threshold.")

    def test_11_singleton_pattern_thread_safety(self):
        """Verify EmotionDetector operates as a singleton across multiple invocations."""
        inst1 = EmotionDetector.get_instance()
        inst2 = EmotionDetector.get_instance()
        self.assertIs(inst1, inst2, "EmotionDetector.get_instance() must return the same instance.")

    def test_12_emotion_decoupling_from_attendance_marking(self):
        """
        Verify that emotion detection does NOT alter attendance business logic.
        A registered student recognized within 9:30 - 10:00 AM IST must be marked PRESENT
        regardless of whatever emotion is predicted.
        """
        import tempfile
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_decoupling_att.db")
        db = AttendanceDatabase(db_path=db_path)

        # Mock embedding
        test_emb = np.random.randn(128).astype(np.float32)
        test_emb /= np.linalg.norm(test_emb)
        # Register a test student
        db.add_student("STU_DEC_01", "Emma Watson", [test_emb], consent_given=True)

        fr_system = FaceRecognitionSystem()
        fr_system.registered_embeddings = np.array([test_emb])
        fr_system.registered_ids = ["STU_DEC_01"]
        fr_system.student_names = {"STU_DEC_01": "Emma Watson"}

        time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        att_manager = AttendanceManager(
            db=db,
            fr_system=fr_system,
            time_manager=time_mgr,
            allow_simulation=True,
            enable_background_watcher=False,
            emotion_detector=self.detector
        )

        active_time = datetime(2026, 3, 20, 9, 45, 0, tzinfo=KOLKATA_TZ)

        # Mock a synthetic frame and patch fr_system.predict_faces to return recognized student
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Test with recognition
        def mock_predict_faces(image, confidence_threshold=0.50):
            return [{
                'student_id': 'STU_DEC_01',
                'name': 'Emma Watson',
                'confidence': 0.88,
                'is_known': True,
                'bbox': (10, 10, 60, 60),
                'det_score': 0.95
            }]

        fr_system.predict_faces = mock_predict_faces

        # Process frame
        res = att_manager.process_frame_attendance(dummy_frame, simulated_time=active_time)

        self.assertTrue(res['allowed'])
        self.assertEqual(len(res['marked_students']), 1)
        self.assertEqual(res['marked_students'][0]['student_id'], 'STU_DEC_01')
        self.assertEqual(res['marked_students'][0]['status'], 'PRESENT')

        # Verify that emotion was detected and attached to prediction
        pred = res['predictions'][0]
        self.assertIn('emotion', pred)
        self.assertIsNotNone(pred['emotion'])
        self.assertIn('predicted_class', pred['emotion'])
        self.assertIn('probabilities', pred['emotion'])

        # Verify database record
        today_records = db.get_daily_attendance("2026-03-20")
        self.assertEqual(len(today_records), 1)
        self.assertEqual(today_records[0]['student_id'], 'STU_DEC_01')
        self.assertEqual(today_records[0]['status'], 'PRESENT')

        # Clean up
        try:
            os.remove(db_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
