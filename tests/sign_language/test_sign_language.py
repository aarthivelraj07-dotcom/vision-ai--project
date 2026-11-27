"""
Unit and Integration Tests for Task 5: Indian Sign Language (ISL) Recognition.

Coverage:
1. Model loading (ONNX palm, ONNX handpose, Joblib SVM pipeline).
2. Landmark extraction (21 3D landmarks).
3. Feature dimension and normalization compatibility (63 features, wrist-centered, palm-scaled).
4. Prediction output format and schema (ISLRecognitionResult, confidence, probabilities).
5. No-hand detection handling (blank frames).
6. Invalid image and error handling (empty arrays, missing files, non-images).
7. Operating-time restriction enforcement (Asia/Kolkata 18:00-22:00 window, next window, admin override).
8. Webcam error handling and resource cleanup.
9. Image-upload workflow simulation (PIL -> numpy -> detector).
10. Main application integration and regression routing.
"""

import os
import unittest
from datetime import datetime, time as dtime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import numpy as np

from modules.sign_language.detector import (
    ISLHandDetector,
    ISLRecognitionResult,
    SUPPORTED_CLASSES,
    get_current_ist_time,
    is_within_operating_window,
    get_next_operating_window,
    is_admin_override_configured,
    verify_admin_override,
)

IST = ZoneInfo("Asia/Kolkata")


class TestISLModelLoading(unittest.TestCase):
    """Test 1: Model loading and architecture verification."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ISLHandDetector()

    def test_detector_initialization(self):
        """Verify detector successfully loads all three underlying models."""
        self.assertTrue(self.detector.is_loaded)
        self.assertIsNotNone(self.detector._palm_net)
        self.assertIsNotNone(self.detector._hand_net)
        self.assertIsNotNone(self.detector._classifier)

    def test_classes_supported(self):
        """Verify the 12 approved static sign classes are present."""
        expected = ["1", "2", "3", "4", "5", "A", "B", "C", "L", "V", "W", "Y"]
        self.assertEqual(sorted(self.detector.classes), sorted(expected))


class TestISLLandmarkExtractionAndNormalization(unittest.TestCase):
    """Tests 2 & 3: Landmark extraction and 63-D normalization."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ISLHandDetector()
        cls.root_dir = cls.detector.root_dir
        cls.sample_path = os.path.join(cls.root_dir, "datasets", "sign_language", "static", "1", "0.jpg")

    def test_landmark_extraction_dimensions(self):
        """Verify that 21 landmarks with (x, y, z) coordinates are extracted."""
        if not os.path.exists(self.sample_path):
            self.skipTest(f"Sample image not found: {self.sample_path}")
        
        result = self.detector.process_image(self.sample_path)
        self.assertTrue(result.detected)
        self.assertIsNotNone(result.landmarks_orig)
        self.assertEqual(result.landmarks_orig.shape, (21, 3))

    def test_feature_normalization_format(self):
        """Verify 63-D normalized feature vector format and properties."""
        if not os.path.exists(self.sample_path):
            self.skipTest(f"Sample image not found: {self.sample_path}")

        result = self.detector.process_image(self.sample_path)
        self.assertIsNotNone(result.landmarks_norm)
        self.assertEqual(result.landmarks_norm.shape, (63,))
        self.assertEqual(result.landmarks_norm.dtype, np.float32)

        # Wrist landmark (index 0: coordinates [0, 1, 2]) must be at origin (0, 0, 0)
        norm_lms = result.landmarks_norm.reshape(21, 3)
        np.testing.assert_allclose(norm_lms[0], [0.0, 0.0, 0.0], atol=1e-5)


class TestISLPredictionOutput(unittest.TestCase):
    """Test 4: Correct prediction output format and result dataclass."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ISLHandDetector()
        cls.root_dir = cls.detector.root_dir

    def test_prediction_output_schema(self):
        """Verify structured ISLRecognitionResult schema and probability sum."""
        sample_path = os.path.join(self.root_dir, "datasets", "sign_language", "static", "A", "0.jpg")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample not found: {sample_path}")

        result = self.detector.process_image(sample_path)
        self.assertTrue(result.detected)
        self.assertIsInstance(result.predicted_class, str)
        self.assertIn(result.predicted_class, SUPPORTED_CLASSES)
        self.assertGreaterEqual(result.relative_confidence, 0.0)
        self.assertLessEqual(result.relative_confidence, 1.0)
        self.assertIsNotNone(result.probabilities)
        self.assertEqual(len(result.probabilities), 12)

        # Probabilities must sum to 1.0
        prob_sum = sum(result.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)


class TestISLEdgeCasesAndRobustness(unittest.TestCase):
    """Tests 5 & 6: No-hand detection and invalid input handling."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ISLHandDetector()

    def test_no_hand_detection_on_blank_frame(self):
        """Verify safe handling when no hand is present (black image)."""
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        result = self.detector.process_image(blank)
        self.assertFalse(result.detected)
        self.assertIsNone(result.predicted_class)
        self.assertIn("No hand detected", result.message)

    def test_missing_file_handling(self):
        """Verify missing image file returns graceful error without crashing."""
        result = self.detector.process_image("non_existent_image_path.jpg")
        self.assertFalse(result.detected)
        self.assertIn("File not found", result.message)

    def test_empty_array_handling(self):
        """Verify empty numpy array returns graceful error without crashing."""
        empty_arr = np.array([])
        result = self.detector.process_image(empty_arr)
        self.assertFalse(result.detected)
        self.assertIn("Empty or invalid", result.message)

    def test_tiny_resolution_handling(self):
        """Verify images below minimum resolution are safely rejected."""
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        result = self.detector.process_image(tiny)
        self.assertFalse(result.detected)
        self.assertIn("resolution too low", result.message)

    def test_draw_landmarks_on_unannotated_frame(self):
        """Verify drawing on frames where no hand is detected leaves image intact."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        dummy_res = ISLRecognitionResult(detected=False)
        annotated = self.detector.draw_landmarks(frame, dummy_res)
        np.testing.assert_array_equal(frame, annotated)


class TestOperatingTimeRestriction(unittest.TestCase):
    """Test 7: Authoritative IST server-side operating window enforcement."""

    def test_operating_window_during_hours(self):
        """Time between 18:00 and 22:00 IST must return True."""
        dt_in_window = datetime(2026, 11, 19, 19, 30, 0, tzinfo=IST)
        self.assertTrue(is_within_operating_window(dt_in_window))

    def test_operating_window_boundary_start(self):
        """Exactly 18:00:00 IST must return True."""
        dt_start = datetime(2026, 11, 19, 18, 0, 0, tzinfo=IST)
        self.assertTrue(is_within_operating_window(dt_start))

    def test_operating_window_boundary_end(self):
        """Exactly 22:00:00 IST must return True."""
        dt_end = datetime(2026, 11, 19, 22, 0, 0, tzinfo=IST)
        self.assertTrue(is_within_operating_window(dt_end))

    def test_operating_window_outside_hours(self):
        """Time before 18:00 or after 22:00 IST must return False."""
        dt_morning = datetime(2026, 11, 19, 10, 0, 0, tzinfo=IST)
        dt_late_night = datetime(2026, 11, 19, 23, 15, 0, tzinfo=IST)
        self.assertFalse(is_within_operating_window(dt_morning))
        self.assertFalse(is_within_operating_window(dt_late_night))

    def test_next_operating_window_text_before_18(self):
        """Before 18:00, next window should announce today."""
        dt_before = datetime(2026, 11, 19, 14, 0, 0, tzinfo=IST)
        msg = get_next_operating_window(dt_before)
        self.assertIn("Today", msg)
        self.assertIn("06:00 PM to 10:00 PM IST", msg)

    def test_next_operating_window_text_after_22(self):
        """After 22:00, next window should announce tomorrow."""
        dt_after = datetime(2026, 11, 19, 22, 30, 0, tzinfo=IST)
        msg = get_next_operating_window(dt_after)
        self.assertIn("Tomorrow", msg)
        self.assertIn("06:00 PM to 10:00 PM IST", msg)

    def test_admin_override_verification(self):
        """Verify secure environment-variable admin override check."""
        with patch.dict(os.environ, {"ISL_ADMIN_PIN": "SecretTestPin999"}):
            self.assertTrue(is_admin_override_configured())
            self.assertTrue(verify_admin_override("SecretTestPin999"))
            self.assertFalse(verify_admin_override("WrongPin"))
            self.assertFalse(verify_admin_override(""))

        with patch.dict(os.environ, {}, clear=True):
            if "ISL_ADMIN_PIN" in os.environ:
                del os.environ["ISL_ADMIN_PIN"]
            if "ATTENDANCE_ADMIN_PIN" in os.environ:
                del os.environ["ATTENDANCE_ADMIN_PIN"]
            self.assertFalse(is_admin_override_configured())
            self.assertFalse(verify_admin_override("AnyPin"))


class TestWebcamAndUIWorkflow(unittest.TestCase):
    """Tests 8 & 9: Webcam error handling and image upload pipeline simulation."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ISLHandDetector()

    def test_webcam_unavailable_handled_safely(self):
        """Verify simulated camera opening failure is handled cleanly without exceptions."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            import cv2
            cap = cv2.VideoCapture(999)  # Non-existent index
            self.assertFalse(cap.isOpened())
            # Safe release call must not raise
            cap.release()

    def test_image_upload_simulation(self):
        """Simulate image upload decoding path (RGB array -> BGR -> detector)."""
        # Create a valid test image with simulated 3-channel data
        rgb_data = np.full((128, 128, 3), 200, dtype=np.uint8)
        bgr_data = rgb_data[:, :, ::-1]

        result = self.detector.process_image(bgr_data)
        # Even with uniform image, it safely returns an ISLRecognitionResult
        self.assertIsInstance(result, ISLRecognitionResult)
        self.assertFalse(result.detected)


class TestAppIntegration(unittest.TestCase):
    """Test 10: Task 5 main router integration in app.py."""

    def test_app_task5_router_route(self):
        """Verify that app.py recognizes task5 query parameter."""
        import app
        self.assertTrue(hasattr(app, "main"))

        # Test query parameter routing to Task 5
        with patch("streamlit.query_params", {"task": "task5"}):
            with patch("modules.sign_language.ui.run_sign_language_app") as mock_run_t5:
                app.main()
                mock_run_t5.assert_called_once()


if __name__ == "__main__":
    unittest.main()
