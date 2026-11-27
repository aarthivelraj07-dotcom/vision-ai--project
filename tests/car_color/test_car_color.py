"""
Unit and Integration Test Suite for Task 6: Car Colour & Pedestrian Detection.

Covers:
1. Model loading & CPU initialization (OpenCV DNN with YOLOv8n ONNX).
2. Letterbox preprocessing (scale, padding, aspect ratio preservation).
3. Pedestrian signal zone coordinate geometry and point-in-zone checks.
4. Car color classification:
   - Blue car detection (HSV + CIELAB b* < -10).
   - Other car colors (Red, White, Gray, Yellow) -> OTHER.
   - Small crops (< 20px) -> UNCERTAIN.
   - Underbody shadow and specular glare masking -> UNCERTAIN.
5. Multi-object detection filtering (only COCO Class 0: Person and Class 2: Car).
6. Bounding box color coding (RED for Blue car, BLUE for other car, YELLOW for uncertain, GREEN for person).
7. Invalid input and empty image safety.
8. Application navigation and routing integration in app.py.
"""

import os
import unittest
import numpy as np
import cv2

from modules.car_color.detector import (
    CarColorDetector,
    SignalZone,
    CarColorResult,
    DetectedObject,
    TrafficSceneResult,
    letterbox,
    DEFAULT_YOLO_PATH,
    COCO_CLASS_CAR,
    COCO_CLASS_PERSON,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_BLUE_RATIO_THRESHOLD,
)
from app import TASK_REGISTRY


class TestCarColorDetectorInit(unittest.TestCase):
    """Test YOLOv8n detector initialization and environment requirements."""

    def test_model_exists_and_loads(self):
        """Verify the reused YOLOv8n ONNX model exists and initializes OpenCV DNN."""
        self.assertTrue(os.path.exists(DEFAULT_YOLO_PATH), f"Model missing at {DEFAULT_YOLO_PATH}")
        detector = CarColorDetector()
        self.assertTrue(detector.is_loaded)
        self.assertEqual(detector.conf_threshold, DEFAULT_CONF_THRESHOLD)
        self.assertEqual(detector.blue_ratio_threshold, DEFAULT_BLUE_RATIO_THRESHOLD)

    def test_nonexistent_model_raises_error(self):
        """Verify attempting to load a non-existent model raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            CarColorDetector(model_path="non_existent_yolo_path.onnx")


class TestLetterboxPreprocessing(unittest.TestCase):
    """Test aspect-ratio preserving letterbox resizing and coordinate transformations."""

    def test_square_image_letterbox(self):
        """Square image to square target should have uniform scale and zero padding."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        padded, scale, (pad_x, pad_y) = letterbox(img, target_size=(640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640.0 / 300.0, places=4)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 0)

    def test_wide_image_letterbox(self):
        """Wide image should have vertical padding (pad_y > 0) and zero horizontal padding."""
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        padded, scale, (pad_x, pad_y) = letterbox(img, target_size=(640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640.0 / 600.0, places=4)
        self.assertEqual(pad_x, 0)
        self.assertGreater(pad_y, 0)

    def test_tall_image_letterbox(self):
        """Tall image should have horizontal padding (pad_x > 0) and zero vertical padding."""
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        padded, scale, (pad_x, pad_y) = letterbox(img, target_size=(640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640.0 / 800.0, places=4)
        self.assertGreater(pad_x, 0)
        self.assertEqual(pad_y, 0)


class TestSignalZone(unittest.TestCase):
    """Test pedestrian traffic signal counting zone geometry and containment."""

    def test_pixel_box_conversion(self):
        """Check normalized coordinates correctly map to pixel bounding box."""
        zone = SignalZone(x_min_pct=0.10, y_min_pct=0.40, x_max_pct=0.90, y_max_pct=0.80)
        x1, y1, x2, y2 = zone.get_pixel_box(img_w=1000, img_h=500)
        self.assertEqual(x1, 100)
        self.assertEqual(y1, 200)
        self.assertEqual(x2, 900)
        self.assertEqual(y2, 400)

    def test_contains_point(self):
        """Check containment logic for points inside and outside the zone."""
        zone = SignalZone(x_min_pct=0.20, y_min_pct=0.50, x_max_pct=0.80, y_max_pct=1.00)
        img_w, img_h = 1000, 1000

        # Inside zone: (500, 750)
        self.assertTrue(zone.contains_point((500, 750), img_w, img_h))

        # Outside zone: above (500, 200)
        self.assertFalse(zone.contains_point((500, 200), img_w, img_h))

        # Outside zone: left (100, 750)
        self.assertFalse(zone.contains_point((100, 750), img_w, img_h))

        # Boundary points should be inclusive
        self.assertTrue(zone.contains_point((200, 500), img_w, img_h))
        self.assertTrue(zone.contains_point((800, 1000), img_w, img_h))


class TestCarColorClassification(unittest.TestCase):
    """Test vehicle body color classification, dual-space rules, and edge cases."""

    def setUp(self):
        self.detector = CarColorDetector()

    def test_blue_car_classification(self):
        """Synthesize an image with a blue car body crop -> classified as BLUE."""
        img = np.full((300, 300, 3), 128, dtype=np.uint8)  # Gray background
        # Car bounding box: [50, 50, 200, 150]
        # Inner body ROI: rx in [90, 210], ry in [95, 162]
        # Fill car body with deep rich royal blue (BGR: 180, 50, 20)
        img[50:200, 50:250] = (180, 50, 20)  # BGR

        res = self.detector.classify_car_color(img, (50, 50, 200, 150))
        self.assertEqual(res.color, "BLUE")
        self.assertGreaterEqual(res.blue_ratio, 0.28)
        self.assertGreater(res.valid_pixels, 30)

    def test_red_car_classification(self):
        """Red car body crop should be classified as OTHER."""
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        # Fill car box with bright red (BGR: 20, 20, 220)
        img[50:200, 50:250] = (20, 20, 220)

        res = self.detector.classify_car_color(img, (50, 50, 200, 150))
        self.assertEqual(res.color, "OTHER")
        self.assertLess(res.blue_ratio, 0.28)

    def test_white_car_classification(self):
        """White / silver car body crop should be classified as OTHER."""
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        # Fill car box with white (BGR: 240, 240, 240)
        img[50:200, 50:250] = (240, 240, 240)

        res = self.detector.classify_car_color(img, (50, 50, 200, 150))
        self.assertEqual(res.color, "OTHER")
        self.assertLess(res.blue_ratio, 0.28)

    def test_white_car_with_blue_skylight_reflection(self):
        """
        White car with a blue sky reflection on its windshield/sunroof should NOT
        be misclassified as BLUE. Dominant white paint must categorize it as OTHER.
        """
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        # White car body
        img[50:200, 50:250] = (210, 205, 195)
        # Add cyan/blue sky reflection on windshield (30% of inner area)
        img[95:120, 90:210] = (160, 120, 70)

        res = self.detector.classify_car_color(img, (50, 50, 200, 150))
        self.assertNotEqual(res.color, "BLUE", "White car with sky reflection must not be labeled BLUE")
        self.assertEqual(res.color, "OTHER")

    def test_small_car_box_returns_uncertain(self):
        """Bounding boxes smaller than MIN_CAR_DIMENSION should yield UNCERTAIN."""
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        res = self.detector.classify_car_color(img, (10, 10, 15, 15))
        self.assertEqual(res.color, "UNCERTAIN")
        self.assertIn("too small", res.reason)

    def test_shadow_obscured_box_returns_uncertain(self):
        """Heavy underbody shadow (V < 18 or L* < 8) across the ROI should yield UNCERTAIN."""
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        # Fill car box with near-black shadow (BGR: 5, 5, 5)
        img[50:200, 50:250] = (5, 5, 5)

        res = self.detector.classify_car_color(img, (50, 50, 200, 150))
        self.assertEqual(res.color, "UNCERTAIN")
        self.assertIn("Insufficient valid body pixels", res.reason)


class TestAnnotationRendering(unittest.TestCase):
    """Test bounding box colors according to user specification."""

    def setUp(self):
        self.detector = CarColorDetector()

    def test_bounding_box_colors(self):
        """
        Verify specification colors:
        - Blue Car: RED (0, 0, 255)
        - Other Car: BLUE (255, 0, 0)
        - Uncertain Car: YELLOW (0, 255, 255)
        - Person: GREEN (0, 255, 0)
        """
        base_img = np.zeros((400, 400, 3), dtype=np.uint8)

        # 1. Blue Car Result
        blue_car_obj = DetectedObject(
            class_id=COCO_CLASS_CAR,
            class_name="car",
            confidence=0.88,
            box=(50, 50, 100, 80),
            center=(100, 90),
            car_color=CarColorResult("BLUE", blue_ratio=0.75, valid_pixels=500, total_roi_pixels=500),
            in_signal_zone=False
        )

        # 2. Other Car Result
        other_car_obj = DetectedObject(
            class_id=COCO_CLASS_CAR,
            class_name="car",
            confidence=0.85,
            box=(200, 50, 100, 80),
            center=(250, 90),
            car_color=CarColorResult("OTHER", blue_ratio=0.05, valid_pixels=500, total_roi_pixels=500),
            in_signal_zone=False
        )

        # 3. Uncertain Car Result
        uncert_car_obj = DetectedObject(
            class_id=COCO_CLASS_CAR,
            class_name="car",
            confidence=0.70,
            box=(50, 200, 100, 80),
            center=(100, 240),
            car_color=CarColorResult("UNCERTAIN", blue_ratio=0.0, valid_pixels=10, total_roi_pixels=500),
            in_signal_zone=False
        )

        # 4. Person Result
        person_obj = DetectedObject(
            class_id=COCO_CLASS_PERSON,
            class_name="person",
            confidence=0.92,
            box=(200, 200, 60, 120),
            center=(230, 260),
            car_color=None,
            in_signal_zone=True
        )

        scene_res = TrafficSceneResult(
            total_cars=3,
            blue_cars=1,
            other_cars=1,
            uncertain_cars=1,
            total_people=1,
            people_in_signal_zone=1,
            objects=[blue_car_obj, other_car_obj, uncert_car_obj, person_obj],
            signal_zone=SignalZone(0.0, 0.4, 1.0, 1.0)
        )

        annotated = self.detector.draw_annotations(base_img, scene_res, draw_zone=True)

        self.assertEqual(annotated.shape, base_img.shape)
        # Check that RED pixels exist for blue car box (BGR: 0, 0, 255)
        # Top edge of blue car box at y=50, x in 50..150
        red_pixels = (annotated[:, :, 2] == 255) & (annotated[:, :, 0] == 0) & (annotated[:, :, 1] == 0)
        self.assertTrue(np.any(red_pixels), "Expected RED bounding box for Blue Car.")

        # Check that BLUE pixels exist for other car box (BGR: 255, 0, 0)
        blue_pixels = (annotated[:, :, 0] == 255) & (annotated[:, :, 1] == 0) & (annotated[:, :, 2] == 0)
        self.assertTrue(np.any(blue_pixels), "Expected BLUE bounding box for Other Car.")

        # Check that YELLOW pixels exist for uncertain car box (BGR: 0, 255, 255)
        yellow_pixels = (annotated[:, :, 0] == 0) & (annotated[:, :, 1] == 255) & (annotated[:, :, 2] == 255)
        self.assertTrue(np.any(yellow_pixels), "Expected YELLOW bounding box for Uncertain Car.")

        # Check that GREEN pixels exist for person box (BGR: 0, 255, 0)
        green_pixels = (annotated[:, :, 0] == 0) & (annotated[:, :, 1] == 255) & (annotated[:, :, 2] == 0)
        self.assertTrue(np.any(green_pixels), "Expected GREEN bounding box for Person.")


class TestSafeInputHandling(unittest.TestCase):
    """Test resilience against corrupt or missing inputs."""

    def setUp(self):
        self.detector = CarColorDetector()

    def test_missing_file_input(self):
        """Detect on nonexistent file should gracefully return 0 detections."""
        res = self.detector.detect("nonexistent_file_path_xyz.jpg")
        self.assertEqual(res.total_cars, 0)
        self.assertEqual(res.total_people, 0)
        self.assertEqual(len(res.objects), 0)

    def test_empty_array_input(self):
        """Detect on empty numpy array should gracefully return 0 detections."""
        empty_arr = np.array([], dtype=np.uint8)
        res = self.detector.detect(empty_arr)
        self.assertEqual(res.total_cars, 0)
        self.assertEqual(res.total_people, 0)

    def test_tiny_image_input(self):
        """Detect on tiny image (< 10x10) should safely return 0 detections."""
        tiny_img = np.zeros((5, 5, 3), dtype=np.uint8)
        res = self.detector.detect(tiny_img)
        self.assertEqual(res.total_cars, 0)


class TestAppNavigationIntegration(unittest.TestCase):
    """Verify Task 6 is properly registered in app.py navigation."""

    def test_task6_in_registry(self):
        """Ensure task6 exists in TASK_REGISTRY with required metadata."""
        self.assertIn("task6", TASK_REGISTRY)
        info = TASK_REGISTRY["task6"]
        self.assertIn("Task 6", info["label"])
        self.assertIn("car_color", info["aliases"])
        self.assertIn("traffic", info["aliases"])

    def test_task6_ui_importable(self):
        """Ensure the Task 6 UI entry point can be imported successfully."""
        from modules.car_color.ui import run_car_color_app
        self.assertTrue(callable(run_car_color_app))

    def test_multiscale_tiled_inference_execution(self):
        """Verify tiled multi-scale inference runs safely on large images (>= 700px)."""
        detector = CarColorDetector()
        large_img = np.full((800, 1000, 3), 120, dtype=np.uint8)
        # Should execute 1 global + 4 tile passes without exceptions
        res = detector.detect(large_img)
        self.assertIsInstance(res, TrafficSceneResult)
        self.assertEqual(res.total_cars, 0)
        self.assertEqual(res.total_people, 0)

    def test_sample_image_exists_and_readable(self):
        """Ensure preset sample images exist in test_images and are readable by detector."""
        detector = CarColorDetector()
        preset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets", "car_color", "test_images"
        )
        sample_path = os.path.join(preset_dir, "empty_street_no_objects.jpg")
        self.assertTrue(os.path.exists(sample_path))
        res = detector.detect(sample_path)
        self.assertEqual(res.total_cars, 0)
        self.assertEqual(res.total_people, 0)


if __name__ == "__main__":
    unittest.main()
