"""
Tests for Dress / Clothing Colour Detection and Uncertainty Fallback.
"""

import unittest
import numpy as np
import cv2

from modules.nationality_detection.dress_color import (
    DressColorAnalyzer,
    DressColorResult,
    MIN_TORSO_PIXELS,
    MIN_DOMINANCE_THRESHOLD,
)


class TestDressColorModule(unittest.TestCase):
    """Verifies torso localization, CIELAB color matching, and 'Unable to determine' fallback."""

    def setUp(self):
        # Create a synthetic canvas: 600x600 with a face at (200, 100, 100, 100)
        # Upper torso region: x in [175, 325], y in [205, 380]
        self.face_box = (200, 100, 100, 100)
        self.canvas_shape = (600, 600, 3)

    def test_01_torso_box_calculation(self):
        box = DressColorAnalyzer.get_torso_roi_box(self.face_box, (600, 600))
        self.assertIsNotNone(box)
        x, y, w, h = box
        # Check anatomical projection
        self.assertEqual(x, 175)  # 200 - 0.25*100
        self.assertEqual(y, 205)  # 100 + 1.05*100
        self.assertEqual(w, 150)  # 325 - 175
        self.assertEqual(h, 175)  # 380 - 205

    def test_02_torso_outside_frame_returns_unable_to_determine(self):
        # Face at the bottom edge of image -> torso will be outside frame
        bottom_face = (200, 550, 100, 100)
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        result = DressColorAnalyzer.analyze(canvas, bottom_face)
        self.assertFalse(result.is_determined)
        self.assertEqual(result.color_name, "Unable to determine")
        self.assertIn("outside image frame", result.reason)

    def test_03_insufficient_pixels_returns_unable_to_determine(self):
        # Image where torso only has a 10x10 area inside image
        edge_face = (200, 500, 80, 80)
        small_canvas = np.zeros((595, 600, 3), dtype=np.uint8)
        result = DressColorAnalyzer.analyze(small_canvas, edge_face)
        self.assertFalse(result.is_determined)
        self.assertEqual(result.color_name, "Unable to determine")

    def test_04_pure_red_garment_detection(self):
        canvas = np.zeros(self.canvas_shape, dtype=np.uint8)
        # Fill torso region with pure vibrant Red (BGR: 0, 0, 220)
        torso_box = DressColorAnalyzer.get_torso_roi_box(self.face_box, (600, 600))
        tx, ty, tw, th = torso_box
        canvas[ty : ty + th, tx : tx + tw] = [0, 0, 220]

        result = DressColorAnalyzer.analyze(canvas, self.face_box)
        self.assertTrue(result.is_determined)
        self.assertEqual(result.color_name, "Red")
        self.assertGreaterEqual(result.confidence, MIN_DOMINANCE_THRESHOLD)
        self.assertIsNotNone(result.hex_code)

    def test_05_pure_blue_garment_detection(self):
        canvas = np.zeros(self.canvas_shape, dtype=np.uint8)
        # Fill torso region with Blue (BGR: 200, 100, 20)
        torso_box = DressColorAnalyzer.get_torso_roi_box(self.face_box, (600, 600))
        tx, ty, tw, th = torso_box
        canvas[ty : ty + th, tx : tx + tw] = [200, 100, 20]

        result = DressColorAnalyzer.analyze(canvas, self.face_box)
        self.assertTrue(result.is_determined)
        self.assertIn(result.color_name, ["Blue", "Navy Blue", "Cyan"])
        self.assertGreaterEqual(result.confidence, MIN_DOMINANCE_THRESHOLD)

    def test_06_low_dominance_complex_pattern_returns_unable_to_determine(self):
        canvas = np.zeros(self.canvas_shape, dtype=np.uint8)
        torso_box = DressColorAnalyzer.get_torso_roi_box(self.face_box, (600, 600))
        tx, ty, tw, th = torso_box
        
        # Fill torso with 6 evenly distributed distinct rainbow colors so no color reaches 40%
        # (each stripe is ~16.6% of pixels)
        strip_h = th // 6
        colors_bgr = [
            [0, 0, 255],    # Red
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 255] # White
        ]
        for i, color in enumerate(colors_bgr):
            y_start = ty + i * strip_h
            y_end = ty + (i + 1) * strip_h if i < 5 else ty + th
            canvas[y_start:y_end, tx : tx + tw] = color

        result = DressColorAnalyzer.analyze(canvas, self.face_box)
        self.assertFalse(result.is_determined)
        self.assertEqual(result.color_name, "Unable to determine")
        self.assertIn("Low color dominance", result.reason)


if __name__ == "__main__":
    unittest.main()
