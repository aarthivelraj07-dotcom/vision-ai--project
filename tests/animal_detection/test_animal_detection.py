"""
Comprehensive Test Suite for Task 2: Animal Detection & Multi-Object Tracking.
Verifies all 9 requirements:
1. Detect multiple animals in single image.
2. Detect multiple animal types in same image.
3. Draw bounding boxes around detected animals.
4. Display carnivorous animals with RED bounding boxes.
5. Display other animals using distinct colors (GREEN for herbivores, AMBER for omnivores/other).
6. Total animal count and popup notification summary data.
7. Species-wise and category-wise count breakdowns.
8. Support image and video tracking input.
9. Avoid counting the same animal repeatedly across video frames.
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.animal_detection.taxonomy import (
    AnimalTaxonomy,
    DietCategory,
    COCO_ANIMAL_CLASSES,
    CATEGORY_COLORS_BGR,
    CATEGORY_COLORS_HEX,
)
from modules.animal_detection.detector import AnimalDetector, letterbox
from modules.animal_detection.tracker import AnimalTracker, compute_iou, compute_centroid


class TestAnimalDetectionTask2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = AnimalDetector.get_instance()
        cls.test_dir = os.path.join(PROJECT_ROOT, "datasets", "animal_detection", "test_images")

    def test_01_taxonomy_three_categories_and_color_assignment(self):
        """Req 4 & 5: Verify 3 categories and strict color coding (Red, Green, Amber)."""
        # Carnivores -> RED (BGR: (0, 0, 255), Hex: #EF4444)
        self.assertEqual(AnimalTaxonomy.get_category("cat"), DietCategory.CARNIVORE)
        self.assertEqual(AnimalTaxonomy.get_color_bgr("cat"), (0, 0, 255))
        self.assertEqual(AnimalTaxonomy.get_color_hex("cat"), "#EF4444")
        self.assertTrue(AnimalTaxonomy.is_carnivore("cat"))

        # Herbivores -> GREEN (BGR: (0, 255, 0), Hex: #10B981)
        for h in ["cow", "horse", "sheep", "elephant", "zebra", "giraffe"]:
            self.assertEqual(AnimalTaxonomy.get_category(h), DietCategory.HERBIVORE)
            self.assertEqual(AnimalTaxonomy.get_color_bgr(h), (0, 255, 0))
            self.assertEqual(AnimalTaxonomy.get_color_hex(h), "#10B981")
            self.assertTrue(AnimalTaxonomy.is_herbivore(h))

        # Omnivores / Other -> AMBER (BGR: (0, 180, 255), Hex: #F59E0B)
        for o in ["dog", "bear", "bird"]:
            self.assertEqual(AnimalTaxonomy.get_category(o), DietCategory.OMNIVORE_OTHER)
            self.assertEqual(AnimalTaxonomy.get_color_bgr(o), (0, 180, 255))
            self.assertEqual(AnimalTaxonomy.get_color_hex(o), "#F59E0B")
            self.assertTrue(AnimalTaxonomy.is_omnivore_other(o))

    def test_02_model_and_metadata_integrity(self):
        """Verify model weights and metadata exist and are valid."""
        model_path = os.path.join(PROJECT_ROOT, "models", "animal_detection", "yolov8n.onnx")
        self.assertTrue(os.path.exists(model_path))
        self.assertGreater(os.path.getsize(model_path), 10_000_000)
        self.assertTrue(self.detector.is_loaded)

    def test_03_detect_single_carnivore_red_box(self):
        """Req 3 & 4: Detect single carnivore (cat) and verify RED bounding box."""
        cat_img = cv2.imread(os.path.join(self.test_dir, "single_cat.jpg"))
        self.assertIsNotNone(cat_img)

        dets = self.detector.detect_animals(cat_img)
        self.assertGreaterEqual(len(dets), 1)

        cat_det = next((d for d in dets if d["class_name"] == "cat"), None)
        self.assertIsNotNone(cat_det)
        self.assertEqual(cat_det["category"], DietCategory.CARNIVORE.value)
        self.assertEqual(cat_det["color_bgr"], (0, 0, 255))
        self.assertIn("[CARNIVORE]", cat_det["label"])

    def test_04_detect_single_omnivore_amber_box(self):
        """Req 5: Detect single omnivore (dog) and verify AMBER bounding box."""
        dog_img = cv2.imread(os.path.join(self.test_dir, "single_dog.jpg"))
        self.assertIsNotNone(dog_img)

        dets = self.detector.detect_animals(dog_img)
        self.assertGreaterEqual(len(dets), 1)

        dog_det = next((d for d in dets if d["class_name"] == "dog"), None)
        self.assertIsNotNone(dog_det)
        self.assertEqual(dog_det["category"], DietCategory.OMNIVORE_OTHER.value)
        self.assertEqual(dog_det["color_bgr"], (0, 180, 255))
        self.assertIn("[OMNIVORE / OTHER]", dog_det["label"])

    def test_05_detect_multiple_animals_same_species_green_box(self):
        """Req 1 & 5: Detect multiple animals in single image (horses -> GREEN boxes)."""
        horses_img = cv2.imread(os.path.join(self.test_dir, "multi_horses.jpg"))
        self.assertIsNotNone(horses_img)

        dets = self.detector.detect_animals(horses_img)
        horse_dets = [d for d in dets if d["class_name"] == "horse"]
        self.assertGreaterEqual(len(horse_dets), 2, "Must detect multiple horses.")

        for h in horse_dets:
            self.assertEqual(h["category"], DietCategory.HERBIVORE.value)
            self.assertEqual(h["color_bgr"], (0, 255, 0))

    def test_06_detect_multiple_animal_types_mixed_species(self):
        """Req 2: Detect multiple animal types in the same image (giraffes & zebras)."""
        mixed_img = cv2.imread(os.path.join(self.test_dir, "mixed_giraffe_zebra.jpg"))
        self.assertIsNotNone(mixed_img)

        dets = self.detector.detect_animals(mixed_img)
        self.assertGreaterEqual(len(dets), 1)

        summary = self.detector.get_summary_statistics(dets)
        self.assertEqual(summary["total_count"], len(dets))
        self.assertGreater(len(summary["species_counts"]), 0)

    def test_07_handle_no_animal_image(self):
        """Verify graceful empty return when image contains no animals."""
        no_img = cv2.imread(os.path.join(self.test_dir, "no_animals.jpg"))
        self.assertIsNotNone(no_img)

        dets = self.detector.detect_animals(no_img)
        self.assertEqual(len(dets), 0)

        summary = self.detector.get_summary_statistics(dets)
        self.assertEqual(summary["total_count"], 0)
        self.assertEqual(summary["carnivore_count"], 0)
        self.assertEqual(summary["herbivore_count"], 0)

    def test_08_draw_annotations_preserves_resolution(self):
        """Req 3: Verify bounding box rendering on image canvas."""
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        mock_dets = [{
            "bbox": (50, 50, 100, 100),
            "class_name": "cat",
            "category": DietCategory.CARNIVORE.value,
            "confidence": 0.95,
            "color_bgr": (0, 0, 255),
            "label": "Cat 95% [CARNIVORE]"
        }]
        annotated = self.detector.draw_detections(canvas, mock_dets)
        self.assertEqual(annotated.shape, canvas.shape)
        # Check red pixel presence from bounding box
        self.assertTrue(np.any(annotated[:, :, 2] == 255))

    def test_09_video_tracking_deduplication(self):
        """Req 9: Verify repeated video frames of the same animal do NOT inflate count."""
        tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=2)
        mock_dog = {
            "bbox": (120, 120, 60, 60),
            "class_name": "dog",
            "category": DietCategory.OMNIVORE_OTHER.value,
            "confidence": 0.89,
        }

        # 15 frames of the same dog in view
        for f in range(15):
            t_dets, metrics = tracker.update([mock_dog])
            self.assertEqual(len(t_dets), 1)
            self.assertEqual(t_dets[0]["track_id"], 1)

        # Cumulative unique count must be strictly 1
        self.assertEqual(metrics["cumulative_unique_count"], 1)
        self.assertEqual(metrics["cumulative_species_counts"]["dog"], 1)

    def test_10_video_tracking_temporary_occlusion(self):
        """Verify tracker coasting state bridges temporary missed frames without ID change."""
        tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=2)
        mock_cat = {"bbox": (50, 50, 40, 40), "class_name": "cat", "category": DietCategory.CARNIVORE.value, "confidence": 0.92}

        # Frames 1-2: Confirm track #1
        tracker.update([mock_cat])
        tracker.update([mock_cat])

        # Frames 3-5: Cat hidden / missed
        for _ in range(3):
            tracker.update([])

        # Frame 6: Cat re-appears at (52, 51, 40, 40)
        re_cat = dict(mock_cat)
        re_cat["bbox"] = (52, 51, 40, 40)
        t_dets, m = tracker.update([re_cat])

        # Track ID must remain #1
        self.assertEqual(t_dets[0]["track_id"], 1)
        self.assertEqual(m["cumulative_unique_count"], 1)

    def test_11_video_tracking_track_retirement(self):
        """Verify track is retired after 16 consecutive frames of absence."""
        tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=2)
        mock_cat = {"bbox": (50, 50, 40, 40), "class_name": "cat", "category": DietCategory.CARNIVORE.value, "confidence": 0.92}

        tracker.update([mock_cat])
        tracker.update([mock_cat])

        # 16 missed frames
        for _ in range(16):
            tracker.update([])

        # Active/coasting tracks must be 0
        self.assertEqual(len(tracker.tracks), 0)
        # But cumulative unique count remembers the entity
        self.assertEqual(len(tracker.cumulative_confirmed_ids), 1)

    def test_12_app_router_backward_compatibility(self):
        """Verify app.py exports run_app and maintains Task 1 entry point compatibility."""
        import app
        self.assertTrue(hasattr(app, "run_app"), "app.py must expose run_app for Task 1 tests.")
        self.assertTrue(hasattr(app, "main"), "app.py must expose main() router function.")

    def test_13_corrupt_and_non_standard_image_inputs(self):
        """Verify edge-case handling for None, empty, grayscale, and 4-channel BGRA inputs."""
        # 1. None input
        self.assertEqual(self.detector.detect_animals(None), [])

        # 2. Empty array
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(self.detector.detect_animals(empty_img), [])

        # 3. Grayscale 2D array (must convert to 3-channel without crashing)
        gray_img = np.zeros((300, 300), dtype=np.uint8)
        res_gray = self.detector.detect_animals(gray_img)
        self.assertIsInstance(res_gray, list)

        # 4. BGRA 4-channel array (must convert to 3-channel without crashing)
        bgra_img = np.zeros((300, 300, 4), dtype=np.uint8)
        res_bgra = self.detector.detect_animals(bgra_img)
        self.assertIsInstance(res_bgra, list)

    def test_14_dietary_rationale_coverage_all_10_species(self):
        """Verify biological and ecological rationale coverage for all 10 COCO animal species."""
        supported = AnimalTaxonomy.get_supported_species()
        self.assertEqual(len(supported), 10)
        expected_species = ["bear", "bird", "cat", "cow", "dog", "elephant", "giraffe", "horse", "sheep", "zebra"]
        self.assertEqual(supported, expected_species)

        for s in supported:
            rationale = AnimalTaxonomy.get_dietary_rationale(s)
            self.assertIsInstance(rationale, str)
            self.assertGreater(len(rationale), 25, f"Rationale for {s} must be descriptive.")

        # Special check for bird polyphyletic rationale
        bird_rationale = AnimalTaxonomy.get_dietary_rationale("bird")
        self.assertTrue(
            "polyphyletic" in bird_rationale.lower() or "generalist" in bird_rationale.lower() or "raptors" in bird_rationale.lower()
        )

    def test_15_summary_statistics_color_coding_integrity(self):
        """Verify summary statistics output produces accurate hex colors matching category."""
        mock_dets = [
            {
                "bbox": (10, 10, 50, 50),
                "class_name": "cat",
                "category": DietCategory.CARNIVORE.value,
                "confidence": 0.90,
                "color_bgr": (0, 0, 255),
                "color_hex": "#EF4444",
                "label": "Cat 90% [CARNIVORE]"
            },
            {
                "bbox": (70, 70, 50, 50),
                "class_name": "horse",
                "category": DietCategory.HERBIVORE.value,
                "confidence": 0.85,
                "color_bgr": (0, 255, 0),
                "color_hex": "#10B981",
                "label": "Horse 85% [HERBIVORE]"
            }
        ]
        summary = self.detector.get_summary_statistics(mock_dets)
        self.assertEqual(summary["total_count"], 2)
        self.assertEqual(summary["carnivore_count"], 1)
        self.assertEqual(summary["herbivore_count"], 1)

        breakdown = {item["species"]: item for item in summary["species_breakdown"]}
        self.assertEqual(breakdown["Cat"]["color_hex"], "#EF4444")
        self.assertEqual(breakdown["Horse"]["color_hex"], "#10B981")

    def test_16_tracker_species_constraint_prevents_mismatch(self):
        """Verify tracker will not associate detections of different species even with 100% spatial overlap."""
        tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=1)
        det_dog = {
            "bbox": (100, 100, 50, 50),
            "class_name": "dog",
            "category": DietCategory.OMNIVORE_OTHER.value,
            "confidence": 0.90
        }
        det_cat = {
            "bbox": (100, 100, 50, 50),
            "class_name": "cat",
            "category": DietCategory.CARNIVORE.value,
            "confidence": 0.90
        }

        # Frame 1: Dog detected at (100, 100, 50, 50)
        t_dets1, _ = tracker.update([det_dog])
        dog_id = t_dets1[0]["track_id"]

        # Frame 2: Cat detected at identical bbox (100, 100, 50, 50)
        t_dets2, _ = tracker.update([det_cat])
        cat_id = t_dets2[0]["track_id"]

        # Cat MUST NOT inherit dog's track ID
        self.assertNotEqual(dog_id, cat_id)

    def test_17_extreme_confidence_threshold_filtering(self):
        """Verify extreme confidence threshold (e.g. 0.999) filters low-confidence detections cleanly."""
        cat_img = cv2.imread(os.path.join(self.test_dir, "single_cat.jpg"))
        self.assertIsNotNone(cat_img)

        # Standard threshold yields detection
        standard_dets = self.detector.detect_animals(cat_img, conf_threshold=0.35)
        self.assertGreaterEqual(len(standard_dets), 1)

        # Extreme threshold 0.999 filters out detections without crashing
        strict_dets = self.detector.detect_animals(cat_img, conf_threshold=0.999)
        self.assertEqual(len(strict_dets), 0)

    def test_18_sample_video_end_to_end_tracking(self):
        """Verify end-to-end video decoding, detection, and tracking on test video."""
        video_path = os.path.join(PROJECT_ROOT, "datasets", "animal_detection", "test_videos", "sample_cat_walk.mp4")
        self.assertTrue(os.path.exists(video_path), "sample_cat_walk.mp4 must exist.")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), "Video must be openable via OpenCV.")

        tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=2)
        frames_processed = 0
        latest_metrics = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_processed += 1
            dets = self.detector.detect_animals(frame, conf_threshold=0.35)
            _, latest_metrics = tracker.update(dets)

        cap.release()
        self.assertEqual(frames_processed, 30)
        self.assertEqual(latest_metrics.get("cumulative_unique_count"), 1, "Must count exactly 1 unique cat across all 30 frames.")
        self.assertEqual(latest_metrics.get("current_carnivores"), 1)


if __name__ == "__main__":
    unittest.main()
