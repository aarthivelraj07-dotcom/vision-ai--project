"""
Standalone Inference Test Suite for Task 2 Animal Detection.
Verifies Phase 1 implementation:
1. AnimalTaxonomy three-category classification and color mapping.
2. Official YOLOv8n ONNX model loading and CPU inference via OpenCV DNN.
3. Preprocessing letterboxing and NMS filtering.
4. Real-image inference:
   - Single carnivore (Cat -> Red bounding box).
   - Single omnivore (Dog -> Amber bounding box).
   - Multi-animal same species (Horses -> Green bounding boxes).
   - Multi-animal mixed species (Giraffes & Zebras).
   - No animals (Fruits -> 0 detections, graceful return).
5. Summary statistics and visual annotation.
"""

import os
import sys
import json
import hashlib
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


class TestAnimalDetectionPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = AnimalDetector.get_instance()
        cls.test_dir = os.path.join(PROJECT_ROOT, "datasets", "animal_detection", "test_images")

    def test_01_taxonomy_three_categories_and_colors(self):
        """Verify the 3 dietary categories and strict color mappings."""
        # 1. Colors
        self.assertEqual(CATEGORY_COLORS_BGR[DietCategory.CARNIVORE], (0, 0, 255))        # Red
        self.assertEqual(CATEGORY_COLORS_BGR[DietCategory.HERBIVORE], (0, 255, 0))        # Green
        self.assertEqual(CATEGORY_COLORS_BGR[DietCategory.OMNIVORE_OTHER], (0, 180, 255)) # Amber

        self.assertEqual(CATEGORY_COLORS_HEX[DietCategory.CARNIVORE], "#EF4444")
        self.assertEqual(CATEGORY_COLORS_HEX[DietCategory.HERBIVORE], "#10B981")
        self.assertEqual(CATEGORY_COLORS_HEX[DietCategory.OMNIVORE_OTHER], "#F59E0B")

        # 2. Confirmed Carnivore
        self.assertEqual(AnimalTaxonomy.get_category("cat"), DietCategory.CARNIVORE)
        self.assertTrue(AnimalTaxonomy.is_carnivore("cat"))
        self.assertEqual(AnimalTaxonomy.get_color_bgr("cat"), (0, 0, 255))
        self.assertEqual(AnimalTaxonomy.get_color_hex("cat"), "#EF4444")

        # 3. Confirmed Herbivores
        herbivores = ["cow", "horse", "sheep", "elephant", "zebra", "giraffe"]
        for h in herbivores:
            self.assertEqual(AnimalTaxonomy.get_category(h), DietCategory.HERBIVORE, f"{h} should be Herbivore")
            self.assertTrue(AnimalTaxonomy.is_herbivore(h))
            self.assertEqual(AnimalTaxonomy.get_color_bgr(h), (0, 255, 0))
            self.assertEqual(AnimalTaxonomy.get_color_hex(h), "#10B981")

        # 4. Omnivore / Other
        omnivores = ["dog", "bear", "bird"]
        for o in omnivores:
            self.assertEqual(AnimalTaxonomy.get_category(o), DietCategory.OMNIVORE_OTHER, f"{o} should be Omnivore/Other")
            self.assertTrue(AnimalTaxonomy.is_omnivore_other(o))
            self.assertEqual(AnimalTaxonomy.get_color_bgr(o), (0, 180, 255))
            self.assertEqual(AnimalTaxonomy.get_color_hex(o), "#F59E0B")

    def test_02_model_file_and_sha256_checksum(self):
        """Verify the official YOLOv8n ONNX model file and its SHA-256 checksum."""
        model_path = os.path.join(PROJECT_ROOT, "models", "animal_detection", "yolov8n.onnx")
        meta_path = os.path.join(PROJECT_ROOT, "models", "animal_detection", "model_meta.json")

        self.assertTrue(os.path.exists(model_path), "yolov8n.onnx must exist.")
        self.assertTrue(os.path.exists(meta_path), "model_meta.json must exist.")

        file_size = os.path.getsize(model_path)
        self.assertGreater(file_size, 10_000_000, "Model size should be ~12.8 MB.")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        with open(model_path, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(computed_sha, meta["sha256"], "SHA-256 hash must match official metadata.")
        self.assertEqual(meta["license"], "AGPL-3.0")

    def test_03_letterbox_preprocessing(self):
        """Verify letterbox maintains aspect ratio and pads to 640x640."""
        # Non-square image 800x400
        test_img = np.full((400, 800, 3), 100, dtype=np.uint8)
        padded, scale, (pad_x, pad_y) = letterbox(test_img, target_size=(640, 640))

        self.assertEqual(padded.shape, (640, 640, 3))
        # Scale = 640 / 800 = 0.8
        self.assertAlmostEqual(scale, 0.8, places=3)
        # New dimensions: 640x320, padding on top/bottom = 320 // 2 = 160
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 160)

    def test_04_real_image_carnivore_detection(self):
        """Verify detection of a real carnivore (cat) with RED bounding box."""
        img_path = os.path.join(self.test_dir, "single_cat.jpg")
        self.assertTrue(os.path.exists(img_path))
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        detections = self.detector.detect_animals(img, conf_threshold=0.35)
        self.assertGreaterEqual(len(detections), 1, "At least one cat should be detected.")

        cat_det = next((d for d in detections if d["class_name"] == "cat"), None)
        self.assertIsNotNone(cat_det, "A 'cat' must be detected.")
        self.assertEqual(cat_det["category"], DietCategory.CARNIVORE.value)
        self.assertEqual(cat_det["color_bgr"], (0, 0, 255), "Carnivore must have RED bounding box.")
        self.assertEqual(cat_det["color_hex"], "#EF4444")
        self.assertIn("CARNIVORE", cat_det["label"])

    def test_05_real_image_omnivore_detection(self):
        """Verify detection of an omnivore (dog) with AMBER bounding box."""
        img_path = os.path.join(self.test_dir, "single_dog.jpg")
        self.assertTrue(os.path.exists(img_path))
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        detections = self.detector.detect_animals(img, conf_threshold=0.35)
        self.assertGreaterEqual(len(detections), 1, "At least one animal should be detected.")

        dog_det = next((d for d in detections if d["class_name"] == "dog"), None)
        self.assertIsNotNone(dog_det, "A 'dog' must be detected.")
        self.assertEqual(dog_det["category"], DietCategory.OMNIVORE_OTHER.value)
        self.assertEqual(dog_det["color_bgr"], (0, 180, 255), "Omnivore must have AMBER bounding box.")
        self.assertEqual(dog_det["color_hex"], "#F59E0B")
        self.assertIn("OMNIVORE / OTHER", dog_det["label"])

    def test_06_real_image_multi_animal_same_species(self):
        """Verify multiple animals of the same species (horses) with GREEN bounding boxes."""
        img_path = os.path.join(self.test_dir, "multi_horses.jpg")
        self.assertTrue(os.path.exists(img_path))
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        detections = self.detector.detect_animals(img, conf_threshold=0.35)
        # The horses image contains multiple horses
        horse_dets = [d for d in detections if d["class_name"] == "horse"]
        self.assertGreaterEqual(len(horse_dets), 2, "Multiple horses should be detected.")

        for h in horse_dets:
            self.assertEqual(h["category"], DietCategory.HERBIVORE.value)
            self.assertEqual(h["color_bgr"], (0, 255, 0), "Herbivore must have GREEN bounding box.")
            self.assertEqual(h["color_hex"], "#10B981")
            self.assertIn("HERBIVORE", h["label"])

    def test_07_real_image_mixed_species_detection(self):
        """Verify detection of multiple animal types in the same image (giraffes / zebras)."""
        img_path = os.path.join(self.test_dir, "mixed_giraffe_zebra.jpg")
        self.assertTrue(os.path.exists(img_path))
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        detections = self.detector.detect_animals(img, conf_threshold=0.35)
        self.assertGreaterEqual(len(detections), 1)

        detected_species = {d["class_name"] for d in detections}
        self.assertTrue("giraffe" in detected_species or "zebra" in detected_species)

        # Check summary breakdown
        summary = self.detector.get_summary_statistics(detections)
        self.assertEqual(summary["total_count"], len(detections))
        self.assertIn("species_breakdown", summary)

    def test_08_no_animals_image_handling(self):
        """Verify graceful return when no animals are present in the image."""
        img_path = os.path.join(self.test_dir, "no_animals.jpg")
        self.assertTrue(os.path.exists(img_path))
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        detections = self.detector.detect_animals(img, conf_threshold=0.35)
        self.assertEqual(len(detections), 0, "No animals should be detected in fruit/background image.")

        summary = self.detector.get_summary_statistics(detections)
        self.assertEqual(summary["total_count"], 0)
        self.assertEqual(summary["carnivore_count"], 0)
        self.assertEqual(summary["herbivore_count"], 0)
        self.assertEqual(summary["omnivore_other_count"], 0)
        self.assertEqual(len(summary["species_breakdown"]), 0)

    def test_09_draw_detections_annotation(self):
        """Verify drawing annotations returns a valid image with matching shape."""
        test_img = np.full((300, 400, 3), 200, dtype=np.uint8)
        mock_dets = [{
            "bbox": (50, 50, 100, 100),
            "class_name": "cat",
            "category": DietCategory.CARNIVORE.value,
            "confidence": 0.92,
            "color_bgr": (0, 0, 255),
            "color_hex": "#EF4444",
            "label": "Cat: 92% [CARNIVORE]"
        }]
        annotated = self.detector.draw_detections(test_img, mock_dets)
        self.assertEqual(annotated.shape, test_img.shape)
        # Pixels inside the bounding box line should be red (0, 0, 255)
        # Check that bounding box was actually drawn
        self.assertTrue(np.any(annotated[:, :, 2] == 255))


if __name__ == "__main__":
    unittest.main()
