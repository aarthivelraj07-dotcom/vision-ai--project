"""
Tests for Task 4 Pipeline Orchestrator, Multi-Face Isolation, and Conditional Profiles.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from modules.nationality_detection.pipeline import Task4Pipeline, Task4Result

_sample_path_1 = os.path.join("datasets", "attendance", "registered_faces", "STU_A", "sample_1.jpg")
_sample_path_2 = os.path.join("datasets", "attendance", "registered_faces", "STU_B", "sample_1.jpg")


class TestTask4Pipeline(unittest.TestCase):
    """Verifies static image processing, multi-face isolation, and conditional output rules."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = Task4Pipeline()

    @patch("modules.nationality_detection.pipeline.EmotionDetector.get_instance")
    def test_pipeline_defers_emotion_model_loading(self, emotion_loader):
        pipeline = Task4Pipeline()

        self.assertIsNone(pipeline.emotion_detector)
        emotion_loader.assert_not_called()

        result = pipeline.process_image(np.zeros((40, 40, 3), dtype=np.uint8), nationality_declaration="India")

        self.assertFalse(result.has_face)
        emotion_loader.assert_not_called()

    def test_pipeline_reuses_emotion_detector_without_reusing_predictions(self):
        pipeline = Task4Pipeline()
        pipeline.detect_all_faces = MagicMock(return_value=[(5, 5, 20, 20)])
        emotion_results = [
            MagicMock(dominant_emotion="happy", confidence=0.8, to_dict=lambda: {"emotion": "happy"}),
            MagicMock(dominant_emotion="sad", confidence=0.7, to_dict=lambda: {"emotion": "sad"}),
        ]
        emotion_detector = MagicMock()
        emotion_detector.predict.side_effect = emotion_results
        dress_result = MagicMock(color_name="Blue", torso_box=None, to_dict=lambda: {"color": "Blue"})

        with patch("modules.nationality_detection.pipeline.EmotionDetector.get_instance", return_value=emotion_detector) as emotion_loader:
            with patch("modules.nationality_detection.pipeline.DressColorAnalyzer.analyze", return_value=dress_result):
                image = np.zeros((40, 40, 3), dtype=np.uint8)
                first = pipeline.process_image(image, nationality_declaration="United States", declared_age=20)
                second = pipeline.process_image(image, nationality_declaration="United States", declared_age=20)

        emotion_loader.assert_called_once_with()
        self.assertIs(first.emotion, emotion_results[0])
        self.assertIs(second.emotion, emotion_results[1])
        self.assertEqual(emotion_detector.predict.call_count, 2)

    def test_01_no_face_image(self):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        result = self.pipeline.process_image(blank, nationality_declaration="India")
        self.assertFalse(result.has_face)
        self.assertEqual(result.num_faces, 0)
        self.assertIn("No face detected", result.error_message)

    def test_02_condition_indian_with_declared_age(self):
        # Indian condition: displayed = [age, dress_colour], suppressed = [emotion]
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")
        
        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled
        padded[220:480, 80:320] = [74, 42, 27]  # Navy Blue shirt

        # Pass user-provided age: 19
        result = self.pipeline.process_image(padded, nationality_declaration="India", declared_age=19)
        
        self.assertTrue(result.has_face)
        self.assertEqual(result.active_condition, "INDIAN")
        self.assertIn("age", result.displayed_fields)
        self.assertIn("dress_colour", result.displayed_fields)
        self.assertNotIn("emotion", result.displayed_fields)
        self.assertIn("emotion", result.suppressed_fields)

        # Verify declared age resolution
        age_dict = result.displayed_fields["age"]
        self.assertEqual(age_dict["declared_age"], 19)
        self.assertEqual(age_dict["source"], "User-provided age")
        self.assertEqual(age_dict["bracket"], "19 years (Young Adult)")

    def test_03_condition_indian_without_declared_age(self):
        # When user does not provide age
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")

        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled
        padded[220:480, 80:320] = [74, 42, 27]

        result = self.pipeline.process_image(padded, nationality_declaration="India", declared_age=None)
        
        self.assertTrue(result.has_face)
        age_dict = result.displayed_fields["age"]
        self.assertFalse(age_dict["is_available"])
        self.assertEqual(age_dict["bracket"], "Age estimation unavailable")
        self.assertEqual(age_dict["source"], "None")

    def test_04_condition_us(self):
        # US condition: displayed = [age, emotion], suppressed = [dress_colour]
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")
        
        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled

        result = self.pipeline.process_image(padded, nationality_declaration="United States", declared_age=25)
        
        self.assertEqual(result.active_condition, "US")
        self.assertIn("age", result.displayed_fields)
        self.assertIn("emotion", result.displayed_fields)
        self.assertNotIn("dress_colour", result.displayed_fields)
        self.assertIn("dress_colour", result.suppressed_fields)
        self.assertEqual(result.displayed_fields["age"]["bracket"], "25 years (Young Adult)")

    def test_05_condition_african_region(self):
        # African regional category (e.g. Nigeria): displayed = [emotion, dress_colour], suppressed = [age]
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")
        
        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled
        padded[220:480, 80:320] = [30, 30, 210]  # Red shirt

        result = self.pipeline.process_image(padded, nationality_declaration="Nigeria")
        
        self.assertEqual(result.active_condition, "AFRICAN_REGION")
        self.assertIn("emotion", result.displayed_fields)
        self.assertIn("dress_colour", result.displayed_fields)
        self.assertNotIn("age", result.displayed_fields)
        self.assertIn("age", result.suppressed_fields)

    def test_06_condition_other(self):
        # Other nationalities (e.g. Japan): displayed = [nationality, emotion], suppressed = [age, dress_colour]
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")
        
        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled

        result = self.pipeline.process_image(padded, nationality_declaration="Japan")
        
        self.assertEqual(result.active_condition, "OTHER")
        self.assertIn("nationality", result.displayed_fields)
        self.assertIn("emotion", result.displayed_fields)
        self.assertNotIn("age", result.displayed_fields)
        self.assertNotIn("dress_colour", result.displayed_fields)

    def test_07_multiple_occupants_strict_age_isolation(self):
        """Verifies two people in one photograph: age is NOT cross-assigned."""
        if not os.path.exists(_sample_path_1) or not os.path.exists(_sample_path_2):
            self.skipTest("Sample faces missing.")

        # Create dual-occupant canvas: 500 height x 800 width
        canvas = np.full((500, 800, 3), 240, dtype=np.uint8)
        
        # Person 1 on left (x=100)
        face1 = cv2.resize(cv2.imread(_sample_path_1), (140, 140))
        canvas[60:200, 100:240] = face1
        canvas[210:480, 70:270] = [74, 42, 27]  # Navy Blue shirt

        # Person 2 on right (x=500)
        face2 = cv2.resize(cv2.imread(_sample_path_2), (140, 140))
        canvas[60:200, 500:640] = face2
        canvas[210:480, 470:670] = [30, 30, 210]  # Red shirt

        # User declares age=19 for Person 1
        result = self.pipeline.process_image(canvas, nationality_declaration="India", declared_age=19)

        self.assertTrue(result.has_face)
        self.assertGreaterEqual(result.num_faces, 2)
        self.assertGreaterEqual(len(result.people), 2)

        p1 = result.people[0]
        p2 = result.people[1]

        # Person 1 receives user-provided age: 19
        self.assertEqual(p1.person_id, "Person 1")
        self.assertTrue(p1.age.is_available)
        self.assertEqual(p1.age.declared_age, 19)
        self.assertEqual(p1.age.source, "User-provided age")

        # Person 2 MUST NOT receive Person 1's age (strictly isolated)
        self.assertEqual(p2.person_id, "Person 2")
        self.assertFalse(p2.age.is_available)
        self.assertIsNone(p2.age.declared_age)
        self.assertEqual(p2.age.bracket, "Age estimation unavailable")
        self.assertEqual(p2.age.source, "None")
        self.assertIn("No user age declared", p2.age.reason)

        # Independent dress colours
        self.assertTrue(p1.dress_color.is_determined)
        self.assertTrue(p2.dress_color.is_determined)
        self.assertEqual(p1.dress_color.color_name, "Navy Blue")
        self.assertEqual(p2.dress_color.color_name, "Red")

    def test_08_low_quality_crop_handling(self):
        # 30x30 tiny blurry crop
        tiny = np.full((30, 30, 3), 128, dtype=np.uint8)
        result = self.pipeline.process_image(tiny, nationality_declaration="India", declared_age=None)
        # Should gracefully return without throwing exceptions
        self.assertIsInstance(result, Task4Result)

    def test_09_multi_person_select_person_2(self):
        """Selecting Person 2 binds declared age (e.g. 19) strictly to Person 2, keeping Person 1 unavailable."""
        if not os.path.exists(_sample_path_1) or not os.path.exists(_sample_path_2):
            self.skipTest("Sample faces missing.")

        canvas = np.full((500, 800, 3), 240, dtype=np.uint8)
        face1 = cv2.resize(cv2.imread(_sample_path_1), (140, 140))
        canvas[60:200, 100:240] = face1
        canvas[210:480, 70:270] = [74, 42, 27]  # Navy Blue shirt (Person 1)

        face2 = cv2.resize(cv2.imread(_sample_path_2), (140, 140))
        canvas[60:200, 500:640] = face2
        canvas[210:480, 470:670] = [30, 30, 210]  # Red shirt (Person 2)

        # Select Person 2 with declared age = 19
        result = self.pipeline.process_image(
            canvas,
            nationality_declaration="India",
            declared_age=19,
            selected_subject="Person 2"
        )

        self.assertTrue(result.has_face)
        self.assertEqual(result.selected_subject, "Person 2")
        self.assertGreaterEqual(result.num_faces, 2)

        p1 = result.people[0]
        p2 = result.people[1]

        # Person 2 (selected subject) MUST receive age 19
        self.assertEqual(p2.person_id, "Person 2")
        self.assertTrue(p2.age.is_available)
        self.assertEqual(p2.age.declared_age, 19)
        self.assertEqual(p2.age.source, "User-provided age")
        self.assertEqual(p2.age.bracket, "19 years (Young Adult)")

        # Person 1 MUST remain unavailable (no cross-assignment)
        self.assertEqual(p1.person_id, "Person 1")
        self.assertFalse(p1.age.is_available)
        self.assertIsNone(p1.age.declared_age)
        self.assertEqual(p1.age.bracket, "Age estimation unavailable")
        self.assertIn("No user age declared for Person 1", p1.age.reason)

        # Top-level result reflects the selected target person
        self.assertEqual(result.age.declared_age, 19)
        self.assertEqual(result.displayed_fields["age"]["declared_age"], 19)

    def test_10_multi_person_select_person_1(self):
        """Selecting Person 1 binds declared age to Person 1, keeping Person 2 unavailable."""
        if not os.path.exists(_sample_path_1) or not os.path.exists(_sample_path_2):
            self.skipTest("Sample faces missing.")

        canvas = np.full((500, 800, 3), 240, dtype=np.uint8)
        face1 = cv2.resize(cv2.imread(_sample_path_1), (140, 140))
        canvas[60:200, 100:240] = face1
        canvas[210:480, 70:270] = [74, 42, 27]

        face2 = cv2.resize(cv2.imread(_sample_path_2), (140, 140))
        canvas[60:200, 500:640] = face2
        canvas[210:480, 470:670] = [30, 30, 210]

        result = self.pipeline.process_image(
            canvas,
            nationality_declaration="India",
            declared_age=25,
            selected_subject="Person 1"
        )

        self.assertEqual(result.selected_subject, "Person 1")
        p1 = result.people[0]
        p2 = result.people[1]

        self.assertTrue(p1.age.is_available)
        self.assertEqual(p1.age.declared_age, 25)
        self.assertFalse(p2.age.is_available)
        self.assertEqual(p2.age.bracket, "Age estimation unavailable")
        self.assertEqual(result.age.declared_age, 25)

    def test_11_single_person_default_selection(self):
        """Single-person image assigns age to Person 1 without selection controls."""
        if not os.path.exists(_sample_path_1):
            self.skipTest("Sample image missing.")

        sample = cv2.imread(_sample_path_1)
        sample_scaled = cv2.resize(sample, (150, 150))
        padded = np.full((500, 400, 3), 240, dtype=np.uint8)
        padded[60:210, 125:275] = sample_scaled

        result = self.pipeline.process_image(padded, nationality_declaration="India", declared_age=30)
        self.assertEqual(result.selected_subject, "Person 1")
        self.assertEqual(result.num_faces, 1)
        self.assertEqual(result.age.declared_age, 30)

    def test_12_no_age_cross_assignment_explicit_dict(self):
        """Supplying explicit declared_ages dictionary correctly assigns per-person ages."""
        if not os.path.exists(_sample_path_1) or not os.path.exists(_sample_path_2):
            self.skipTest("Sample faces missing.")

        canvas = np.full((500, 800, 3), 240, dtype=np.uint8)
        face1 = cv2.resize(cv2.imread(_sample_path_1), (140, 140))
        canvas[60:200, 100:240] = face1
        face2 = cv2.resize(cv2.imread(_sample_path_2), (140, 140))
        canvas[60:200, 500:640] = face2

        result = self.pipeline.process_image(
            canvas,
            nationality_declaration="India",
            declared_ages={"Person 1": 45, "Person 2": 19},
            selected_subject="Person 2"
        )

        p1 = result.people[0]
        p2 = result.people[1]
        self.assertEqual(p1.age.declared_age, 45)
        self.assertEqual(p2.age.declared_age, 19)
        self.assertEqual(result.selected_subject, "Person 2")
        self.assertEqual(result.age.declared_age, 19)

    def test_13_five_detected_faces_dynamic_selection(self):
        """Verifies dynamic handling of 5 detected faces: selecting Person 4 binds age exclusively to Person 4."""
        group_5_path = os.path.join("datasets", "nationality_detection", "test_images", "portrait_group_5_test.jpg")
        if not os.path.exists(group_5_path):
            self.skipTest("portrait_group_5_test.jpg missing.")

        img5 = cv2.imread(group_5_path)
        result = self.pipeline.process_image(
            img5,
            nationality_declaration="India",
            declared_age=38,
            selected_subject="Person 4"
        )

        self.assertTrue(result.has_face)
        self.assertEqual(result.num_faces, 5)
        self.assertEqual(len(result.people), 5)
        self.assertEqual(result.selected_subject, "Person 4")

        # Person 4 MUST have declared age 38
        p4 = result.people[3]
        self.assertEqual(p4.person_id, "Person 4")
        self.assertTrue(p4.age.is_available)
        self.assertEqual(p4.age.declared_age, 38)
        self.assertEqual(p4.age.bracket, "38 years (Middle-aged Adult)")
        self.assertEqual(result.age.declared_age, 38)

        # All other 4 occupants MUST be unavailable
        for i, idx in enumerate([0, 1, 2, 4]):
            other = result.people[idx]
            self.assertFalse(other.age.is_available, f"{other.person_id} should be unavailable")
            self.assertIsNone(other.age.declared_age)
            self.assertEqual(other.age.bracket, "Age estimation unavailable")

    def test_14_ten_detected_faces_dynamic_selection(self):
        """Verifies dynamic handling of 10 detected faces: selecting Person 7 binds age exclusively to Person 7."""
        group_10_path = os.path.join("datasets", "nationality_detection", "test_images", "portrait_group_10_test.jpg")
        if not os.path.exists(group_10_path):
            self.skipTest("portrait_group_10_test.jpg missing.")

        img10 = cv2.imread(group_10_path)
        result = self.pipeline.process_image(
            img10,
            nationality_declaration="India",
            declared_age=21,
            selected_subject="Person 7"
        )

        self.assertTrue(result.has_face)
        self.assertEqual(result.num_faces, 10)
        self.assertEqual(len(result.people), 10)
        self.assertEqual(result.selected_subject, "Person 7")

        # Person 7 MUST have declared age 21
        p7 = result.people[6]
        self.assertEqual(p7.person_id, "Person 7")
        self.assertTrue(p7.age.is_available)
        self.assertEqual(p7.age.declared_age, 21)
        self.assertEqual(p7.age.bracket, "21 years (Young Adult)")
        self.assertEqual(result.age.declared_age, 21)

        # All other 9 occupants MUST be unavailable
        for i in range(10):
            if i != 6:
                other = result.people[i]
                self.assertFalse(other.age.is_available, f"{other.person_id} should be unavailable")
                self.assertIsNone(other.age.declared_age)
                self.assertEqual(other.age.bracket, "Age estimation unavailable")

    def test_15_invalid_selection_out_of_bounds(self):
        """Selecting out-of-bounds 'Person 99' safely rejects attribution to any detected occupant."""
        group_5_path = os.path.join("datasets", "nationality_detection", "test_images", "portrait_group_5_test.jpg")
        if not os.path.exists(group_5_path):
            self.skipTest("portrait_group_5_test.jpg missing.")

        img5 = cv2.imread(group_5_path)
        result = self.pipeline.process_image(
            img5,
            nationality_declaration="India",
            declared_age=30,
            selected_subject="Person 99"
        )

        self.assertTrue(result.has_face)
        self.assertEqual(result.num_faces, 5)
        self.assertEqual(result.selected_subject, "Person 99")

        # NONE of the 5 people should receive the declared age
        for p in result.people:
            self.assertFalse(p.age.is_available, f"{p.person_id} should NOT receive out-of-bounds age")
            self.assertIsNone(p.age.declared_age)
            self.assertEqual(p.age.bracket, "Age estimation unavailable")

        # Top-level result defaults to Person 1 which is unavailable
        self.assertFalse(result.age.is_available)
        self.assertEqual(result.age.bracket, "Age estimation unavailable")

    def test_16_invalid_selection_unrecognized_string(self):
        """Selecting an unrecognized string safely ensures no face is falsely attributed."""
        group_5_path = os.path.join("datasets", "nationality_detection", "test_images", "portrait_group_5_test.jpg")
        if not os.path.exists(group_5_path):
            self.skipTest("portrait_group_5_test.jpg missing.")

        img5 = cv2.imread(group_5_path)
        result = self.pipeline.process_image(
            img5,
            nationality_declaration="India",
            declared_age=40,
            selected_subject="Invalid Person String"
        )

        for p in result.people:
            self.assertFalse(p.age.is_available)
            self.assertIsNone(p.age.declared_age)
        self.assertFalse(result.age.is_available)

    def test_17_raw_formatted_selection_string_normalized(self):
        """Verifies UI string format 'Person 3 (Face 3 from left)' is normalized into canonical 'Person 3'."""
        group_5_path = os.path.join("datasets", "nationality_detection", "test_images", "portrait_group_5_test.jpg")
        if not os.path.exists(group_5_path):
            self.skipTest("portrait_group_5_test.jpg missing.")

        img5 = cv2.imread(group_5_path)
        result = self.pipeline.process_image(
            img5,
            nationality_declaration="India",
            declared_age=29,
            selected_subject="Person 3 (Face 3 from left)"
        )

        self.assertEqual(result.selected_subject, "Person 3")
        self.assertTrue(result.people[2].age.is_available)
        self.assertEqual(result.people[2].age.declared_age, 29)
        self.assertFalse(result.people[0].age.is_available)
        self.assertFalse(result.people[1].age.is_available)
        self.assertFalse(result.people[3].age.is_available)
        self.assertFalse(result.people[4].age.is_available)


if __name__ == "__main__":
    unittest.main()
