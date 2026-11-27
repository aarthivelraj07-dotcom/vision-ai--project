"""
Comprehensive Test Suite for Phase 2: Student Registration and Face Recognition.

Validates all 10 core requirements:
1. Registering a new student.
2. Duplicate student ID rejection.
3. Multiple images per student.
4. Recognizing a registered student on unseen test images.
5. Rejecting an unknown person below confidence threshold.
6. Recognizing multiple registered students in a single frame.
7. Handling images without faces.
8. Handling poor-quality images (blur/lighting).
9. Saving and loading registered student data from SQLite.
10. Adding new students without losing existing records.
"""

import os
import sys
import shutil
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager

TEST_DB_PATH = os.path.join("datasets", "attendance", "test_attendance.db")
TEST_CLASSIFIER_PATH = os.path.join("models", "attendance", "test_classifier.pkl")
TEST_FACES_DIR = os.path.join("datasets", "attendance", "test_faces")


def generate_face_canvas(
    center=(160, 160),
    head_size=(70, 95),
    skin_tone=(180, 190, 225),
    hair_color=(30, 30, 30),
    eye_offset=25,
    eye_y=145,
    eye_size=8,
    mouth_y=205,
    mouth_w=28,
    mouth_h=12,
    img_size=(320, 320),
    blur=0
) -> np.ndarray:
    """Helper to generate realistic face canvas for automated testing."""
    w, h = img_size
    img = np.full((h, w, 3), (220, 220, 220), dtype=np.uint8)
    cx, cy = center
    hw, hh = head_size

    cv2.ellipse(img, (cx, cy), (hw, hh), 0, 0, 360, skin_tone, -1)
    cv2.ellipse(img, (cx, cy - int(hh * 0.65)), (int(hw * 1.05), int(hh * 0.5)), 0, 180, 360, hair_color, -1)
    left_x = cx - eye_offset
    right_x = cx + eye_offset
    cv2.circle(img, (left_x, eye_y), eye_size, (255, 255, 255), -1)
    cv2.circle(img, (right_x, eye_y), eye_size, (255, 255, 255), -1)
    cv2.circle(img, (left_x, eye_y), max(2, eye_size // 2), (40, 40, 40), -1)
    cv2.circle(img, (right_x, eye_y), max(2, eye_size // 2), (40, 40, 40), -1)
    cv2.line(img, (left_x - 12, eye_y - 12), (left_x + 12, eye_y - 10), hair_color, 3)
    cv2.line(img, (right_x - 12, eye_y - 10), (right_x + 12, eye_y - 12), hair_color, 3)
    nose_y = cy + 15
    cv2.line(img, (cx, eye_y), (cx, nose_y), (int(skin_tone[0]*0.8), int(skin_tone[1]*0.8), int(skin_tone[2]*0.8)), 3)
    cv2.circle(img, (cx, nose_y), 6, (int(skin_tone[0]*0.75), int(skin_tone[1]*0.75), int(skin_tone[2]*0.75)), -1)
    cv2.ellipse(img, (cx, mouth_y), (mouth_w, mouth_h), 0, 0, 180, (70, 70, 180), -1)

    if blur > 0:
        img = cv2.GaussianBlur(img, (blur * 2 + 1, blur * 2 + 1), 0)

    return img


class TestStudentRegistrationAndFaceRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare clean test environment and test face samples."""
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_CLASSIFIER_PATH):
            os.remove(TEST_CLASSIFIER_PATH)

        cls.db = AttendanceDatabase(db_path=TEST_DB_PATH)
        cls.fr_system = FaceRecognitionSystem(classifier_path=TEST_CLASSIFIER_PATH)
        cls.reg_manager = StudentRegistrationManager(db=cls.db, fr_system=cls.fr_system)

        # Generate distinct subject images
        # Alice (Fair skin, dark hair)
        cls.alice_img1 = generate_face_canvas(skin_tone=(190, 205, 235), hair_color=(25, 25, 25), eye_offset=26)
        cls.alice_img2 = generate_face_canvas(skin_tone=(195, 210, 240), hair_color=(25, 25, 25), eye_offset=27)
        cls.alice_test = generate_face_canvas(skin_tone=(185, 200, 230), hair_color=(25, 25, 25), eye_offset=26)

        # Bob (Warm skin tone, brown hair, larger eyes)
        cls.bob_img1 = generate_face_canvas(skin_tone=(140, 170, 215), hair_color=(20, 60, 110), eye_offset=30, eye_size=10)
        cls.bob_img2 = generate_face_canvas(skin_tone=(145, 175, 220), hair_color=(20, 60, 110), eye_offset=31, eye_size=10)
        cls.bob_img3 = generate_face_canvas(skin_tone=(135, 165, 210), hair_color=(20, 60, 110), eye_offset=30, eye_size=10)
        cls.bob_test = generate_face_canvas(skin_tone=(142, 172, 217), hair_color=(20, 60, 110), eye_offset=30, eye_size=10)

        # Unknown Person (Tan skin, blonde hair, narrow eyes)
        cls.unknown_img = generate_face_canvas(skin_tone=(120, 150, 190), hair_color=(80, 160, 210), eye_offset=20, eye_size=6)

        # Multi-Face frame: Alice and Bob side-by-side in one frame
        cls.multi_face_frame = np.full((360, 640, 3), (225, 225, 225), dtype=np.uint8)
        cls.multi_face_frame[60:300, 60:300] = cls.alice_test[40:280, 40:280]
        cls.multi_face_frame[60:300, 380:620] = cls.bob_test[40:280, 40:280]

        # No Face
        cls.no_face_img = np.full((320, 320, 3), (120, 150, 120), dtype=np.uint8)

        # Blurry Face
        cls.blurry_face_img = generate_face_canvas(blur=15)

    def test_01_register_new_student(self):
        """1. Register a new student successfully."""
        res = self.reg_manager.register_student(
            student_id="STU001",
            name="Alice Smith",
            images=[self.alice_img1, self.alice_img2],
            consent_given=True
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['student_id'], "STU001")
        self.assertEqual(res['accepted_count'], 2)
        self.assertEqual(self.db.get_student_count(), 1)

    def test_02_duplicate_student_id_rejection(self):
        """2. Reject duplicate student ID."""
        res = self.reg_manager.register_student(
            student_id="STU001",  # Same ID
            name="Duplicate Alice",
            images=[self.alice_img1],
            consent_given=True
        )
        self.assertFalse(res['success'])
        self.assertEqual(res['error_type'], 'DUPLICATE_ID')
        # Ensure count did not increase
        self.assertEqual(self.db.get_student_count(), 1)

    def test_03_multiple_images_per_student(self):
        """3. Support multiple registration images per student."""
        res = self.reg_manager.register_student(
            student_id="STU002",
            name="Bob Jones",
            images=[self.bob_img1, self.bob_img2, self.bob_img3],
            consent_given=True
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['accepted_count'], 3)
        self.assertEqual(self.db.get_student_count(), 2)

    def test_04_recognize_registered_student(self):
        """4. Recognize registered student on a separate test image."""
        preds = self.fr_system.predict_faces(self.alice_test, confidence_threshold=0.75)
        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertTrue(pred['is_known'])
        self.assertEqual(pred['student_id'], "STU001")
        self.assertEqual(pred['name'], "Alice Smith")
        self.assertGreaterEqual(pred['confidence'], 0.75)

    def test_05_reject_unknown_person(self):
        """5. Reject unregistered/unknown individual."""
        preds = self.fr_system.predict_faces(self.unknown_img, confidence_threshold=0.75)
        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertFalse(pred['is_known'])
        self.assertEqual(pred['student_id'], "UNKNOWN")
        self.assertEqual(pred['name'], "Unknown Student")
        # Ensure confidence is below threshold
        self.assertLess(pred['confidence'], 0.75)

    def test_06_recognize_multiple_faces_in_one_frame(self):
        """6. Recognize multiple registered students in a single frame."""
        preds = self.fr_system.predict_faces(self.multi_face_frame, confidence_threshold=0.75)
        self.assertEqual(len(preds), 2)
        
        found_ids = {p['student_id'] for p in preds}
        self.assertIn("STU001", found_ids)
        self.assertIn("STU002", found_ids)
        for p in preds:
            self.assertTrue(p['is_known'])
            self.assertGreaterEqual(p['confidence'], 0.75)

    def test_07_handle_images_without_faces(self):
        """7. Handle image without faces during registration and recognition."""
        # Registration rejection
        res = self.reg_manager.register_student(
            student_id="STU999",
            name="Ghost User",
            images=[self.no_face_img],
            consent_given=True
        )
        self.assertFalse(res['success'])
        self.assertEqual(res['error_type'], 'NO_VALID_FACES')

        # Recognition handling
        preds = self.fr_system.predict_faces(self.no_face_img)
        self.assertEqual(len(preds), 0)

    def test_08_handle_poor_quality_images(self):
        """8. Handle poor-quality images (reject blurry image)."""
        is_valid, msg, face = self.reg_manager.validate_image_for_registration(self.blurry_face_img)
        self.assertFalse(is_valid)
        self.assertIn("blurry", msg.lower())

    def test_09_saving_and_loading_data(self):
        """9. Persistence: reload DB and classifier from disk."""
        # Create fresh DB instance pointing to same file
        reloaded_db = AttendanceDatabase(db_path=TEST_DB_PATH)
        self.assertEqual(reloaded_db.get_student_count(), 2)

        stu1 = reloaded_db.get_student("STU001")
        self.assertIsNotNone(stu1)
        self.assertEqual(stu1['name'], "Alice Smith")

        # Create fresh FaceRecognitionSystem loading saved classifier
        reloaded_fr = FaceRecognitionSystem(classifier_path=TEST_CLASSIFIER_PATH)
        preds = reloaded_fr.predict_faces(self.bob_test, confidence_threshold=0.75)
        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0]['is_known'])
        self.assertEqual(preds[0]['student_id'], "STU002")

    def test_10_add_new_student_without_losing_existing_records(self):
        """10. Add new student without destroying existing registrations."""
        # Charlie (distinct features)
        charlie_img1 = generate_face_canvas(skin_tone=(150, 185, 230), hair_color=(10, 10, 10), eye_offset=24)
        charlie_img2 = generate_face_canvas(skin_tone=(155, 190, 235), hair_color=(10, 10, 10), eye_offset=24)
        
        res = self.reg_manager.register_student(
            student_id="STU003",
            name="Charlie Brown",
            images=[charlie_img1, charlie_img2],
            consent_given=True
        )
        self.assertTrue(res['success'])
        self.assertEqual(self.db.get_student_count(), 3)

        # Check Alice is STILL recognized
        preds_alice = self.fr_system.predict_faces(self.alice_test, confidence_threshold=0.75)
        self.assertTrue(preds_alice[0]['is_known'])
        self.assertEqual(preds_alice[0]['student_id'], "STU001")

        # Check Bob is STILL recognized
        preds_bob = self.fr_system.predict_faces(self.bob_test, confidence_threshold=0.75)
        self.assertTrue(preds_bob[0]['is_known'])
        self.assertEqual(preds_bob[0]['student_id'], "STU002")

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        import gc
        gc.collect()
        try:
            if os.path.exists(TEST_DB_PATH):
                os.remove(TEST_DB_PATH)
        except Exception:
            pass
        try:
            if os.path.exists(TEST_CLASSIFIER_PATH):
                os.remove(TEST_CLASSIFIER_PATH)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
