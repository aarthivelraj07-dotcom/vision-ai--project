"""
Persistence and Application Restart Verification Test.
Tests that student identity records, biometric embeddings, and classifier state
persist across application restarts and process boundaries.
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import fetch_olivetti_faces
from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager
from training.attendance.evaluate_large_scale import prepare_face_image

PERSIST_DB = os.path.join("datasets", "attendance", "persist_test.db")
PERSIST_PKL = os.path.join("models", "attendance", "persist_test.pkl")


class TestPersistenceAcrossRestart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(PERSIST_DB):
            os.remove(PERSIST_DB)
        if os.path.exists(PERSIST_PKL):
            os.remove(PERSIST_PKL)

        # Load 2 subjects from Olivetti dataset
        dataset = fetch_olivetti_faces(data_home='datasets/attendance/scikit_learn_data', download_if_missing=False)
        cls.img_alice_1 = prepare_face_image(dataset.images[0])
        cls.img_alice_2 = prepare_face_image(dataset.images[1])
        cls.img_alice_test = prepare_face_image(dataset.images[4])

        cls.img_bob_1 = prepare_face_image(dataset.images[10])
        cls.img_bob_2 = prepare_face_image(dataset.images[11])
        cls.img_bob_test = prepare_face_image(dataset.images[14])

    def test_01_session_1_enrollment(self):
        """Session 1: Register students and store biometrics."""
        db1 = AttendanceDatabase(db_path=PERSIST_DB)
        fr1 = FaceRecognitionSystem(classifier_path=PERSIST_PKL)
        reg1 = StudentRegistrationManager(db=db1, fr_system=fr1)

        res_a = reg1.register_student("STU_A", "Alice Student", [self.img_alice_1, self.img_alice_2], consent_given=True)
        self.assertTrue(res_a['success'])

        res_b = reg1.register_student("STU_B", "Bob Student", [self.img_bob_1, self.img_bob_2], consent_given=True)
        self.assertTrue(res_b['success'])

        self.assertEqual(db1.get_student_count(), 2)
        # Test recognition in Session 1
        p_a = fr1.predict_faces(self.img_alice_test, confidence_threshold=0.50)
        self.assertTrue(p_a[0]['is_known'])
        self.assertEqual(p_a[0]['student_id'], "STU_A")
        self.assertAlmostEqual(p_a[0]['confidence'], 0.85, delta=0.15)

        # Store baseline similarity for session 2 comparison
        self.__class__.baseline_conf = p_a[0]['confidence']

        # Simulate total application shutdown: delete instances
        del reg1
        del fr1
        del db1

    def test_02_session_2_cold_restart(self):
        """Session 2: Cold restart from clean process state."""
        # Open completely fresh database connection
        db2 = AttendanceDatabase(db_path=PERSIST_DB)
        self.assertEqual(db2.get_student_count(), 2)

        # Verify Alice record
        stu_a = db2.get_student("STU_A")
        self.assertIsNotNone(stu_a)
        self.assertEqual(stu_a['name'], "Alice Student")
        self.assertEqual(stu_a['face_count'], 2)
        self.assertEqual(stu_a['consent_given'], 1)

        # Verify Bob record
        stu_b = db2.get_student("STU_B")
        self.assertIsNotNone(stu_b)
        self.assertEqual(stu_b['name'], "Bob Student")
        self.assertEqual(stu_b['face_count'], 2)

        # Verify 128-D raw embeddings persistence
        X, ids, names = db2.get_all_embeddings()
        self.assertEqual(X.shape, (4, 128))
        self.assertEqual(len(ids), 4)
        self.assertIn("STU_A", ids)
        self.assertIn("STU_B", ids)

        # Verify SFace recognizer cold load
        fr2 = FaceRecognitionSystem(classifier_path=PERSIST_PKL)
        self.assertTrue(fr2.load_classifier())
        self.assertEqual(len(fr2.registered_ids), 4)

        # Predict with reloaded model
        preds = fr2.predict_faces(self.img_alice_test, confidence_threshold=0.50)
        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0]['is_known'])
        self.assertEqual(preds[0]['student_id'], "STU_A")
        self.assertEqual(preds[0]['name'], "Alice Student")
        # Ensure exact match with baseline confidence
        self.assertAlmostEqual(preds[0]['confidence'], self.__class__.baseline_conf, places=5)

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        try:
            if os.path.exists(PERSIST_DB):
                os.remove(PERSIST_DB)
        except Exception:
            pass
        try:
            if os.path.exists(PERSIST_PKL):
                os.remove(PERSIST_PKL)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
