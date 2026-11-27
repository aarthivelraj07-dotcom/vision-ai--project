"""
End-to-End Functional Verification of Application Capabilities:
1. Image upload and webcam frame ingestion
2. Registration and Duplicate ID prevention
3. Recognition of registered vs unknown persons
4. Multi-face recognition in a single frame
5. Persistence across cold restart
"""

import os
import sys
import unittest
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import fetch_olivetti_faces
from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager
from training.attendance.evaluate_large_scale import prepare_face_image

E2E_DB = os.path.join("datasets", "attendance", "e2e_test.db")
E2E_PKL = os.path.join("models", "attendance", "e2e_test.pkl")


class TestApplicationE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(E2E_DB):
            os.remove(E2E_DB)
        if os.path.exists(E2E_PKL):
            os.remove(E2E_PKL)

        dataset = fetch_olivetti_faces(data_home='datasets/attendance/scikit_learn_data', download_if_missing=False)
        cls.img_alice_1 = prepare_face_image(dataset.images[0])
        cls.img_alice_2 = prepare_face_image(dataset.images[1])
        cls.img_alice_test = prepare_face_image(dataset.images[4])

        cls.img_bob_1 = prepare_face_image(dataset.images[10])
        cls.img_bob_2 = prepare_face_image(dataset.images[11])
        cls.img_bob_test = prepare_face_image(dataset.images[14])

        cls.img_unknown = prepare_face_image(dataset.images[300])

        # Composite multi-face image: Alice on left, Bob on right
        h, w = cls.img_alice_test.shape[:2]
        cls.multi_face_frame = np.full((h + 40, w * 2 + 60, 3), (230, 230, 230), dtype=np.uint8)
        cls.multi_face_frame[20:20+h, 20:20+w] = cls.img_alice_test
        cls.multi_face_frame[20:20+h, 40+w:40+2*w] = cls.img_bob_test

    def test_e2e_flow(self):
        # 1. Initialize subsystem
        db = AttendanceDatabase(db_path=E2E_DB)
        fr = FaceRecognitionSystem(classifier_path=E2E_PKL)
        reg = StudentRegistrationManager(db=db, fr_system=fr)

        # 2. Ingestion / Upload validation
        valid, msg, face = reg.validate_image_for_registration(self.img_alice_1)
        self.assertTrue(valid, f"Alice image 1 should be valid: {msg}")

        # 3. Registration
        res_a = reg.register_student("STU_201", "Alice Wonder", [self.img_alice_1, self.img_alice_2], consent_given=True)
        self.assertTrue(res_a['success'])
        self.assertEqual(res_a['accepted_count'], 2)

        res_b = reg.register_student("STU_202", "Bob Builder", [self.img_bob_1, self.img_bob_2], consent_given=True)
        self.assertTrue(res_b['success'])

        # 4. Duplicate ID Rejection
        res_dup = reg.register_student("STU_201", "Impostor Alice", [self.img_alice_1], consent_given=True)
        self.assertFalse(res_dup['success'])
        self.assertEqual(res_dup['error_type'], 'DUPLICATE_ID')

        # 5. Recognition of registered student
        preds_a = fr.predict_faces(self.img_alice_test, confidence_threshold=0.50)
        self.assertEqual(len(preds_a), 1)
        self.assertTrue(preds_a[0]['is_known'])
        self.assertEqual(preds_a[0]['student_id'], "STU_201")
        self.assertEqual(preds_a[0]['name'], "Alice Wonder")

        # 6. Rejection of unknown person
        preds_u = fr.predict_faces(self.img_unknown, confidence_threshold=0.50)
        self.assertEqual(len(preds_u), 1)
        self.assertFalse(preds_u[0]['is_known'])
        self.assertEqual(preds_u[0]['student_id'], "UNKNOWN")
        self.assertEqual(preds_u[0]['name'], "Unknown Student")

        # 7. Multiple-Face Recognition
        preds_multi = fr.predict_faces(self.multi_face_frame, confidence_threshold=0.50)
        self.assertEqual(len(preds_multi), 2)
        identified_ids = {p['student_id'] for p in preds_multi}
        self.assertEqual(identified_ids, {"STU_201", "STU_202"})
        for p in preds_multi:
            self.assertTrue(p['is_known'])
            self.assertGreaterEqual(p['confidence'], 0.50)

        # 8. Persistence across cold restart
        del reg
        del fr
        del db

        db_reloaded = AttendanceDatabase(db_path=E2E_DB)
        fr_reloaded = FaceRecognitionSystem(classifier_path=E2E_PKL)
        self.assertEqual(db_reloaded.get_student_count(), 2)
        p_reloaded = fr_reloaded.predict_faces(self.img_bob_test, confidence_threshold=0.50)
        self.assertEqual(len(p_reloaded), 1)
        self.assertTrue(p_reloaded[0]['is_known'])
        self.assertEqual(p_reloaded[0]['student_id'], "STU_202")

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        try:
            if os.path.exists(E2E_DB):
                os.remove(E2E_DB)
        except Exception:
            pass
        try:
            if os.path.exists(E2E_PKL):
                os.remove(E2E_PKL)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
