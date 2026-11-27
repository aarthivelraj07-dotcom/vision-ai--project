"""
Student Registration Pipeline.
Handles validation of student ID, biometric consent verification, multi-image face detection,
quality filtering, biometric embedding extraction, and persistence in SQLite.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem

FACES_DIR = os.path.join("datasets", "attendance", "registered_faces")


class StudentRegistrationManager:
    def __init__(self, db: Optional[AttendanceDatabase] = None, fr_system: Optional[FaceRecognitionSystem] = None):
        self.db = db if db is not None else AttendanceDatabase()
        self.fr_system = fr_system if fr_system is not None else FaceRecognitionSystem()
        os.makedirs(FACES_DIR, exist_ok=True)

    def validate_image_for_registration(self, image: np.ndarray) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate a single image before registration.
        Must contain exactly one clear, high-quality face.
        """
        if image is None or image.size == 0:
            return False, "Empty or invalid image data.", None

        faces = self.fr_system.detect_faces(image)
        if len(faces) == 0:
            return False, "No face detected. Ensure face is clearly visible to the camera.", None

        if len(faces) > 1:
            return False, f"Multiple faces ({len(faces)}) detected. Registration image must contain exactly one person.", None

        face = faces[0]
        is_valid, quality_msg = self.fr_system.validate_face_quality(image, face)
        if not is_valid:
            return False, f"Face quality check failed: {quality_msg}", None

        return True, "Valid face detected.", face

    def register_student(
        self,
        student_id: str,
        name: str,
        images: List[np.ndarray],
        consent_given: bool = True
    ) -> Dict[str, Any]:
        """
        Register a student with multiple consented face images.
        Extracts embeddings, persists student info and embeddings in SQLite,
        and re-trains the face classifier.
        """
        student_id = student_id.strip()
        name = name.strip()

        # 1. Input Validation
        if not student_id:
            return {'success': False, 'message': "Student ID cannot be empty.", 'error_type': 'EMPTY_ID'}
        if not name:
            return {'success': False, 'message': "Student Name cannot be empty.", 'error_type': 'EMPTY_NAME'}
        if not consent_given:
            return {'success': False, 'message': "Biometric consent is mandatory before registering face data.", 'error_type': 'NO_CONSENT'}

        # 2. Duplicate ID Check
        if self.db.student_exists(student_id):
            return {
                'success': False,
                'message': f"Student ID '{student_id}' already exists in the system. Duplicate IDs are rejected.",
                'error_type': 'DUPLICATE_ID'
            }

        if not images:
            return {'success': False, 'message': "Please provide at least one face image for registration.", 'error_type': 'NO_IMAGES'}

        # 3. Process Each Image
        valid_embeddings = []
        accepted_count = 0
        rejected_count = 0
        rejection_reasons = []

        student_face_dir = os.path.join(FACES_DIR, student_id)
        os.makedirs(student_face_dir, exist_ok=True)

        for idx, img in enumerate(images, 1):
            is_valid, msg, face_info = self.validate_image_for_registration(img)
            if not is_valid:
                rejected_count += 1
                rejection_reasons.append(f"Image #{idx}: {msg}")
                continue

            embedding = self.fr_system.extract_embedding(img, face_info)
            if embedding is None:
                rejected_count += 1
                rejection_reasons.append(f"Image #{idx}: Could not extract facial features.")
                continue

            valid_embeddings.append(embedding)
            accepted_count += 1

            # Save aligned face crop locally for audit & verification
            raw_face = face_info.get('raw_face')
            if raw_face is not None and self.fr_system.recognizer is not None:
                aligned = self.fr_system.recognizer.alignCrop(img, raw_face)
                save_path = os.path.join(student_face_dir, f"sample_{accepted_count}.jpg")
                cv2.imwrite(save_path, aligned)

        # 4. Enforce minimum valid images
        if accepted_count == 0:
            return {
                'success': False,
                'message': f"No valid faces could be registered. Issues: {'; '.join(rejection_reasons)}",
                'accepted_count': 0,
                'rejected_count': rejected_count,
                'reasons': rejection_reasons,
                'error_type': 'NO_VALID_FACES'
            }

        # 5. Save Student and Embeddings in SQLite
        try:
            self.db.add_student(student_id, name, valid_embeddings, consent_given=consent_given)
        except Exception as e:
            return {
                'success': False,
                'message': f"Database error during registration: {str(e)}",
                'error_type': 'DB_ERROR'
            }

        # 6. Re-train / update Face Recognition Classifier
        all_embeddings, all_ids, names_map = self.db.get_all_embeddings()
        self.fr_system.train_classifier(all_embeddings, all_ids, names_map)

        return {
            'success': True,
            'student_id': student_id,
            'name': name,
            'accepted_count': accepted_count,
            'rejected_count': rejected_count,
            'reasons': rejection_reasons,
            'message': f"Successfully registered {name} (ID: {student_id}) with {accepted_count} valid face embedding(s)."
        }
