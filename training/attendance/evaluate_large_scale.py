"""
Large-Scale Face Recognition Benchmark and Evaluation Suite.
Uses the AT&T Olivetti Faces Benchmark Dataset (40 subjects, 400 real face photos).
Enrollment: 20 subjects x 4 photos = 80 images.
Testing: 240 completely held-out images:
  - 120 genuine known student queries (20 students x 6 unseen photos).
  - 120 unknown impostor queries (20 unregistered subjects x 6 photos).
Computes FAR, FRR, Precision, Recall, F1-Score, Confusion Matrix, and ROC/Threshold Analysis.
"""

import os
import sys
import numpy as np
import cv2

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import fetch_olivetti_faces
from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager

BENCH_DB_PATH = os.path.join("datasets", "attendance", "bench_attendance.db")
BENCH_CLASSIFIER_PATH = os.path.join("models", "attendance", "bench_classifier.pkl")


def prepare_face_image(img_raw: np.ndarray) -> np.ndarray:
    """Convert 64x64 float [0, 1] into 160x160 uint8 [0, 255] 3-channel BGR."""
    img64 = (img_raw * 255).astype(np.uint8)
    img_resized = cv2.resize(img64, (160, 160))
    return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)


def run_large_scale_benchmark():
    # Clean previous benchmark artifacts
    if os.path.exists(BENCH_DB_PATH):
        os.remove(BENCH_DB_PATH)
    if os.path.exists(BENCH_CLASSIFIER_PATH):
        os.remove(BENCH_CLASSIFIER_PATH)

    print("Loading AT&T Olivetti Faces Dataset...")
    dataset = fetch_olivetti_faces(data_home='datasets/attendance/scikit_learn_data', download_if_missing=False)
    images = dataset.images
    targets = dataset.target

    # Subjects 0 to 19: Enrolled Students (20 students)
    # Subjects 20 to 39: Unknown Individuals (20 individuals)
    enrolled_subject_ids = list(range(20))
    unknown_subject_ids = list(range(20, 40))

    db = AttendanceDatabase(db_path=BENCH_DB_PATH)
    fr = FaceRecognitionSystem(classifier_path=BENCH_CLASSIFIER_PATH)
    reg = StudentRegistrationManager(db=db, fr_system=fr)

    print("\n--- PHASE 1: ENROLLING 20 STUDENTS (4 Images Each = 80 Enrollment Photos) ---")
    enrolled_embeddings_count = 0
    for subj_id in enrolled_subject_ids:
        subj_indices = np.where(targets == subj_id)[0]
        # Strict partition: indices 0, 1, 2, 3 used for ENROLLMENT ONLY
        enrollment_indices = subj_indices[:4]
        enrollment_imgs = [prepare_face_image(images[idx]) for idx in enrollment_indices]

        student_id = f"STU_{subj_id:03d}"
        student_name = f"Student Subject #{subj_id:02d}"

        res = reg.register_student(
            student_id=student_id,
            name=student_name,
            images=enrollment_imgs,
            consent_given=True
        )
        if not res['success']:
            print(f"Failed to register {student_id}: {res['message']}")
        else:
            enrolled_embeddings_count += res['accepted_count']

    print(f"Successfully enrolled 20 students. Total stored embeddings: {enrolled_embeddings_count}")

    # Build Held-Out Test Sets (Indices 4, 5, 6, 7, 8, 9 for all subjects)
    # ZERO overlap with enrollment!
    print("\n--- PHASE 2: PREPARING HELD-OUT TEST DATASET (240 Images) ---")
    known_test_set = []  # List of (expected_student_id, image, subject_id, photo_idx)
    for subj_id in enrolled_subject_ids:
        subj_indices = np.where(targets == subj_id)[0]
        test_indices = subj_indices[4:]  # 6 unseen test photos
        for p_idx, idx in enumerate(test_indices, 4):
            img = prepare_face_image(images[idx])
            known_test_set.append((f"STU_{subj_id:03d}", img, subj_id, p_idx))

    unknown_test_set = []  # List of (image, subject_id, photo_idx)
    for subj_id in unknown_subject_ids:
        subj_indices = np.where(targets == subj_id)[0]
        test_indices = subj_indices[4:]  # 6 photos of unregistered individuals
        for p_idx, idx in enumerate(test_indices, 4):
            img = prepare_face_image(images[idx])
            unknown_test_set.append((img, subj_id, p_idx))

    print(f"Known Student Test Queries   : {len(known_test_set)} images (20 students x 6 photos)")
    print(f"Unknown Impostor Test Queries: {len(unknown_test_set)} images (20 unknown subjects x 6 photos)")
    print(f"Total Test Set Size          : {len(known_test_set) + len(unknown_test_set)} separate test images")

    # Extract all test predictions once to speed up threshold evaluation
    print("\nRunning inference forward pass across all 240 test queries...")
    known_results = []  # List of (expected_id, best_matched_id, similarity)
    for expected_id, img, s_id, p_idx in known_test_set:
        # Detect and match
        faces = fr.detect_faces(img, score_threshold=0.3)
        if not faces:
            known_results.append((expected_id, "NONE", 0.0))
            continue
        emb = fr.extract_embedding(img, faces[0])
        if emb is None:
            known_results.append((expected_id, "NONE", 0.0))
            continue
        sims = np.dot(fr.registered_embeddings, emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = fr.registered_ids[best_idx]
        known_results.append((expected_id, best_id, best_sim))

    unknown_results = []  # List of (best_matched_id, similarity)
    for img, s_id, p_idx in unknown_test_set:
        faces = fr.detect_faces(img, score_threshold=0.3)
        if not faces:
            unknown_results.append(("NONE", 0.0))
            continue
        emb = fr.extract_embedding(img, faces[0])
        if emb is None:
            unknown_results.append(("NONE", 0.0))
            continue
        sims = np.dot(fr.registered_embeddings, emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = fr.registered_ids[best_idx]
        unknown_results.append((best_id, best_sim))

    # Evaluate across a sweep of thresholds
    thresholds = [0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]
    print("\n=========================================================================================================")
    print("THRESHOLD SWEEP BENCHMARK (240 Separate Test Images: 120 Known, 120 Unknown)")
    print("=========================================================================================================")
    print(f"{'Threshold':>9s} | {'TP':>4s} | {'FN':>4s} | {'TN':>4s} | {'FP':>4s} | {'FAR (%)':>8s} | {'FRR (%)':>8s} | {'Precision':>9s} | {'Recall':>7s} | {'F1':>6s} | {'Accuracy':>8s}")
    print("---------------------------------------------------------------------------------------------------------")

    sweep_data = {}
    for thresh in thresholds:
        tp = 0
        fn = 0
        wrong_id_fp = 0
        for exp_id, best_id, sim in known_results:
            if sim >= thresh:
                if best_id == exp_id:
                    tp += 1
                else:
                    wrong_id_fp += 1
            else:
                fn += 1  # False Rejection

        tn = 0
        impostor_fp = 0
        for best_id, sim in unknown_results:
            if sim >= thresh:
                impostor_fp += 1  # False Acceptance
            else:
                tn += 1  # Correct Rejection

        fp = wrong_id_fp + impostor_fp
        total = tp + fn + tn + fp
        far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        frr = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = ((tp + tn) / total) * 100

        sweep_data[thresh] = {
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'far': far, 'frr': frr, 'prec': prec, 'rec': rec, 'f1': f1 / 100, 'acc': acc
        }
        print(f"{thresh:9.2f} | {tp:4d} | {fn:4d} | {tn:4d} | {fp:4d} | {far:7.2f}% | {frr:7.2f}% | {prec:8.2f}% | {rec:6.2f}% | {f1/100:6.4f} | {acc:7.2f}%")

    print("=========================================================================================================")

    # Select Optimal Operational Threshold (e.g., 0.48 - 0.50)
    opt_thresh = 0.48
    d = sweep_data[opt_thresh]
    print(f"\n======================================================")
    print(f"DETAILED METRICS AT OPERATIONAL THRESHOLD (tau = {opt_thresh:.2f})")
    print(f"======================================================")
    print(f"1. Confusion Matrix:")
    print(f"                   Actual Known    Actual Unknown")
    print(f"   Predicted Known   TP = {d['tp']:3d}          FP = {d['fp']:3d}")
    print(f"   Predicted Unknown FN = {d['fn']:3d}          TN = {d['tn']:3d}")
    print(f"------------------------------------------------------")
    print(f"2. Security & Verification Metrics:")
    print(f"   False Acceptance Rate (FAR) : {d['far']:.2f}%  ({d['fp']} impostors accepted)")
    print(f"   False Rejection Rate (FRR)  : {d['frr']:.2f}%  ({d['fn']} legitimate students rejected)")
    print(f"   Precision                   : {d['prec']:.2f}%")
    print(f"   Recall (Sensitivity)        : {d['rec']:.2f}%")
    print(f"   Specificity                 : {(d['tn'] / (d['tn'] + d['fp'])) * 100:.2f}%")
    print(f"   F1-Score                    : {d['f1']:.4f}")
    print(f"   Overall System Accuracy     : {d['acc']:.2f}%")
    print(f"======================================================")

    # Clean up benchmark artifacts
    import gc
    gc.collect()
    try:
        if os.path.exists(BENCH_DB_PATH):
            os.remove(BENCH_DB_PATH)
        if os.path.exists(BENCH_CLASSIFIER_PATH):
            os.remove(BENCH_CLASSIFIER_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    run_large_scale_benchmark()
