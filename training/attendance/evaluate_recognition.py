"""
Evaluation and Metrics Benchmark for Face Recognition Subsystem.
Evaluates recognition accuracy, open-set unknown rejection, precision, recall, and F1-score
using distinct held-out test face sets.
"""

import os
import sys
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager
from tests.attendance.test_phase2 import generate_face_canvas

EVAL_DB_PATH = os.path.join("datasets", "attendance", "eval_attendance.db")
EVAL_CLASSIFIER_PATH = os.path.join("models", "attendance", "eval_classifier.pkl")


def run_threshold_sweep():
    print("==========================================================================")
    print("THRESHOLD SWEEP BENCHMARK: PRECISION, RECALL, SPECIFICITY, ACCURACY")
    print("==========================================================================")
    print(f"{'Threshold':>10s} | {'TP':>4s} | {'TN':>4s} | {'FP':>4s} | {'FN':>4s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>7s} | {'F1-Score':>8s}")
    print("--------------------------------------------------------------------------")
    for thresh in [0.75, 0.80, 0.82, 0.85, 0.88, 0.90]:
        # Run sweep
        run_evaluation(confidence_threshold=thresh, verbose=False)
    print("==========================================================================")


def run_evaluation(confidence_threshold: float = 0.85, verbose: bool = True):
    if os.path.exists(EVAL_DB_PATH):
        os.remove(EVAL_DB_PATH)
    if os.path.exists(EVAL_CLASSIFIER_PATH):
        os.remove(EVAL_CLASSIFIER_PATH)

    db = AttendanceDatabase(db_path=EVAL_DB_PATH)
    fr = FaceRecognitionSystem(classifier_path=EVAL_CLASSIFIER_PATH)
    reg = StudentRegistrationManager(db=db, fr_system=fr)

    # 1. Register 5 distinct students with 3 enrollment images each
    students_config = [
        {"id": "STU101", "name": "Aarav Sharma", "skin": (145, 175, 220), "hair": (20, 20, 20), "eye_off": 24, "eye_sz": 8},
        {"id": "STU102", "name": "Bhavna Patel", "skin": (160, 190, 235), "hair": (30, 25, 25), "eye_off": 27, "eye_sz": 7},
        {"id": "STU103", "name": "Chetan Kumar", "skin": (130, 160, 210), "hair": (15, 45, 80), "eye_off": 29, "eye_sz": 9},
        {"id": "STU104", "name": "Divya Rao", "skin": (175, 205, 245), "hair": (40, 30, 30), "eye_off": 25, "eye_sz": 8},
        {"id": "STU105", "name": "Eshan Verma", "skin": (138, 168, 215), "hair": (25, 25, 25), "eye_off": 28, "eye_sz": 10},
    ]

    for s in students_config:
        imgs = [
            generate_face_canvas(skin_tone=s["skin"], hair_color=s["hair"], eye_offset=s["eye_off"], eye_size=s["eye_sz"]),
            generate_face_canvas(skin_tone=s["skin"], hair_color=s["hair"], eye_offset=s["eye_off"]+1, eye_size=s["eye_sz"]),
            generate_face_canvas(skin_tone=s["skin"], hair_color=s["hair"], eye_offset=s["eye_off"]-1, eye_size=s["eye_sz"])
        ]
        res = reg.register_student(s["id"], s["name"], imgs, consent_given=True)
        if not res['success']:
            print(f"Error registering {s['name']}: {res['message']}")

    # 2. Generate Separate Held-out Test Set
    # 10 Known Test Queries (2 novel test variations per registered student)
    known_queries = []
    for s in students_config:
        t1 = generate_face_canvas(skin_tone=s["skin"], hair_color=s["hair"], eye_offset=s["eye_off"], eye_size=s["eye_sz"], blur=1)
        t2 = generate_face_canvas(skin_tone=s["skin"], hair_color=s["hair"], eye_offset=s["eye_off"]+1, eye_size=s["eye_sz"], blur=2)
        known_queries.append((s["id"], t1))
        known_queries.append((s["id"], t2))

    # 10 Unknown / Impostor Queries (unregistered individuals with novel features)
    unknown_queries = [
        generate_face_canvas(skin_tone=(110, 140, 185), hair_color=(80, 160, 220), eye_offset=20, eye_size=6),
        generate_face_canvas(skin_tone=(180, 170, 190), hair_color=(90, 40, 30), eye_offset=32, eye_size=11),
        generate_face_canvas(skin_tone=(115, 145, 190), hair_color=(60, 120, 180), eye_offset=21, eye_size=6),
        generate_face_canvas(skin_tone=(195, 215, 250), hair_color=(120, 120, 120), eye_offset=33, eye_size=7),
        generate_face_canvas(skin_tone=(105, 135, 180), hair_color=(10, 10, 10), eye_offset=19, eye_size=5),
        generate_face_canvas(skin_tone=(150, 150, 210), hair_color=(70, 70, 70), eye_offset=22, eye_size=7),
        generate_face_canvas(skin_tone=(125, 155, 200), hair_color=(100, 180, 240), eye_offset=34, eye_size=10),
        generate_face_canvas(skin_tone=(165, 180, 225), hair_color=(50, 20, 10), eye_offset=23, eye_size=6),
        generate_face_canvas(skin_tone=(135, 145, 195), hair_color=(40, 80, 120), eye_offset=31, eye_size=11),
        generate_face_canvas(skin_tone=(170, 195, 235), hair_color=(30, 70, 100), eye_offset=26, eye_size=9)
    ]

    tp = 0  # Registered student recognized correctly
    fp = 0  # Unknown person recognized as student or wrong student
    tn = 0  # Unknown person correctly rejected as Unknown
    fn = 0  # Registered student rejected as Unknown

    known_scores = []
    unknown_scores = []

    # Evaluate Known Queries
    for expected_id, img in known_queries:
        preds = fr.predict_faces(img, confidence_threshold=confidence_threshold)
        if preds and preds[0]['is_known'] and preds[0]['student_id'] == expected_id:
            tp += 1
            known_scores.append(preds[0]['confidence'])
        else:
            fn += 1
            if preds:
                known_scores.append(preds[0]['confidence'])

    # Evaluate Unknown Queries
    for img in unknown_queries:
        preds = fr.predict_faces(img, confidence_threshold=confidence_threshold)
        if not preds or not preds[0]['is_known']:
            tn += 1
            if preds:
                unknown_scores.append(preds[0]['confidence'])
        else:
            fp += 1
            unknown_scores.append(preds[0]['confidence'])

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if verbose:
        print("======================================================")
        print("FACE RECOGNITION BENCHMARK EVALUATION RESULTS")
        print("======================================================")
        print(f"Total Test Instances        : {total}")
        print(f"Enrolled Students           : {len(students_config)} (3 face embeddings each)")
        print(f"Confidence Threshold        : {confidence_threshold:.2f}")
        print("------------------------------------------------------")
        print(f"True Positives (TP)         : {tp:2d} / 10  (Registered recognized)")
        print(f"True Negatives (TN)         : {tn:2d} / 10  (Unknown correctly rejected)")
        print(f"False Positives (FP)        : {fp:2d} / 10  (Impostor false acceptance)")
        print(f"False Negatives (FN)        : {fn:2d} / 10  (Registered false rejection)")
        print("------------------------------------------------------")
        print(f"Classification Accuracy     : {accuracy * 100:.2f}%")
        print(f"Precision                   : {precision * 100:.2f}%")
        print(f"Recall (Sensitivity)        : {recall * 100:.2f}%")
        print(f"Specificity                 : {specificity * 100:.2f}%")
        print(f"F1-Score                    : {f1:.4f}")
        print("------------------------------------------------------")
        if known_scores:
            print(f"Mean Known Cosine Similarity: {np.mean(known_scores):.4f} (Min: {np.min(known_scores):.4f})")
        if unknown_scores:
            print(f"Mean Unknown Cosine Score   : {np.mean(unknown_scores):.4f} (Max: {np.max(unknown_scores):.4f})")
        print("======================================================")
    else:
        print(f"{confidence_threshold:10.2f} | {tp:4d} | {tn:4d} | {fp:4d} | {fn:4d} | {accuracy*100:7.1f}% | {precision*100:8.1f}% | {recall*100:6.1f}% | {f1:8.4f}")

    # Cleanup
    import gc
    gc.collect()
    try:
        if os.path.exists(EVAL_DB_PATH):
            os.remove(EVAL_DB_PATH)
        if os.path.exists(EVAL_CLASSIFIER_PATH):
            os.remove(EVAL_CLASSIFIER_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    run_threshold_sweep()
    print("\nDetailed breakdown at optimal threshold (0.85):")
    run_evaluation(confidence_threshold=0.85, verbose=True)
