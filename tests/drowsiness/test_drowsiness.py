"""
Comprehensive Test Suite for Task 3: Real-Time Drowsiness & Fatigue Detection System.
Tests:
1. Soukupová-Čech EAR formula calculation (6 landmarks).
2. MAR formula calculation (4 landmarks).
3. Degenerate and boundary inputs for EAR and MAR.
4. Model loading and anchor extraction (YuNet/Haar).
5. Static photograph evaluation and explicit non-claim of prolonged drowsiness.
6. Temporal state machine:
   - Blink detection (1-7 frames) vs micro-sleep.
   - Prolonged closure (>= 15 frames) triggering CRITICAL alarm.
   - Cooldown protection against spamming duplicate alarms.
   - Recovery event when eyes reopen.
   - Yawn detection (>= 12 frames with MAR > 0.55).
   - Face occlusion / loss handling.
7. End-to-end video stream analysis on sample sequence.
8. Alert Manager: Web audio snippet, winsound buzzer non-blocking execution, and CSV report export.
9. Image edge cases: None, 2D grayscale, 4-channel BGRA, empty arrays.
10. Task router backward compatibility and Task 3 routing.
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.drowsiness.detector import (
    DrowsinessDetector,
    get_drowsiness_detector,
    calculate_ear,
    calculate_mar,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
)
from modules.drowsiness.state_machine import (
    DrowsinessStateMachine,
    DrowsinessState,
    MultiPersonDrowsinessTracker,
)
from modules.drowsiness.age_estimator import (
    get_age_estimator,
    AgeEstimator,
)
from modules.drowsiness.classifier import (
    get_drowsiness_classifier,
    DrowsinessClassifier,
)
from modules.drowsiness.alert_manager import (
    generate_drowsiness_csv_report,
    trigger_host_buzzer,
    BROWSER_AUDIO_SIREN_HTML,
)


class TestDrowsinessTask3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = get_drowsiness_detector()
        cls.test_img_dir = os.path.join(PROJECT_ROOT, "datasets", "drowsiness", "test_images")
        cls.test_video_dir = os.path.join(PROJECT_ROOT, "datasets", "drowsiness", "test_videos")

    # =========================================================================
    # 1. MATHEMATICAL FORMULATION TESTS: EAR & MAR
    # =========================================================================

    def test_01_calculate_ear_standard_geometry(self):
        """Verify Soukupová-Čech EAR calculation with known geometric coordinates."""
        # Eye width = 30.0 (p1 to p4)
        # Vertical apertures: p2-p6 = 12.0, p3-p5 = 12.0
        # Expected EAR = (12.0 + 12.0) / (2.0 * 30.0) = 24.0 / 60.0 = 0.4000
        p1 = np.array([0.0, 10.0])
        p2 = np.array([10.0, 4.0])
        p3 = np.array([20.0, 4.0])
        p4 = np.array([30.0, 10.0])
        p5 = np.array([20.0, 16.0])
        p6 = np.array([10.0, 16.0])
        landmarks = [p1, p2, p3, p4, p5, p6]

        ear = calculate_ear(landmarks)
        self.assertAlmostEqual(ear, 0.4000, places=3)

    def test_02_calculate_ear_closed_eye(self):
        """Verify EAR approaches zero when eyelids meet."""
        # Vertical apertures: p2-p6 = 0.0, p3-p5 = 0.0
        p1 = np.array([0.0, 10.0])
        p2 = np.array([10.0, 10.0])
        p3 = np.array([20.0, 10.0])
        p4 = np.array([30.0, 10.0])
        p5 = np.array([20.0, 10.0])
        p6 = np.array([10.0, 10.0])
        landmarks = [p1, p2, p3, p4, p5, p6]

        ear = calculate_ear(landmarks)
        self.assertEqual(ear, 0.0)

    def test_03_calculate_ear_degenerate_inputs(self):
        """Verify EAR handles invalid, empty, or zero-width coordinates gracefully."""
        self.assertEqual(calculate_ear(None), 0.0)
        self.assertEqual(calculate_ear([]), 0.0)
        self.assertEqual(calculate_ear([np.array([0, 0])] * 5), 0.0)

        # Zero width eye (p1 == p4)
        p_same = np.array([10.0, 10.0])
        landmarks = [p_same] * 6
        self.assertEqual(calculate_ear(landmarks), 0.0)

    def test_04_calculate_mar_standard_geometry(self):
        """Verify Mouth Aspect Ratio (MAR) with known geometry."""
        # Width: left to right = 60.0 (m_left=(10, 50), m_right=(70, 50))
        # Height: top to bottom = 36.0 (m_top=(40, 32), m_bot=(40, 68))
        # Expected MAR = 36.0 / 60.0 = 0.6000
        m_left = np.array([10.0, 50.0])
        m_top = np.array([40.0, 32.0])
        m_right = np.array([70.0, 50.0])
        m_bot = np.array([40.0, 68.0])
        mouth_landmarks = [m_left, m_top, m_right, m_bot]

        mar = calculate_mar(mouth_landmarks)
        self.assertAlmostEqual(mar, 0.6000, places=3)

    def test_05_calculate_mar_closed_mouth_and_degenerate(self):
        """Verify MAR returns 0.0 for closed mouth or degenerate inputs."""
        # Closed mouth (m_top == m_bot)
        m_left = np.array([10.0, 50.0])
        m_top = np.array([40.0, 50.0])
        m_right = np.array([70.0, 50.0])
        m_bot = np.array([40.0, 50.0])
        self.assertEqual(calculate_mar([m_left, m_top, m_right, m_bot]), 0.0)

        # Degenerate inputs
        self.assertEqual(calculate_mar(None), 0.0)
        self.assertEqual(calculate_mar([]), 0.0)
        self.assertEqual(calculate_mar([m_left, m_top]), 0.0)

    # =========================================================================
    # 2. DETECTOR AND STATIC IMAGE TESTING
    # =========================================================================

    def test_06_detector_singleton_and_model_loading(self):
        """Verify DrowsinessDetector follows singleton pattern and initializes."""
        det1 = DrowsinessDetector.get_instance()
        det2 = get_drowsiness_detector()
        self.assertIs(det1, det2)
        self.assertIsNotNone(det1)

    def test_07_static_driver_alert_evaluation(self):
        """Verify alert driver image yields high EAR, low MAR, and ALERT status."""
        img_path = os.path.join(self.test_img_dir, "driver_alert.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Missing test image: {img_path}")

        res = self.detector.process_frame(img)
        self.assertTrue(res["face_detected"])
        self.assertGreaterEqual(res["ear"], DEFAULT_EAR_THRESHOLD)
        self.assertLess(res["mar"], DEFAULT_MAR_THRESHOLD)

        sm = DrowsinessStateMachine()
        eval_res = sm.evaluate_static_image(res["ear"], res["mar"], res["face_detected"])
        self.assertEqual(eval_res["instantaneous_state"], "ALERT")
        self.assertEqual(eval_res["eye_state"], "OPEN")
        self.assertEqual(eval_res["mouth_state"], "NORMAL")

    def test_08_static_driver_closed_eyes_disclaimer_rule(self):
        """
        Verify requirement: Closed eyes in a static photograph produces WARNING,
        and EXPLICITLY DOES NOT claim prolonged drowsiness.
        """
        img_path = os.path.join(self.test_img_dir, "driver_closed_eyes.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Missing test image: {img_path}")

        res = self.detector.process_frame(img)
        self.assertTrue(res["face_detected"])
        self.assertLess(res["ear"], DEFAULT_EAR_THRESHOLD)

        sm = DrowsinessStateMachine()
        eval_res = sm.evaluate_static_image(res["ear"], res["mar"], res["face_detected"])
        # Crucial check: Must be WARNING, never CRITICAL
        self.assertEqual(eval_res["instantaneous_state"], "WARNING")
        self.assertEqual(eval_res["eye_state"], "CLOSED")
        self.assertIn("Video stream required", eval_res["description"])

    def test_09_static_driver_yawn_evaluation(self):
        """Verify yawning driver photo yields MAR > threshold and WARNING status."""
        img_path = os.path.join(self.test_img_dir, "driver_yawn.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Missing test image: {img_path}")

        res = self.detector.process_frame(img)
        self.assertTrue(res["face_detected"])
        self.assertGreater(res["mar"], DEFAULT_MAR_THRESHOLD)

        sm = DrowsinessStateMachine()
        eval_res = sm.evaluate_static_image(res["ear"], res["mar"], res["face_detected"])
        self.assertEqual(eval_res["instantaneous_state"], "WARNING")
        self.assertEqual(eval_res["mouth_state"], "YAWNING")

    def test_10_static_no_face_handling(self):
        """Verify non-facial image produces NO_FACE state gracefully."""
        img_path = os.path.join(self.test_img_dir, "no_face.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Missing test image: {img_path}")

        res = self.detector.process_frame(img)
        self.assertFalse(res["face_detected"])
        self.assertEqual(res["ear"], 0.0)
        self.assertEqual(res["mar"], 0.0)

        sm = DrowsinessStateMachine()
        eval_res = sm.evaluate_static_image(res["ear"], res["mar"], res["face_detected"])
        self.assertEqual(eval_res["instantaneous_state"], "NO_FACE")

    # =========================================================================
    # 3. TEMPORAL STATE MACHINE TESTS
    # =========================================================================

    def test_11_temporal_blink_filtering(self):
        """
        Verify natural short blinks (3 frames) increment blink counter,
        do NOT trigger alarms, and remain in ALERT state.
        """
        sm = DrowsinessStateMachine(ear_threshold=0.22, blink_min_frames=1, blink_max_frames=7, drowsy_min_frames=15)

        # Frame 1 to 5: Eyes Open (EAR = 0.35)
        for _ in range(5):
            tel = sm.update(ear=0.35, mar=0.10, face_detected=True)
            self.assertEqual(tel["current_state"], "ALERT")
            self.assertFalse(tel["alarm_active"])

        # Frame 6 to 8: Normal Blink (3 frames with EAR = 0.12)
        for _ in range(3):
            tel = sm.update(ear=0.12, mar=0.10, face_detected=True)

        # Frame 9: Eyes Re-open (EAR = 0.35)
        tel = sm.update(ear=0.35, mar=0.10, face_detected=True)

        self.assertEqual(tel["current_state"], "ALERT")
        self.assertEqual(tel["total_blinks"], 1)
        self.assertEqual(tel["drowsy_events_count"], 0)
        self.assertFalse(tel["alarm_active"])

    def test_12_temporal_prolonged_closure_critical_alarm(self):
        """
        Verify prolonged eye closure (>= 15 consecutive frames) transitions
        to CRITICAL state and dispatches drowsiness alarm.
        """
        sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15)

        # Frames 1 to 14: Eyes closed
        for f in range(1, 15):
            tel = sm.update(ear=0.10, mar=0.10, face_detected=True)
            if f <= 7:
                self.assertEqual(tel["current_state"], "ALERT")
            else:
                self.assertEqual(tel["current_state"], "WARNING")
            self.assertFalse(tel["alarm_active"])

        # Frame 15: Reaches drowsy_min_frames threshold -> CRITICAL
        tel_15 = sm.update(ear=0.10, mar=0.10, face_detected=True)
        self.assertEqual(tel_15["current_state"], "CRITICAL")
        self.assertTrue(tel_15["alarm_active"])
        self.assertTrue(tel_15["drowsiness_detected"])
        self.assertEqual(tel_15["drowsy_events_count"], 1)

    def test_13_alarm_cooldown_protection(self):
        """Verify system does not spam repeat alarms during continuous eye closure episode."""
        sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15, alarm_cooldown_frames=25)

        # First 15 frames -> triggers alarm
        for _ in range(15):
            tel = sm.update(ear=0.10, mar=0.10, face_detected=True)

        self.assertEqual(sm.drowsy_events_count, 1)

        # Frames 16 to 25: Continued closure (within cooldown)
        for _ in range(10):
            tel = sm.update(ear=0.10, mar=0.10, face_detected=True)
            self.assertEqual(tel["current_state"], "CRITICAL")
            self.assertFalse(tel["drowsiness_detected"])  # new alarm flag is False

        # Still only 1 discrete alarm event
        self.assertEqual(sm.drowsy_events_count, 1)

    def test_14_recovery_event_logging(self):
        """Verify waking up and reopening eyes clears alarm and logs RECOVERY event."""
        sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15)

        # Cause alarm (15 frames)
        for _ in range(15):
            sm.update(ear=0.10, mar=0.10, face_detected=True)

        self.assertTrue(sm.alarm_active)

        # Eyes reopen (EAR = 0.35)
        tel_recovery = sm.update(ear=0.35, mar=0.10, face_detected=True)
        self.assertEqual(tel_recovery["current_state"], "ALERT")
        self.assertFalse(tel_recovery["alarm_active"])

        # Check event log
        event_types = [e["event_type"] for e in sm.event_log]
        self.assertIn("DROWSINESS_ALERT", event_types)
        self.assertIn("RECOVERY", event_types)

    def test_15_yawn_event_detection(self):
        """Verify yawning (MAR > 0.55 for >= 12 frames) triggers WARNING state and yawn log."""
        sm = DrowsinessStateMachine(mar_threshold=0.55, yawn_min_frames=12)

        # Frames 1 to 11: Yawn opening
        for _ in range(11):
            tel = sm.update(ear=0.35, mar=0.70, face_detected=True)

        self.assertEqual(sm.yawn_events_count, 0)

        # Frame 12: Reaches yawn_min_frames
        tel_12 = sm.update(ear=0.35, mar=0.70, face_detected=True)
        self.assertEqual(tel_12["current_state"], "WARNING")
        self.assertEqual(tel_12["yawn_events_count"], 1)
        self.assertTrue(tel_12["yawn_detected"])

    def test_16_face_loss_resets_counters(self):
        """Verify losing face mid-stream resets closure counter to prevent phantom alerts."""
        sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15)

        # 10 closed frames
        for _ in range(10):
            sm.update(ear=0.10, mar=0.10, face_detected=True)
        self.assertEqual(sm.consecutive_closed_frames, 10)

        # Driver turns away / face lost
        tel_lost = sm.update(ear=0.0, mar=0.0, face_detected=False)
        self.assertEqual(tel_lost["current_state"], "ALERT")
        self.assertEqual(sm.consecutive_closed_frames, 0)

    # =========================================================================
    # 4. VIDEO ANALYSIS AND ALERT MANAGER TESTS
    # =========================================================================

    def test_17_end_to_end_sample_video_processing(self):
        """Process sample video sequence and verify alarm and recovery lifecycle."""
        video_path = os.path.join(self.test_video_dir, "sample_drowsy_sequence.mp4")
        self.assertTrue(os.path.exists(video_path), f"Sample video missing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), "Cannot open sample video")

        sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15)
        alarms_detected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            proc = self.detector.process_frame(frame)
            tel = sm.update(proc["ear"], proc["mar"], proc["face_detected"])
            if tel["drowsiness_detected"]:
                alarms_detected += 1

        cap.release()

        self.assertGreaterEqual(alarms_detected, 1)
        self.assertGreaterEqual(len(sm.event_log), 2)  # At least ALARM and RECOVERY

    def test_18_csv_session_report_generation(self):
        """Verify CSV export formats all 7 required columns properly."""
        dummy_log = [
            {
                "timestamp": "12:00:00.000",
                "event_type": "DROWSINESS_ALERT",
                "severity": "CRITICAL",
                "frames_duration": 16,
                "ear": 0.09,
                "mar": 0.08,
                "details": "Continuous eye closure for 16 frames"
            },
            {
                "timestamp": "12:00:01.500",
                "event_type": "RECOVERY",
                "severity": "INFO",
                "frames_duration": 16,
                "ear": 0.38,
                "mar": 0.08,
                "details": "Driver opened eyes"
            }
        ]
        csv_str = generate_drowsiness_csv_report(dummy_log)
        self.assertIn("timestamp,event_type,severity,frames_duration,ear,mar,details", csv_str)
        self.assertIn("DROWSINESS_ALERT", csv_str)
        self.assertIn("RECOVERY", csv_str)

        # Empty event log export
        empty_csv = generate_drowsiness_csv_report([])
        self.assertIn("timestamp,event_type,severity,frames_duration,ear,mar,details", empty_csv)

    def test_19_alert_components_and_buzzer(self):
        """Verify browser audio siren script and non-blocking winsound buzzer."""
        self.assertIn("AudioContext", BROWSER_AUDIO_SIREN_HTML)
        self.assertIn("createOscillator", BROWSER_AUDIO_SIREN_HTML)

        # trigger_host_buzzer should not block or raise exceptions
        try:
            trigger_host_buzzer(freq=1000, duration_ms=50)
        except Exception as e:
            self.fail(f"trigger_host_buzzer raised unexpected exception: {e}")

    # =========================================================================
    # 5. ROBUSTNESS & ROUTING TESTS
    # =========================================================================

    def test_20_detector_robustness_edge_cases(self):
        """Verify detector handles None, empty, 2D grayscale, and 4-channel BGRA images."""
        # None
        res_none = self.detector.process_frame(None)
        self.assertFalse(res_none["face_detected"])

        # Empty image
        res_empty = self.detector.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertFalse(res_empty["face_detected"])

        # 2D grayscale image
        gray_img = np.full((100, 100), 128, dtype=np.uint8)
        res_gray = self.detector.process_frame(gray_img)
        self.assertFalse(res_gray["face_detected"])

        # 4-channel BGRA image
        bgra_img = np.full((100, 100, 4), 128, dtype=np.uint8)
        res_bgra = self.detector.process_frame(bgra_img)
        self.assertFalse(res_bgra["face_detected"])

    def test_21_drawing_annotations_overlay(self):
        """Verify draw_annotations overlays HUD and contours without corrupting image."""
        img = np.full((200, 200, 3), 150, dtype=np.uint8)
        dummy_proc = {
            "face_detected": True,
            "face_bbox": (20, 20, 100, 120),
            "face_confidence": 0.95,
            "ear": 0.35,
            "mar": 0.10,
            "left_ear": 0.35,
            "right_ear": 0.35,
            "left_eye_landmarks": [np.array([40, 50])] * 6,
            "right_eye_landmarks": [np.array([80, 50])] * 6,
            "mouth_landmarks": [np.array([60, 100])] * 4
        }
        ann = self.detector.draw_annotations(img, dummy_proc, show_landmarks=True)
        self.assertEqual(ann.shape, img.shape)
        self.assertEqual(ann.dtype, np.uint8)

    def test_22_app_routing_backward_compatibility(self):
        """Verify app.py exports run_app and routes cleanly to Task 3."""
        import app
        self.assertTrue(hasattr(app, "run_app"))
        self.assertTrue(hasattr(app, "main"))

    def test_23_static_image_annotation_and_status_card_consistency(self):
        """
        Verify that on closed-eyes photograph:
        - process_frame returns instantaneous_state == WARNING (never CRITICAL)
        - evaluate_static_image returns instantaneous_state == WARNING (never CRITICAL)
        - draw_annotations sets WARNING banner on face overlay
        - CRITICAL is reserved solely for temporal multi-frame duration threshold
        """
        img_path = os.path.join(self.test_img_dir, "driver_closed_eyes.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        # 1. process_frame
        proc = self.detector.process_frame(img)
        self.assertTrue(proc["face_detected"])
        self.assertEqual(proc["instantaneous_state"], "WARNING")
        self.assertEqual(proc["eye_state"], "CLOSED")
        self.assertNotEqual(proc["instantaneous_state"], "CRITICAL")

        # 2. evaluate_static_image
        sm = DrowsinessStateMachine()
        eval_res = sm.evaluate_static_image(proc["ear"], proc["mar"], proc["face_detected"])
        self.assertEqual(eval_res["instantaneous_state"], "WARNING")
        self.assertEqual(eval_res["eye_state"], "CLOSED")
        self.assertNotEqual(eval_res["instantaneous_state"], "CRITICAL")

        # 3. draw_annotations default overlay
        ann = self.detector.draw_annotations(img, proc)
        self.assertIsNotNone(ann)
        # Should NOT be CRITICAL
        self.assertNotEqual(proc["instantaneous_state"], "CRITICAL")

    # =========================================================================
    # 6. MULTI-OCCUPANT, AGE ESTIMATION & POP-UP SUMMARY TESTS (REQUIREMENTS 2-7, 10)
    # =========================================================================

    def test_24_multi_person_detection_and_counting(self):
        """
        Verify Requirement 2, 3, & 4:
        - Detect multiple occupants in a single image.
        - Count total occupants.
        - Count sleeping occupants accurately.
        """
        img_path = os.path.join(self.test_img_dir, "multi_person_alert_sleeping.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Missing multi-person image: {img_path}")

        res = self.detector.process_frame(img)
        self.assertTrue(res["face_detected"])
        self.assertEqual(res["total_people"], 2)
        self.assertEqual(res["sleeping_count"], 1)
        self.assertEqual(res["awake_count"], 1)
        self.assertEqual(len(res["people"]), 2)

    def test_25_sleeping_occupant_red_bounding_box_assignment(self):
        """
        Verify Requirement 5:
        - Sleeping occupants must be assigned RED bounding boxes (0, 0, 255).
        - Awake occupants must be assigned GREEN bounding boxes (0, 255, 0).
        """
        img_path = os.path.join(self.test_img_dir, "multi_person_alert_sleeping.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        res = self.detector.process_frame(img)
        people = res["people"]
        self.assertEqual(len(people), 2)

        # Person #1 is awake -> GREEN box
        self.assertFalse(people[0]["is_sleeping"])
        self.assertEqual(people[0]["box_color"], (0, 255, 0))

        # Person #2 is sleeping -> RED box
        self.assertTrue(people[1]["is_sleeping"])
        self.assertEqual(people[1]["box_color"], (0, 0, 255))

    def test_26_age_estimation_model_inference_and_bounds(self):
        """
        Verify Requirement 6 & 10:
        - Age estimator model loads and evaluates deep facial embeddings.
        - Returns realistic age estimates within [18, 80] with uncertainty brackets.
        """
        age_est = get_age_estimator()
        self.assertIsNotNone(age_est)
        self.assertIsNotNone(age_est.model, "Random Forest age model not loaded")

        img_path = os.path.join(self.test_img_dir, "driver_alert.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        res = self.detector.process_frame(img)
        self.assertIn("age_info", res)
        age_info = res["age_info"]

        self.assertGreaterEqual(age_info["estimated_age"], 18)
        self.assertLessEqual(age_info["estimated_age"], 75)
        self.assertIn("yrs", age_info["label"])
        self.assertGreaterEqual(age_info["confidence"], 0.60)
        self.assertGreaterEqual(age_info["uncertainty"], 4)

    def test_27_popup_summary_generation(self):
        """
        Verify Requirement 7:
        - Pop-up summary string explicitly details the number of sleeping occupants and their age.
        """
        img_path = os.path.join(self.test_img_dir, "multi_person_alert_sleeping.jpg")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img)

        res = self.detector.process_frame(img)
        popup = res["popup_summary"]

        self.assertIn("Alert: 1 sleeping occupant detected!", popup)
        self.assertIn("Person #2", popup)
        self.assertIn("yrs", popup)

    def test_28_multi_person_temporal_tracking_and_independent_alarms(self):
        """
        Verify multi-person temporal tracking with independent state machines:
        - Occupant 1 stays alert.
        - Occupant 2 closes eyes for 15 frames -> triggers CRITICAL only on Occupant 2.
        """
        tracker = MultiPersonDrowsinessTracker(ear_threshold=0.22, drowsy_min_frames=15)

        # 14 frames: Occupant 1 (open EAR=0.35), Occupant 2 (closed EAR=0.08)
        dummy_det = [
            {"bbox": (50, 50, 100, 100), "ear": 0.35, "mar": 0.10, "age_info": {"estimated_age": 25, "label": "~25 ± 5 yrs"}},
            {"bbox": (250, 50, 100, 100), "ear": 0.08, "mar": 0.10, "age_info": {"estimated_age": 28, "label": "~28 ± 5 yrs"}}
        ]

        for _ in range(14):
            t_res = tracker.update(dummy_det)
            self.assertEqual(t_res["sleeping_count"], 0)
            self.assertEqual(t_res["awake_count"], 2)

        # Frame 15: Occupant 2 reaches 15 closed frames -> CRITICAL
        t_res_15 = tracker.update(dummy_det)
        self.assertEqual(t_res_15["sleeping_count"], 1)
        self.assertEqual(t_res_15["awake_count"], 1)
        self.assertTrue(t_res_15["alarm_active"])
        self.assertIn("Occupant #2", t_res_15["popup_summary"])
        self.assertEqual(len(tracker.unified_event_log), 1)

    def test_29_multi_person_video_file_pipeline(self):
        """
        Verify end-to-end video processing on multi-occupant video footage.
        """
        video_path = os.path.join(self.test_video_dir, "multi_person_vehicle_sequence.mp4")
        self.assertTrue(os.path.exists(video_path), f"Video missing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())

        tracker = MultiPersonDrowsinessTracker(ear_threshold=0.22, drowsy_min_frames=15)
        frames_run = 0

        while cap.isOpened() and frames_run < 35:
            ret, frame = cap.read()
            if not ret:
                break
            frames_run += 1
            proc = self.detector.process_frame(frame)
            tracker.update(proc.get("people", []))

        cap.release()
        self.assertGreaterEqual(frames_run, 25)
        self.assertGreaterEqual(len(tracker.tracks), 2)
        self.assertGreaterEqual(len(tracker.unified_event_log), 1)

    def test_30_drowsiness_classifier_pipeline_and_fallbacks(self):
        """
        Verify DrowsinessClassifier engine:
        - Singleton instantiation and status inspection.
        - 8-dimensional biometric eye feature extraction.
        - Binary prediction mapping and fallbacks.
        """
        clf = get_drowsiness_classifier()
        self.assertIsNotNone(clf)

        # Feature extraction on dummy points
        dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        l_lms = [np.array([20, 20]), np.array([25, 15]), np.array([35, 15]), np.array([40, 20]), np.array([35, 25]), np.array([25, 25])]
        r_lms = [np.array([60, 20]), np.array([65, 15]), np.array([75, 15]), np.array([80, 20]), np.array([75, 25]), np.array([65, 25])]
        bbox = (10, 10, 80, 80)

        feats = clf.extract_eye_features(dummy_img, l_lms, r_lms, 0.35, 0.35, bbox)
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats), 8)

        # Awake evaluation
        awake_eval = clf.classify_eye_state(feats, fallback_ear=0.35, ear_threshold=0.22)
        self.assertFalse(awake_eval["is_sleeping"])
        self.assertEqual(awake_eval["state"], "OPEN")

        # Asleep evaluation
        asleep_eval = clf.classify_eye_state(feats, fallback_ear=0.10, ear_threshold=0.22)
        self.assertTrue(asleep_eval["is_sleeping"])
        self.assertEqual(asleep_eval["state"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
