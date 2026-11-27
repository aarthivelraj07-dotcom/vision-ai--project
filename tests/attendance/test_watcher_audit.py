"""
Runtime Audit and Stress Test Suite for BackgroundSessionWatcher & Database Concurrency.

Focuses specifically on:
1. Thread lifecycle and singleton guarantee: Multiple AttendanceManager instances and Streamlit sessions cannot spawn duplicate watcher threads.
2. Autonomous background finalization without any user interaction or camera frame input.
3. SQLite thread safety, connection isolation, transaction safety, and concurrent finalization stress.
4. Cold restart recovery when stopped before 10:00 AM and restarted after 10:00 AM.
5. Watcher repetition idempotency: repeated checks cannot duplicate records or corrupt data.
6. Strict rejection of simulated time in production mode (allow_simulation=False).
"""

import os
import sys
import time
import threading
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.attendance_manager import (
    AttendanceManager,
    AttendanceTimeManager,
    BackgroundSessionWatcher,
    KOLKATA_TZ
)
from modules.attendance.registration import StudentRegistrationManager
from tests.attendance.test_phase2 import generate_face_canvas

AUDIT_DB = os.path.join("datasets", "attendance", "test_audit_attendance.db")
AUDIT_PKL = os.path.join("models", "attendance", "test_audit_classifier.pkl")


class TestBackgroundSessionWatcherRuntimeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare fresh isolated database and enrolled test students once."""
        BackgroundSessionWatcher.stop_singleton()
        if os.path.exists(AUDIT_DB):
            try:
                os.remove(AUDIT_DB)
            except Exception:
                pass
        if os.path.exists(AUDIT_PKL):
            try:
                os.remove(AUDIT_PKL)
            except Exception:
                pass

        cls.db = AttendanceDatabase(db_path=AUDIT_DB)
        cls.fr = FaceRecognitionSystem(classifier_path=AUDIT_PKL)
        cls.reg = StudentRegistrationManager(db=cls.db, fr_system=cls.fr)

        # Enroll 3 students: STU_A, STU_B, STU_C
        img1 = generate_face_canvas(skin_tone=(190, 205, 235), hair_color=(25, 25, 25))
        img2 = generate_face_canvas(skin_tone=(195, 210, 240), hair_color=(25, 25, 25))
        cls.reg.register_student("STU_A", "Student A", [img1, img2], consent_given=True)

        img3 = generate_face_canvas(skin_tone=(140, 170, 215), hair_color=(20, 60, 110))
        img4 = generate_face_canvas(skin_tone=(145, 175, 220), hair_color=(20, 60, 110))
        cls.reg.register_student("STU_B", "Student B", [img3, img4], consent_given=True)

        img5 = generate_face_canvas(skin_tone=(150, 185, 230), hair_color=(10, 10, 10))
        img6 = generate_face_canvas(skin_tone=(155, 190, 235), hair_color=(10, 10, 10))
        cls.reg.register_student("STU_C", "Student C", [img5, img6], consent_given=True)

    def setUp(self):
        """Reset state and clean attendance records before each test."""
        BackgroundSessionWatcher.stop_singleton()
        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM attendance")
                conn.execute("DELETE FROM attendance_sessions")
        finally:
            conn.close()

    def tearDown(self):
        """Ensure singleton watcher is stopped after each test."""
        BackgroundSessionWatcher.stop_singleton()

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        BackgroundSessionWatcher.stop_singleton()
        import gc
        gc.collect()
        if os.path.exists(AUDIT_DB):
            try:
                os.remove(AUDIT_DB)
            except Exception:
                pass
        if os.path.exists(AUDIT_PKL):
            try:
                os.remove(AUDIT_PKL)
            except Exception:
                pass

    def test_01_singleton_watcher_guarantee_across_multiple_sessions(self):
        """
        1. Verify that multiple AttendanceManager instances (simulating multiple browser tabs
        and Streamlit reruns) cannot create duplicate background watcher threads.
        Only exactly one thread named 'AttendanceSessionWatcher' must exist.
        """
        managers = []
        # Simulate 10 simultaneous browser connections initializing AttendanceManager
        def create_manager():
            mgr = AttendanceManager(
                db=self.db,
                fr_system=self.fr,
                allow_simulation=False,
                enable_background_watcher=True
            )
            managers.append(mgr)

        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(managers), 10)

        # Count active watcher threads across the entire process
        watcher_threads = [t for t in threading.enumerate() if t.name == "AttendanceSessionWatcher" and t.is_alive()]
        self.assertEqual(
            len(watcher_threads),
            1,
            f"Expected exactly 1 watcher thread, but found {len(watcher_threads)}!"
        )

        active_watcher = BackgroundSessionWatcher.get_active_instance()
        self.assertIsNotNone(active_watcher)
        self.assertTrue(active_watcher.is_alive())

    def test_02_autonomous_watcher_finalization_without_user_or_camera(self):
        """
        2. Verify that the background watcher continues checking the clock and executes
        automatic finalization at 10:00 AM IST without any user interaction or camera frames.
        """
        test_date = "2026-11-13"
        # Record Student A present at 09:35 AM
        self.db.record_attendance(test_date, "STU_A", "Student A", "PRESENT", "09:35:00", 0.85)
        self.assertFalse(self.db.is_session_finalized(test_date))

        # Time manager simulating 10:00:01 AM IST
        sim_time = datetime(2026, 11, 13, 10, 0, 1, tzinfo=KOLKATA_TZ)
        time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        mgr = AttendanceManager(
            db=self.db,
            fr_system=self.fr,
            time_manager=time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

        # Patch time call to return simulated 10:00:01 AM
        time_mgr.get_time = lambda s=None: sim_time

        # Start watcher thread with fast check interval (0.05s)
        watcher = BackgroundSessionWatcher(mgr, check_interval_sec=0.05)
        watcher.start()

        try:
            # Sleep 0.25s with ZERO user interaction, zero camera frames, and zero manual method calls
            time.sleep(0.25)

            # Check that the watcher autonomously finalized the session in the background
            self.assertTrue(self.db.is_session_finalized(test_date))

            # Verify that STU_B and STU_C were autonomously marked ABSENT
            records = self.db.get_daily_attendance(test_date)
            status_map = {r['student_id']: r['status'] for r in records}
            self.assertEqual(status_map.get("STU_A"), "PRESENT")
            self.assertEqual(status_map.get("STU_B"), "ABSENT")
            self.assertEqual(status_map.get("STU_C"), "ABSENT")
        finally:
            watcher.stop()

    def test_03_sqlite_thread_safety_and_concurrent_finalization_stress(self):
        """
        3. Stress test SQLite thread safety, connection handling, and concurrent finalization.
        Multiple threads attempt to finalize and record attendance concurrently.
        Must succeed without OperationalError: database is locked or duplicate entries.
        """
        test_date = "2026-11-13"
        errors = []

        def worker_finalize():
            try:
                self.db.finalize_session(test_date)
            except Exception as e:
                errors.append(e)

        def worker_record(sid, name):
            try:
                self.db.record_attendance(test_date, sid, name, "PRESENT", "09:35:00", 0.90)
            except Exception as e:
                errors.append(e)

        threads = []
        # Threads trying to record attendance
        threads.append(threading.Thread(target=worker_record, args=("STU_A", "Student A")))
        threads.append(threading.Thread(target=worker_record, args=("STU_B", "Student B")))
        # Threads trying to finalize the session concurrently
        for _ in range(8):
            threads.append(threading.Thread(target=worker_finalize))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Zero unhandled exceptions or database locks allowed
        self.assertEqual(len(errors), 0, f"Encountered concurrency errors: {errors}")

        # The session must be finalized
        self.assertTrue(self.db.is_session_finalized(test_date))

        # Exactly 3 records must exist, one for each student (no duplicates)
        records = self.db.get_daily_attendance(test_date)
        self.assertEqual(len(records), 3)
        self.assertEqual(set(r['student_id'] for r in records), {"STU_A", "STU_B", "STU_C"})

    def test_04_cold_restart_recovery_after_10am(self):
        """
        4. Verify that when the application was stopped before 10:00 AM (leaving an unfinished session)
        and restarted after 10:00 AM, the session is reconciled and finalized automatically upon startup.
        """
        unfinished_date = "2026-11-10"
        # Simulate attendance taken at 09:32 AM on that date, then app stopped
        self.db.record_attendance(unfinished_date, "STU_A", "Student A", "PRESENT", "09:32:10", 0.88)
        self.assertFalse(self.db.is_session_finalized(unfinished_date))

        # Restart app after 10:00 AM (or on a future day)
        restart_time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        restart_mgr = AttendanceManager(
            db=self.db,
            fr_system=self.fr,
            time_manager=restart_time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

        restart_now = datetime(2026, 11, 13, 10, 15, 0, tzinfo=KOLKATA_TZ)
        restart_mgr.auto_check_finalization(simulated_time=restart_now)

        # The past unfinished session must now be automatically finalized
        self.assertTrue(self.db.is_session_finalized(unfinished_date))
        records = self.db.get_daily_attendance(unfinished_date)
        status_map = {r['student_id']: r['status'] for r in records}
        self.assertEqual(status_map.get("STU_A"), "PRESENT")
        self.assertEqual(status_map.get("STU_B"), "ABSENT")
        self.assertEqual(status_map.get("STU_C"), "ABSENT")

    def test_05_repeated_watcher_checks_do_not_duplicate_or_corrupt(self):
        """
        5. Verify that repeated watcher checks (e.g. firing every 5 seconds past 10:00 AM)
        cannot create duplicate records or corrupt the database.
        """
        test_date = "2026-11-13"
        time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        mgr = AttendanceManager(
            db=self.db,
            fr_system=self.fr,
            time_manager=time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

        sim_post_close = datetime(2026, 11, 13, 10, 5, 0, tzinfo=KOLKATA_TZ)

        # Fire auto_check_finalization 50 times in a tight loop
        for _ in range(50):
            mgr.auto_check_finalization(simulated_time=sim_post_close)

        # Database must have exactly 3 records, status finalized, zero duplicate rows
        records = self.db.get_daily_attendance(test_date)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r['status'] == 'ABSENT' for r in records))
        self.assertTrue(self.db.is_session_finalized(test_date))

    def test_06_production_code_strictly_rejects_simulated_timestamps(self):
        """
        6. Verify that production configuration (allow_simulation=False) unconditionally
        rejects simulated timestamps in all APIs and always evaluates true system time.
        """
        prod_mgr = AttendanceManager(
            db=self.db,
            fr_system=self.fr,
            allow_simulation=False,
            enable_background_watcher=False
        )

        fake_morning = datetime(2026, 11, 13, 9, 35, 0, tzinfo=KOLKATA_TZ)

        # Pass fake morning timestamp to get_session_status
        status_info = prod_mgr.get_session_status(simulated_time=fake_morning)

        # In production mode, fake_morning must be ignored and actual system time returned
        real_now = datetime.now(KOLKATA_TZ)
        self.assertEqual(status_info['date'], real_now.strftime("%Y-%m-%d"))
        # Real system clock hour must match returned time, NOT the fake 9 AM
        self.assertEqual(int(status_info['time_only_str'].split(":")[0]), real_now.hour)


if __name__ == "__main__":
    unittest.main()
