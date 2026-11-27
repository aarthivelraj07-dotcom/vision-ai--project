"""
Comprehensive Tests for Registration Timing, Session Eligibility, and Finalization Recovery.
Validates:
1. Students registered before 10:00 AM IST are marked ABSENT at finalization.
2. Students registered after 10:00 AM IST are excluded from that day's session and not marked ABSENT.
3. Finalized sessions strictly lock and remain immutable when new students register later.
4. Late application startup and cold-start recovery accurately reconcile sessions based on created_at timestamps.
5. Accurate separation of total system students vs session-eligible students in summary.
"""

import os
import sys
import unittest
import numpy as np
from datetime import datetime, time
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase, is_student_eligible_for_session
from modules.attendance.attendance_manager import AttendanceManager, AttendanceTimeManager, KOLKATA_TZ

TEST_ELIG_DB = os.path.join("datasets", "attendance", "test_eligibility_recovery.db")


class TestSessionEligibilityAndRecovery(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_ELIG_DB):
            os.remove(TEST_ELIG_DB)
        self.db = AttendanceDatabase(db_path=TEST_ELIG_DB)
        self.time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        self.att_mgr = AttendanceManager(
            db=self.db,
            fr_system=None,
            time_manager=self.time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

    def tearDown(self):
        if os.path.exists(TEST_ELIG_DB):
            os.remove(TEST_ELIG_DB)

    def test_01_eligibility_helper_function(self):
        """1. Verify is_student_eligible_for_session across boundary timestamps."""
        session_date = "2026-11-14"

        # Registered earlier date
        self.assertTrue(is_student_eligible_for_session("2026-11-13T16:00:00", session_date))

        # Registered on session date before 10:00 AM IST
        self.assertTrue(is_student_eligible_for_session("2026-11-14T09:15:00", session_date))
        self.assertTrue(is_student_eligible_for_session("2026-11-14T09:59:59", session_date))
        self.assertTrue(is_student_eligible_for_session("2026-11-14T10:00:00", session_date))

        # Registered on session date AFTER 10:00 AM IST (post-window)
        self.assertFalse(is_student_eligible_for_session("2026-11-14T10:00:01", session_date))
        self.assertFalse(is_student_eligible_for_session("2026-11-14T15:30:00", session_date))
        self.assertFalse(is_student_eligible_for_session("2026-11-14T19:27:13.554865", session_date))

        # Same post-window student is eligible on the NEXT day
        self.assertTrue(is_student_eligible_for_session("2026-11-14T19:27:13.554865", "2026-11-15"))

    def test_02_registration_before_window_marked_absent(self):
        """2. Student registered before 10:00 AM IST is marked ABSENT if unrecorded at finalization."""
        test_date = "2026-11-14"

        # Register student with created_at at 09:15 AM IST
        dummy_emb = [np.zeros(128, dtype=np.float32)]
        self.db.add_student("EARLY_01", "Early Bird", dummy_emb, consent_given=True)
        # Explicitly set created_at in database to 09:15 AM
        conn = self.db.get_connection()
        conn.execute("UPDATE students SET created_at = '2026-11-14T09:15:00' WHERE student_id = 'EARLY_01'")
        conn.commit()
        conn.close()

        # Finalize session at 10:00 AM IST
        sim_close = datetime(2026, 11, 14, 10, 0, 1, tzinfo=KOLKATA_TZ)
        fin_res = self.att_mgr.auto_check_finalization(simulated_time=sim_close)
        self.assertIsNotNone(fin_res)
        self.assertTrue(fin_res['success'])

        # Verify student is marked ABSENT
        records = self.db.get_daily_attendance(test_date)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['student_id'], "EARLY_01")
        self.assertEqual(records[0]['status'], "ABSENT")

        # Summary check
        summary = self.db.get_attendance_summary(test_date)
        self.assertEqual(summary['total_registered'], 1)
        self.assertEqual(summary['present_count'], 0)
        self.assertEqual(summary['absent_count'], 1)
        self.assertEqual(summary['post_session_registered'], 0)

    def test_03_registration_after_window_excluded_from_today_session(self):
        """3. Student registered after 10:00 AM IST is NOT marked absent for today's finalized session."""
        test_date = "2026-11-14"

        # Session finalizes when 0 students exist
        sim_close = datetime(2026, 11, 14, 10, 0, 1, tzinfo=KOLKATA_TZ)
        fin_res = self.att_mgr.auto_check_finalization(simulated_time=sim_close)
        self.assertIsNotNone(fin_res)
        self.assertTrue(fin_res['success'])

        # Now register a student in the evening at 19:27 IST
        dummy_emb = [np.zeros(128, dtype=np.float32)]
        self.db.add_student("LATE_01", "Late Student", dummy_emb, consent_given=True)
        conn = self.db.get_connection()
        conn.execute("UPDATE students SET created_at = '2026-11-14T19:27:13' WHERE student_id = 'LATE_01'")
        conn.commit()
        conn.close()

        # Re-check session status
        sim_evening = datetime(2026, 11, 14, 19, 30, 0, tzinfo=KOLKATA_TZ)
        status_info = self.att_mgr.get_session_status(simulated_time=sim_evening)
        summary = status_info['summary']

        # Verify today's attendance table remains empty (0 rows for today)
        records = self.db.get_daily_attendance(test_date)
        self.assertEqual(len(records), 0, "Post-window student must not have retroactive attendance record.")

        # Verify summary distinguishes system total from session eligible
        self.assertEqual(summary['total_system_students'], 1)
        self.assertEqual(summary['session_eligible_students'], 0)
        self.assertEqual(summary['total_registered'], 0)
        self.assertEqual(summary['present_count'], 0)
        self.assertEqual(summary['absent_count'], 0)
        self.assertEqual(summary['post_session_registered'], 1)

        # Context-aware status message check
        msg = status_info['status_message']
        self.assertIn("0 eligible students", msg)
        self.assertIn("1 student registered after 10:00 AM", msg)

    def test_04_session_immutability_preserved(self):
        """4. Finalized session strictly locks and cannot be modified by subsequent calls."""
        test_date = "2026-11-14"

        # Finalize with 0 students
        res1 = self.db.finalize_session(test_date)
        self.assertTrue(res1['success'])
        self.assertFalse(res1['already_finalized'])

        # Register late student
        dummy_emb = [np.zeros(128, dtype=np.float32)]
        self.db.add_student("LATE_02", "Late Bob", dummy_emb, consent_given=True)

        # Calling finalize again must return already_finalized=True and NOT add records
        res2 = self.db.finalize_session(test_date)
        self.assertTrue(res2['success'])
        self.assertTrue(res2['already_finalized'])

        records = self.db.get_daily_attendance(test_date)
        self.assertEqual(len(records), 0)

    def test_05_mixed_roster_eligibility(self):
        """5. Mixed roster: early student marked ABSENT, late student excluded."""
        test_date = "2026-11-14"
        dummy_emb = [np.zeros(128, dtype=np.float32)]

        # Early student (enrolled 09:10 AM)
        self.db.add_student("EARLY_02", "Early Alice", dummy_emb, consent_given=True)
        # Late student (enrolled 11:30 AM)
        self.db.add_student("LATE_03", "Late Charlie", dummy_emb, consent_given=True)

        conn = self.db.get_connection()
        conn.execute("UPDATE students SET created_at = '2026-11-14T09:10:00' WHERE student_id = 'EARLY_02'")
        conn.execute("UPDATE students SET created_at = '2026-11-14T11:30:00' WHERE student_id = 'LATE_03'")
        conn.commit()
        conn.close()

        # Finalize session at 15:00 IST (late startup recovery)
        sim_late = datetime(2026, 11, 14, 15, 0, 0, tzinfo=KOLKATA_TZ)
        fin_res = self.att_mgr.auto_check_finalization(simulated_time=sim_late)
        self.assertIsNotNone(fin_res)

        records = self.db.get_daily_attendance(test_date)
        # Only EARLY_02 must be in attendance table as ABSENT
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['student_id'], "EARLY_02")
        self.assertEqual(records[0]['status'], "ABSENT")

        # Summary counts
        summary = self.db.get_attendance_summary(test_date)
        self.assertEqual(summary['total_system_students'], 2)
        self.assertEqual(summary['session_eligible_students'], 1)
        self.assertEqual(summary['absent_count'], 1)
        self.assertEqual(summary['post_session_registered'], 1)


if __name__ == "__main__":
    unittest.main()
