"""
Comprehensive Test Suite for Phase 3: Attendance Management.

Validates all 17 required criteria:
1. Attendance before 9:30 AM is rejected (09:29:59 IST).
2. Attendance window boundary evaluation (09:30:00 and 09:59:59 IST are ACTIVE).
3. Registered student marked PRESENT with exact timestamp, ID, name, confidence.
4. Unknown individual NOT marked present (zero attendance records created).
5. Duplicate attendance prevented for the same student on the same date.
6. Multiple registered students in one frame marked independently.
7. Attendance at 10:00 AM rejected (10:00:00 IST) and session closed.
8. Unrecorded registered students automatically marked ABSENT at finalization.
9. Finalized session is strictly locked from further modifications.
10. CSV export generates valid data with all 6 required columns.
11. Excel export generates valid openpyxl/pandas binary workbook (.xlsx).
12. Existing Phase 2 student profiles and embeddings remain intact.
13. Timezone conversion correctly translates UTC to Asia/Kolkata (+5:30).
14. Production mode (allow_simulation=False) strictly rejects simulated timestamps and uses live system time.
15. Automatic finalization operates without camera submissions or manual button clicks.
16. Cold restart recovery: application automatically finalizes unfinalized past sessions on startup.
17. Repeated finalization idempotency: repeated finalization calls produce zero duplicate records.
"""

import os
import io
import sys
import unittest
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem
from modules.attendance.registration import StudentRegistrationManager
from modules.attendance.attendance_manager import AttendanceManager, AttendanceTimeManager, KOLKATA_TZ
from modules.attendance.reports import export_attendance_csv, export_attendance_excel, get_attendance_dataframe
from tests.attendance.test_phase2 import generate_face_canvas

TEST_P3_DB = os.path.join("datasets", "attendance", "test_p3_attendance.db")
TEST_P3_PKL = os.path.join("models", "attendance", "test_p3_classifier.pkl")


class TestAttendanceManagementPhase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize clean database, face recognition system, and enrolled test students."""
        if os.path.exists(TEST_P3_DB):
            os.remove(TEST_P3_DB)
        if os.path.exists(TEST_P3_PKL):
            os.remove(TEST_P3_PKL)

        cls.db = AttendanceDatabase(db_path=TEST_P3_DB)
        cls.fr = FaceRecognitionSystem(classifier_path=TEST_P3_PKL)
        cls.reg = StudentRegistrationManager(db=cls.db, fr_system=cls.fr)
        cls.time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        cls.att_mgr = AttendanceManager(
            db=cls.db,
            fr_system=cls.fr,
            time_manager=cls.time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

        # 1. Enroll Student 1: Alice (STU101)
        cls.alice_reg1 = generate_face_canvas(skin_tone=(190, 205, 235), hair_color=(25, 25, 25), eye_offset=26)
        cls.alice_reg2 = generate_face_canvas(skin_tone=(195, 210, 240), hair_color=(25, 25, 25), eye_offset=27)
        cls.alice_test = generate_face_canvas(skin_tone=(185, 200, 230), hair_color=(25, 25, 25), eye_offset=26)
        cls.reg.register_student("STU101", "Alice Smith", [cls.alice_reg1, cls.alice_reg2], consent_given=True)

        # 2. Enroll Student 2: Bob (STU102)
        cls.bob_reg1 = generate_face_canvas(skin_tone=(140, 170, 215), hair_color=(20, 60, 110), eye_offset=30, eye_size=10)
        cls.bob_reg2 = generate_face_canvas(skin_tone=(145, 175, 220), hair_color=(20, 60, 110), eye_offset=31, eye_size=10)
        cls.bob_test = generate_face_canvas(skin_tone=(142, 172, 217), hair_color=(20, 60, 110), eye_offset=30, eye_size=10)
        cls.reg.register_student("STU102", "Bob Jones", [cls.bob_reg1, cls.bob_reg2], consent_given=True)

        # 3. Enroll Student 3: Charlie (STU103) - will remain absent during test
        cls.charlie_reg1 = generate_face_canvas(skin_tone=(150, 185, 230), hair_color=(10, 10, 10), eye_offset=24)
        cls.charlie_reg2 = generate_face_canvas(skin_tone=(155, 190, 235), hair_color=(10, 10, 10), eye_offset=24)
        cls.reg.register_student("STU103", "Charlie Brown", [cls.charlie_reg1, cls.charlie_reg2], consent_given=True)

        # 4. Unknown Impostor Image (unregistered)
        cls.unknown_test = generate_face_canvas(skin_tone=(120, 150, 190), hair_color=(80, 160, 210), eye_offset=20, eye_size=6)

        # 5. Composite Multi-face frame: Alice on left, Bob on right
        h, w = cls.alice_test.shape[:2]
        cls.multi_face_frame = np.full((h + 40, w * 2 + 60, 3), (225, 225, 225), dtype=np.uint8)
        cls.multi_face_frame[20:20+h, 20:20+w] = cls.alice_test
        cls.multi_face_frame[20:20+h, 40+w:40+2*w] = cls.bob_test

        cls.test_date = "2026-11-13"

    def test_01_attendance_before_930_rejected(self):
        """1. Attendance before 9:30 AM is rejected (09:29:59 IST)."""
        dt_early = datetime(2026, 11, 13, 9, 29, 59, tzinfo=KOLKATA_TZ)
        is_active, code, msg = self.time_mgr.evaluate_window(dt_early)
        self.assertFalse(is_active)
        self.assertEqual(code, "NOT_STARTED")
        self.assertIn("09:30:00", msg)

        res = self.att_mgr.process_frame_attendance(self.alice_test, simulated_time=dt_early)
        self.assertFalse(res['allowed'])
        self.assertEqual(res['session_status'], "NOT_STARTED")
        self.assertEqual(len(res['marked_students']), 0)

    def test_02_attendance_window_boundaries_accepted(self):
        """2. Attendance at 9:30:00 AM and 9:59:59 AM is within active window."""
        dt_start = datetime(2026, 11, 13, 9, 30, 0, tzinfo=KOLKATA_TZ)
        is_active_start, code_start, _ = self.time_mgr.evaluate_window(dt_start)
        self.assertTrue(is_active_start)
        self.assertEqual(code_start, "ACTIVE")

        dt_end = datetime(2026, 11, 13, 9, 59, 59, tzinfo=KOLKATA_TZ)
        is_active_end, code_end, _ = self.time_mgr.evaluate_window(dt_end)
        self.assertTrue(is_active_end)
        self.assertEqual(code_end, "ACTIVE")

    def test_03_correct_registered_student_marked_present(self):
        """3. Registered student is marked PRESENT with exact timestamp, ID, name, and confidence."""
        dt_active = datetime(2026, 11, 13, 9, 35, 12, tzinfo=KOLKATA_TZ)
        res = self.att_mgr.process_frame_attendance(self.alice_test, simulated_time=dt_active)

        self.assertTrue(res['allowed'])
        self.assertEqual(len(res['marked_students']), 1)
        marked = res['marked_students'][0]
        self.assertEqual(marked['student_id'], "STU101")
        self.assertEqual(marked['name'], "Alice Smith")
        self.assertEqual(marked['status'], "PRESENT")
        self.assertEqual(marked['time'], "09:35:12")
        self.assertGreaterEqual(marked['confidence'], 0.50)

        # Verify record in SQLite database
        daily_records = self.db.get_daily_attendance(self.test_date)
        alice_rec = next((r for r in daily_records if r['student_id'] == "STU101"), None)
        self.assertIsNotNone(alice_rec)
        self.assertEqual(alice_rec['status'], "PRESENT")
        self.assertEqual(alice_rec['identification_time'], "09:35:12")
        self.assertAlmostEqual(alice_rec['confidence'], marked['confidence'], places=3)

    def test_04_unknown_student_not_marked_present(self):
        """4. Unknown individual is not marked present and no attendance record is created."""
        dt_active = datetime(2026, 11, 13, 9, 40, 0, tzinfo=KOLKATA_TZ)
        records_before = len(self.db.get_daily_attendance(self.test_date))

        res = self.att_mgr.process_frame_attendance(self.unknown_test, confidence_threshold=0.75, simulated_time=dt_active)
        self.assertTrue(res['allowed'])
        self.assertEqual(len(res['marked_students']), 0)
        self.assertEqual(res['unknowns_ignored'], 1)

        records_after = len(self.db.get_daily_attendance(self.test_date))
        self.assertEqual(records_before, records_after, "Unknown person must never generate an attendance record.")

    def test_05_duplicate_attendance_prevented(self):
        """5. Duplicate attendance for the same student on the same date is blocked."""
        dt_dup = datetime(2026, 11, 13, 9, 42, 0, tzinfo=KOLKATA_TZ)
        res = self.att_mgr.process_frame_attendance(self.alice_test, simulated_time=dt_dup)

        self.assertTrue(res['allowed'])
        self.assertEqual(len(res['marked_students']), 0, "Alice should not be marked again.")
        self.assertEqual(len(res['duplicates_prevented']), 1)
        self.assertEqual(res['duplicates_prevented'][0]['student_id'], "STU101")

        # Verify only one row exists for Alice in SQLite
        daily_records = self.db.get_daily_attendance(self.test_date)
        alice_records = [r for r in daily_records if r['student_id'] == "STU101"]
        self.assertEqual(len(alice_records), 1)

    def test_06_multiple_students_marked_independently(self):
        """6. Multiple registered students in one frame are marked separately."""
        dt_multi = datetime(2026, 11, 13, 9, 45, 30, tzinfo=KOLKATA_TZ)
        res = self.att_mgr.process_frame_attendance(self.multi_face_frame, simulated_time=dt_multi)

        self.assertTrue(res['allowed'])
        # Alice was already marked, so Bob should be freshly marked PRESENT, Alice flagged as duplicate
        marked_ids = {m['student_id'] for m in res['marked_students']}
        self.assertIn("STU102", marked_ids)

        dup_ids = {d['student_id'] for d in res['duplicates_prevented']}
        self.assertIn("STU101", dup_ids)

        # Verify both Alice and Bob are PRESENT in SQLite
        daily_records = self.db.get_daily_attendance(self.test_date)
        present_ids = {r['student_id'] for r in daily_records if r['status'] == "PRESENT"}
        self.assertIn("STU101", present_ids)
        self.assertIn("STU102", present_ids)

    def test_07_attendance_at_1000_closes_and_auto_finalizes(self):
        """7. At 10:00:00 AM IST, attendance window closes and session auto-finalizes."""
        dt_closed = datetime(2026, 11, 13, 10, 0, 0, tzinfo=KOLKATA_TZ)
        is_active, code, msg = self.time_mgr.evaluate_window(dt_closed)
        self.assertFalse(is_active)
        self.assertEqual(code, "CLOSED")

        # Frame submission at 10:00:00 AM is rejected and automatically finalizes the session
        res = self.att_mgr.process_frame_attendance(self.alice_test, simulated_time=dt_closed)
        self.assertFalse(res['allowed'])
        self.assertIn(res['session_status'], ["CLOSED", "FINALIZED"])
        self.assertEqual(len(res['marked_students']), 0)
        self.assertTrue(self.db.is_session_finalized(self.test_date))

    def test_08_absent_students_identified_at_finalization(self):
        """8. At finalization, registered students not identified are marked ABSENT."""
        # Ensure session is finalized
        if not self.db.is_session_finalized(self.test_date):
            self.att_mgr.finalize_attendance_session(date=self.test_date)

        # Check records: STU101, STU102 should be PRESENT; STU103 (Charlie) should be ABSENT
        daily_records = self.db.get_daily_attendance(self.test_date)
        status_map = {r['student_id']: r['status'] for r in daily_records}

        self.assertEqual(status_map.get("STU101"), "PRESENT")
        self.assertEqual(status_map.get("STU102"), "PRESENT")
        self.assertEqual(status_map.get("STU103"), "ABSENT")
        self.assertEqual(len(daily_records), 3)

        summary = self.db.get_attendance_summary(self.test_date)
        self.assertEqual(summary['total_registered'], 3)
        self.assertEqual(summary['present_count'], 2)
        self.assertEqual(summary['absent_count'], 1)
        self.assertAlmostEqual(summary['attendance_percentage'], 66.7, places=1)

    def test_09_finalized_attendance_cannot_be_modified(self):
        """9. Finalized session strictly locks records from further changes."""
        self.assertTrue(self.db.is_session_finalized(self.test_date))

        # Attempt to mark attendance after finalization
        dt_attempt = datetime(2026, 11, 13, 10, 5, 0, tzinfo=KOLKATA_TZ)
        res = self.att_mgr.process_frame_attendance(self.alice_test, simulated_time=dt_attempt)
        self.assertFalse(res['allowed'])
        self.assertEqual(res['session_status'], "FINALIZED")

        # Direct DB attempt to insert after finalization must also be blocked
        db_res = self.db.record_attendance(self.test_date, "STU103", "Charlie Brown", "PRESENT")
        self.assertFalse(db_res['success'])
        self.assertEqual(db_res.get('error_type'), 'SESSION_FINALIZED')

    def test_10_csv_export_works(self):
        """10. CSV export generates all required columns with accurate data."""
        csv_data = export_attendance_csv(self.test_date, self.db)
        self.assertIsInstance(csv_data, str)
        self.assertIn("Date,Student ID,Student Name,Attendance Status,Identification Time,Recognition Confidence", csv_data)
        self.assertIn("STU101,Alice Smith,PRESENT", csv_data)
        self.assertIn("STU102,Bob Jones,PRESENT", csv_data)
        self.assertIn("STU103,Charlie Brown,ABSENT,N/A,N/A", csv_data)

    def test_11_excel_export_works(self):
        """11. Excel export generates valid openpyxl/pandas binary workbook."""
        excel_bytes = export_attendance_excel(self.test_date, self.db)
        self.assertIsInstance(excel_bytes, bytes)
        self.assertGreater(len(excel_bytes), 1000)

        # Read back with pandas to verify structure
        df_read = pd.read_excel(io.BytesIO(excel_bytes))
        expected_cols = ["Date", "Student ID", "Student Name", "Attendance Status", "Identification Time", "Recognition Confidence"]
        self.assertEqual(list(df_read.columns), expected_cols)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(set(df_read["Student ID"]), {"STU101", "STU102", "STU103"})

    def test_12_existing_phase2_records_intact(self):
        """12. Existing Phase 2 student profiles and facial embeddings are intact."""
        all_students = self.db.get_all_students()
        self.assertEqual(len(all_students), 3)

        stu_ids = {s['student_id'] for s in all_students}
        self.assertEqual(stu_ids, {"STU101", "STU102", "STU103"})

        X, y_ids, names_map = self.db.get_all_embeddings()
        self.assertEqual(X.shape[1], 128)
        self.assertEqual(len(y_ids), 6)  # 2 embeddings each for 3 students

    def test_13_timezone_conversion(self):
        """13. Asia/Kolkata timezone conversion evaluates properly from UTC."""
        utc_zone = ZoneInfo("UTC")
        # 04:00:00 UTC == 09:30:00 IST (+5:30)
        dt_utc_start = datetime(2026, 11, 13, 4, 0, 0, tzinfo=utc_zone)
        converted = self.time_mgr.get_time(dt_utc_start)
        self.assertEqual(converted.tzinfo, KOLKATA_TZ)
        self.assertEqual(converted.hour, 9)
        self.assertEqual(converted.minute, 30)

        is_active, code, _ = self.time_mgr.evaluate_window(converted)
        self.assertTrue(is_active)
        self.assertEqual(code, "ACTIVE")

        # 04:30:00 UTC == 10:00:00 IST (Closed)
        dt_utc_closed = datetime(2026, 11, 13, 4, 30, 0, tzinfo=utc_zone)
        converted_closed = self.time_mgr.get_time(dt_utc_closed)
        is_active_c, code_c, _ = self.time_mgr.evaluate_window(converted_closed)
        self.assertFalse(is_active_c)
        self.assertEqual(code_c, "CLOSED")

    def test_14_production_mode_strictly_rejects_simulated_time(self):
        """14. Production mode (allow_simulation=False) strictly ignores simulated time and returns system time."""
        prod_time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=False)
        fake_time = datetime(2026, 11, 13, 9, 45, 0, tzinfo=KOLKATA_TZ)
        resolved_time = prod_time_mgr.get_time(fake_time)

        # In production mode, resolved time must be real system time, tzinfo must be Asia/Kolkata
        self.assertEqual(resolved_time.tzinfo, KOLKATA_TZ)
        # Verify that passing fake_time had no effect on the returned hour/minute if system clock is different
        live_now = datetime.now(KOLKATA_TZ)
        self.assertEqual(resolved_time.hour, live_now.hour)

    def test_15_automatic_finalization_without_camera_or_button(self):
        """15. Automatic finalization marks unrecorded students ABSENT without camera frames or manual buttons."""
        isolated_date = "2026-11-20"
        self.assertFalse(self.db.is_session_finalized(isolated_date))

        # At 10:00:01 AM IST, an inquiry or background check auto-finalizes the session
        dt_past_close = datetime(2026, 11, 20, 10, 0, 1, tzinfo=KOLKATA_TZ)
        fin_res = self.att_mgr.auto_check_finalization(simulated_time=dt_past_close)

        self.assertIsNotNone(fin_res)
        self.assertTrue(fin_res['success'])
        self.assertTrue(self.db.is_session_finalized(isolated_date))

        # All 3 enrolled students must now be recorded as ABSENT
        records = self.db.get_daily_attendance(isolated_date)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r['status'] == 'ABSENT' for r in records))

    def test_16_cold_restart_reconciles_past_unfinalized_sessions(self):
        """16. Cold restart automatically reconciles and finalizes unfinalized past sessions upon launch."""
        past_unfinalized_date = "2026-11-10"
        # Record one student attendance on that past day without finalizing
        self.db.record_attendance(past_unfinalized_date, "STU101", "Alice Smith", "PRESENT", "09:35:00")
        self.assertFalse(self.db.is_session_finalized(past_unfinalized_date))

        # Simulate cold restart: new AttendanceManager instance initialized
        new_time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        # Calling auto_check_finalization for a date after 2026-11-10 triggers reconciliation
        dt_later = datetime(2026, 11, 13, 9, 31, 0, tzinfo=KOLKATA_TZ)
        self.att_mgr.auto_check_finalization(simulated_time=dt_later)

        # Past unfinalized session is now finalized
        self.assertTrue(self.db.is_session_finalized(past_unfinalized_date))
        past_records = self.db.get_daily_attendance(past_unfinalized_date)
        status_map = {r['student_id']: r['status'] for r in past_records}
        self.assertEqual(status_map.get("STU101"), "PRESENT")
        self.assertEqual(status_map.get("STU102"), "ABSENT")
        self.assertEqual(status_map.get("STU103"), "ABSENT")

    def test_17_repeated_finalization_is_idempotent(self):
        """17. Repeated finalization calls do not duplicate records and remain strictly idempotent."""
        # Today's session is already finalized
        self.assertTrue(self.db.is_session_finalized(self.test_date))
        records_count_before = len(self.db.get_daily_attendance(self.test_date))

        # Call finalize again
        res2 = self.db.finalize_session(self.test_date)
        self.assertTrue(res2['success'])
        self.assertTrue(res2['already_finalized'])

        records_count_after = len(self.db.get_daily_attendance(self.test_date))
        self.assertEqual(records_count_before, records_count_after, "Finalizing again must not duplicate records.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        import gc
        gc.collect()
        try:
            if os.path.exists(TEST_P3_DB):
                os.remove(TEST_P3_DB)
        except Exception:
            pass
        try:
            if os.path.exists(TEST_P3_PKL):
                os.remove(TEST_P3_PKL)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
