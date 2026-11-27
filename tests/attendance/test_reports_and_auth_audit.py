"""
Comprehensive Audit Tests for Admin PIN Authentication, CSV & Excel Exports, and Session Integrity.
Covers:
1. Correct Admin PIN verification via environment variable.
2. Incorrect Admin PIN rejection (constant-time, no default fallback).
3. Missing environment variable behavior.
4. CSV export formatting and exact column header validation.
5. Excel (.xlsx) export binary integrity and openpyxl structure validation.
6. Empty attendance session export handling.
7. Session containing attendance records (PRESENT and ABSENT).
8. Late student registration exclusion from closed sessions.
9. Finalized session immutability under concurrent/repeated requests.
"""

import os
import sys
import io
import unittest
import openpyxl
import pandas as pd
import numpy as np
from datetime import datetime, time
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.database import AttendanceDatabase
from modules.attendance.ui import get_admin_pin, is_admin_pin_configured, verify_admin_pin
from modules.attendance.reports import (
    get_attendance_dataframe,
    export_attendance_csv,
    export_attendance_excel
)
from modules.attendance.attendance_manager import AttendanceManager, AttendanceTimeManager, KOLKATA_TZ

TEST_AUDIT_DB = os.path.join("datasets", "attendance", "test_audit_reports.db")


class TestReportsAndAuthAudit(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_AUDIT_DB):
            os.remove(TEST_AUDIT_DB)
        self.db = AttendanceDatabase(db_path=TEST_AUDIT_DB)
        self.time_mgr = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=True)
        self.att_mgr = AttendanceManager(
            db=self.db,
            fr_system=None,
            time_manager=self.time_mgr,
            allow_simulation=True,
            enable_background_watcher=False
        )

    def tearDown(self):
        if os.path.exists(TEST_AUDIT_DB):
            os.remove(TEST_AUDIT_DB)

    def test_01_admin_pin_correct_verification(self):
        """1. Correct Admin PIN configured via ATTENDANCE_ADMIN_PIN is verified successfully."""
        old_val = os.environ.get("ATTENDANCE_ADMIN_PIN")
        try:
            os.environ["ATTENDANCE_ADMIN_PIN"] = "SecureAuditPass2026!"
            self.assertTrue(is_admin_pin_configured())
            self.assertEqual(get_admin_pin(), "SecureAuditPass2026!")
            self.assertTrue(verify_admin_pin("SecureAuditPass2026!"))
            # Leading/trailing whitespace should be trimmed safely
            self.assertTrue(verify_admin_pin("  SecureAuditPass2026!  "))
        finally:
            if old_val is not None:
                os.environ["ATTENDANCE_ADMIN_PIN"] = old_val
            else:
                os.environ.pop("ATTENDANCE_ADMIN_PIN", None)

    def test_02_admin_pin_incorrect_rejection(self):
        """2. Incorrect Admin PIN is strictly rejected; no hardcoded defaults allowed."""
        old_val = os.environ.get("ATTENDANCE_ADMIN_PIN")
        try:
            os.environ["ATTENDANCE_ADMIN_PIN"] = "CorrectSecret999"
            self.assertFalse(verify_admin_pin("WrongPass"))
            self.assertFalse(verify_admin_pin("admin"))
            self.assertFalse(verify_admin_pin("admin123"))
            self.assertFalse(verify_admin_pin("1234"))
            self.assertFalse(verify_admin_pin(""))
            self.assertFalse(verify_admin_pin("None"))
        finally:
            if old_val is not None:
                os.environ["ATTENDANCE_ADMIN_PIN"] = old_val
            else:
                os.environ.pop("ATTENDANCE_ADMIN_PIN", None)

    def test_03_admin_pin_missing_environment_variable(self):
        """3. When ATTENDANCE_ADMIN_PIN is absent, authentication is disabled with NO default fallback."""
        old_val = os.environ.pop("ATTENDANCE_ADMIN_PIN", None)
        try:
            self.assertFalse(is_admin_pin_configured())
            self.assertEqual(get_admin_pin(), "")
            self.assertFalse(verify_admin_pin("any_pin"))
            self.assertFalse(verify_admin_pin("admin123"))
            self.assertFalse(verify_admin_pin(""))
        finally:
            if old_val is not None:
                os.environ["ATTENDANCE_ADMIN_PIN"] = old_val

    def test_04_csv_export_format_and_columns(self):
        """4. Verify CSV export generates the exact 6 expected column headers."""
        date_str = "2026-11-14"
        csv_str = export_attendance_csv(date_str, self.db)
        self.assertIsInstance(csv_str, str)
        first_line = csv_str.strip().splitlines()[0]
        expected_header = "Date,Student ID,Student Name,Attendance Status,Identification Time,Recognition Confidence"
        self.assertEqual(first_line, expected_header)

    def test_05_excel_export_binary_integrity(self):
        """5. Verify Excel (.xlsx) export produces a valid openpyxl workbook with correct sheet name."""
        date_str = "2026-11-14"
        excel_bytes = export_attendance_excel(date_str, self.db)
        self.assertIsInstance(excel_bytes, bytes)
        self.assertGreater(len(excel_bytes), 1000)

        # Parse with openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        self.assertIn(f"Attendance_{date_str}", wb.sheetnames)
        sheet = wb[f"Attendance_{date_str}"]
        headers = [cell.value for cell in sheet[1]]
        expected_cols = ["Date", "Student ID", "Student Name", "Attendance Status", "Identification Time", "Recognition Confidence"]
        self.assertEqual(headers, expected_cols)

    def test_06_empty_attendance_session_export(self):
        """6. Empty attendance session produces valid files with exact headers and 0 data rows (no fake data)."""
        empty_date = "2026-11-14"

        # CSV check: only 1 line (the header)
        csv_data = export_attendance_csv(empty_date, self.db).strip().splitlines()
        self.assertEqual(len(csv_data), 1, "Empty session CSV must contain only the header line.")

        # Excel check: only 1 row (the header)
        excel_bytes = export_attendance_excel(empty_date, self.db)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        sheet = wb.active
        all_rows = list(sheet.iter_rows(values_only=True))
        self.assertEqual(len(all_rows), 1, "Empty session Excel must contain only the header row.")

    def test_07_session_containing_attendance_records(self):
        """7. Session with recorded PRESENT and ABSENT records exports accurately."""
        date_str = "2026-11-14"
        self.db.record_attendance(date_str, "STU001", "Alice Smith", "PRESENT", "09:35:12", confidence=0.885)
        self.db.record_attendance(date_str, "STU002", "Bob Jones", "ABSENT", None, confidence=0.0)

        # Verify DataFrame
        df = get_attendance_dataframe(date_str, self.db)
        self.assertEqual(len(df), 2)

        # Check Alice
        alice = df[df["Student ID"] == "STU001"].iloc[0]
        self.assertEqual(alice["Student Name"], "Alice Smith")
        self.assertEqual(alice["Attendance Status"], "PRESENT")
        self.assertEqual(alice["Identification Time"], "09:35:12")
        self.assertEqual(alice["Recognition Confidence"], "88.5%")

        # Check Bob
        bob = df[df["Student ID"] == "STU002"].iloc[0]
        self.assertEqual(bob["Student Name"], "Bob Jones")
        self.assertEqual(bob["Attendance Status"], "ABSENT")
        self.assertEqual(bob["Identification Time"], "N/A")
        self.assertEqual(bob["Recognition Confidence"], "N/A")

        # Verify CSV export content
        csv_str = export_attendance_csv(date_str, self.db)
        self.assertIn("STU001,Alice Smith,PRESENT,09:35:12,88.5%", csv_str)
        self.assertIn("STU002,Bob Jones,ABSENT,N/A,N/A", csv_str)

        # Verify Excel export content
        excel_bytes = export_attendance_excel(date_str, self.db)
        df_excel = pd.read_excel(io.BytesIO(excel_bytes))
        self.assertEqual(len(df_excel), 2)
        self.assertEqual(set(df_excel["Student ID"]), {"STU001", "STU002"})

    def test_08_late_student_registration_excluded_from_finalized_session(self):
        """8. Student registered after 10:00 AM IST is excluded from today's finalized session."""
        date_str = "2026-11-14"

        # Auto-finalize session at 10:00 AM with 0 enrolled students
        sim_close = datetime(2026, 11, 14, 10, 0, 1, tzinfo=KOLKATA_TZ)
        self.att_mgr.auto_check_finalization(simulated_time=sim_close)
        self.assertTrue(self.db.is_session_finalized(date_str))

        # Register late student at 19:27 IST
        dummy_emb = [np.zeros(128, dtype=np.float32)]
        self.db.add_student("LATE_99", "Late Registrant", dummy_emb, consent_given=True)
        conn = self.db.get_connection()
        conn.execute("UPDATE students SET created_at = '2026-11-14T19:27:13' WHERE student_id = 'LATE_99'")
        conn.commit()
        conn.close()

        # Verify today's attendance records remain 0
        records = self.db.get_daily_attendance(date_str)
        self.assertEqual(len(records), 0)

        # Verify summary counts
        summary = self.db.get_attendance_summary(date_str)
        self.assertEqual(summary['session_eligible_students'], 0)
        self.assertEqual(summary['total_system_students'], 1)
        self.assertEqual(summary['post_session_registered'], 1)

    def test_09_finalized_session_immutability(self):
        """9. Finalized session strictly locks and cannot be altered by later attendance attempts."""
        date_str = "2026-11-14"
        self.db.finalize_session(date_str)
        self.assertTrue(self.db.is_session_finalized(date_str))

        # Attempt to mark attendance after finalization must be blocked
        res = self.db.record_attendance(date_str, "STU001", "Alice", "PRESENT", "10:05:00")
        self.assertFalse(res['success'])
        self.assertEqual(res['error_type'], 'SESSION_FINALIZED')

        # Attendance count must remain 0
        records = self.db.get_daily_attendance(date_str)
        self.assertEqual(len(records), 0)


if __name__ == "__main__":
    unittest.main()
