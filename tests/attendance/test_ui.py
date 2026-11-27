"""
UI and Application Configuration Verification Tests.
Validates Streamlit UI helpers, mandatory environment-based Admin PIN authentication,
session state lifecycle, and app entry point stability.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.attendance.ui import get_admin_pin, is_admin_pin_configured, verify_admin_pin
from modules.attendance import ui as attendance_ui


class TestUIAndConfiguration(unittest.TestCase):
    def test_01_unconfigured_admin_pin_behavior(self):
        """Verify that missing ATTENDANCE_ADMIN_PIN disables access and provides NO default fallback."""
        old_val = os.environ.pop("ATTENDANCE_ADMIN_PIN", None)
        try:
            self.assertFalse(is_admin_pin_configured())
            self.assertEqual(get_admin_pin(), "")
            # All attempts including old default 'admin123' must be rejected
            self.assertFalse(verify_admin_pin("admin123"))
            self.assertFalse(verify_admin_pin("1234"))
            self.assertFalse(verify_admin_pin(""))
            self.assertFalse(verify_admin_pin("any_password"))
        finally:
            if old_val is not None:
                os.environ["ATTENDANCE_ADMIN_PIN"] = old_val

    def test_02_environment_configured_admin_pin(self):
        """Verify dynamic configuration of Admin PIN strictly via environment variable."""
        old_val = os.environ.get("ATTENDANCE_ADMIN_PIN")
        try:
            os.environ["ATTENDANCE_ADMIN_PIN"] = "SecureUniPass2026!"
            self.assertTrue(is_admin_pin_configured())
            self.assertEqual(get_admin_pin(), "SecureUniPass2026!")
            self.assertTrue(verify_admin_pin("SecureUniPass2026!"))
            # Old default must be rejected
            self.assertFalse(verify_admin_pin("admin123"))
            self.assertFalse(verify_admin_pin("wrong_pass"))
        finally:
            if old_val is not None:
                os.environ["ATTENDANCE_ADMIN_PIN"] = old_val
            else:
                os.environ.pop("ATTENDANCE_ADMIN_PIN", None)

    def test_03_app_module_import(self):
        """Verify app entry point imports cleanly without exceptions."""
        import app
        self.assertTrue(hasattr(app, 'run_app'))

    def test_04_ui_module_attributes(self):
        """Verify that UI contains all necessary section rendering functions."""
        import modules.attendance.ui as ui
        self.assertTrue(callable(ui.render_registration_tab))
        self.assertTrue(callable(ui.render_recognition_tab))
        self.assertTrue(callable(ui.render_directory_tab))
        self.assertTrue(callable(ui.run_app))

    def test_system_instances_defer_emotion_model_loading(self):
        session_state = MagicMock()
        session_state.__contains__.return_value = False
        with (
            patch.object(attendance_ui.st, "session_state", session_state),
            patch.object(attendance_ui, "AttendanceDatabase") as database_type,
            patch.object(attendance_ui, "FaceRecognitionSystem") as face_type,
            patch.object(attendance_ui, "StudentRegistrationManager") as registration_type,
            patch.object(attendance_ui, "AttendanceManager") as attendance_type,
            patch.object(attendance_ui, "get_emotion_detector") as emotion_loader,
        ):
            database = database_type.return_value
            face_system = face_type.return_value
            attendance_ui.get_system_instances()

        emotion_loader.assert_not_called()
        registration_type.assert_called_once_with(db=database, fr_system=face_system)
        attendance_type.assert_called_once_with(
            db=database,
            fr_system=face_system,
            emotion_detector=None
        )

    def test_run_app_renders_only_open_tab_and_reuses_session_summary(self):
        tabs = [MagicMock() for _ in range(5)]
        for tab in tabs:
            tab.open = False
        tabs[0].open = True

        summary = {
            "total_registered": 0,
            "total_system_students": 0,
            "present_count": 0,
            "absent_count": 0,
        }
        session_info = {
            "summary": summary,
            "is_active": False,
            "status": "CLOSED",
        }
        db = MagicMock()
        face_system = MagicMock()
        registration_manager = MagicMock()
        attendance_manager = MagicMock()
        attendance_manager.get_session_status.return_value = session_info
        mock_st = MagicMock()
        mock_st.tabs.return_value = tabs
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]

        with (
            patch.object(attendance_ui, "st", mock_st),
            patch.object(attendance_ui, "apply_theme"),
            patch.object(attendance_ui, "render_page_header"),
            patch.object(
                attendance_ui,
                "get_system_instances",
                return_value=(db, face_system, registration_manager, attendance_manager),
            ),
            patch.object(attendance_ui, "render_attendance_dashboard_tab") as dashboard,
            patch.object(attendance_ui, "render_registration_tab") as registration,
            patch.object(attendance_ui, "render_reports_tab") as reports,
            patch.object(attendance_ui, "render_recognition_tab") as recognition,
            patch.object(attendance_ui, "render_directory_tab") as directory,
        ):
            attendance_ui.run_app()

        self.assertEqual(mock_st.tabs.call_args.kwargs["on_change"], "rerun")
        attendance_manager.get_session_status.assert_called_once_with()
        db.get_attendance_summary.assert_not_called()
        db.get_student_count.assert_not_called()
        dashboard.assert_called_once_with(
            attendance_manager,
            face_system,
            db,
            session_info=session_info,
        )
        registration.assert_not_called()
        reports.assert_not_called()
        recognition.assert_not_called()
        directory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
