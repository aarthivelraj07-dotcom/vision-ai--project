"""
Tests for Task 4 Streamlit User Interface Module.
"""

import unittest
from unittest.mock import patch, MagicMock

import modules.nationality_detection.ui as ui_module


class TestTask4UI(unittest.TestCase):
    """Verifies UI module structure and helper functions."""

    def test_01_ui_module_imports_cleanly(self):
        self.assertTrue(hasattr(ui_module, "run_nationality_app"))

    @patch("modules.nationality_detection.ui.st")
    def test_02_run_nationality_app_renders_header(self, mock_st):
        # Configure mocks to simulate initial page state without uploaded file
        mock_st.sidebar = MagicMock()
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.file_uploader.return_value = None
        mock_st.selectbox.return_value = "India (IND)"
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        ui_module.run_nationality_app()

        # Verify page configuration and header were called
        mock_st.set_page_config.assert_called_once()
        mock_st.title.assert_any_call("🌍 Task 4: Nationality Detection & Conditional Profile")


if __name__ == "__main__":
    unittest.main()
