"""
Attendance Report Generation and Export.
Generates compliant CSV and Excel (.xlsx) reports with date, student ID, student name,
attendance status, exact identification time, and recognition confidence.
Biometric data is strictly protected and never exported.
"""

import io
import pandas as pd
from typing import Optional, Dict, Any, List
from modules.attendance.database import AttendanceDatabase


def get_attendance_dataframe(date: str, db: AttendanceDatabase) -> pd.DataFrame:
    """Retrieve and format attendance records into a structured pandas DataFrame."""
    records = db.get_daily_attendance(date)
    if not records:
        return pd.DataFrame(columns=[
            "Date", "Student ID", "Student Name", "Attendance Status", "Identification Time", "Recognition Confidence"
        ])

    rows = []
    for r in records:
        conf_val = r.get("confidence")
        if conf_val is not None and conf_val > 0.0:
            conf_str = f"{conf_val * 100:.1f}%"
        else:
            conf_str = "N/A"

        ident_time = r.get("identification_time") or "N/A"

        rows.append({
            "Date": r.get("date", date),
            "Student ID": r.get("student_id", ""),
            "Student Name": r.get("name", ""),
            "Attendance Status": r.get("status", "ABSENT"),
            "Identification Time": ident_time,
            "Recognition Confidence": conf_str
        })

    df = pd.DataFrame(rows)
    return df


def export_attendance_csv(date: str, db: AttendanceDatabase) -> str:
    """Generate attendance report in CSV format."""
    df = get_attendance_dataframe(date, db)
    return df.to_csv(index=False, encoding="utf-8")


def export_attendance_excel(date: str, db: AttendanceDatabase) -> bytes:
    """Generate attendance report in Excel (.xlsx) binary format."""
    df = get_attendance_dataframe(date, db)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sheet_name = f"Attendance_{date}"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 4, 15)
    return buffer.getvalue()
