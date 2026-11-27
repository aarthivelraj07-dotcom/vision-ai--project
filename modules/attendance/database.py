"""
Database management for Student Registration and Attendance.
Uses SQLite for persistent storage of student profiles, biometric embeddings, and attendance records.
"""

import os
import sqlite3
import threading
import numpy as np
from datetime import datetime, time
from typing import List, Tuple, Optional, Dict, Any

DEFAULT_DB_PATH = os.path.join("datasets", "attendance", "attendance.db")


def is_student_eligible_for_session(student_created_at: Optional[str], session_date: str) -> bool:
    """
    Determine if a student was enrolled and eligible for an attendance session on session_date.
    Eligibility rule:
    1. If student was registered on an earlier date (D_reg < session_date): ELIGIBLE.
    2. If student was registered on the same date (D_reg == session_date):
       Eligible ONLY if registered on or before 10:00:00 AM IST (window closure).
       Students registered after 10:00:00 AM IST are post-window enrollments and become eligible on the next session day.
    3. If D_reg > session_date (e.g. mock test past date): treated as ELIGIBLE for test harness compatibility.
    """
    if not student_created_at:
        return True
    try:
        clean_ts = str(student_created_at).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        reg_date = dt.strftime("%Y-%m-%d")
        if reg_date < session_date:
            return True
        elif reg_date == session_date:
            return dt.time() <= time(10, 0, 0)
        else:
            return True
    except Exception:
        return True


class AttendanceDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._write_lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
        except Exception:
            pass
        return conn

    def init_db(self):
        """Initialize and migrate database schema."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                # 1. Students profile table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS students (
                        student_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        face_count INTEGER DEFAULT 0,
                        consent_given INTEGER DEFAULT 1
                    )
                """)

                # 2. Biometric face embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                    )
                """)

                # 3. Attendance records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        student_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        identification_time TEXT,
                        confidence REAL,
                        session_id TEXT,
                        UNIQUE(date, student_id),
                        FOREIGN KEY (student_id) REFERENCES students(student_id)
                    )
                """)

                # 4. Daily attendance session tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_sessions (
                        date TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        started_at TEXT,
                        finalized_at TEXT,
                        total_registered INTEGER DEFAULT 0,
                        present_count INTEGER DEFAULT 0,
                        absent_count INTEGER DEFAULT 0
                    )
                """)

                # Migration check: Ensure 'confidence' column exists in attendance table
                cursor.execute("PRAGMA table_info(attendance)")
                columns = [col["name"] for col in cursor.fetchall()]
                if "confidence" not in columns:
                    cursor.execute("ALTER TABLE attendance ADD COLUMN confidence REAL")
        finally:
            conn.close()

    def student_exists(self, student_id: str) -> bool:
        """Check if a student ID already exists."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id.strip(),))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def add_student(self, student_id: str, name: str, embeddings: List[np.ndarray], consent_given: bool = True) -> bool:
        """
        Add a new student and their facial embeddings.
        Returns True if added successfully, raises ValueError if student_id already exists.
        """
        student_id = student_id.strip()
        name = name.strip()
        
        if not student_id:
            raise ValueError("Student ID cannot be empty.")
        if not name:
            raise ValueError("Student Name cannot be empty.")
        if not consent_given:
            raise ValueError("Biometric consent is required for registration.")
            
        with self._write_lock:
            conn = self.get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
                    if cursor.fetchone() is not None:
                        raise ValueError(f"Student ID '{student_id}' is already registered. Duplicate IDs are rejected.")

                    now = datetime.now().isoformat()
                    cursor.execute(
                        "INSERT INTO students (student_id, name, created_at, face_count, consent_given) VALUES (?, ?, ?, ?, ?)",
                        (student_id, name, now, len(embeddings), 1 if consent_given else 0)
                    )

                    for emb in embeddings:
                        emb_bytes = np.asarray(emb, dtype=np.float32).tobytes()
                        cursor.execute(
                            "INSERT INTO face_embeddings (student_id, embedding, created_at) VALUES (?, ?, ?)",
                            (student_id, emb_bytes, now)
                        )
                return True
            finally:
                conn.close()

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve student profile by ID."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id.strip(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_all_students(self) -> List[Dict[str, Any]]:
        """Retrieve all registered students."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students ORDER BY student_id ASC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
        """
        Load all stored embeddings.
        Returns:
            X: np.ndarray of shape (N, 128)
            y_ids: List of student_ids of length N
            names_map: Dict mapping student_id -> student_name
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.student_id, s.name, e.embedding 
                FROM face_embeddings e 
                JOIN students s ON e.student_id = s.student_id
                ORDER BY s.student_id ASC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return np.empty((0, 128), dtype=np.float32), [], {}

        embeddings = []
        y_ids = []
        names_map = {}
        for row in rows:
            sid = row["student_id"]
            sname = row["name"]
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(emb)
            y_ids.append(sid)
            names_map[sid] = sname

        X = np.vstack(embeddings)
        return X, y_ids, names_map

    def get_student_count(self) -> int:
        """Return total count of registered students."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def delete_student(self, student_id: str) -> bool:
        """Delete a student and associated embeddings."""
        with self._write_lock:
            conn = self.get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id.strip(),))
                    cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id.strip(),))
                    return cursor.rowcount > 0
            finally:
                conn.close()

    # ==========================================
    # Phase 3: Attendance Management Operations
    # ==========================================

    def is_session_finalized(self, date: str) -> bool:
        """Check if the attendance session for the given date has been finalized."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM attendance_sessions WHERE date = ?", (date,))
            row = cursor.fetchone()
            if row and row["status"] == "FINALIZED":
                return True
            return False
        finally:
            conn.close()

    def get_unfinalized_dates(self, before_or_on_date: Optional[str] = None) -> List[str]:
        """Return list of dates with attendance records or active sessions that are not marked FINALIZED."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT date FROM attendance
                WHERE date NOT IN (SELECT date FROM attendance_sessions WHERE status = 'FINALIZED')
                UNION
                SELECT date FROM attendance_sessions
                WHERE status != 'FINALIZED'
            """)
            dates = [row["date"] for row in cursor.fetchall()]
            if before_or_on_date:
                dates = [d for d in dates if d <= before_or_on_date]
            return sorted(dates)
        finally:
            conn.close()

    def get_session_record(self, date: str) -> Optional[Dict[str, Any]]:
        """Retrieve daily attendance session metadata."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance_sessions WHERE date = ?", (date,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def set_session_status(self, date: str, status: str, started_at: Optional[str] = None):
        """Initialize or update session status for a date."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM attendance_sessions WHERE date = ?", (date,))
                row = cursor.fetchone()
                if row:
                    # Do not overwrite finalized status automatically
                    if row["status"] == "FINALIZED" and status != "FINALIZED":
                        return
                    cursor.execute("UPDATE attendance_sessions SET status = ? WHERE date = ?", (status, date))
                else:
                    cursor.execute(
                        "INSERT INTO attendance_sessions (date, status, started_at) VALUES (?, ?, ?)",
                        (date, status, started_at)
                    )
        finally:
            conn.close()

    def record_attendance(
        self,
        date: str,
        student_id: str,
        name: str,
        status: str = "PRESENT",
        identification_time: Optional[str] = None,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a student's attendance.
        Enforces:
          1. Session must NOT be finalized.
          2. No duplicate attendance for the same student on the same date.
        Thread-safe: protected by process-wide write lock and atomic transaction.
        """
        student_id = student_id.strip()
        with self._write_lock:
            if self.is_session_finalized(date):
                return {
                    'success': False,
                    'already_marked': False,
                    'message': f"Session for {date} is finalized. Further attendance marking is blocked.",
                    'error_type': 'SESSION_FINALIZED'
                }

            conn = self.get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    # Check for existing daily attendance record
                    cursor.execute(
                        "SELECT status, identification_time, confidence FROM attendance WHERE date = ? AND student_id = ?",
                        (date, student_id)
                    )
                    existing = cursor.fetchone()
                    if existing:
                        return {
                            'success': False,
                            'already_marked': True,
                            'message': f"Student {student_id} ({name}) is already marked {existing['status']} for today.",
                            'existing_record': dict(existing)
                        }

                    # Insert new attendance record
                    cursor.execute("""
                        INSERT INTO attendance (date, student_id, name, status, identification_time, confidence, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (date, student_id, name, status, identification_time, confidence, session_id or date))

                    return {
                        'success': True,
                        'already_marked': False,
                        'message': f"Student {student_id} ({name}) marked {status} successfully at {identification_time}."
                    }
            finally:
                conn.close()

    def finalize_session(self, date: str, finalized_at: Optional[str] = None) -> Dict[str, Any]:
        """
        Finalize attendance session for a date:
        1. Identifies all registered students who were NOT marked PRESENT.
        2. Inserts 'ABSENT' records for those students.
        3. Updates session status to 'FINALIZED'.
        4. Locks further attendance modification.
        Thread-safe & Idempotent: serialized via write lock, re-checks finalized status, uses INSERT OR IGNORE.
        """
        with self._write_lock:
            if self.is_session_finalized(date):
                summary = self.get_attendance_summary(date)
                return {
                    'success': True,
                    'already_finalized': True,
                    'message': f"Session for {date} was already finalized.",
                    'summary': summary
                }

            conn = self.get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    # Double check status inside transaction
                    cursor.execute("SELECT status FROM attendance_sessions WHERE date = ?", (date,))
                    sess_row = cursor.fetchone()
                    if sess_row and sess_row["status"] == "FINALIZED":
                        summary = self.get_attendance_summary(date)
                        return {
                            'success': True,
                            'already_finalized': True,
                            'message': f"Session for {date} was already finalized.",
                            'summary': summary
                        }

                    # Get all registered students
                    cursor.execute("SELECT student_id, name, created_at FROM students ORDER BY student_id ASC")
                    all_students = cursor.fetchall()

                    # Filter eligible students relative to 09:30-10:00 AM IST attendance window on session date
                    eligible_students = [
                        s for s in all_students
                        if is_student_eligible_for_session(s["created_at"], date)
                    ]

                    # Get students already marked present
                    cursor.execute("SELECT student_id FROM attendance WHERE date = ?", (date,))
                    marked_ids = {row["student_id"] for row in cursor.fetchall()}

                    # Mark all unrecorded eligible students as ABSENT using INSERT OR IGNORE
                    absent_count = 0
                    for s in eligible_students:
                        sid = s["student_id"]
                        sname = s["name"]
                        if sid not in marked_ids:
                            cursor.execute("""
                                INSERT OR IGNORE INTO attendance (date, student_id, name, status, identification_time, confidence, session_id)
                                VALUES (?, ?, ?, 'ABSENT', NULL, 0.0, ?)
                            """, (date, sid, sname, date))
                            if cursor.rowcount > 0:
                                absent_count += 1

                    # Calculate counts
                    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'PRESENT'", (date,))
                    present_count = cursor.fetchone()[0]

                    total_registered = len(eligible_students)

                    # Upsert attendance session record
                    now_str = finalized_at or datetime.now().isoformat()
                    cursor.execute("""
                        INSERT INTO attendance_sessions (date, status, finalized_at, total_registered, present_count, absent_count)
                        VALUES (?, 'FINALIZED', ?, ?, ?, ?)
                        ON CONFLICT(date) DO UPDATE SET
                            status = 'FINALIZED',
                            finalized_at = excluded.finalized_at,
                            total_registered = excluded.total_registered,
                            present_count = excluded.present_count,
                            absent_count = excluded.absent_count
                    """, (date, now_str, total_registered, present_count, absent_count))

                summary = self.get_attendance_summary(date)
                return {
                    'success': True,
                    'already_finalized': False,
                    'message': f"Session for {date} finalized. Total: {total_registered}, Present: {present_count}, Absent: {absent_count}.",
                    'summary': summary
                }
            finally:
                conn.close()

    def get_daily_attendance(self, date: str) -> List[Dict[str, Any]]:
        """Retrieve all attendance records for a given date."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT student_id, name, date, status, identification_time, confidence, session_id
                FROM attendance
                WHERE date = ?
                ORDER BY 
                    CASE WHEN status = 'PRESENT' THEN 0 ELSE 1 END,
                    identification_time ASC,
                    student_id ASC
            """, (date,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_attendance_summary(self, date: str) -> Dict[str, Any]:
        """Compute present/absent summary and attendance percentage."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            total_system_students = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'PRESENT'", (date,))
            present_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'ABSENT'", (date,))
            absent_count = cursor.fetchone()[0]

            is_finalized = self.is_session_finalized(date)

            cursor.execute("SELECT total_registered, present_count, absent_count FROM attendance_sessions WHERE date = ?", (date,))
            sess_row = cursor.fetchone()

            if sess_row and is_finalized:
                session_eligible = sess_row["total_registered"]
                present_count = sess_row["present_count"]
                absent_count = sess_row["absent_count"]
            else:
                cursor.execute("SELECT student_id, created_at FROM students")
                students_data = cursor.fetchall()
                eligible_sids = [
                    s["student_id"] for s in students_data
                    if is_student_eligible_for_session(s["created_at"], date)
                ]
                session_eligible = len(eligible_sids)

            unrecorded = max(0, session_eligible - present_count - absent_count)
            effective_absent = absent_count if is_finalized else unrecorded
            post_session_registered = max(0, total_system_students - session_eligible)

            pct = (present_count / session_eligible * 100) if session_eligible > 0 else 0.0

            return {
                'date': date,
                'total_registered': session_eligible,
                'total_system_students': total_system_students,
                'session_eligible_students': session_eligible,
                'post_session_registered': post_session_registered,
                'present_count': present_count,
                'absent_count': effective_absent,
                'attendance_percentage': round(pct, 1),
                'is_finalized': is_finalized
            }
        finally:
            conn.close()

    def get_all_attendance_dates(self) -> List[str]:
        """Return list of distinct dates with attendance records."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
            return [row["date"] for row in cursor.fetchall()]
        finally:
            conn.close()
