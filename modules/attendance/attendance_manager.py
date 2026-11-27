"""
Attendance Management and Session Controller.
Enforces the mandatory 9:30 AM to 10:00 AM local time window (Asia/Kolkata timezone).
Features:
- Strict production mode: always uses real Asia/Kolkata system time.
- Automated 10:00 AM finalization: automatically marks unrecorded registered students as ABSENT without manual button presses.
- Background session watcher daemon for live unattended finalization.
- Cold-restart session recovery if application starts/resumes after 10:00 AM.
- Duplicate attendance prevention and unknown face rejection.
"""

import os
import threading
from zoneinfo import ZoneInfo
from datetime import datetime, time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem, DEFAULT_CONFIDENCE_THRESHOLD
from modules.attendance.emotion_detector import EmotionDetector

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
SESSION_START_TIME = time(9, 30, 0)   # 09:30:00 AM IST
SESSION_END_TIME = time(10, 0, 0)    # 10:00:00 AM IST


class AttendanceTimeManager:
    """Manages timezone-aware time evaluation. Rejects simulated timestamps in production."""
    def __init__(self, timezone: ZoneInfo = KOLKATA_TZ, allow_simulation: bool = False):
        self.timezone = timezone
        self.allow_simulation = allow_simulation

    def get_time(self, simulated_time: Optional[datetime] = None) -> datetime:
        """
        Return current datetime in Asia/Kolkata timezone.
        Simulated timestamps are strictly rejected in production (when allow_simulation=False).
        Only automated test suites explicitly configured with allow_simulation=True may mock time.
        """
        if self.allow_simulation and simulated_time is not None:
            if simulated_time.tzinfo is None:
                return simulated_time.replace(tzinfo=self.timezone)
            return simulated_time.astimezone(self.timezone)
        # Production: always live system time in Asia/Kolkata
        return datetime.now(self.timezone)

    def evaluate_window(self, dt: datetime) -> Tuple[bool, str, str]:
        """
        Evaluate if a given datetime is within the 9:30 AM - 10:00 AM attendance window.
        Returns:
            (is_active, session_code, message)
        """
        if dt.tzinfo is not None:
            dt = dt.astimezone(self.timezone)
        t = dt.time()
        # Strictly before 09:30:00 AM IST
        if t < SESSION_START_TIME:
            return (
                False,
                "NOT_STARTED",
                f"Attendance session is NOT started. Permitted window opens at 09:30:00 AM IST (current: {dt.strftime('%H:%M:%S')} IST)."
            )
        # At or after 10:00:00 AM IST
        elif t >= SESSION_END_TIME:
            return (
                False,
                "CLOSED",
                f"Attendance session is CLOSED. Window closed at 10:00:00 AM IST (current: {dt.strftime('%H:%M:%S')} IST)."
            )
        # 09:30:00 <= t < 10:00:00
        else:
            return (
                True,
                "ACTIVE",
                f"Attendance session is ACTIVE until 10:00:00 AM IST (current: {dt.strftime('%H:%M:%S')} IST)."
            )


class BackgroundSessionWatcher:
    """
    Process-wide singleton background thread.
    Guarantees that across all Streamlit sessions, browser tabs, and AttendanceManager instances,
    EXACTLY ONE background watcher thread is active in the entire application process.
    """
    _lock = threading.RLock()
    _instance: Optional["BackgroundSessionWatcher"] = None

    def __init__(self, attendance_manager, check_interval_sec: float = 5.0):
        self.manager = attendance_manager
        self.interval = check_interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_or_create(cls, attendance_manager, check_interval_sec: float = 5.0) -> "BackgroundSessionWatcher":
        """
        Thread-safe singleton accessor.
        Returns the existing running watcher if active, or creates and starts a single watcher thread.
        Prevents duplicate threads across Streamlit reruns and multi-browser sessions.
        """
        with cls._lock:
            if cls._instance is None or not cls._instance.is_alive():
                cls._instance = cls(attendance_manager, check_interval_sec=check_interval_sec)
                cls._instance.start()
            else:
                # Update manager reference if needed, without creating duplicate threads
                cls._instance.manager = attendance_manager
            return cls._instance

    @classmethod
    def get_active_instance(cls) -> Optional["BackgroundSessionWatcher"]:
        with cls._lock:
            if cls._instance is not None and cls._instance.is_alive():
                return cls._instance
            return None

    @classmethod
    def stop_singleton(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        with BackgroundSessionWatcher._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name="AttendanceSessionWatcher"
                )
                self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with BackgroundSessionWatcher._lock:
            if BackgroundSessionWatcher._instance is self:
                BackgroundSessionWatcher._instance = None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if self.manager is not None:
                    self.manager.auto_check_finalization()
            except Exception:
                pass
            self._stop_event.wait(self.interval)


class AttendanceManager:
    """Orchestrates face-recognition attendance, session tracking, and automatic finalization."""
    def __init__(
        self,
        db: Optional[AttendanceDatabase] = None,
        fr_system: Optional[FaceRecognitionSystem] = None,
        time_manager: Optional[AttendanceTimeManager] = None,
        allow_simulation: bool = False,
        enable_background_watcher: bool = True,
        emotion_detector: Optional[EmotionDetector] = None
    ):
        self.db = db if db is not None else AttendanceDatabase()
        self.fr_system = fr_system if fr_system is not None else FaceRecognitionSystem()
        self.allow_simulation = allow_simulation
        self.emotion_detector = emotion_detector
        if time_manager is not None:
            self.time_manager = time_manager
            if allow_simulation:
                self.time_manager.allow_simulation = True
        else:
            self.time_manager = AttendanceTimeManager(timezone=KOLKATA_TZ, allow_simulation=allow_simulation)

        # Cold-start reconciliation: finalize past or already closed sessions upon launch
        if not self.allow_simulation:
            self.auto_check_finalization()

        # Start or reuse live process-wide background watcher thread
        self.watcher = None
        if enable_background_watcher and not self.allow_simulation:
            self.watcher = BackgroundSessionWatcher.get_or_create(self, check_interval_sec=5.0)

    def auto_check_finalization(self, simulated_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Automatic finalization check:
        1. Automatically reconciles any past unfinalized dates.
        2. If current time is at or past 10:00 AM IST and today's session is not finalized:
           Automatically marks unrecorded registered students as ABSENT and locks the session.
        Does NOT require any manual button click or camera submission.
        """
        now = self.time_manager.get_time(simulated_time if self.allow_simulation else None)
        today_date = now.strftime("%Y-%m-%d")

        # 1. Past unfinalized dates reconciliation
        unfinalized_dates = self.db.get_unfinalized_dates(before_or_on_date=today_date)
        for past_d in unfinalized_dates:
            if past_d < today_date:
                self.db.finalize_session(date=past_d, finalized_at=now.isoformat())

        # 2. Today's session finalization check if at or after 10:00:00 AM IST
        if now.time() >= SESSION_END_TIME:
            if not self.db.is_session_finalized(today_date):
                # Automatically finalize session
                res = self.db.finalize_session(
                    date=today_date,
                    finalized_at=now.isoformat()
                )
                return res
        return None

    def get_session_status(self, simulated_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Retrieve live session state, time information, and student counts for today.
        Triggers automatic finalization check seamlessly on inquiry.
        """
        # Run automatic check
        self.auto_check_finalization(simulated_time=simulated_time)

        now = self.time_manager.get_time(simulated_time if self.allow_simulation else None)
        today_date = now.strftime("%Y-%m-%d")

        is_finalized = self.db.is_session_finalized(today_date)
        is_active, code, msg = self.time_manager.evaluate_window(now)

        summary = self.db.get_attendance_summary(today_date)

        if is_finalized:
            display_status = "FINALIZED"
            eligible_c = summary.get('total_registered', 0)
            present_c = summary.get('present_count', 0)
            absent_c = summary.get('absent_count', 0)
            post_reg_c = summary.get('post_session_registered', 0)

            if eligible_c == 0:
                if post_reg_c > 0:
                    display_msg = (
                        f"Attendance session for {today_date} finalized with 0 eligible students enrolled during the 09:30–10:00 AM window. "
                        f"{post_reg_c} student{'s' if post_reg_c != 1 else ''} registered after 10:00 AM will become eligible starting on the next session day."
                    )
                else:
                    display_msg = f"Attendance session for {today_date} is FINALIZED and locked (0 students enrolled)."
            elif absent_c > 0:
                display_msg = (
                    f"Attendance session for {today_date} is FINALIZED and locked. "
                    f"Roster: {eligible_c} | Present: {present_c} | Unrecorded Marked Absent: {absent_c}."
                )
            else:
                display_msg = (
                    f"Attendance session for {today_date} is FINALIZED and locked. "
                    f"All {eligible_c} enrolled students were marked PRESENT (100% attendance)."
                )
        elif is_active:
            display_status = "ACTIVE"
            display_msg = msg
        elif code == "NOT_STARTED":
            display_status = "NOT_STARTED"
            display_msg = msg
        else:
            display_status = "CLOSED"
            display_msg = msg

        return {
            'date': today_date,
            'current_time_str': now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            'time_only_str': now.strftime("%H:%M:%S"),
            'status': display_status,
            'status_message': display_msg,
            'is_active': is_active and not is_finalized,
            'is_finalized': is_finalized,
            'summary': summary
        }

    def process_frame_attendance(
        self,
        image: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        simulated_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Process a live webcam frame or uploaded image:
        1. Checks time window (9:30 - 10:00 AM Asia/Kolkata).
        2. Detects all faces in frame.
        3. Identifies registered students and automatically records attendance.
        4. Prevents duplicate entries.
        5. Displays UNKNOWN STUDENT for unregistered faces without recording attendance.
        """
        now = self.time_manager.get_time(simulated_time if self.allow_simulation else None)
        today_date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        session_state = self.get_session_status(simulated_time=simulated_time)

        # Detect and classify faces
        predictions = self.fr_system.predict_faces(image, confidence_threshold=confidence_threshold)

        # Predict emotions for each detected face independently (does NOT alter attendance logic)
        if predictions and image is not None and len(image.shape) >= 2:
            if self.emotion_detector is None:
                try:
                    self.emotion_detector = EmotionDetector.get_instance()
                except Exception:
                    self.emotion_detector = None

            if self.emotion_detector is not None:
                for p in predictions:
                    bbox = p.get('bbox')
                    if bbox:
                        x, y, w, h = bbox
                        ih, iw = image.shape[:2]
                        x1, y1 = max(0, int(x)), max(0, int(y))
                        x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
                        if x2 > x1 and y2 > y1:
                            face_crop = image[y1:y2, x1:x2]
                            p['emotion'] = self.emotion_detector.predict_emotion(face_crop)
                        else:
                            p['emotion'] = None
                    else:
                        p['emotion'] = None

        annotated_image = self.fr_system.draw_annotations(image, predictions)

        # If session is NOT active, do not mark attendance
        if not session_state['is_active']:
            return {
                'allowed': False,
                'session_status': session_state['status'],
                'message': session_state['status_message'],
                'predictions': predictions,
                'annotated_image': annotated_image,
                'marked_students': [],
                'duplicates_prevented': [],
                'unknowns_ignored': sum(1 for p in predictions if not p['is_known']),
                'time_str': time_str,
                'date': today_date
            }

        marked_students = []
        duplicates_prevented = []
        unknowns_ignored = 0

        # Process each face independently
        for p in predictions:
            if p['is_known']:
                sid = p['student_id']
                sname = p['name']
                conf = p['confidence']

                rec_result = self.db.record_attendance(
                    date=today_date,
                    student_id=sid,
                    name=sname,
                    status="PRESENT",
                    identification_time=time_str,
                    confidence=conf,
                    session_id=today_date
                )

                if rec_result['success']:
                    marked_students.append({
                        'student_id': sid,
                        'name': sname,
                        'status': 'PRESENT',
                        'time': time_str,
                        'confidence': conf
                    })
                elif rec_result.get('already_marked'):
                    duplicates_prevented.append({
                        'student_id': sid,
                        'name': sname,
                        'message': rec_result['message']
                    })
            else:
                unknowns_ignored += 1

        summary = self.db.get_attendance_summary(today_date)

        return {
            'allowed': True,
            'session_status': session_state['status'],
            'message': f"Processed {len(predictions)} face(s): {len(marked_students)} marked PRESENT, {len(duplicates_prevented)} duplicate(s) ignored, {unknowns_ignored} unknown(s) ignored.",
            'predictions': predictions,
            'annotated_image': annotated_image,
            'marked_students': marked_students,
            'duplicates_prevented': duplicates_prevented,
            'unknowns_ignored': unknowns_ignored,
            'time_str': time_str,
            'date': today_date,
            'summary': summary
        }

    def finalize_attendance_session(
        self,
        date: Optional[str] = None,
        simulated_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Finalize session:
        Enrolled students not identified are marked ABSENT.
        Locks further attendance changes. Idempotent: repeated calls do not duplicate records.
        """
        now = self.time_manager.get_time(simulated_time if self.allow_simulation else None)
        target_date = date if date is not None else now.strftime("%Y-%m-%d")

        result = self.db.finalize_session(
            date=target_date,
            finalized_at=now.isoformat()
        )
        return result
