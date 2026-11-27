"""
Streamlit User Interface for Attendance & Emotion Detection Project.
Phase 2 & Phase 3 Modules:
- Student Registration (Admin secured, Biometric consent)
- Live Attendance Dashboard (9:30 - 10:00 AM IST time window, automatic presence marking, duplicate prevention, unknown rejection)
- Session Finalization (Mark unrecorded registered students as ABSENT at 10:00 AM)
- Face Recognition Testing & Live Verification
- Reports & History Export (CSV and Excel formats)
- Registered Students Directory
"""

import os
import hmac
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple
import streamlit as st

from modules.attendance.database import AttendanceDatabase
from modules.attendance.face_recognition import FaceRecognitionSystem, DEFAULT_CONFIDENCE_THRESHOLD
from modules.attendance.registration import StudentRegistrationManager
from modules.attendance.attendance_manager import AttendanceManager, AttendanceTimeManager, KOLKATA_TZ
from modules.attendance.reports import export_attendance_csv, export_attendance_excel, get_attendance_dataframe
from modules.attendance.emotion_detector import EmotionDetector, EMOTION_CLASSES
from modules.common.theme import apply_theme, render_page_header

# Secure Environment-Configured Admin Credential
def get_admin_pin() -> str:
    """Retrieve administrator PIN from environment variable. Returns empty string if not configured."""
    return os.environ.get("ATTENDANCE_ADMIN_PIN", "").strip()


def is_admin_pin_configured() -> bool:
    """Check if administrator PIN has been set in the environment."""
    return len(get_admin_pin()) > 0


def verify_admin_pin(input_pin: str) -> bool:
    """Constant-time verification of admin PIN against configured secret."""
    if not input_pin or not is_admin_pin_configured():
        return False
    configured_pin = get_admin_pin()
    return hmac.compare_digest(input_pin.strip(), configured_pin)


def render_admin_pin_setup_guide(section_name: str = "this section"):
    """Render clear, professional setup instructions when ATTENDANCE_ADMIN_PIN is not configured."""
    st.error(f"🔒 **{section_name.title()} Locked: Administrator PIN Not Configured**")
    st.markdown(
        f"""
        To protect student biometric facial embeddings and attendance records, **{section_name}** 
        requires authentication with an Administrator PIN.
        
        The `ATTENDANCE_ADMIN_PIN` environment variable has not been configured in the running server environment.
        """
    )
    with st.expander("🛠️ How to Configure Administrator PIN on Windows (Step-by-Step)", expanded=True):
        st.markdown(
            """
            To enable administrator features, configure the PIN in your Windows terminal and restart Streamlit:

            **Option 1: Windows PowerShell (Recommended)**
            ```powershell
            # 1. Stop current Streamlit app (Ctrl+C in terminal)
            # 2. Set your custom secure PIN:
            $env:ATTENDANCE_ADMIN_PIN = "YourSecurePIN123"
            # 3. Launch Streamlit:
            python -m streamlit run app.py
            ```

            **Option 2: Windows Command Prompt (cmd.exe)**
            ```cmd
            :: 1. Stop current Streamlit app (Ctrl+C in terminal)
            :: 2. Set your custom secure PIN:
            set ATTENDANCE_ADMIN_PIN=YourSecurePIN123
            :: 3. Launch Streamlit:
            python -m streamlit run app.py
            ```

            **Option 3: Persistent Windows User Environment Variable**
            1. Press `Win + R`, type `sysdm.cpl`, and hit **Enter**.
            2. Go to the **Advanced** tab and click **Environment Variables**.
            3. Under **User variables for [YourUser]**, click **New...**:
               - **Variable name:** `ATTENDANCE_ADMIN_PIN`
               - **Variable value:** `YourSecurePIN123` *(choose any secure PIN)*
            4. Click **OK** to save. Open a new terminal window and run `python -m streamlit run app.py`.

            *Note: Never hardcode PINs or use default passwords. The configured PIN is never displayed in the UI or logs.*
            """
        )


# Professional Theme Styling for Task 1 — Teal + Tangerine + Pale Aqua
STREAMLIT_CSS = """
<style>
    /* Card containers */
    .custom-card {
        background: #FFFFFF !important;
        border: 1.5px solid #B7DCD7 !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(24, 59, 69, 0.06);
        color: #183B45 !important;
    }
    
    .lavender-card {
        background: #F4FBF9 !important;
        border: 1.5px solid #B7DCD7 !important;
        border-left: 4px solid #087E8B !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }
    
    .coral-card {
        background: #FFF8F4 !important;
        border: 1.5px solid #FF914D !important;
        border-left: 4px solid #FF914D !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }
    
    .success-card {
        background: #F0FAF7 !important;
        border: 1.5px solid #10B981 !important;
        border-left: 4px solid #10B981 !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }
    
    /* Status pill */
    .stat-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .pill-active {
        background-color: rgba(16, 185, 129, 0.18);
        color: #065F46;
        border: 1px solid #10B981;
    }
    .pill-not-started {
        background-color: rgba(255, 145, 77, 0.18);
        color: #C25E00;
        border: 1px solid #FF914D;
    }
    .pill-closed {
        background-color: rgba(8, 126, 139, 0.15);
        color: #183B45;
        border: 1px solid #087E8B;
    }
    .pill-finalized {
        background-color: rgba(69, 99, 106, 0.18);
        color: #183B45;
        border: 1px solid #45636A;
    }
</style>
"""


def get_emotion_detector():
    """Cache and return singleton EmotionDetector instance."""
    if 'emotion_detector' not in st.session_state:
        try:
            st.session_state.emotion_detector = EmotionDetector.get_instance()
        except Exception:
            st.session_state.emotion_detector = None
    return st.session_state.emotion_detector


def get_system_instances():
    """Cache and return shared instances of DB, Face Recognition, Registration, and Attendance Manager."""
    if 'db' not in st.session_state:
        st.session_state.db = AttendanceDatabase()
    if 'fr_system' not in st.session_state:
        st.session_state.fr_system = FaceRecognitionSystem()
    if 'reg_manager' not in st.session_state:
        st.session_state.reg_manager = StudentRegistrationManager(
            db=st.session_state.db,
            fr_system=st.session_state.fr_system
        )
    if 'att_manager' not in st.session_state:
        st.session_state.att_manager = AttendanceManager(
            db=st.session_state.db,
            fr_system=st.session_state.fr_system,
            emotion_detector=None
        )
    return st.session_state.db, st.session_state.fr_system, st.session_state.reg_manager, st.session_state.att_manager


def render_attendance_dashboard_tab(
    att_manager: AttendanceManager,
    fr_system: FaceRecognitionSystem,
    db: AttendanceDatabase,
    session_info: Optional[Dict[str, Any]] = None
):
    """Render the Main Live Attendance Dashboard with 9:30 - 10:00 AM IST enforcement."""
    st.subheader("Attendance Dashboard")
    st.markdown("Automated face-recognition attendance with mandatory time-window control (09:30 AM - 10:00 AM IST).")

    # Current Session Status (Automatic finalization is performed seamlessly if current time >= 10:00 AM IST)
    if session_info is None:
        session_info = att_manager.get_session_status()
    summary = session_info['summary']
    status_code = session_info['status']

    # Header Metric Cards
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.markdown(f"**Session Status**")
        if status_code == "ACTIVE":
            st.markdown('<span class="stat-pill pill-active">ACTIVE (9:30 - 10:00 AM)</span>', unsafe_allow_html=True)
        elif status_code == "NOT_STARTED":
            st.markdown('<span class="stat-pill pill-not-started">NOT STARTED (Opens 9:30)</span>', unsafe_allow_html=True)
        elif status_code == "FINALIZED":
            st.markdown('<span class="stat-pill pill-finalized">FINALIZED (Locked)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="stat-pill pill-closed">CLOSED (10:00 AM)</span>', unsafe_allow_html=True)
    with col_s2:
        st.metric("Current Time (IST)", session_info['time_only_str'])
    with col_s3:
        total_sys = summary.get('total_system_students', summary['total_registered'])
        eligible_sess = summary['total_registered']
        st.metric(
            "Session Eligible",
            eligible_sess,
            help=f"Students enrolled on or before 10:00 AM IST who were eligible for this session. (Total in System: {total_sys})"
        )
    with col_s4:
        st.metric("Present Today", summary['present_count'])
    with col_s5:
        st.metric("Absent Today", summary['absent_count'])

    # Post-window enrollment notice if applicable
    post_reg_count = summary.get('post_session_registered', 0)
    total_sys = summary.get('total_system_students', 0)
    if post_reg_count > 0 and session_info['is_finalized']:
        st.markdown(f"""
        <div class="lavender-card" style="padding: 12px 18px; margin-top: 10px;">
            ℹ️ <strong>Roster Notice:</strong> {post_reg_count} student{'s' if post_reg_count != 1 else ''} 
            (Total in System: <strong>{total_sys}</strong>) registered <em>after</em> the 09:30–10:00 AM IST attendance session closed.
            Per institutional policy, their attendance tracking begins on their first active session day (tomorrow at 09:30 AM IST).
        </div>
        """, unsafe_allow_html=True)

    # Session Status Banner
    if status_code == "ACTIVE":
        st.success(session_info['status_message'])
    elif status_code == "FINALIZED":
        st.info(session_info['status_message'])
    elif status_code == "NOT_STARTED":
        st.warning(session_info['status_message'])
    else:
        st.error(session_info['status_message'])

    # Action Toolbar: Refresh
    col_act1, col_act2 = st.columns([1, 3])
    with col_act1:
        if st.button("Refresh State", use_container_width=True):
            st.rerun()
    with col_act2:
        if session_info['is_finalized']:
            st.caption("🔒 Session is automatically finalized and locked. Records cannot be altered.")
        elif status_code == "ACTIVE":
            st.caption("🟢 Live session active. Present registered faces will be recorded automatically.")
        else:
            st.caption("⏸️ Session is inactive. Face detection is active for preview only; attendance recording is disabled.")

    st.markdown("---")

    # Live Attendance Webcam / Upload Section
    col_cam, col_records = st.columns([1, 1], gap="large")

    with col_cam:
        st.markdown("##### 📹 Live Camera & Attendance Ingestion")
        cam_mode = st.radio("Attendance Ingestion Source:", ["Live Webcam Feed", "Upload Classroom Image"], horizontal=True)
        frame_input = None

        if cam_mode == "Live Webcam Feed":
            col_tgl, col_ind = st.columns([2, 1])
            with col_tgl:
                cam_enabled = st.toggle(
                    "📷 Start Camera & Recognition Stream",
                    value=False,
                    key="attendance_cam_toggle",
                    help="Toggle ON to open live camera preview. Toggle OFF to immediately stop recognition and release the webcam hardware."
                )
            with col_ind:
                if cam_enabled:
                    st.markdown("<span class='stat-pill pill-active'>🟢 CAMERA LIVE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='stat-pill pill-closed'>🔴 CAMERA STANDBY</span>", unsafe_allow_html=True)

            if cam_enabled:
                st.caption("🔒 Accessing via `http://localhost:8501`. Center face in preview and click **Take Photo** to capture and evaluate for attendance.")
                cam_data = st.camera_input("Capture live classroom frame", key="attendance_camera_input")
                if cam_data is not None:
                    bytes_data = cam_data.getvalue()
                    frame_input = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                st.info("Camera is currently **OFF** and hardware resources are released. Toggle **Start Camera & Recognition Stream** above to activate live camera ingestion.")
        else:
            st.markdown("<span class='stat-pill pill-active'>📁 FILE INGESTION MODE</span>", unsafe_allow_html=True)
            upload_frame = st.file_uploader("Upload single or multi-student classroom frame:", type=["jpg", "jpeg", "png"], key="attendance_file_upload")
            if upload_frame is not None:
                bytes_data = upload_frame.read()
                frame_input = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if frame_input is not None:
            st.markdown("---")
            st.markdown("##### 🔍 Recognition & Evaluation Results")

            # Process Frame strictly using live Asia/Kolkata system time
            proc_res = att_manager.process_frame_attendance(
                image=frame_input,
                confidence_threshold=0.50
            )

            # Display annotated image
            annotated_rgb = cv2.cvtColor(proc_res['annotated_image'], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Recognition Preview (Lavender: Registered, Coral: Unknown)", use_container_width=True)

            # High-visibility Attendance Evaluation Summary
            predictions = proc_res.get('predictions', [])
            total_faces = len(predictions)
            st.markdown(f"**Total Faces Detected:** `{total_faces}`")

            if total_faces == 0:
                st.info("ℹ️ No human faces detected in the provided frame. Please ensure faces are unobstructed and well-lit.")
            else:
                for idx, p in enumerate(predictions, 1):
                    is_known = p['is_known']
                    conf = p['confidence']
                    name = p['name']
                    sid = p['student_id']
                    emo = p.get('emotion')

                    emo_str = f"Observed Emotion: <strong>{emo['predicted_class']}</strong> ({emo['confidence']:.1%})" if emo else "Emotion: N/A"

                    if is_known:
                        st.markdown(f"""
                        <div class="lavender-card">
                            <strong>Face #{idx}: Registered Student — {name} (ID: {sid})</strong><br>
                            Match Confidence: <strong>{conf:.1%}</strong> (Threshold: 50.0%)<br>
                            <em>{emo_str}</em>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="coral-card">
                            <strong>Face #{idx}: UNREGISTERED / UNKNOWN INDIVIDUAL</strong><br>
                            Match Confidence: <strong>{conf:.1%}</strong> (Below 50.0% threshold)<br>
                            <span style="color:#B91C1C; font-weight:600;">Attendance Blocked: Unknown individuals cannot be marked present.</span><br>
                            <em>{emo_str}</em>
                        </div>
                        """, unsafe_allow_html=True)

                    if emo and 'probabilities' in emo:
                        with st.expander(f"Face #{idx} ({name}) Emotion Breakdown"):
                            for e_name, e_prob in emo['probabilities'].items():
                                st.progress(min(1.0, max(0.0, float(e_prob))), text=f"{e_name}: {e_prob:.1%}")

            # Attendance Recording Outcome
            if proc_res['allowed']:
                if proc_res['marked_students']:
                    for s in proc_res['marked_students']:
                        st.success(f"✅ **Marked PRESENT:** {s['name']} (ID: {s['student_id']}) at {s['time']} IST (Confidence: {s['confidence']:.1%})")
                if proc_res['duplicates_prevented']:
                    for d in proc_res['duplicates_prevented']:
                        st.warning(f"⚠️ **Duplicate Prevented:** {d['message']}")
                if proc_res['unknowns_ignored'] > 0:
                    st.error(f"🚫 **Rejected {proc_res['unknowns_ignored']} unknown face(s).** No attendance records created.")
            else:
                st.error(f"🔒 **Attendance Recording Closed:** {proc_res['message']}")
                st.markdown(
                    """
                    > **Notice on Attendance Hours:** Official attendance marking is restricted strictly to **09:30 AM – 10:00 AM IST** local time.
                    > Outside this window, facial recognition and emotion analysis remain functional for identification preview, but no official attendance records are created or altered.
                    """
                )
                st.info("💡 **Tip:** To test face recognition, student identification, and emotion detection at any hour without time restrictions, open **'Face Recognition Testing'** in the sidebar navigation.")

    with col_records:
        st.markdown(f"##### Today's Attendance Table ({session_info['date']})")
        today_records = db.get_daily_attendance(session_info['date'])
        if today_records:
            table_rows = []
            for r in today_records:
                conf = r.get("confidence")
                table_rows.append({
                    "Student ID": r["student_id"],
                    "Student Name": r["name"],
                    "Status": r["status"],
                    "Identification Time": r.get("identification_time") or "N/A",
                    "Confidence": f"{conf*100:.1f}%" if conf and conf > 0 else "N/A"
                })
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display, use_container_width=True, height=450)
        else:
            total_students_enrolled = db.get_student_count()
            eligible_count = summary.get('total_registered', 0)
            post_reg_count = summary.get('post_session_registered', 0)

            if session_info['is_finalized'] and eligible_count == 0:
                st.info(
                    f"📋 **No attendance records for {session_info['date']}**\n\n"
                    f"• **Session Eligibility:** 0 eligible students were enrolled when today's 09:30–10:00 AM IST session closed.\n\n"
                    f"• **System Registrations:** {total_students_enrolled} student(s) enrolled in the system.\n\n"
                    f"• **Policy Note:** Students registered after 10:00 AM IST ({post_reg_count} student{'s' if post_reg_count != 1 else ''}) "
                    f"will participate starting with the next active attendance window (09:30 AM tomorrow)."
                )
            else:
                st.info(
                    f"No attendance records recorded yet for today ({session_info['date']}).\n\n"
                    f"• Eligible Students for this session: **{eligible_count}** (System Total: **{total_students_enrolled}**)\n\n"
                    f"• During the **09:30 AM – 10:00 AM IST** window, recognized students are automatically recorded here as **PRESENT**.\n\n"
                    f"• At **10:00 AM IST**, all remaining unrecorded eligible students are automatically marked **ABSENT**."
                )


def render_reports_tab(db: AttendanceDatabase):
    """Render Attendance Reports & Data Export (CSV / Excel)."""
    st.subheader("Attendance History & Reports")
    st.markdown("Inspect historical attendance sessions and download official CSV and Excel reports.")

    # Secure Admin Access Check
    if not is_admin_pin_configured():
        render_admin_pin_setup_guide("Attendance Reports & Official Data Export")
        return

    all_dates = db.get_all_attendance_dates()
    today_str = datetime.now(KOLKATA_TZ).strftime("%Y-%m-%d")
    if today_str not in all_dates:
        all_dates.insert(0, today_str)

    is_authenticated = st.session_state.get('admin_authenticated', False)

    col_sel, col_pin = st.columns([2, 1])
    with col_sel:
        selected_date = st.selectbox("Select Attendance Date:", all_dates)
    with col_pin:
        if not is_authenticated:
            pin_input = st.text_input("Admin PIN to Unlock Export:", type="password", key="export_pin")
            is_authorized = verify_admin_pin(pin_input)
            if pin_input:
                if is_authorized:
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated.")
                else:
                    st.error("Invalid Admin PIN. Access denied.")
        else:
            st.success("Admin Session Authenticated")
            is_authorized = True
            pin_input = ""

    summary = db.get_attendance_summary(selected_date)

    # Summary Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Session Eligible",
            summary['total_registered'],
            help=f"Students enrolled on or before 10:00 AM IST on {selected_date}. (Total in System: {summary.get('total_system_students', summary['total_registered'])})"
        )
    with c2:
        st.metric("Present Count", summary['present_count'])
    with c3:
        st.metric("Absent Count", summary['absent_count'])
    with c4:
        st.metric("Attendance %", f"{summary['attendance_percentage']}%")

    df = get_attendance_dataframe(selected_date, db)

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        if summary.get('is_finalized') and summary.get('total_registered', 0) == 0:
            st.info(
                f"ℹ️ **No attendance records for {selected_date}.**\n\n"
                f"The attendance session on this date finalized with **0 eligible students** enrolled during the 09:30–10:00 AM IST window. "
                f"Students registered after the window are not retroactively recorded as absent."
            )
        else:
            st.info(f"No records found for date {selected_date}.")

    st.markdown("---")
    st.markdown("##### Download Attendance Reports")

    if not is_authorized:
        st.info("Enter the valid Admin PIN above to unlock official CSV and Excel download exports.")
    else:
        st.success("Admin authorized. Select report format to download:")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_content = export_attendance_csv(selected_date, db)
            st.download_button(
                label="Download CSV Report",
                data=csv_content,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with col_dl2:
            excel_bytes = export_attendance_excel(selected_date, db)
            st.download_button(
                label="Download Excel (.xlsx) Report",
                data=excel_bytes,
                file_name=f"attendance_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="secondary",
                use_container_width=True
            )


def render_registration_tab(reg_manager: StudentRegistrationManager, fr_system: FaceRecognitionSystem, db: AttendanceDatabase):
    """Render Student Registration with Admin security and Biometric consent."""
    st.subheader("Student Registration")
    st.markdown("Register authorized students with verified facial profile photos.")

    # Check if ATTENDANCE_ADMIN_PIN is configured in environment
    if not is_admin_pin_configured():
        render_admin_pin_setup_guide("Student Registration & Profile Enrollment")
        return

    # Biometric Security / Access Restriction
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        with st.container():
            st.warning("Registration functions are restricted to authorized administrators.")
            col1, col2 = st.columns([3, 1])
            with col1:
                pin_input = st.text_input("Enter Admin PIN to unlock registration:", type="password", key="admin_pin_input")
            with col2:
                st.write("")
                st.write("")
                if st.button("Unlock Access", type="primary"):
                    if verify_admin_pin(pin_input):
                        st.session_state.admin_authenticated = True
                        st.success("Access granted.")
                        st.rerun()
                    else:
                        st.error("Invalid PIN. Access denied.")
            return

    # Registration Form
    with st.expander("Admin Session Active (Click to Lock)", expanded=False):
        if st.button("Lock Registration"):
            st.session_state.admin_authenticated = False
            st.rerun()

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("##### 1. Student Identification")
        student_id = st.text_input("Unique Student ID *", placeholder="e.g. STU1001", help="Must be unique alphanumeric ID.")
        student_name = st.text_input("Student Full Name *", placeholder="e.g. Jane Doe")

        st.markdown("##### 2. Student Consent Agreement")
        consent = st.checkbox(
            "I confirm that the student has provided explicit consent for facial profile verification and local identification.",
            value=False,
            help="Institutional regulations require recorded consent before capturing facial profiles."
        )

        st.markdown("##### 3. Capture / Upload Consented Images")
        upload_mode = st.radio("Face Capture Method:", ["Webcam Camera", "Upload Image Files"], horizontal=True)

        captured_images = []
        if upload_mode == "Webcam Camera":
            st.info("Take 3 to 5 clear snapshots with slight head variations (frontal, slight left, slight right).")
            reg_cam_enabled = st.toggle(
                "📷 Turn On Registration Camera",
                value=False,
                key="registration_cam_toggle",
                help="Toggle ON to open live camera preview. Toggle OFF to release webcam."
            )
            if reg_cam_enabled:
                camera_img = st.camera_input("Take Snapshot", key="registration_camera_input")
                st.caption("🔒 Accessing via `http://localhost:8501`. If preview is blank, check browser camera permissions and ensure other apps are closed.")
                if camera_img is not None:
                    bytes_data = camera_img.getvalue()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Maintain list of captured webcam images in session state
                    if 'webcam_images' not in st.session_state:
                        st.session_state.webcam_images = []
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Add Snapshot to Registration", type="secondary"):
                            is_valid, msg, _ = reg_manager.validate_image_for_registration(cv_img)
                            if is_valid:
                                st.session_state.webcam_images.append(cv_img)
                                st.success(f"Snapshot #{len(st.session_state.webcam_images)} accepted!")
                            else:
                                st.error(f"Snapshot rejected: {msg}")
                    with col_btn2:
                        if st.button("Clear Snapshots"):
                            st.session_state.webcam_images = []
                            st.rerun()
            else:
                st.info("Camera is currently OFF and released. Toggle the switch above to activate your webcam.")

            captured_images = st.session_state.get('webcam_images', [])
            st.write(f"Accumulated snapshots: **{len(captured_images)}**")

        else:
            uploaded_files = st.file_uploader(
                "Upload 1 to 5 face photos (.jpg, .jpeg, .png):",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            if uploaded_files:
                for uf in uploaded_files:
                    bytes_data = uf.read()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        captured_images.append(cv_img)

        st.markdown("---")
        submit_btn = st.button("Complete Registration & Fit Model", type="primary", use_container_width=True)

        if submit_btn:
            if not student_id.strip():
                st.error("Please enter a valid Student ID.")
            elif not student_name.strip():
                st.error("Please enter the student's full name.")
            elif not consent:
                st.error("Cannot proceed: Consent checkbox must be confirmed.")
            elif len(captured_images) == 0:
                st.error("Please capture or upload at least one face photo.")
            else:
                with st.spinner("Validating images, processing facial profile, and updating records..."):
                    result = reg_manager.register_student(
                        student_id=student_id,
                        name=student_name,
                        images=captured_images,
                        consent_given=consent
                    )
                if result['success']:
                    st.success(f"{result['message']}")
                    st.balloons()
                    # Reset accumulated images
                    if 'webcam_images' in st.session_state:
                        st.session_state.webcam_images = []
                    st.rerun()
                else:
                    st.error(f"Registration Failed: {result['message']}")

    with col_preview:
        st.markdown("##### Real-Time Validation Feedback")
        if captured_images:
            st.write(f"Sample images queued for registration: **{len(captured_images)}**")
            for idx, img in enumerate(captured_images[:4], 1):
                is_valid, msg, face = reg_manager.validate_image_for_registration(img)
                preview_copy = img.copy()
                if face is not None:
                    fx, fy, fw, fh = face['bbox']
                    color = (34, 197, 94) if is_valid else (94, 63, 244)
                    cv2.rectangle(preview_copy, (fx, fy), (fx+fw, fy+fh), color, 2)
                
                rgb_preview = cv2.cvtColor(preview_copy, cv2.COLOR_BGR2RGB)
                st.image(rgb_preview, caption=f"Sample {idx}: {msg}", use_container_width=True)
        else:
            st.info("Capture snapshots or upload images to preview real-time face detection and quality validation.")


def render_recognition_tab(fr_system: FaceRecognitionSystem, db: AttendanceDatabase, emo_detector: Optional[EmotionDetector] = None):
    """Render Face Recognition & Emotion Testing with adjustable confidence threshold."""
    st.subheader("Face Recognition & Emotion Testing")
    st.markdown("Test multi-face recognition, emotion classification, and unknown student rejection.")

    col_ctrl, col_stats = st.columns([2, 1])
    with col_ctrl:
        conf_threshold = st.slider(
            "Recognition Confidence Threshold (Cosine Similarity):",
            min_value=0.20,
            max_value=0.85,
            value=float(DEFAULT_CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Threshold below which faces are classified as 'Unknown Student'. Default is 0.50."
        )
    with col_stats:
        total_students = db.get_student_count()
        st.metric("Registered Students in Model", f"{total_students}")

    test_mode = st.radio("Select Recognition Input Source:", ["Webcam Snapshot", "Upload Test Image"], horizontal=True)
    test_image = None

    if test_mode == "Webcam Snapshot":
        test_cam_enabled = st.toggle(
            "📷 Turn On Testing Camera",
            value=False,
            key="testing_cam_toggle",
            help="Toggle ON to open live camera preview. Toggle OFF to release webcam."
        )
        if test_cam_enabled:
            cam_shot = st.camera_input("Capture frame for recognition", key="testing_camera_input")
            st.caption("🔒 Accessing via `http://localhost:8501`. If preview is blank, check browser camera permissions and ensure other apps are closed.")
            if cam_shot is not None:
                bytes_data = cam_shot.getvalue()
                test_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            st.info("Camera is currently OFF and released. Toggle the switch above to activate your webcam.")
    else:
        uploaded_test = st.file_uploader("Upload single or group image for face recognition:", type=["jpg", "jpeg", "png"])
        if uploaded_test is not None:
            bytes_data = uploaded_test.read()
            test_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if test_image is None:
        st.markdown(
            """
            <div class="custom-card" style="text-align:center; color:#94A3B8; padding:35px 20px;">
                <p style="font-size:1.05rem; font-weight:600; margin-bottom:6px;">👤 No Test Image Loaded</p>
                <p style="font-size:0.85rem; margin:0;">Upload a test image or activate your webcam above to test facial verification and emotion classification.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Step 1: Preview & Action
        col_prev, col_action = st.columns([1, 1], gap="medium")
        with col_prev:
            st.markdown("##### 1. Input Image Preview")
            st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), caption="Test Image Preview", use_container_width=True)

        with col_action:
            st.markdown("##### 2. Verify Face Recognition & Emotion")
            st.write(f"Confidence Threshold: **{conf_threshold:.0%}** (Cosine Similarity)")
            st.caption("Click the button below to detect faces, identify registered students, predict emotional expression, and evaluate similarity.")
            verify_btn = st.button("🔍 Verify Face & Emotion", type="primary", key="btn_run_face_verify", use_container_width=True)

        att_test_token = f"{test_image.shape}_{float(np.mean(test_image)):.2f}_{conf_threshold:.2f}"
        active_att_test = st.session_state.get("att_test_active_result")

        if verify_btn:
            with st.spinner("Detecting facial landmarks, matching identity embeddings, and predicting emotions..."):
                predictions = fr_system.predict_faces(test_image, confidence_threshold=conf_threshold)
                if predictions:
                    emo_detector = emo_detector or get_emotion_detector()
                if emo_detector is not None and predictions:
                    ih, iw = test_image.shape[:2]
                    for p in predictions:
                        bbox = p.get('bbox')
                        if bbox:
                            x, y, w, h = bbox
                            x1, y1 = max(0, int(x)), max(0, int(y))
                            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
                            if x2 > x1 and y2 > y1:
                                crop = test_image[y1:y2, x1:x2]
                                p['emotion'] = emo_detector.predict_emotion(crop)
                            else:
                                p['emotion'] = None

                annotated_bgr = fr_system.draw_annotations(test_image, predictions)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.session_state["att_test_active_result"] = (att_test_token, predictions, annotated_rgb)
                active_att_test = st.session_state["att_test_active_result"]

        if active_att_test is not None and active_att_test[0] == att_test_token:
            _, predictions, annotated_rgb = active_att_test
            st.markdown("---")
            st.markdown("##### 3. Verification & Emotion Breakdown")
            col_img, col_res = st.columns([3, 2], gap="large")
            with col_img:
                st.image(annotated_rgb, caption="Recognized Faces & Emotions (Lavender: Known Student, Coral: Unknown Student)", use_container_width=True)

            with col_res:
                st.markdown("##### Recognition Breakdown")
                if not predictions:
                    st.warning("No faces detected in the provided image.")
                else:
                    st.write(f"Total Faces Detected: **{len(predictions)}**")
                    for i, p in enumerate(predictions, 1):
                        emo = p.get('emotion')
                        emo_html = ""
                        if emo:
                            emo_html = f"<br><strong>Emotion:</strong> {emo['predicted_class']} ({emo['confidence']:.1%})"

                        if p['is_known']:
                            st.markdown(f"""
                            <div class="lavender-card">
                                <strong>Face #{i}: Registered Student</strong><br>
                                <strong>Name:</strong> {p['name']}<br>
                                <strong>Student ID:</strong> {p['student_id']}<br>
                                <strong>Recognition Conf:</strong> {p['confidence']:.1%}{emo_html}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="coral-card">
                                <strong>Face #{i}: Unknown Student</strong><br>
                                <em>Similarity ({p['confidence']:.1%}) below threshold ({conf_threshold:.0%})</em>{emo_html}<br>
                                <span class="stat-pill stat-coral">Rejected / Not Present</span>
                            </div>
                            """, unsafe_allow_html=True)

                        if emo and 'probabilities' in emo:
                            with st.expander(f"Face #{i} Emotion Probability Distribution"):
                                for e_name, e_prob in emo['probabilities'].items():
                                    st.progress(min(1.0, max(0.0, float(e_prob))), text=f"{e_name}: {e_prob:.1%}")
        else:
            st.info("👈 Click **Verify Face & Emotion** above to test facial verification and emotion classification.")


def render_directory_tab(db: AttendanceDatabase, fr_system: FaceRecognitionSystem):
    """Render Registered Students Directory."""
    st.subheader("Registered Students Directory")
    students = db.get_all_students()

    if not students:
        st.info("No students have been registered yet. Navigate to 'Student Registration' to add students.")
        return

    st.write(f"Total Active Registrations: **{len(students)}**")
    
    # Table of registered students
    formatted_data = []
    for s in students:
        formatted_data.append({
            "Student ID": s["student_id"],
            "Student Name": s["name"],
            "Face Samples": s["face_count"],
            "Registration Date": s["created_at"][:19].replace("T", " ") if s.get("created_at") else "N/A",
            "Consent Recorded": "Yes" if s.get("consent_given") else "No"
        })
    st.dataframe(formatted_data, use_container_width=True)

    # Admin removal feature
    with st.expander("Manage Registrations (Delete Student)"):
        st.warning("Removing a student permanently deletes their registration profile.")
        if not is_admin_pin_configured():
            st.warning("🔒 Student deletion requires an Administrator PIN, but `ATTENDANCE_ADMIN_PIN` is not configured in the system environment.")
        else:
            is_authenticated = st.session_state.get('admin_authenticated', False)
            if not is_authenticated:
                del_pin = st.text_input("Enter Admin PIN to confirm deletion:", type="password", key="del_pin")
            else:
                del_pin = ""
                st.caption("Admin session active. Authenticated for student deletion.")

            del_id = st.selectbox("Select Student to Remove:", [s["student_id"] for s in students])
            if st.button("Delete Student", type="primary"):
                if is_authenticated or verify_admin_pin(del_pin):
                    db.delete_student(del_id)
                    # Re-train model
                    all_emb, all_ids, names_map = db.get_all_embeddings()
                    fr_system.train_classifier(all_emb, all_ids, names_map)
                    st.success(f"Student '{del_id}' removed and model updated.")
                    st.rerun()
                else:
                    st.error("Invalid Admin PIN. Access denied.")


def run_app():
    """Main Streamlit Entry Point for Task 1."""
    try:
        st.set_page_config(
            page_title="VISION AI — Attendance & Emotion",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    apply_theme()

    db, fr_system, reg_manager, att_manager = get_system_instances()

    render_page_header(
        "Attendance System",
        "Automated student identity verification and scheduled session attendance tracking",
        "ACTIVE"
    )

    # Top KPI Strip
    session_info = att_manager.get_session_status()
    today_summary = session_info['summary']
    total_count = today_summary.get('total_system_students')
    if total_count is None:
        total_count = db.get_student_count()
    present_count = today_summary.get('present_count', 0)
    absent_count = today_summary.get('absent_count', 0)
    is_active = session_info.get('is_active', False)
    status_label = "Open (09:30 - 10:00 AM)" if is_active else "Closed (Off-Hours)"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Enrolled Students</div>
            <div class="metric-value">{total_count}</div>
            <div class="metric-desc">Registered Student Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Present Today</div>
            <div class="metric-value">{present_count}</div>
            <div class="metric-desc">Marked Present Today</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Marked Absent</div>
            <div class="metric-value">{absent_count}</div>
            <div class="metric-desc">Finalized Absentees</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Operating Window</div>
            <div class="metric-value" style="font-size: 1.15rem; margin-top: 6px;">09:30 - 10:00 IST</div>
            <div class="metric-desc">Status: {status_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Tabbed Workspace
    tabs = st.tabs([
        "📊 Attendance Dashboard",
        "📝 Student Registration",
        "📈 Attendance Reports & Export",
        "🔍 Face Recognition & Emotion Testing",
        "📋 Registered Student Directory"
    ], on_change="rerun")

    if tabs[0].open:
        with tabs[0]:
            render_attendance_dashboard_tab(att_manager, fr_system, db, session_info=session_info)
    if tabs[1].open:
        with tabs[1]:
            render_registration_tab(reg_manager, fr_system, db)
    if tabs[2].open:
        with tabs[2]:
            render_reports_tab(db)
    if tabs[3].open:
        with tabs[3]:
            render_recognition_tab(fr_system, db)
    if tabs[4].open:
        with tabs[4]:
            render_directory_tab(db, fr_system)


if __name__ == "__main__":
    run_app()
