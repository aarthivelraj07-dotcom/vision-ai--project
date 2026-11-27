"""
Streamlit User Interface for Task 3: Drowsiness & Fatigue Detection System.
Adheres to the application's light, lavender, and coral design language.

Color System:
- ALERT (Safe): Green (#10B981) / Lavender background (#F5F3FF)
- WARNING (Yawn / Low EAR): Amber (#F59E0B) / Soft Gold (#FFFBEB)
- CRITICAL (Drowsiness Alarm): Red (#EF4444) / Coral background (#FFF1F2)

Features:
- Tab 1: Static Image Testing (Biometric landmark extraction, EAR, MAR, consistent instantaneous warning, static disclaimer).
- Tab 2: Live Camera / Video Stream Analysis (Real-time temporal state machine, blink filter, prolonged closure alarm, EAR trend curve, Web Audio siren, Windows host buzzer, CSV session log export).
- Tab 3: Drowsiness Analytics & Reference (Soukupová-Čech EAR formula, MAR yawn geometry, temporal transition logic, architecture guide).
"""

import os
import time
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.drowsiness.detector import (
    DrowsinessDetector,
    get_drowsiness_detector,
    calculate_ear,
    calculate_mar,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
)
from modules.drowsiness.state_machine import (
    DrowsinessStateMachine,
    DrowsinessState,
    MultiPersonDrowsinessTracker,
)
from modules.drowsiness.age_estimator import (
    get_age_estimator,
    AgeEstimator,
)
from modules.drowsiness.alert_manager import (
    render_browser_audio_alert,
    trigger_host_buzzer,
    generate_drowsiness_csv_report,
)
from modules.common.theme import apply_theme, render_page_header


# CSS styling adhering to Teal (#087E8B) + Tangerine (#FF914D) + Pale Aqua (#DDF5F2)
DROWSINESS_UI_CSS = """
<style>
    /* Metric cards styling */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1.5px solid #B7DCD7 !important;
        border-radius: 12px;
        padding: 12px 18px;
        box-shadow: 0 4px 14px rgba(24, 59, 69, 0.06);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] label span {
        color: #45636A !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #183B45 !important;
        font-weight: 800 !important;
    }

    /* Custom Card Containers */
    .custom-card {
        background: #FFFFFF !important;
        border: 1.5px solid #B7DCD7 !important;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(24, 59, 69, 0.06);
        color: #183B45 !important;
    }
    .lavender-card {
        background: #F4FBF9 !important;
        border: 1.5px solid #B7DCD7 !important;
        border-left: 4px solid #087E8B !important;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }
    .coral-card {
        background: #FFF8F4 !important;
        border: 1.5px solid #FF914D !important;
        border-left: 4px solid #FF914D !important;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }
    .amber-card {
        background: #FFF9F2 !important;
        border: 1.5px solid #FF914D !important;
        border-left: 4px solid #FF914D !important;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        color: #183B45 !important;
    }

    /* Badges with high visual contrast */
    .badge-alert {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.18);
        color: #065F46;
        border: 1.5px solid #10B981;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-warning {
        display: inline-block;
        background-color: rgba(255, 145, 77, 0.18);
        color: #C25E00;
        border: 1.5px solid #FF914D;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-critical {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.2);
        color: #991B1B;
        border: 1.5px solid #EF4444;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        animation: pulse 1.5s infinite;
    }
    .badge-neutral {
        display: inline-block;
        background-color: #DDF5F2;
        color: #183B45;
        border: 1.5px solid #B7DCD7;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Flashing banner for critical alert */
    .critical-alarm-banner {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
        color: #FFFFFF;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4);
        border: 2px solid #EF4444;
    }

    /* Section divider spacing */
    .section-spacer {
        margin-top: 24px;
        margin-bottom: 16px;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
</style>
"""


def _get_badge_html(state: str) -> str:
    """Return colored HTML badge for a given state string."""
    state_upper = (state or "").upper()
    if state_upper in ("ALERT", "OPEN", "NORMAL"):
        return f"<span class='badge-alert'>✓ {state_upper}</span>"
    elif state_upper in ("WARNING", "YAWNING", "YAWN"):
        return f"<span class='badge-warning'>⚠ {state_upper}</span>"
    elif state_upper in ("CRITICAL", "CLOSED", "DROWSY"):
        return f"<span class='badge-critical'>🚨 {state_upper}</span>"
    else:
        return f"<span class='badge-neutral'>{state_upper}</span>"


def render_image_testing_tab(detector: DrowsinessDetector, ear_thresh: float, mar_thresh: float):
    """Render Tab 1: Static Image Testing & Instantaneous Analysis."""
    st.subheader("Static Image Alertness Analysis")
    st.markdown(
        "Upload a photo or capture a camera snapshot to evaluate eye openness, yawn indications, and occupant alertness."
    )

    # Static photo disclaimer banner
    st.markdown("""
    <div class="lavender-card" style="margin-bottom: 18px;">
        <h4 style="margin-top:0; color: #5B21B6; font-size: 1.05rem;">ℹ️ Static Photograph Verification Notice</h4>
        <p style="margin-bottom:0; color: #4B5563; font-size: 0.90rem;">
            Drowsiness and micro-sleep are strictly <strong>temporal physiological states</strong> requiring continuous multi-frame continuity 
            (≥ 15 consecutive frames / ~500ms). In a static photograph, closed eyes indicate an instantaneous closed-eye state 
            (such as a natural 100ms blink or pose), which flags a <strong>WARNING (Eyes Closed – Photo Snapshot)</strong>, but 
            <strong>never claims prolonged drowsiness (CRITICAL)</strong>. Continuous video tracking is required to establish drowsiness.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion mode
    input_mode = st.radio(
        "Select Image Input Source:",
        ["📁 Upload Image File", "📷 Webcam Snapshot", "📂 Sample Test Dataset"],
        horizontal=True
    )

    image_bytes = None

    if input_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader(
            "Choose a facial photograph (JPG, PNG, JPEG, BMP):",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()

    elif input_mode == "📷 Webcam Snapshot":
        camera_snap = st.camera_input("Capture instant photo:")
        if camera_snap is not None:
            image_bytes = camera_snap.read()

    elif input_mode == "📂 Sample Test Dataset":
        test_dir = os.path.join("datasets", "drowsiness", "test_images")
        sample_options = {}
        if os.path.exists(test_dir):
            for fname in sorted(os.listdir(test_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_options[fname] = os.path.join(test_dir, fname)

        # Fallback to attendance images if drowsiness dataset empty
        if not sample_options:
            att_reg_dir = os.path.join("datasets", "attendance", "registered_faces")
            if os.path.exists(att_reg_dir):
                for fname in sorted(os.listdir(att_reg_dir))[:4]:
                    sample_options[f"Attendance: {fname}"] = os.path.join(att_reg_dir, fname)

        if sample_options:
            chosen_sample = st.selectbox("Select pre-staged sample:", list(sample_options.keys()))
            sample_path = sample_options[chosen_sample]
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        else:
            st.info("No sample images found in datasets directory.")

    if image_bytes is None:
        st.markdown("""
        <div class="custom-card" style="text-align: center; color: #64748B; padding: 40px 20px;">
            <p style="font-size: 1.1rem; margin-bottom: 6px;">👈 Please upload an image or select a sample above.</p>
            <p style="font-size: 0.85rem; margin: 0;">Supported formats: JPG, PNG, WEBP, BMP</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        st.error("Failed to decode image. Please check file format.")
        return

    col_prev, col_action = st.columns([1, 1], gap="medium")
    with col_prev:
        st.markdown("##### 1. Source Image Preview")
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Photograph Preview", use_container_width=True)

    with col_action:
        st.markdown("##### 2. Run Alertness Analysis")
        st.write(f"Active EAR Threshold: **{ear_thresh:.2f}** | MAR Threshold: **{mar_thresh:.2f}**")
        st.caption("Click the button below to extract ocular landmarks, detect eye closure, and evaluate driver alertness.")
        run_btn = st.button("🔍 Analyse Driver Alertness & Facial State", type="primary", key="btn_run_drowsiness_analysis", use_container_width=True)

    drowsy_token = f"{frame_bgr.shape}_{float(np.mean(frame_bgr)):.2f}_{ear_thresh:.2f}_{mar_thresh:.2f}"
    active_drowsy = st.session_state.get("drowsiness_active_result")

    if run_btn:
        with st.spinner("Analyzing biometric landmarks, calculating EAR/MAR, and evaluating driver status..."):
            processed_res = detector.process_frame(
                frame_bgr,
                ear_thresh=ear_thresh,
                mar_thresh=mar_thresh
            )
            st.session_state["drowsiness_active_result"] = (drowsy_token, processed_res)
            active_drowsy = st.session_state["drowsiness_active_result"]

    if active_drowsy is not None and active_drowsy[0] == drowsy_token:
        _, processed_res = active_drowsy
        total_people = processed_res.get("total_people", 0)
        sleeping_count = processed_res.get("sleeping_count", 0)
        awake_count = processed_res.get("awake_count", 0)
        people = processed_res.get("people", [])
        popup_summary = processed_res.get("popup_summary", "")

        # Requirement 7: Display a pop-up showing the number of sleeping people and their age
        if sleeping_count > 0:
            st.toast(f"🚨 {popup_summary}", icon="⚠️")
            st.markdown(f"""
            <div class="critical-alarm-banner" style="margin-bottom: 20px;">
                🚨 POP-UP ALERT: {popup_summary}
            </div>
            """, unsafe_allow_html=True)
        elif total_people > 0:
            st.markdown(f"""
            <div class="lavender-card" style="margin-bottom: 20px; text-align: center;">
                <span style="font-weight: 700; color: #047857; font-size: 1.05rem;">
                    ✓ POP-UP STATUS: {popup_summary}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Multi-person summary metrics row (Requirements 2, 3, 4)
        m_sum1, m_sum2, m_sum3, m_sum4 = st.columns(4)
        with m_sum1:
            st.metric("Total Occupants", total_people)
        with m_sum2:
            st.metric(
                "Sleeping (RED)",
                sleeping_count,
                delta="Hazardous" if sleeping_count > 0 else "None",
                delta_color="inverse" if sleeping_count > 0 else "off"
            )
        with m_sum3:
            st.metric(
                "Awake (GREEN)",
                awake_count,
                delta="Active" if awake_count > 0 else "None",
                delta_color="normal"
            )
        with m_sum4:
            driver_age = people[0]["age_info"]["label"] if people else "N/A"
            st.metric("Driver Est. Age", driver_age)

        # Static Evaluation: Guarantees WARNING for closed eyes and never CRITICAL
        sm = DrowsinessStateMachine(ear_threshold=ear_thresh, mar_threshold=mar_thresh)
        static_eval = sm.evaluate_static_image(
            ear=processed_res["ear"],
            mar=processed_res["mar"],
            face_detected=processed_res["face_detected"]
        )

        col_img, col_metrics = st.columns([1.2, 1.0], gap="large")

        with col_img:
            # Tabs for Requirement 9: Display a preview of the input image
            tab_hud, tab_raw = st.tabs(["🎯 Annotated HUD & Age Overlays", "📷 Raw Input Image Preview"])

            with tab_hud:
                annotated_bgr = processed_res.get("annotated_frame", frame_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                st.caption("Overlay Legend: 🔴 RED Box = Sleeping Occupant | 🟢 GREEN Box = Alert Occupant | Cyan/Amber = Facial Landmarks.")

            with tab_raw:
                raw_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                st.image(raw_rgb, use_container_width=True)
                st.caption("Original unannotated image preview.")

        with col_metrics:
            st.markdown("**Occupant Status & Estimated Age:**")

            if people:
                # Build occupancy dataframe
                occ_data = []
                for p in people:
                    is_sleep = p.get("is_sleeping", False)
                    occ_data.append({
                        "Occupant": f"#{p.get('person_id', 1)}",
                        "Status": "🔴 ASLEEP" if is_sleep else "🟢 AWAKE",
                        "Predicted Age": f"{p.get('estimated_age', 25)} yrs",
                        "Age Bracket": p.get("age_info", {}).get("bracket", "20–30 yrs"),
                        "EAR": f"{p.get('ear', 0.0):.3f}",
                        "MAR": f"{p.get('mar', 0.0):.3f}",
                        "Box Color": "RED" if is_sleep else "GREEN"
                    })
                st.dataframe(pd.DataFrame(occ_data), use_container_width=True, hide_index=True)

            state_badge = _get_badge_html(static_eval["instantaneous_state"])
            card_class = "coral-card" if sleeping_count > 0 else ("amber-card" if static_eval["instantaneous_state"] == "WARNING" else ("lavender-card" if static_eval["instantaneous_state"] == "ALERT" else "custom-card"))

            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-weight:700; color:#172033; font-size:1.0rem;">Occupant Fleet Status:</span>
                    {state_badge}
                </div>
                <div style="font-size:0.92rem; color:#475467; margin-bottom:8px;">
                    <strong>Interpretation:</strong> {static_eval['status_text']}
                </div>
                <p style="font-size:0.85rem; color:#667085; margin:0;">
                    {static_eval['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 4 Key Metrics for primary occupant
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(
                    label="Primary EAR",
                    value=f"{processed_res['ear']:.3f}",
                    delta=f"Threshold: {ear_thresh:.2f}",
                    delta_color="normal" if processed_res['ear'] >= ear_thresh else "inverse"
                )
            with mcol2:
                st.metric(
                    label="Primary MAR",
                    value=f"{processed_res['mar']:.3f}",
                    delta=f"Threshold: {mar_thresh:.2f}",
                    delta_color="normal" if processed_res['mar'] <= mar_thresh else "inverse"
                )

            # Detailed Landmark Breakdown
            with st.expander("🔍 Detailed Landmark Coordinates & Metrics"):
                st.markdown(f"- **Total Occupants Detected:** `{total_people}`")
                for i, p in enumerate(people):
                    bbox = p.get("bbox", (0, 0, 0, 0))
                    st.markdown(f"**Occupant #{p.get('person_id', i+1)}:**")
                    st.markdown(f"  - Face Bounding Box: `[x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}]`")
                    st.markdown(f"  - Eye Aspect Ratio (EAR): `{p.get('ear', 0.0):.4f}`")
                    st.markdown(f"  - Mouth Aspect Ratio (MAR): `{p.get('mar', 0.0):.4f}`")
                    st.markdown(f"  - Age Estimate: `{p.get('age_info', {}).get('label')}` (Confidence: {p.get('age_info', {}).get('confidence', 0.85)*100:.0f}%)")
    else:
        st.info("👈 Click **Analyse Driver Alertness & Facial State** above to begin analysis.")


def render_video_analysis_tab(
    detector: DrowsinessDetector,
    ear_thresh: float,
    mar_thresh: float,
    drowsy_frames: int,
    yawn_frames: int,
    enable_browser_sound: bool,
    enable_host_buzzer: bool
):
    """Render Tab 2: Live Camera & Video Stream Temporal Tracking."""
    st.subheader("Live Temporal Tracking & Drowsiness Alerting")
    st.markdown(
        r"Stream video or webcam feed through the **Drowsiness State Machine** to monitor real-time "
        r"blink frequency, detect prolonged micro-sleep ($\ge 15$ frames), and trigger multi-modal alarms."
    )

    # Ingestion choice (Req 8: Proper GUI for image and video uploads)
    source_choice = st.radio(
        "Select Video Input:",
        [
            "🚗 Sample 1: Multi-Occupant Vehicle Sequence (Driver Alert + Passenger Asleep)",
            "🎬 Sample 2: Single Driver Drowsiness Sequence",
            "📁 Upload Custom Video (MP4/AVI/MOV)",
            "📷 Webcam Snapshot Mode"
        ],
        horizontal=False
    )

    video_path = None
    video_bytes = None

    if "Multi-Occupant" in source_choice:
        multi_vid = os.path.join("datasets", "drowsiness", "test_videos", "multi_person_vehicle_sequence.mp4")
        if os.path.exists(multi_vid):
            video_path = multi_vid
            st.info(f"Loaded verified multi-occupant vehicle sequence: `{multi_vid}`")
        else:
            st.warning("Multi-occupant sequence file not found.")

    elif "Single Driver" in source_choice:
        sample_path = os.path.join("datasets", "drowsiness", "test_videos", "sample_drowsy_sequence.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
            st.info(f"Loaded verified sample sequence: `{sample_path}`")
        else:
            st.warning("Sample sequence file not yet generated in `datasets/drowsiness/test_videos/`.")

    elif "Upload Custom Video" in source_choice:
        uploaded_video = st.file_uploader(
            "Upload driver video footage (MP4, AVI, MOV):",
            type=["mp4", "avi", "mov", "m4v"]
        )
        if uploaded_video is not None:
            video_bytes = uploaded_video.getvalue()

    elif "Webcam Snapshot" in source_choice:
        st.markdown("**Single Webcam Ingestion:**")
        cam_shot = st.camera_input("Capture live webcam frame for biometric check:")
        if cam_shot is not None:
            nparr = np.frombuffer(cam_shot.read(), np.uint8)
            live_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if live_frame is not None:
                proc = detector.process_frame(live_frame, ear_thresh=ear_thresh, mar_thresh=mar_thresh)
                ann = proc.get("annotated_frame", live_frame)
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.metric("Live EAR", f"{proc['ear']:.3f}", delta=f"Threshold {ear_thresh}")
        return

    if video_path is None and video_bytes is None:
        st.markdown("""
        <div class="custom-card" style="text-align: center; color: #64748B; padding: 40px 20px;">
            <p style="font-size: 1.1rem; margin-bottom: 6px;">Select a sample video sequence or upload your driver video above.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Requirement 9: Display a preview of the input image or video
    with st.expander("▶ Input Video Preview (Requirement 9)", expanded=False):
        st.video(video_bytes if video_bytes is not None else video_path)

    # Execution controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    with ctrl_col1:
        start_btn = st.button("▶ Run Multi-Occupant Video Analysis", type="primary", use_container_width=True)
    with ctrl_col2:
        frame_stride = st.selectbox("Processing Rate:", [1, 2, 3], index=0, format_func=lambda x: f"Every {x} frame(s)")
    with ctrl_col3:
        max_frames_to_run = st.number_input("Max Frames to Process:", min_value=15, max_value=600, value=90, step=15)

    if start_btn:
        temp_video_path = None
        if video_bytes is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_bytes)
                temp_video_path = temp_video_file.name
            processing_video_path = temp_video_path
        else:
            processing_video_path = video_path

        cap = cv2.VideoCapture(processing_video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video file at: {processing_video_path}")
            cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Multi-occupant video tracker
        tracker = MultiPersonDrowsinessTracker(
            ear_threshold=ear_thresh,
            mar_threshold=mar_thresh,
            drowsy_min_frames=drowsy_frames
        )

        progress_bar = st.progress(0.0)
        status_text_elem = st.empty()

        # Dynamic containers for live UI update
        alert_banner_slot = st.empty()
        sound_slot = st.empty()
        frame_slot = st.empty()

        # Telemetry metrics row (Req 2, 3, 4)
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            state_metric_slot = st.empty()
        with m_col2:
            sleeping_metric_slot = st.empty()
        with m_col3:
            awake_metric_slot = st.empty()
        with m_col4:
            ear_metric_slot = st.empty()
        with m_col5:
            alarm_metric_slot = st.empty()

        frame_idx = 0
        processed_count = 0
        ear_trend_records: List[Dict[str, Any]] = []

        while cap.isOpened() and processed_count < max_frames_to_run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            processed_count += 1
            prog_val = min(1.0, float(processed_count) / float(min(total_frames, max_frames_to_run)))
            progress_bar.progress(prog_val)

            # Process frame with multi-person detector
            proc_res = detector.process_frame(frame, ear_thresh=ear_thresh, mar_thresh=mar_thresh)
            tracker_res = tracker.update(proc_res.get("people", []))

            sleeping_count = tracker_res["sleeping_count"]
            total_people = tracker_res["total_people"]
            awake_count = tracker_res["awake_count"]
            popup_summary = tracker_res["popup_summary"]

            # Record trend data for EAR graph
            ear_trend_records.append({
                "Frame": processed_count,
                "EAR": proc_res["ear"],
                "Closure Threshold": ear_thresh,
                "MAR": proc_res["mar"]
            })

            # Check Alert State (Requirement 5 & 7)
            if sleeping_count > 0:
                alert_banner_slot.markdown(
                    f"""<div class="critical-alarm-banner">🚨 POP-UP ALERT: {popup_summary}</div>""",
                    unsafe_allow_html=True
                )
                if enable_browser_sound and tracker_res["alarm_active"]:
                    render_browser_audio_alert()
                if enable_host_buzzer and tracker_res["alarm_active"]:
                    trigger_host_buzzer()
            elif total_people > 0:
                alert_banner_slot.markdown(
                    f"""<div class="lavender-card" style="text-align:center; color:#047857; font-weight:700;">✓ STATUS: {popup_summary}</div>""",
                    unsafe_allow_html=True
                )
            else:
                alert_banner_slot.empty()

            # Render live frame and metrics
            annotated_frame = proc_res.get("annotated_frame", frame)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(annotated_rgb, use_container_width=True)

            state_metric_slot.metric("Occupants", total_people)
            sleeping_metric_slot.metric(
                "Sleeping (RED)",
                sleeping_count,
                delta="Hazardous" if sleeping_count > 0 else "Safe",
                delta_color="inverse" if sleeping_count > 0 else "off"
            )
            awake_metric_slot.metric("Awake (GREEN)", awake_count)
            ear_metric_slot.metric("Primary EAR", f"{proc_res['ear']:.3f}", f"thr: {ear_thresh:.2f}")
            alarm_metric_slot.metric("Alarms Logged", len(tracker.unified_event_log))

            time.sleep(0.02)

        cap.release()
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        progress_bar.progress(1.0)
        status_text_elem.success(f"Video analysis complete! Processed {processed_count} frames across {len(tracker.tracks)} active occupant track(s).")

        # --- EAR Trend Graph ---
        if ear_trend_records:
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            st.subheader("📈 Real-Time Eye Aspect Ratio (EAR) Trend")
            st.markdown(
                "Continuous frame-by-frame EAR trajectory showing aperture dynamics and prolonged closure "
                "relative to the critical drowsiness threshold."
            )
            df_trend = pd.DataFrame(ear_trend_records)
            st.line_chart(
                df_trend.set_index("Frame")[["EAR", "Closure Threshold"]],
                color=["#7C3AED", "#EF4444"],
                use_container_width=True
            )
            st.caption("🟣 Purple Curve: Actual measured EAR | 🔴 Red Line: Configured closure threshold (0.22)")

        # --- Session Summary Metrics Card ---
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        elapsed_total = processed_count / fps
        total_occupants_tracked = len(tracker.tracks)
        total_drowsy_alarms = len(tracker.unified_event_log)

        st.markdown(f"""
        <div class="lavender-card">
            <h4 style="margin-top:0; color:#5B21B6;">📊 Multi-Occupant Session Performance Summary</h4>
            <div style="display: flex; gap: 32px; flex-wrap: wrap; margin-top: 10px;">
                <div><strong>Total Frames Processed:</strong> <code>{processed_count}</code></div>
                <div><strong>Stream Duration:</strong> <code>{elapsed_total:.1f}s</code></div>
                <div><strong>Occupants Tracked:</strong> <code>{total_occupants_tracked}</code></div>
                <div><strong>Critical Drowsiness Alarms:</strong> <code>{total_drowsy_alarms}</code></div>
            </div>
            <p style="font-size:0.85rem; color:#6B7280; margin-top:8px; margin-bottom:0;">
                ℹ️ Real-time spatial tracking assigns stable occupant IDs across frames and monitors each passenger independently.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Post-analysis session event log & CSV export ---
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        event_log = tracker.unified_event_log
        st.subheader(f"📋 Multi-Occupant Session Event Log ({len(event_log)} Events Recorded)")

        if event_log:
            df_events = pd.DataFrame(event_log)
            st.dataframe(df_events, use_container_width=True)

            csv_data = generate_drowsiness_csv_report(event_log)
            st.download_button(
                label="📥 Download Multi-Occupant Session Report (CSV)",
                data=csv_data,
                file_name=f"multi_occupant_drowsiness_report_{int(time.time())}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.markdown("""
            <div class="lavender-card">
                <p style="margin:0; color:#4B5563;">
                    ✓ <strong>Clean Driving Session:</strong> No prolonged eye-closure anomalies or micro-sleep episodes were logged. All occupants maintained alert status.
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_analytics_reference_tab():
    """Render Tab 3: Scientific Methodology & Reference Guide."""
    st.subheader("Scientific Methodology & Physiological Foundations")
    st.markdown("Mathematical and algorithmic foundations of computer-vision driver fatigue monitoring and biometric age estimation.")

    # 1. Biometric Age Estimation
    st.markdown("""
    <div class="lavender-card">
        <h4 style="margin-top:0; color: #C4B5FD;">1. Biometric Age Analysis & Demographic Profiling</h4>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            The drowsiness system incorporates facial structure analysis and anthropometric biometrics to estimate occupant age brackets alongside real-time alertness monitoring:
        </p>
        <div style="background-color: rgba(15, 23, 42, 0.7); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 10px 0;">
            <ul style="margin: 0; padding-left: 20px; font-size: 0.92rem; color: #CBD5E1;">
                <li><strong>Craniofacial Geometric Profiling:</strong> Evaluates facial aspect ratio, interocular distance, and lower-face height proportions.</li>
                <li><strong>Demographic Confidence Brackets:</strong> Generates calibrated demographic age ranges (e.g. <code>~24 ± 5 yrs</code>) to assist occupant safety logging.</li>
                <li><strong>Multi-Occupant Isolation:</strong> Independently monitors each occupant's age bracket and alertness state across the vehicle cabin.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Soukupova-Cech EAR
    st.markdown("""
    <div class="lavender-card">
        <h4 style="margin-top:0; color: #C4B5FD;">2. Eye Aspect Ratio (EAR) — Soukupová & Čech (2016)</h4>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            The Eye Aspect Ratio (EAR) is a dimensionless, scale-invariant geometric metric representing the ratio of 
            vertical eyelid aperture to horizontal eye width. For each eye defined by 6 anatomical landmarks 
            <code>p1, p2, p3, p4, p5, p6</code>:
        </p>
        <div style="background-color: rgba(15, 23, 42, 0.7); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); font-family: monospace; font-size: 1.05rem; margin: 10px 0; color: #38BDF8;">
            EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )
        </div>
        <p style="color: #94A3B8; font-size: 0.90rem; margin-bottom: 0;">
            - <strong>Open Eyes:</strong> EAR typically ranges between <code>0.28 and 0.42</code>.<br>
            - <strong>Closed Eyes / Micro-Sleep:</strong> EAR drops rapidly below <code>0.20</code>.<br>
            - <strong>Normal Blink:</strong> Rapid drop and recovery lasting 100ms – 250ms (1 to 6 frames at 30 FPS).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Mouth Aspect Ratio (MAR)
    st.markdown("""
    <div class="amber-card">
        <h4 style="margin-top:0; color: #FCD34D;">3. Mouth Aspect Ratio (MAR) — Yawn Geometry</h4>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            The Mouth Aspect Ratio (MAR) quantifies oral cavity dilation during yawning or deep vocalization:
        </p>
        <div style="background-color: rgba(15, 23, 42, 0.7); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3); font-family: monospace; font-size: 1.05rem; margin: 10px 0; color: #FBBF24;">
            MAR = ||m_top - m_bottom|| / ||m_left - m_right||
        </div>
        <p style="color: #94A3B8; font-size: 0.90rem; margin-bottom: 0;">
            - <strong>Normal Resting Mouth:</strong> MAR ranges between <code>0.00 and 0.25</code>.<br>
            - <strong>Talking / Smiling:</strong> MAR ranges between <code>0.25 and 0.45</code>.<br>
            - <strong>Fatigue Yawning:</strong> MAR expands beyond <code>0.55</code> for continuous frames (≥ 12 frames).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Temporal State Machine Comparison Table
    st.markdown("""
    <div class="custom-card">
        <h4 style="margin-top:0; color: #F8FAFC;">4. Temporal State Machine Logic & Duration Rules</h4>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
            <thead>
                <tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.1); text-align: left; color: #94A3B8;">
                    <th style="padding: 8px 12px;">Physiological Event</th>
                    <th style="padding: 8px 12px;">Condition</th>
                    <th style="padding: 8px 12px;">Frame Duration</th>
                    <th style="padding: 8px 12px;">System State</th>
                    <th style="padding: 8px 12px;">Action Taken</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 12px; font-weight:600;">Normal Blink</td>
                    <td style="padding: 8px 12px;">EAR &lt; 0.22</td>
                    <td style="padding: 8px 12px;">1 – 7 frames (30–230ms)</td>
                    <td style="padding: 8px 12px;"><span class="badge-alert">✓ ALERT</span></td>
                    <td style="padding: 8px 12px; color:#059669;">Increment blink counter; no siren</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 12px; font-weight:600;">Eye Sluggishness</td>
                    <td style="padding: 8px 12px;">EAR &lt; 0.22</td>
                    <td style="padding: 8px 12px;">8 – 14 frames</td>
                    <td style="padding: 8px 12px;"><span class="badge-warning">⚠ WARNING</span></td>
                    <td style="padding: 8px 12px; color:#D97706;">Yellow indicator; monitor progression</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; background-color: #FEF2F2;">
                    <td style="padding: 8px 12px; font-weight:700; color:#DC2626;">Drowsiness / Micro-Sleep</td>
                    <td style="padding: 8px 12px;">EAR &lt; 0.22</td>
                    <td style="padding: 8px 12px;">≥ 15 frames (≥ 500ms)</td>
                    <td style="padding: 8px 12px;"><span class="badge-critical">🚨 CRITICAL</span></td>
                    <td style="padding: 8px 12px; font-weight:700; color:#DC2626;">Dispatch Audio Siren + Host Buzzer</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; font-weight:600;">Fatigue Yawning</td>
                    <td style="padding: 8px 12px;">MAR &gt; 0.55</td>
                    <td style="padding: 8px 12px;">≥ 12 frames (≥ 400ms)</td>
                    <td style="padding: 8px 12px;"><span class="badge-warning">⚠ WARNING</span></td>
                    <td style="padding: 8px 12px; color:#D97706;">Log yawn event; suggest rest stop</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def run_drowsiness_app():
    """Main Streamlit Entry Point for Task 3: Drowsiness Detection."""
    try:
        st.set_page_config(
            page_title="VISION AI — Drowsiness & Fatigue Detection",
            page_icon="😴",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.markdown(DROWSINESS_UI_CSS, unsafe_allow_html=True)
    apply_theme()
    detector = get_drowsiness_detector()

    render_page_header(
        "Drowsiness Detection",
        "Real-time driver fatigue monitoring, eye alertness tracking, and safety alarm dispatch",
        "ACTIVE"
    )

    # Workspace Threshold & Alarm Controls Expander
    with st.expander("⚙️ Alert Sensitivity & Notification Settings", expanded=False):
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            ear_thresh = st.slider(
                "Eye Closure Sensitivity (EAR):",
                min_value=0.15,
                max_value=0.35,
                value=float(DEFAULT_EAR_THRESHOLD),
                step=0.01,
                help="Eyes with EAR below this value are considered closed."
            )
            mar_thresh = st.slider(
                "Yawn Detection Sensitivity (MAR):",
                min_value=0.30,
                max_value=0.80,
                value=float(DEFAULT_MAR_THRESHOLD),
                step=0.05,
                help="Mouth aperture exceeding this value indicates yawning."
            )
        with col_cfg2:
            drowsy_frames = st.slider(
                "Alarm Trigger Duration (Frames):",
                min_value=10,
                max_value=30,
                value=15,
                step=1,
                help="Consecutive closed frames required to trigger critical alarm."
            )
            yawn_frames = st.slider(
                "Yawn Duration (Frames):",
                min_value=8,
                max_value=25,
                value=12,
                step=1
            )
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                enable_browser_sound = st.checkbox("🔊 Web Audio Siren", value=True)
            with sub_col2:
                enable_host_buzzer = st.checkbox("💻 Windows Buzzer", value=True)

    tab_image, tab_video, tab_analytics = st.tabs([
        "📷 Static Image Testing",
        "🎥 Live Camera / Video Analysis",
        "📖 Drowsiness Analytics & Reference"
    ])

    with tab_image:
        render_image_testing_tab(detector, ear_thresh, mar_thresh)

    with tab_video:
        render_video_analysis_tab(
            detector,
            ear_thresh,
            mar_thresh,
            drowsy_frames,
            yawn_frames,
            enable_browser_sound,
            enable_host_buzzer
        )

    with tab_analytics:
        render_analytics_reference_tab()


if __name__ == "__main__":
    run_drowsiness_app()
