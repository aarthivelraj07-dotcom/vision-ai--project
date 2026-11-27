"""
Streamlit User Interface for Task 5: Indian Sign Language (ISL) Static Sign Recognition.

Capabilities:
1. Static Image Analysis: Image upload (JPG, JPEG, PNG), landmark skeleton visualization,
   prominent sign prediction badge, and relative model confidence breakdown.
2. Real-Time Camera: Live video stream with OpenCV VideoCapture, Start/Stop toggle,
   safe resource release, and native Streamlit camera_input fallback.
3. Operating Window Enforcement: Authoritative server-side IST check (6:00 PM - 10:00 PM IST),
   blocking analysis outside permitted hours with secure environment-configured admin override.
4. Supported Classes: 12 static handshape classes ('1', '2', '3', '4', '5', 'A', 'B', 'C', 'L', 'V', 'W', 'Y').
"""

import os
import time
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
import streamlit as st

from .detector import (
    ISLHandDetector,
    ISLRecognitionResult,
    SUPPORTED_CLASSES,
    get_current_ist_time,
    is_within_operating_window,
    get_next_operating_window,
    is_admin_override_configured,
    verify_admin_override,
)
from modules.common.theme import apply_theme, render_page_header

_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom Styling for Task 5 — Teal + Tangerine + Pale Aqua
TASK5_CSS = """
<style>
    .isl-card {
        background: #FFFFFF !important;
        border: 1.5px solid #B7DCD7 !important;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(24, 59, 69, 0.06);
        color: #183B45 !important;
    }
    .isl-badge-active {
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
    .isl-badge-locked {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.18);
        color: #991B1B;
        border: 1.5px solid #EF4444;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .sign-display-box {
        background: #FFFFFF !important;
        border: 2px solid #FF914D !important;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(255, 145, 77, 0.15) !important;
    }
    .sign-large-text {
        font-size: 5rem;
        font-weight: 900;
        color: #FF914D !important;
        line-height: 1.0;
        margin-bottom: 8px;
        text-shadow: none !important;
    }
    .sign-subtitle {
        font-size: 1.05rem;
        font-weight: 700;
        color: #183B45 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_cached_detector() -> ISLHandDetector:
    """Cache the OpenCV DNN models and SVM classifier to avoid reloads across reruns."""
    return ISLHandDetector()


def run_sign_language_app():
    """Main Streamlit entry point for Task 5: Indian Sign Language Recognition."""
    try:
        st.set_page_config(
            page_title="VISION AI — Indian Sign Language",
            page_icon="🤟",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.markdown(TASK5_CSS, unsafe_allow_html=True)
    apply_theme()

    render_page_header(
        "Sign Language Recognition",
        "Recognize Indian Sign Language hand gestures from photos or live camera",
        "ACTIVE"
    )

    # Status & Operating Hours Header Banner
    now_ist = get_current_ist_time()
    window_open = is_within_operating_window(now_ist)
    status_badge = (
        "<span class='isl-badge-active'>🟢 OPERATING WINDOW OPEN (6:00 PM - 10:00 PM IST)</span>"
        if window_open
        else "<span class='isl-badge-locked'>🔴 RESTRICTED (PERMITTED HOURS: 6:00 PM - 10:00 PM IST)</span>"
    )

    st.markdown(f"""
    <div style="background: rgba(19, 27, 46, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div style="font-size: 0.88rem; color: #CBD5E1;">
            <strong>Server Time (IST):</strong> <code>{now_ist.strftime('%I:%M:%S %p')}</code> &nbsp;|&nbsp; 
            <strong>Classes:</strong> <code>12 (1-5, A, B, C, L, V, W, Y)</code> &nbsp;|&nbsp; 
            <strong>Recognition:</strong> <code>Hand Gesture Tracking</code>
        </div>
        <div>{status_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # Authoritative Operating Time Enforcement
    now_ist = get_current_ist_time()
    window_open = is_within_operating_window(now_ist)
    admin_active = st.session_state.get("isl_admin_override", False)
    access_granted = window_open or admin_active

    if not window_open:
        next_window = get_next_operating_window(now_ist)
        if not admin_active:
            st.warning(
                "🕒 **Sign Language Recognition is available from 6:00 PM to 10:00 PM IST.**\n\n"
                "Please return during the authorized operating hours or use the authorized administrator access below to evaluate the task.\n\n"
                f"• **Current Server Time:** `{now_ist.strftime('%I:%M:%S %p %Z')} ({now_ist.strftime('%Y-%m-%d')})`\n\n"
                f"• **Next Permitted Window:** `{next_window}`\n\n"
                f"Interactive image recognition and live webcam streams are restricted outside operating hours per institutional assignment policy."
            )

            # Administrator / Evaluator Override Section
            if is_admin_override_configured():
                with st.expander("🔐 Evaluator / Administrator Window Override", expanded=True):
                    st.caption("An administrator PIN is configured in the environment. Enter your authorized PIN to unlock evaluation.")
                    col_p1, col_p2 = st.columns([3, 1])
                    with col_p1:
                        pin_in = st.text_input("Administrator PIN:", type="password", key="isl_override_pin")
                    with col_p2:
                        st.write("")
                        st.write("")
                        if st.button("Unlock Session", use_container_width=True):
                            if verify_admin_override(pin_in):
                                st.session_state.isl_admin_override = True
                                st.success("Authorized: Admin override session activated.")
                                st.rerun()
                            else:
                                st.error("Access Denied: Invalid Administrator PIN.")
            else:
                with st.expander("🛠️ How to Enable Evaluator PIN for Testing Outside 6:00–10:00 PM IST", expanded=False):
                    st.markdown("""
                    To evaluate Task 5 during daytime hours without waiting until 6:00 PM IST, configure a temporary local administrator PIN in your terminal. This stays on your local machine and is never committed to GitHub.

                    **Option 1: Windows PowerShell (Recommended)**
                    ```powershell
                    # 1. Stop current Streamlit app (Ctrl+C in terminal)
                    # 2. Set your custom evaluator PIN:
                    $env:ISL_ADMIN_PIN = "YourSecurePIN123"
                    # 3. Restart Streamlit:
                    python -m streamlit run app.py
                    ```

                    **Option 2: Windows Command Prompt (cmd.exe)**
                    ```cmd
                    :: 1. Stop current Streamlit app (Ctrl+C in terminal)
                    :: 2. Set your custom evaluator PIN:
                    set ISL_ADMIN_PIN=YourSecurePIN123
                    :: 3. Restart Streamlit:
                    python -m streamlit run app.py
                    ```

                    *Note: You may choose any PIN. Once set, enter that exact PIN in the unlock box above.*
                    """)
        else:
            st.info(
                f"🛡️ **Admin Override Active:** Operating outside the authorized window (`{now_ist.strftime('%I:%M:%S %p IST')}`). "
                f"Testing access is granted."
            )
            if st.button("🔒 End Admin Session", key="btn_end_admin"):
                st.session_state.isl_admin_override = False
                st.rerun()

    # 5. Core Interface Tabs
    tab_upload, tab_webcam, tab_guide = st.tabs([
        "📁 Static Image Analysis",
        "📹 Real-Time Live Camera",
        "ℹ️ Supported Classes & Documentation"
    ])

    # =========================================================================
    # TAB 1: Static Image Analysis
    # =========================================================================
    with tab_upload:
        st.subheader("1. Static Image Handshape Recognition")
        st.markdown(
            "Upload a clear photo of an Indian Sign Language hand gesture to extract landmark keypoints "
            "and predict the corresponding letter or number."
        )

        col_up_left, col_up_right = st.columns([1, 1])

        with col_up_left:
            st.markdown("##### 📤 Image Upload")
            uploaded_file = st.file_uploader(
                "Select a static gesture image (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"],
                key="isl_uploader",
                disabled=not access_granted
            )

            # Pre-loaded sample option
            st.markdown("---")
            st.markdown("##### 🧪 Or Test a Sample Exemplar:")
            sample_class = st.selectbox(
                "Select a sample class:",
                list(SUPPORTED_CLASSES),
                key="isl_sample_select",
                disabled=not access_granted
            )
            sample_path = os.path.join(
                _root_dir, "datasets", "sign_language", "static", sample_class, "0.jpg"
            )
            use_sample = st.button("Load Sample Exemplar", key="btn_load_sample", disabled=not access_granted)

            image_to_process = None
            image_source_label = ""

            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file)
                    image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    image_source_label = f"Uploaded file: `{uploaded_file.name}`"
                except Exception as err:
                    st.error(f"Error reading uploaded image: {err}")
            elif use_sample or st.session_state.get("active_sample_path") == sample_path:
                if os.path.exists(sample_path):
                    st.session_state.active_sample_path = sample_path
                    image_to_process = cv2.imread(sample_path)
                    image_source_label = f"Sample exemplar: Class `{sample_class}` (`{os.path.basename(sample_path)}`)"
                else:
                    st.warning(f"Sample file not found at: `{sample_path}`")

            if image_to_process is not None:
                st.caption(f"**Source:** {image_source_label}")
                st.image(
                    cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB),
                    caption="Source Image Preview",
                    use_container_width=True
                )

        with col_up_right:
            st.markdown("##### 🔍 Recognition Results")
            if not access_granted:
                st.info("Recognition is currently blocked due to server operating-hour restrictions.")
            elif image_to_process is None:
                st.info("Upload an image or choose a sample exemplar on the left to begin recognition.")
            else:
                analyze_btn = st.button("🔍 Recognize Hand Gesture", type="primary", use_container_width=True)

                if analyze_btn or st.session_state.get("isl_auto_run", False):
                    try:
                        detector = get_cached_detector()
                    except Exception as e:
                        st.error(f"❌ Failed to load ISL detection models: {e}")
                        return
                    with st.spinner("Analyzing hand gesture and evaluating prediction..."):
                        t0 = time.time()
                        result: ISLRecognitionResult = detector.process_image(image_to_process)
                        latency_ms = (time.time() - t0) * 1000.0

                    if not result.detected:
                        st.error(f"❌ **Hand Detection Failed:** {result.message}")
                        st.info("💡 **Tips for better detection:** Ensure the hand is upright, unobstructed, and well-lit.")
                    else:
                        # Draw annotated skeleton
                        annotated_bgr = detector.draw_landmarks(image_to_process, result)
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                        # Prominent Prediction Box
                        st.markdown(
                            f"""
                            <div class="sign-display-box">
                                <div class="sign-subtitle">Predicted ISL Static Sign</div>
                                <div class="sign-large-text">{result.predicted_class}</div>
                                <div style="color: #1E3A8A; font-weight: 600;">
                                    Recognition Confidence: <strong>{result.relative_confidence * 100:.1f}%</strong>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Display Annotated Image
                        st.image(annotated_rgb, caption="Hand Gesture Tracking & Recognition Preview", use_container_width=True)

                        # Metadata and Diagnostics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Hand Detection Score", f"{result.palm_score:.3f}")
                        with col_m2:
                            st.metric("Gesture Alignment", f"{result.hand_score:.3f}")
                        with col_m3:
                            st.metric("Processing Speed", f"{latency_ms:.1f} ms")

                        # Relative Confidence Table
                        with st.expander("📊 Classification Confidence Breakdown", expanded=False):
                            st.caption(
                                "⚠️ **Calibration Notice:** Values represent relative Platt-scaled posterior probabilities. "
                                "They reflect relative classification separation and have not undergone empirical calibration testing."
                            )
                            if result.probabilities:
                                sorted_probs = sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True)
                                for c_name, p_val in sorted_probs[:5]:
                                    st.progress(min(1.0, max(0.0, p_val)), text=f"Sign '{c_name}': {p_val * 100:.1f}%")

    # =========================================================================
    # TAB 2: Real-Time Live Camera
    # =========================================================================
    with tab_webcam:
        st.subheader("2. Real-Time Static Sign Recognition via Camera")
        st.markdown(
            "Recognizes static hand gestures in individual video frames in real time. "
            "Includes rate-limiting (~10 FPS) to minimize CPU load and safe camera release upon stopping."
        )

        if not access_granted:
            st.warning("⚠️ Live camera recognition is currently unavailable outside the authorized operating window (06:00 PM – 10:00 PM IST).")
        else:
            col_cam_ctrl, col_cam_feed = st.columns([1, 2])

            with col_cam_ctrl:
                st.markdown("##### 📹 Camera Controls")
                cam_mode = st.radio(
                    "Select Camera Access Method:",
                    ["Hardware Stream (OpenCV VideoCapture)", "Browser Snapshot (Streamlit camera_input)"],
                    key="isl_cam_mode"
                )

                st.markdown("---")
                st.markdown("##### 💡 Best Practices")
                st.caption(
                    "1. Position your hand approximately 1.5–2.5 feet from the lens.\n"
                    "2. Hold your hand pose steady for 1 second.\n"
                    "3. Ensure the palm or back of the hand faces the camera directly.\n"
                    "4. Avoid cluttered or low-contrast backgrounds."
                )

            with col_cam_feed:
                if cam_mode == "Browser Snapshot (Streamlit camera_input)":
                    st.markdown("##### 📸 Browser Snapshot Mode")
                    cam_snapshot = st.camera_input("Capture frame for sign recognition:", key="isl_browser_cam")
                    if cam_snapshot is not None:
                        try:
                            detector = get_cached_detector()
                            snap_bytes = cam_snapshot.getvalue()
                            snap_array = cv2.imdecode(np.frombuffer(snap_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if snap_array is not None:
                                res = detector.process_image(snap_array)
                                if not res.detected:
                                    st.warning(f"⚠️ {res.message}")
                                else:
                                    annotated = detector.draw_landmarks(snap_array, res)
                                    st.markdown(
                                        f"<div class='sign-display-box'>"
                                        f"<div class='sign-subtitle'>Recognized Sign</div>"
                                        f"<div class='sign-large-text'>{res.predicted_class}</div>"
                                        f"<div>Relative Confidence: <strong>{res.relative_confidence * 100:.1f}%</strong></div>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error processing camera snapshot: {e}")

                else:
                    st.markdown("##### 🎥 Hardware Video Stream")
                    run_stream = st.checkbox("🟢 Start Live Camera Stream", key="isl_stream_active")

                    frame_placeholder = st.empty()
                    status_placeholder = st.empty()

                    if run_stream:
                        cap = None
                        try:
                            detector = get_cached_detector()
                            # Initialize camera device 0
                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                            if not cap.isOpened():
                                # Try standard backend if DirectShow fails
                                cap = cv2.VideoCapture(0)

                            if not cap.isOpened():
                                status_placeholder.error(
                                    "❌ **Camera Unavailable:** Could not open local webcam device index 0. "
                                    "Please verify that another application is not using the camera, "
                                    "or switch to 'Browser Snapshot' mode above."
                                )
                            else:
                                status_placeholder.info("🟢 Camera feed active. Hold a static sign handshape in view. Uncheck to stop.")
                                frame_count = 0

                                # Streaming loop with controlled framerate
                                while st.session_state.get("isl_stream_active", False):
                                    ret, frame = cap.read()
                                    if not ret or frame is None:
                                        status_placeholder.warning("⚠️ Failed to capture frame from webcam.")
                                        break

                                    frame_count += 1
                                    # Process every 2nd frame to optimize CPU utilization
                                    if frame_count % 2 == 0:
                                        res = detector.process_image(frame)
                                        if res.detected:
                                            frame = detector.draw_landmarks(frame, res)
                                            status_placeholder.success(
                                                f"✨ **Recognized Sign:** `{res.predicted_class}` "
                                                f"({res.relative_confidence * 100:.1f}% relative confidence) | "
                                                f"Pose Score: {res.hand_score:.2f}"
                                            )
                                        else:
                                            status_placeholder.info("🔎 Searching for hand... Keep your hand visible and steady.")

                                    # Mirror frame for natural self-view
                                    flipped = cv2.flip(frame, 1)
                                    frame_placeholder.image(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB), use_container_width=True)

                                    # Rate limiter: sleep ~60ms (~15 FPS max)
                                    time.sleep(0.06)

                        except Exception as stream_err:
                            status_placeholder.error(f"Camera stream encountered an error: {stream_err}")
                        finally:
                            if cap is not None and cap.isOpened():
                                cap.release()
                            status_placeholder.caption("🔒 Camera hardware released.")
                    else:
                        frame_placeholder.info("Camera is currently **OFF** and hardware resources are released. Check the box above to start.")

    # =========================================================================
    # TAB 3: Supported Classes & Documentation
    # =========================================================================
    with tab_guide:
        st.subheader("3. Indian Sign Language (ISL) Class Specifications")
        st.markdown(
            "This module recognizes **12 static ISL handshapes**: 5 numerical digits and 7 alphabet letters."
        )

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("##### 🔢 Numerical Digits (5 Classes)")
            st.markdown(
                """
                | Class | Handshape Description |
                | :---: | :--- |
                | **1** | Index finger fully extended upward; thumb and other 3 fingers folded. |
                | **2** | Index and middle fingers extended in a 'V' shape; thumb folds ring/pinky. |
                | **3** | Thumb, index, and middle fingers extended; ring and pinky fingers folded. |
                | **4** | Four fingers extended upright; thumb folded across the palm. |
                | **5** | All five fingers fully extended and spread outward (open palm). |
                """
            )

        with col_g2:
            st.markdown("##### 🔤 Alphabetic Letters (7 Classes)")
            st.markdown(
                """
                | Class | Handshape Description |
                | :---: | :--- |
                | **A** | Closed fist with the thumb resting upright alongside the index finger. |
                | **B** | Flat open hand with four fingers upright together; thumb tucked across palm. |
                | **C** | Fingers and thumb curved into a smooth 'C' arc facing sideways. |
                | **L** | Index finger pointing straight up and thumb extended at a 90° angle ('L' shape). |
                | **V** | Index and middle fingers separated upright in a 'V'; thumb folds ring/pinky. |
                | **W** | Index, middle, and ring fingers extended upward in a 'W'; pinky folded by thumb. |
                | **Y** | Thumb and pinky fingers extended outward; middle three fingers folded into palm. |
                """
            )

        st.markdown("---")
        st.markdown("##### 💡 Usage & Operational Notes")
        st.info(
            "1. **Static Hand Gestures:** This system evaluates static ISL handshapes in individual frames. Ensure the hand is positioned upright, well-lit, and clearly within frame.\n"
            "2. **Optimal Detection:** Keep the hand centered in front of the camera against a neutral background for maximum recognition confidence.\n"
            "3. **Recognition Confidence:** Indicated scores reflect gesture similarity relative to standard ISL reference handshape geometry."
        )
