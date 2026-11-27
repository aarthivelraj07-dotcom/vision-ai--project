"""
Streamlit User Interface for Task 2: Animal Detection & Tracking.
Adheres to the application's white, lavender, and coral visual theme.
Dietary color indicators:
- Carnivore: Red (#EF4444)
- Herbivore: Green (#10B981)
- Omnivore / Other: Amber/Gold (#F59E0B)

Features:
- Single image ingestion (Upload and Webcam snapshot).
- Video ingestion with real-time multi-object tracking and deduplication.
- Dynamic popup toast notification and banner alerts.
- Metric cards for Total Animals, Carnivores, Herbivores, and Omnivores.
- Species-wise breakdown table with average confidence.
- Complete 10-species taxonomy and dietary guide.
"""

import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.animal_detection.taxonomy import (
    AnimalTaxonomy,
    DietCategory,
    CATEGORY_COLORS_HEX,
    COCO_ANIMAL_CLASSES,
)
from modules.animal_detection.detector import AnimalDetector, DEFAULT_CONF_THRESHOLD
from modules.animal_detection.tracker import AnimalTracker
from modules.common.theme import apply_theme, render_page_header


# Dedicated UI CSS preserving the white, lavender, and coral design
ANIMAL_UI_CSS = """
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

    /* Card containers */
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

    /* Classification badge pills */
    .badge-carnivore {
        background-color: rgba(239, 68, 68, 0.18);
        color: #991B1B;
        border: 1.5px solid #EF4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
    }
    .badge-herbivore {
        background-color: rgba(16, 185, 129, 0.18);
        color: #065F46;
        border: 1.5px solid #10B981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
    }
    .badge-omnivore {
        background-color: rgba(255, 145, 77, 0.18);
        color: #C25E00;
        border: 1.5px solid #FF914D;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
    }

    /* Responsive Visual Legend Card */
    .legend-card {
        background: #FFFFFF;
        border: 1.5px solid #B7DCD7;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(24, 59, 69, 0.05);
        color: #183B45;
    }
    .legend-title {
        font-weight: 700;
        font-size: 0.8rem;
        color: #45636A;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .legend-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }

    /* Responsive Species-Wise Table */
    .table-container {
        width: 100%;
        overflow-x: auto;
        border-radius: 12px;
        border: 1.5px solid #B7DCD7;
        background-color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(24, 59, 69, 0.05);
        margin-bottom: 20px;
    }
    .species-table {
        width: 100%;
        border-collapse: collapse;
        font-family: inherit;
    }
    .species-table th {
        background-color: #DDF5F2;
        color: #183B45;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 12px 14px;
        border-bottom: 1.5px solid #B7DCD7;
        white-space: normal;
        line-height: 1.25;
    }
    .species-table td {
        padding: 12px 14px;
        font-size: 0.88rem;
        color: #183B45;
        border-bottom: 1px solid #B7DCD7;
        vertical-align: middle;
    }
    .species-table tr:last-child td {
        border-bottom: none;
    }
    .species-table tr:hover {
        background-color: #F4FBF9;
    }
</style>
"""


def render_species_breakdown_table(breakdown: List[Dict[str, Any]]):
    """
    Render a responsive, beautifully styled species breakdown table without horizontal overflow.
    Every column, including Average Confidence, is guaranteed 100% visible.
    """
    if not breakdown:
        return

    rows = []
    for item in breakdown:
        species = item["species"]
        cat = item["category"]
        count = item["count"]
        avg_conf = item["avg_confidence"] * 100

        if cat == DietCategory.CARNIVORE.value:
            badge_class = "badge-carnivore"
        elif cat == DietCategory.HERBIVORE.value:
            badge_class = "badge-herbivore"
        else:
            badge_class = "badge-omnivore"

        rows.append(
            f"<tr>"
            f"<td style='font-weight:600; color:#F1F5F9;'>{species}</td>"
            f"<td><span class='{badge_class}'>{cat}</span></td>"
            f"<td style='text-align:center; font-weight:600; color:#F1F5F9;'>{count}</td>"
            f"<td style='text-align:right; font-weight:700; color:#38BDF8;'>{avg_conf:.1f}%</td>"
            f"</tr>"
        )

    table_html = (
        "<div class='table-container'>"
        "<table class='species-table'>"
        "<thead>"
        "<tr>"
        "<th style='width: 25%; text-align: left;'>Species</th>"
        "<th style='width: 38%; text-align: left;'>Dietary Category</th>"
        "<th style='width: 15%; text-align: center;'>Count</th>"
        "<th style='width: 22%; text-align: right;'>Avg Confidence</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_cumulative_species_table(species_dict: Dict[str, int]):
    """Render responsive cumulative unique species table for video tracking."""
    if not species_dict:
        return

    rows = []
    for s_name, count in sorted(species_dict.items(), key=lambda x: -x[1]):
        cat = AnimalTaxonomy.get_category(s_name)
        cat_val = cat.value
        if cat == DietCategory.CARNIVORE:
            badge_class = "badge-carnivore"
        elif cat == DietCategory.HERBIVORE:
            badge_class = "badge-herbivore"
        else:
            badge_class = "badge-omnivore"

        rows.append(
            f"<tr>"
            f"<td style='font-weight:600; color:#F1F5F9;'>{s_name.capitalize()}</td>"
            f"<td><span class='{badge_class}'>{cat_val}</span></td>"
            f"<td style='text-align:right; font-weight:700; color:#38BDF8;'>{count}</td>"
            f"</tr>"
        )

    table_html = (
        "<div class='table-container'>"
        "<table class='species-table'>"
        "<thead>"
        "<tr>"
        "<th style='width: 38%; text-align: left;'>Species</th>"
        "<th style='width: 38%; text-align: left;'>Dietary Category</th>"
        "<th style='width: 24%; text-align: right;'>Unique Individuals</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector_instance() -> AnimalDetector:
    """Cache and retrieve singleton AnimalDetector instance."""
    return AnimalDetector.get_instance()


def get_tracker_instance() -> AnimalTracker:
    """Retrieve persistent video tracker instance."""
    if "animal_tracker" not in st.session_state:
        st.session_state.animal_tracker = AnimalTracker()
    return st.session_state.animal_tracker


def render_image_detection_tab(detector: AnimalDetector):
    """Render Image Ingestion & Static Multi-Animal Detection."""
    st.subheader("Image Animal Detection")
    st.markdown("Upload a photo or capture a camera snapshot to detect multiple animals and classify their dietary types.")

    col_ctrl, col_info = st.columns([1, 1], gap="large")
    with col_ctrl:
        conf_thresh = st.slider(
            "Detection Confidence Threshold:",
            min_value=0.20,
            max_value=0.90,
            value=float(DEFAULT_CONF_THRESHOLD),
            step=0.05,
            help="Minimum confidence score to accept an animal detection. Default is 0.35."
        )
    with col_info:
        st.markdown(
            """
            <div class="legend-card">
                <div class="legend-title">Dietary Classification Legend</div>
                <div class="legend-badges">
                    <span class="badge-carnivore">● Carnivore (Red)</span>
                    <span class="badge-herbivore">● Herbivore (Green)</span>
                    <span class="badge-omnivore">● Omnivore / Other (Amber)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    input_source = st.radio("Select Image Input Source:", ["Upload Image File", "Capture Webcam Snapshot"], horizontal=True)
    image_input = None

    if input_source == "Upload Image File":
        uploaded_file = st.file_uploader("Choose an image with single, multiple, or mixed animals:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            if len(bytes_data) > 0:
                image_input = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if image_input is None:
                st.error("Uploaded file could not be decoded as a valid image. Please provide an uncorrupted JPG, JPEG, or PNG file.")
    else:
        cam_snap = st.camera_input("Capture camera snapshot")
        if cam_snap is not None:
            bytes_data = cam_snap.getvalue()
            if len(bytes_data) > 0:
                image_input = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if image_input is None:
                st.error("Camera snapshot could not be processed as an image.")

    if image_input is None:
        st.markdown(
            """
            <div class="custom-card" style="text-align:center; color:#94A3B8; padding:35px 20px;">
                <p style="font-size:1.05rem; font-weight:600; margin-bottom:6px;">📷 No Image Loaded Yet</p>
                <p style="font-size:0.85rem; margin:0;">Upload a photo or capture a camera snapshot above to preview and analyze wildlife.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Step 1: Preview & Action Controls
        col_prev, col_action = st.columns([1, 1], gap="medium")
        with col_prev:
            st.markdown("##### 1. Source Image Preview")
            st.image(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB), caption="Uploaded Image Preview", use_container_width=True)

        with col_action:
            st.markdown("##### 2. Run Animal Detection")
            st.write(f"Active Confidence Threshold: **{conf_thresh:.2f}**")
            st.caption("Click the button below to detect animal species, draw colored bounding boxes (🔴 Carnivore, 🟢 Herbivore, 🟡 Omnivore), and compute dietary statistics.")
            run_btn = st.button("🔍 Analyse Animal Species & Diet", type="primary", key="btn_run_animal_detection", use_container_width=True)

        # Cache key to synchronize image state
        img_token = f"{image_input.shape}_{float(np.mean(image_input)):.2f}_{conf_thresh}"
        active_res = st.session_state.get("animal_active_result")

        if run_btn:
            with st.spinner("Analyzing image, detecting animal species, and evaluating dietary classifications..."):
                detections = detector.detect_animals(image_input, conf_threshold=conf_thresh)
                summary = detector.get_summary_statistics(detections)
                annotated_bgr = detector.draw_detections(image_input, detections)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.session_state["animal_active_result"] = (img_token, detections, summary, annotated_rgb)
                active_res = st.session_state["animal_active_result"]

        if active_res is not None and active_res[0] == img_token:
            _, detections, summary, annotated_rgb = active_res
            total_count = summary["total_count"]

            # Trigger notification / toast popup
            if total_count > 0:
                st.toast(
                    f"🐾 Detected {total_count} animal(s): {summary['carnivore_count']} Carnivore(s), "
                    f"{summary['herbivore_count']} Herbivore(s), {summary['omnivore_other_count']} Omnivore(s)!",
                    icon="🦁"
                )
                st.success(
                    f"**Total Animals Detected: {total_count}** — "
                    f"{summary['carnivore_count']} Carnivore(s) (Red), "
                    f"{summary['herbivore_count']} Herbivore(s) (Green), "
                    f"{summary['omnivore_other_count']} Omnivore / Other (Amber)."
                )
            else:
                st.info("No animals detected in the provided image. Adjust the confidence slider if needed.")

            # Metric Cards Layout with spacing
            st.markdown("<div style='margin-top: 10px; margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Animals", f"{total_count}")
            with col_m2:
                st.metric("Carnivores (Red)", f"{summary['carnivore_count']}")
            with col_m3:
                st.metric("Herbivores (Green)", f"{summary['herbivore_count']}")
            with col_m4:
                st.metric("Omnivores / Other (Amber)", f"{summary['omnivore_other_count']}")

            st.markdown("<div style='margin-top: 18px; margin-bottom: 8px;'></div>", unsafe_allow_html=True)

            # Results Section Layout: Balanced 1:1 side-by-side columns
            col_img, col_breakdown = st.columns([1, 1], gap="large")
            with col_img:
                st.markdown("#### 🖼️ Detection Preview")
                st.image(
                    annotated_rgb,
                    caption=f"Detection Preview ({total_count} Animal{'s' if total_count != 1 else ''} Identified)",
                    use_container_width=True
                )

            with col_breakdown:
                st.markdown("#### 📊 Species-Wise Breakdown")
                if summary["species_breakdown"]:
                    render_species_breakdown_table(summary["species_breakdown"])

                    st.markdown("#### 🐾 Detected Entities Detail")
                    for i, det in enumerate(detections, 1):
                        cat_val = det["category"]
                        if cat_val == DietCategory.CARNIVORE.value:
                            badge_html = "<span class='badge-carnivore'>CARNIVORE</span>"
                            card_class = "coral-card"
                        elif cat_val == DietCategory.HERBIVORE.value:
                            badge_html = "<span class='badge-herbivore'>HERBIVORE</span>"
                            card_class = "lavender-card"
                        else:
                            badge_html = "<span class='badge-omnivore'>OMNIVORE / OTHER</span>"
                            card_class = "custom-card"

                        card_html = (
                            f"<div class='{card_class}'>"
                            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
                            f"<span style='font-size:1rem; font-weight:700; color:#172033;'>Entity #{i}: {det['class_name'].capitalize()}</span>"
                            f"{badge_html}"
                            f"</div>"
                            f"<div style='color:#667085; font-size:0.88rem; line-height:1.6;'>"
                            f"<strong>Confidence:</strong> {det['confidence']:.1%}<br>"
                            f"<strong>Bounding Box (x, y, w, h):</strong> <code style='background:#EFF6FF; color:#1D4ED8; padding:2px 6px; border-radius:4px;'>{det['bbox']}</code>"
                            f"</div>"
                            f"</div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                        with st.expander(f"Dietary Rationale: {det['class_name'].capitalize()}"):
                            st.write(AnimalTaxonomy.get_dietary_rationale(det["class_name"]))
                else:
                    empty_card_html = (
                        "<div class='custom-card' style='text-align:center; color:#64748B; padding:28px;'>"
                        "<p style='font-size:1.05rem; font-weight:600; margin-bottom:6px;'>🐾 No Animals Identified</p>"
                        "<p style='font-size:0.86rem; margin-bottom:0;'>Try lowering the detection confidence threshold slider or upload another image containing animals.</p>"
                        "</div>"
                    )
                    st.markdown(empty_card_html, unsafe_allow_html=True)
        else:
            st.info("👈 Click **Analyse Animal Species & Diet** above to begin detection and view results.")


def render_video_tracking_tab(detector: AnimalDetector):
    """Render Video Ingestion with Multi-Object Tracking to Prevent Double Counting."""
    st.subheader("Video Animal Tracking & Deduplication")
    st.markdown("Upload video footage to detect and track animals over time without repeatedly counting the same individual.")

    st.info(
        "💡 **Movement Tracking:** Intelligently tracks animals across frames to avoid double-counting the same individual."
    )

    uploaded_video = st.file_uploader("Upload video file (.mp4, .avi, .mov):", type=["mp4", "avi", "mov"])
    conf_thresh = st.slider(
        "Video Detection Confidence:",
        min_value=0.20,
        max_value=0.85,
        value=0.35,
        step=0.05,
        key="video_conf_slider"
    )

    process_video = uploaded_video is not None and st.button("Process & Track Video", type="primary")
    if uploaded_video is not None and process_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.getvalue())
        temp_video_path = tfile.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Uploaded video could not be opened. Please provide a valid, uncorrupted video file (.mp4, .avi, .mov).")
            cap.release()
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
            return

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_video_frames <= 0:
            st.error("Uploaded video contains no readable frames.")
            cap.release()
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
            return

        st.write(f"Video Loaded: **{total_video_frames} frames** at ~{fps:.1f} FPS.")

        if process_video:
            tracker = get_tracker_instance()
            tracker.reset()

            progress_bar = st.progress(0, text="Processing video frames...")
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()

            frame_idx = 0
            sample_rate = max(1, int(fps // 10))  # Process 10 frames per second of video

            latest_metrics = {}
            annotated_preview = None

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % sample_rate != 0:
                        continue

                    raw_dets = detector.detect_animals(frame, conf_threshold=conf_thresh)
                    tracked_dets, latest_metrics = tracker.update(raw_dets)
                    annotated_frame = detector.draw_detections(frame, tracked_dets)
                    annotated_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    if frame_idx % (sample_rate * 3) == 0:
                        frame_placeholder.image(
                            annotated_preview,
                            caption=f"Frame {frame_idx}/{total_video_frames} — Tracked Entities",
                            use_container_width=True
                        )
                        metrics_placeholder.markdown(f"""
                        **Current Frame Count:** `{latest_metrics.get('current_frame_count', 0)}` animals currently in view | 
                        **Cumulative Unique Animals:** `{latest_metrics.get('cumulative_unique_count', 0)}` distinct entities tracked
                        """)

                    progress_val = min(1.0, frame_idx / max(1, total_video_frames))
                    progress_bar.progress(progress_val, text=f"Processing frame {frame_idx}/{total_video_frames}...")
            finally:
                cap.release()
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

            progress_bar.progress(1.0, text="Video processing complete!")

            # Final Results Display
            st.success(
                f"**Video Tracking Analysis Complete!** Total Unique Animals Tracked: "
                f"**{latest_metrics.get('cumulative_unique_count', 0)}**"
            )

            # Dual Counter Cards
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("Cumulative Unique Animals", latest_metrics.get("cumulative_unique_count", 0))
            with col_d2:
                carn_unique = latest_metrics.get("cumulative_category_counts", {}).get(DietCategory.CARNIVORE.value, 0)
                st.metric("Unique Carnivores (Red)", carn_unique)
            with col_d3:
                herb_unique = latest_metrics.get("cumulative_category_counts", {}).get(DietCategory.HERBIVORE.value, 0)
                st.metric("Unique Herbivores (Green)", herb_unique)

            # Cumulative Species Table
            st.markdown("#### 📊 Cumulative Unique Species Count")
            species_dict = latest_metrics.get("cumulative_species_counts", {})
            if species_dict:
                render_cumulative_species_table(species_dict)
            else:
                st.info("No unique animals were tracked in this video sequence.")

            # Documented Tracking Bounds Notice
            with st.expander("ℹ️ Tracking Bounds & Known Limitations"):
                st.markdown("""
                - **Temporary Occlusions:** Tracks tolerate up to 15 missed frames (~0.5s). If an animal is hidden behind foliage for longer, it receives a new track ID upon re-emergence.
                - **Scene Exits & Re-entries:** When an animal walks completely off camera and returns later, it cannot be distinguished from a new animal of the same species.
                - **Dense Herds:** If animals overlap closely or cross paths directly, identity switching may occur.
                - **Notice:** Unique counts represent spatial continuity tracking estimates; they are not biological guarantees under severe occlusions.
                """)


def render_taxonomy_guide_tab():
    """Render the Species Reference & Dietary Taxonomy Guide."""
    st.subheader("Animal Dietary Taxonomy Guide")
    st.markdown("Scientific and dietary classification guide for recognized wildlife species.")

    species_list = AnimalTaxonomy.get_supported_species()
    for s_name in species_list:
        cat = AnimalTaxonomy.get_category(s_name)
        color_hex = AnimalTaxonomy.get_color_hex(cat)
        rationale = AnimalTaxonomy.get_dietary_rationale(s_name)

        if cat == DietCategory.CARNIVORE:
            card_class = "coral-card"
            badge_html = "<span class='badge-carnivore'>CARNIVORE (RED)</span>"
        elif cat == DietCategory.HERBIVORE:
            card_class = "lavender-card"
            badge_html = "<span class='badge-herbivore'>HERBIVORE (GREEN)</span>"
        else:
            card_class = "custom-card"
            badge_html = "<span class='badge-omnivore'>OMNIVORE / OTHER (AMBER)</span>"

        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="margin-top:0; color: #F8FAFC;">{s_name.capitalize()} {badge_html}</h4>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 0;">{rationale}</p>
        </div>
        """, unsafe_allow_html=True)


def run_animal_app():
    """Main Streamlit Entry Point for Task 2: Animal Detection."""
    try:
        st.set_page_config(
            page_title="VISION AI — Animal Detection",
            page_icon="🐾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.markdown(ANIMAL_UI_CSS, unsafe_allow_html=True)
    apply_theme()
    detector = get_detector_instance()

    render_page_header(
        "Animal Detection",
        "Identify wild animals and track movements across image and video scenes",
        "ACTIVE"
    )

    # Dietary Indicators Badge Bar
    st.markdown("""
    <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; padding: 12px 18px; background: rgba(19, 27, 46, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">DIETARY INDICATORS:</span>
        <span class="badge-carnivore">🔴 Carnivore (Red Alert)</span>
        <span class="badge-herbivore">🟢 Herbivore (Green)</span>
        <span class="badge-omnivore">🟡 Omnivore / Other (Amber)</span>
    </div>
    """, unsafe_allow_html=True)

    tab_image, tab_video, tab_taxonomy = st.tabs([
        "📷 Static Image Analysis",
        "🎥 Video Detection & Tracking",
        "📖 Dietary Taxonomy Reference"
    ])

    with tab_image:
        render_image_detection_tab(detector)

    with tab_video:
        render_video_tracking_tab(detector)

    with tab_taxonomy:
        render_taxonomy_guide_tab()
