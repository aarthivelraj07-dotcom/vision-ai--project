"""
Streamlit User Interface for Task 6: Car Colour & Pedestrian Detection.

Features:
1. Image Ingestion: Upload JPG, JPEG, PNG, or load pre-configured traffic scene presets.
2. Parameter Controls:
   - Detection confidence threshold slider.
   - Interactive pedestrian signal counting zone rectangle sliders (X%, Y%).
3. Multi-Object Detection & Slicing:
   - YOLOv8n ONNX via OpenCV DNN on CPU.
   - Slices COCO Class 2 ('car') and Class 0 ('person').
4. Vehicle Color Classification:
   - Inner body sampling ROI.
   - Blue vs. Other classification with Uncertain safeguard.
5. Strict Color-Coded Bounding Boxes:
   - Blue Car: RED (0, 0, 255)
   - Other Car: BLUE (255, 0, 0)
   - Uncertain Car: YELLOW (0, 255, 255)
   - Person: GREEN (0, 255, 0)
   - Signal Zone: AMBER (0, 165, 255)
6. Real-Time Metric Dashboards & Summary Table.
"""

import os
import time
from typing import Optional
from PIL import Image
import cv2
import numpy as np
import streamlit as st

from .detector import (
    CarColorDetector,
    SignalZone,
    TrafficSceneResult,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_BLUE_RATIO_THRESHOLD,
)
from modules.common.theme import apply_theme, render_page_header

# Custom Styling — Teal + Tangerine + Pale Aqua
TASK6_CSS = """
<style>
    .metric-card-car {
        background: #FFFFFF !important;
        border: 1.5px solid #B7DCD7 !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        text-align: center !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 14px rgba(24, 59, 69, 0.06) !important;
    }
    .metric-card-blue {
        background: #FFF8F4 !important;
        border: 1.5px solid #FF914D !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        text-align: center !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 14px rgba(24, 59, 69, 0.06) !important;
    }
    .metric-card-person {
        background: #F4FBF9 !important;
        border: 1.5px solid #087E8B !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        text-align: center !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 14px rgba(24, 59, 69, 0.06) !important;
    }
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        line-height: 1.0 !important;
        margin-top: 4px !important;
        color: #183B45 !important;
    }
    .metric-label {
        font-size: 0.80rem !important;
        font-weight: 700 !important;
        color: #45636A !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .tag-blue-car {
        background-color: rgba(239, 68, 68, 0.18);
        color: #991B1B;
        border: 1.5px solid #EF4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.80rem;
    }
    .tag-other-car {
        background-color: rgba(8, 126, 139, 0.15);
        color: #183B45;
        border: 1.5px solid #087E8B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.80rem;
    }
    .tag-person {
        background-color: rgba(16, 185, 129, 0.18);
        color: #065F46;
        border: 1.5px solid #10B981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.80rem;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_cached_detector() -> CarColorDetector:
    """Cache the YOLOv8n detector instance to optimize reload speed."""
    return CarColorDetector()


def run_car_color_app():
    """Main Streamlit entry point for Task 6: Car Colour & Pedestrian Detection."""
    try:
        st.set_page_config(
            page_title="VISION AI — Car Colour & Pedestrian Detection",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.markdown(TASK6_CSS, unsafe_allow_html=True)
    apply_theme()

    render_page_header(
        "Car Colour Detection",
        "Vehicle detection, car colour classification, and pedestrian crosswalk monitoring",
        "ACTIVE"
    )

    # Color legend bar
    st.markdown("""
    <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; padding: 12px 18px; background: rgba(19, 27, 46, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; flex-wrap: wrap;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.04em;">BOUNDING BOX LEGEND:</span>
        <span style="background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">🔴 Blue Car (Red Box)</span>
        <span style="background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">🔵 Other Car (Blue Box)</span>
        <span style="background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">🟡 Uncertain (Yellow Box)</span>
        <span style="background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">🟢 Pedestrian (Green Box)</span>
        <span style="background: #FFEDD5; color: #9A3412; border: 1px solid #FDBA74; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">🟠 Signal Crossing Zone</span>
    </div>
    """, unsafe_allow_html=True)

    # Parameter Controls Expander
    with st.expander("⚙️ Detection Parameters & Signal Crossing Zone Configuration", expanded=False):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            conf_thresh = st.slider(
                "Vehicle Detection Confidence:",
                min_value=0.10,
                max_value=0.85,
                value=DEFAULT_CONF_THRESHOLD,
                step=0.05,
                help="Minimum confidence threshold for vehicle and pedestrian detection."
            )
            blue_thresh = st.slider(
                "Blue Colour Ratio Threshold:",
                min_value=0.15,
                max_value=0.50,
                value=DEFAULT_BLUE_RATIO_THRESHOLD,
                step=0.02,
                help="Minimum fraction of body pixels required to classify a car as Blue."
            )
        with col_p2:
            st.caption("Adjust rectangular counting zone for pedestrians at the traffic signal:")
            sub1, sub2 = st.columns(2)
            with sub1:
                z_y1 = st.slider("Zone Top (Y min %):", 0, 100, 45, step=5)
                z_x1 = st.slider("Zone Left (X min %):", 0, 100, 0, step=5)
            with sub2:
                z_y2 = st.slider("Zone Bottom (Y max %):", 0, 100, 100, step=5)
                z_x2 = st.slider("Zone Right (X max %):", 0, 100, 100, step=5)

    signal_zone = SignalZone(
        x_min_pct=z_x1 / 100.0,
        y_min_pct=z_y1 / 100.0,
        x_max_pct=z_x2 / 100.0,
        y_max_pct=z_y2 / 100.0
    )

    # Initialize Detector
    try:
        detector = get_cached_detector()
        detector.conf_threshold = conf_thresh
        detector.blue_ratio_threshold = blue_thresh
    except Exception as e:
        st.error(f"❌ Failed to load vehicle detection network: {e}")
        return

    # 3. Main Workflow: Upload & Presets
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.subheader("1. Ingest Traffic Scene Image")
        source_mode = st.radio(
            "Select Image Source:",
            ["📁 Upload Traffic Image", "🧪 Pre-loaded Sample Scene"],
            horizontal=True,
            key="car_color_source_mode"
        )

        image_to_process = None
        source_label = ""

        preset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets", "car_color", "test_images"
        )
        preset_files = [f for f in sorted(os.listdir(preset_dir)) if f.endswith((".jpg", ".png"))] if os.path.exists(preset_dir) else []

        if source_mode == "📁 Upload Traffic Image":
            uploaded_file = st.file_uploader(
                "Choose a traffic scene photo (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"],
                key="car_color_uploader"
            )
            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file)
                    image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    source_label = f"Uploaded file: `{uploaded_file.name}`"
                except Exception as err:
                    st.error(f"Error reading image: {err}")
            else:
                st.info("👆 Please upload a traffic scene photo above to begin vehicle and pedestrian analysis.")
        else:
            if preset_files:
                chosen_preset = st.selectbox(
                    "Select a pre-loaded sample traffic image:",
                    preset_files,
                    key="car_color_preset_select"
                )
                preset_path = os.path.join(preset_dir, chosen_preset)
                if os.path.exists(preset_path):
                    image_to_process = cv2.imread(preset_path)
                    source_label = f"Sample scene: `{chosen_preset}`"
            else:
                st.warning("No sample images found in test_images directory.")

        if image_to_process is not None:
            st.caption(f"**Active Source:** {source_label} ({image_to_process.shape[1]}×{image_to_process.shape[0]} px)")
            st.image(
                cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB),
                caption=f"Source Image Preview: {source_label}",
                use_container_width=True
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(19, 27, 46, 0.7); border: 1.5px dashed rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 25px; text-align: center; color: #94A3B8; margin-top: 15px;">
                    📷 <strong>No image loaded yet</strong><br/>
                    Select a sample scene or upload a traffic photo to inspect detections.
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_results:
        st.subheader("2. Detection & Counting Results")

        if image_to_process is None:
            st.info("👈 Upload an image or select a sample scene on the left to begin traffic analysis.")
        else:
            detect_btn = st.button("🔍 Detect Vehicles & Count Pedestrians", type="primary", use_container_width=True)

            if detect_btn:
                with st.spinner("Analyzing traffic scene vehicles and pedestrian zones..."):
                    t0 = time.time()
                    result: TrafficSceneResult = detector.detect(
                        image_to_process, signal_zone=signal_zone, conf_threshold=conf_thresh
                    )
                    annotated_bgr = detector.draw_annotations(image_to_process, result)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    latency_ms = (time.time() - t0) * 1000.0
                    st.session_state["car_color_active_result"] = (result, annotated_rgb, latency_ms, source_label)

            active_result_data = st.session_state.get("car_color_active_result")

            if active_result_data is not None and active_result_data[3] == source_label:
                result, annotated_rgb, latency_ms, _ = active_result_data

                # Metric Cards
                st.markdown("##### 📊 Real-Time Traffic Counts")
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.markdown(
                        f"<div class='metric-card-car'>"
                        f"<div class='metric-label'>Total Cars</div>"
                        f"<div class='metric-value' style='color:#38BDF8;'>{result.total_cars}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m_c2:
                    st.markdown(
                        f"<div class='metric-card-blue'>"
                        f"<div class='metric-label'>Blue Cars (Red Box)</div>"
                        f"<div class='metric-value' style='color:#F87171;'>{result.blue_cars}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m_c3:
                    st.markdown(
                        f"<div class='metric-card-car'>"
                        f"<div class='metric-label'>Other Cars (Blue Box)</div>"
                        f"<div class='metric-value' style='color:#60A5FA;'>{result.other_cars}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                m_p1, m_p2, m_p3 = st.columns(3)
                with m_p1:
                    st.markdown(
                        f"<div class='metric-card-person'>"
                        f"<div class='metric-label'>Total People</div>"
                        f"<div class='metric-value' style='color:#34D399;'>{result.total_people}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m_p2:
                    st.markdown(
                        f"<div class='metric-card-person'>"
                        f"<div class='metric-label'>In Signal Zone</div>"
                        f"<div class='metric-value' style='color:#10B981;'>{result.people_in_signal_zone}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m_p3:
                    st.markdown(
                        f"<div class='metric-card-car'>"
                        f"<div class='metric-label'>Processing Speed</div>"
                        f"<div class='metric-value' style='color:#38BDF8; font-size:1.8rem;'>{latency_ms:.0f} ms</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                if result.uncertain_cars > 0:
                    st.caption(f"⚠️ **Note:** {result.uncertain_cars} vehicle(s) were flagged as UNCERTAIN (Yellow Box) due to small size or extreme shadow/glare.")

                # Annotated Visual Display
                st.markdown("##### 🖼️ Annotated Detections")
                st.image(
                    annotated_rgb,
                    caption=f"Annotated Scene (Cars: {result.total_cars} | Blue: {result.blue_cars} | Pedestrians: {result.total_people})",
                    use_container_width=True
                )

                # Legend Guide
                st.markdown(
                    """
                    <div style="background: rgba(19, 27, 46, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 12px 16px; margin-top: 10px; font-size: 0.85rem; color: #E2E8F0;">
                        <strong style="color: #F8FAFC;">Bounding Box Legend:</strong><br/>
                        • <span class="tag-blue-car">RED BOX</span> ➔ <strong style="color:#FCA5A5;">Blue Car</strong> (Verified Blue Finish)<br/>
                        • <span class="tag-other-car">BLUE BOX</span> ➔ <strong style="color:#93C5FD;">Other Car</strong> (White, Black, Red, Silver, etc.)<br/>
                        • <span style="background-color:rgba(234, 179, 8, 0.25); color:#FDE047; border:1px solid rgba(234, 179, 8, 0.45); padding:3px 8px; border-radius:6px; font-weight:700;">YELLOW BOX</span> ➔ <strong style="color:#FDE047;">Uncertain Car</strong> (Shadow or low lighting)<br/>
                        • <span class="tag-person">GREEN BOX</span> ➔ <strong style="color:#6EE7B7;">Pedestrian</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Detailed Object Breakdown Table
                with st.expander("📋 Detailed Object Breakdown", expanded=False):
                    if not result.objects:
                        st.write("No cars or pedestrians detected in the current image.")
                    else:
                        table_data = []
                        for idx, obj in enumerate(result.objects):
                            c_info = obj.car_color
                            color_desc = c_info.color if c_info else "N/A"
                            blue_pct = f"{c_info.blue_ratio * 100:.1f}%" if c_info else "N/A"
                            zone_desc = "YES" if obj.in_signal_zone else ("NO" if obj.class_name == "person" else "-")
                            table_data.append({
                                "#": idx + 1,
                                "Class": obj.class_name.upper(),
                                "Confidence": f"{obj.confidence * 100:.1f}%",
                                "Bounding Box": f"({obj.box[0]}, {obj.box[1]}, {obj.box[2]}×{obj.box[3]})",
                                "Color Assigned": color_desc,
                                "Blue Ratio": blue_pct,
                                "In Signal Zone": zone_desc
                            })
                        st.table(table_data)
            else:
                st.info("👆 Click **'🔍 Detect Vehicles & Count Pedestrians'** above to run detection on this traffic scene.")
