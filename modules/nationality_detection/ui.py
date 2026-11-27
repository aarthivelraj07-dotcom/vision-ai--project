"""
Streamlit User Interface for Task 4: Nationality Detection & Conditional Profile Analysis.

Core Static Image Workflow:
1. Static portrait image upload (JPG, PNG, JPEG) or pre-loaded test presets.
2. Explicit user nationality declaration via searchable ISO country catalog.
3. Optional user-provided age declaration (automated heuristic prediction disabled).
4. Multi-occupant detection and strict person-to-age isolation (no cross-person attribution).
5. Optional MRZ travel document text parser (with strict non-authenticity disclaimer).
6. Automated face detection, CIELAB dress color extraction, and FER CNN emotion prediction.
7. Strict enforcement of the 4 assignment conditional output rules:
   - Indian: Age and Dress Colour
   - US: Age and Emotion
   - African category: Emotion and Dress Colour
   - Other nationalities: Nationality and Emotion
"""

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from .constants import ISO_COUNTRIES, is_african_nation, classify_nationality_category
from .nationality import NationalityResolver, NationalityInfo
from .pipeline import Task4Pipeline, Task4Result, PersonProfile
from modules.common.theme import apply_theme

_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@st.cache_resource(show_spinner=False)
def get_cached_pipeline() -> Task4Pipeline:
    """Cache Task 4 pipeline instance to avoid re-initializing on reruns."""
    return Task4Pipeline()


def run_nationality_app():
    """Main Streamlit entry point for Task 4."""
    try:
        st.set_page_config(
            page_title="VISION AI — Nationality & Profile Analysis",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    apply_theme()

    # Main Header
    st.title("🌍 Task 4: Nationality Detection & Conditional Profile")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 1rem; margin-top: -10px;'>"
        "Extracts visual biometrics (dress colour and emotion) and conditionally presents "
        "profile attributes dictated by the declared nationality.</p>",
        unsafe_allow_html=True
    )

    # Assignment Conditions & Ethical Invariance Strip
    cond_col1, cond_col2 = st.columns([3, 2])
    with cond_col1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1.5px solid #B7DCD7; border-left: 4px solid #087E8B; padding: 14px 18px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 3px 12px rgba(24, 59, 69, 0.06);">
                <strong style="color: #183B45; font-size: 0.92rem;">Assignment Conditional Output Rules:</strong>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; font-size: 0.85rem; color: #183B45;">
                    <div>🇮🇳 <strong>Indian:</strong> Age & Dress Colour</div>
                    <div>🇺🇸 <strong>US:</strong> Age & Facial Emotion</div>
                    <div>🌍 <strong>African:</strong> Emotion & Dress Colour</div>
                    <div>🌐 <strong>Other:</strong> Nationality & Emotion</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cond_col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1.5px solid #B7DCD7; border-left: 4px solid #FF914D; padding: 14px 18px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 3px 12px rgba(24, 59, 69, 0.06);">
                <strong style="color: #183B45; font-size: 0.92rem;">⚖️ Ethical Biometrics Invariance:</strong>
                <p style="margin: 6px 0 0 0; color: #45636A; font-size: 0.82rem; line-height: 1.4;">
                    Nationality is a legal citizenship status obtained via declaration or standardized documents. 
                    It is <strong>never</strong> inferred from facial appearance, skin tone, or clothing.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Initialize Pipeline (cached resource)
    pipeline = get_cached_pipeline()

    # Section 1: Inputs
    col_input_left, col_input_right = st.columns([1, 1])

    with col_input_left:
        st.subheader("1. Upload Portrait Image")
        uploaded_file = st.file_uploader(
            "Choose a clear portrait image (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Upload an image showing one or more people."
        )

        preset_options = [
            "None (Upload via file uploader)",
            "Preset 1: Indian National (Navy Blue Garment)",
            "Preset 2: US National (Gray Garment)",
            "Preset 3: Nigerian National / African Region (Red Garment)",
            "Preset 4: Japanese National / Other Sovereign (Green Garment)",
            "Preset 5: Poor Quality / Cropped Fallback",
            "Preset 6: Multi-Person Test (2 Occupants)",
            "Preset 7: Group Photograph (5 Occupants)",
            "Preset 8: Crowd / Large Group (10 Occupants)",
        ]
        selected_preset = st.selectbox(
            "Or select sample test preset:",
            options=preset_options,
            index=0,
            help="Load pre-calibrated portrait images for instant interactive verification."
        )

        # Resolve image immediately so face count & bounding boxes are available for selection
        image_bgr = None
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif selected_preset != "None (Upload via file uploader)":
            preset_map = {
                "Preset 1: Indian National (Navy Blue Garment)": "portrait_indian_test.jpg",
                "Preset 2: US National (Gray Garment)": "portrait_us_test.jpg",
                "Preset 3: Nigerian National / African Region (Red Garment)": "portrait_african_test.jpg",
                "Preset 4: Japanese National / Other Sovereign (Green Garment)": "portrait_other_test.jpg",
                "Preset 5: Poor Quality / Cropped Fallback": "portrait_poor_quality.jpg",
                "Preset 6: Multi-Person Test (2 Occupants)": "portrait_multi_person_test.jpg",
                "Preset 7: Group Photograph (5 Occupants)": "portrait_group_5_test.jpg",
                "Preset 8: Crowd / Large Group (10 Occupants)": "portrait_group_10_test.jpg",
            }
            filename = preset_map.get(selected_preset)
            if filename:
                preset_path = os.path.join(_root_dir, "datasets", "nationality_detection", "test_images", filename)
                if os.path.exists(preset_path):
                    image_bgr = cv2.imread(preset_path)

        # Pre-pass face detection for interactive preview and selection
        detected_faces = pipeline.detect_all_faces(image_bgr) if image_bgr is not None else []
        num_detected = len(detected_faces)

        if image_bgr is not None:
            preview_img = image_bgr.copy()
            box_colors = [
                (0, 255, 0),      # Bright Green
                (0, 165, 255),    # Orange
                (255, 105, 180),  # Hot Pink
                (255, 255, 0),    # Cyan/Yellow
                (0, 255, 255),    # Yellow
                (255, 191, 0),    # Deep Sky
                (147, 112, 219),  # Medium Purple
                (50, 205, 50),    # Lime Green
                (255, 99, 71),    # Tomato Red
                (0, 191, 255),    # Deep Sky Blue
            ]
            f_scale = 0.50 if num_detected > 4 else 0.65
            for f_i, (fx, fy, fw, fh) in enumerate(detected_faces):
                p_label = f"Person {f_i + 1}"
                c = box_colors[f_i % len(box_colors)]
                cv2.rectangle(preview_img, (fx, fy), (fx + fw, fy + fh), c, 2)
                cv2.putText(preview_img, p_label, (fx, max(22, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 0, 0), 3)
                cv2.putText(preview_img, p_label, (fx, max(22, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, f_scale, c, 2)

            caption = (
                f"Image Preview: {num_detected} face(s) successfully detected (ordered left-to-right)"
                if num_detected > 0
                else "Image Preview (No faces detected)"
            )
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)

    with col_input_right:
        st.subheader("2. Subject Declarations")
        
        # Build country options list
        country_display_list = [f"{c['name']} ({c['alpha_3']})" for c in ISO_COUNTRIES]
        default_index = 0
        if "Indian" in selected_preset:
            default_index = [i for i, c in enumerate(country_display_list) if "India" in c][0]
        elif "US" in selected_preset:
            default_index = [i for i, c in enumerate(country_display_list) if "United States" in c][0]
        elif "Nigerian" in selected_preset or "African" in selected_preset:
            default_index = [i for i, c in enumerate(country_display_list) if "Nigeria" in c][0]
        elif "Japanese" in selected_preset:
            default_index = [i for i, c in enumerate(country_display_list) if "Japan" in c][0]

        selected_country_str = st.selectbox(
            "Select Declared Nationality:",
            options=country_display_list,
            index=default_index,
            help="Explicitly declare the individual's legal citizenship. Never inferred from appearance."
        )
        
        # Extract name from string
        declared_country_name = selected_country_str.split(" (")[0]

        # Subject Selection & Age Declaration Controls
        if num_detected > 1:
            st.markdown(
                f"<div style='background-color: #EFF6FF; border: 1px solid #93C5FD; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;'>"
                f"👥 <strong>{num_detected} Occupants Detected:</strong> Bounding box labels indicate Person 1 (left) through Person {num_detected} (right). "
                f"Select target subject to assign user-declared age.</div>",
                unsafe_allow_html=True
            )
            subject_options = [f"Person {i + 1} (Face {i + 1} from left)" for i in range(num_detected)]
            if num_detected <= 10:
                selected_subject_str = st.radio(
                    "Whose age are you entering?",
                    options=subject_options,
                    index=0,
                    horizontal=True,
                    help="Select which detected individual you are providing the declared age for."
                )
            else:
                selected_subject_str = st.selectbox(
                    "Whose age are you entering?",
                    options=subject_options,
                    index=0,
                    help="Select which detected individual you are providing the declared age for."
                )
            selected_subject = pipeline.normalize_subject_id(selected_subject_str)

            declared_age_input = st.number_input(
                f"Optional User-Provided Age for {selected_subject} (Years):",
                min_value=0,
                max_value=120,
                value=0,
                step=1,
                help=f"Explicitly declare age for {selected_subject} (leave 0 for unavailable). Other occupants without declared age will display 'Age estimation unavailable'."
            )
        else:
            selected_subject = "Person 1"
            declared_age_input = st.number_input(
                "Optional User-Provided Age (Years):",
                min_value=0,
                max_value=120,
                value=0,
                step=1,
                help="Explicitly declare age (leave 0 for unavailable). Automated biometric age prediction is disabled to eliminate demographic bias."
            )

        # Optional Document Data Expander
        with st.expander("📄 Optional: Extract from Travel Document MRZ (TD3)"):
            st.markdown(
                "<span style='color: #DC2626; font-size: 0.85rem;'>⚠️ "
                "<strong>Notice:</strong> Optical extraction does NOT establish document authenticity. "
                "CSCA cryptographic validation is not performed.</span>",
                unsafe_allow_html=True
            )
            mrz_line1 = st.text_input(
                "MRZ Line 1 (44 chars, e.g. P<UTOERIKSSON<<ANNA...)",
                placeholder="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
            )
            mrz_line2 = st.text_input(
                "MRZ Line 2 (44 chars, e.g. L898902C36IND...)",
                placeholder="L898902C36IND7408122F1204159ZE184226B<<<<<10"
            )
            use_mrz = st.checkbox("Use Extracted MRZ Data as Source", value=False)

    if image_bgr is None:
        st.info("👆 Please upload an image or select a test preset to begin static profile analysis.")
        return

    # Determine input data
    mrz_tuple = (mrz_line1, mrz_line2) if use_mrz and mrz_line1 and mrz_line2 else None
    declared_age_val = int(declared_age_input) if declared_age_input > 0 else None

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run_btn = st.button("🔍 Analyse Profile & Conditions", type="primary", key="btn_run_task4_analysis", use_container_width=True)

    task4_token = f"{image_bgr.shape}_{float(np.mean(image_bgr)):.2f}_{declared_country_name}_{declared_age_val}_{selected_subject}_{mrz_tuple}"
    active_t4 = st.session_state.get("task4_active_result")

    if run_btn:
        with st.spinner("Analyzing portrait features and evaluating conditional profile..."):
            result: Task4Result = pipeline.process_image(
                image=image_bgr,
                nationality_declaration=declared_country_name,
                mrz_data=mrz_tuple,
                declared_age=declared_age_val,
                selected_subject=selected_subject
            )
            st.session_state["task4_active_result"] = (task4_token, result)
            active_t4 = st.session_state["task4_active_result"]

    if active_t4 is not None and active_t4[0] == task4_token:
        _, result = active_t4
        st.markdown("---")
        st.subheader("3. Analysis Results & Conditional Presentation")

        if not result.has_face:
            st.error(f"❌ {result.error_message}")
            st.info("ℹ️ Person-specific biometric analysis (age, dress colour, emotion) cannot be performed without a detected face.")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image (No faces detected)", use_container_width=True)
            return

        # Multi-occupant Banner
        if result.num_faces > 1:
            st.info(
                f"👥 **Group / Multi-Occupant Analysis ({result.num_faces} faces successfully localized):** "
                "Biometric features and age are tracked independently per person. "
                "In crowded or group imagery, partial occlusions, profile angles, or low lighting may prevent some individuals from being detected; "
                f"analysis is restricted strictly to the {result.num_faces} verified face regions. "
                f"User-declared age is mapped exclusively to **{result.selected_subject}** and is never assigned to other occupants."
            )

        # Visual Presentation Columns
        col_vis, col_profile = st.columns([1, 1])

        with col_vis:
            st.markdown("#### 📷 Visual Localization")
            annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"Detected Occupants ({result.num_faces}) & Localization", use_container_width=True)

            # Torso Status for Primary Occupant
            if result.dress_color:
                if result.dress_color.is_determined:
                    st.success(f"👗 Torso ROI Located: {result.dress_color.pixel_count:,} valid pixels analyzed")
                else:
                    st.warning(f"👗 Torso Status: {result.dress_color.reason}")

        with col_profile:
            st.markdown("#### 👤 Conditional Profile Display")

            # Active Condition Badge
            condition_badge_color = {
                "INDIAN": "#1E3A8A",
                "US": "#065F46",
                "AFRICAN_REGION": "#9A3412",
                "OTHER": "#4C1D95",
            }.get(result.active_condition, "#374151")

            condition_label = {
                "INDIAN": "🇮🇳 INDIAN NATIONAL CONDITION",
                "US": "🇺🇸 US NATIONAL CONDITION",
                "AFRICAN_REGION": "🌍 AFRICAN REGIONAL CATEGORY CONDITION",
                "OTHER": "🌐 OTHER SOVEREIGN NATIONALITY CONDITION",
            }.get(result.active_condition, result.active_condition)

            st.markdown(
                f"""
                <div style="background-color: {condition_badge_color}; color: white; padding: 10px 14px; border-radius: 6px; font-weight: bold; margin-bottom: 16px;">
                    {condition_label}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display Data Provenance
            st.markdown(f"**Data Provenance:** `{result.nationality_info.source}`")
            if result.nationality_info.document_issuing_country:
                st.markdown(f"**Document Issuing State:** `{result.nationality_info.document_issuing_country}` *(Separate from citizenship)*")

            st.markdown("---")

            # Render function for a person profile
            def render_person_card(person: PersonProfile, is_multi: bool = False):
                if is_multi:
                    if person.person_id == result.selected_subject:
                        st.markdown(f"### 👤 {person.person_id} *(Target Subject)*")
                    else:
                        st.markdown(f"### 👤 {person.person_id}")

                # Display Condition 1: INDIAN -> Age & Dress Colour
                if result.active_condition == "INDIAN":
                    st.markdown(f"##### 📌 Displayed Fields ({person.person_id} - Indian Profile)")
                    
                    # Age Card
                    if person.age.is_available:
                        st.markdown(f"**Age:** `{person.age.bracket}` *(Source: {person.age.source})*")
                    else:
                        st.markdown(f"**Age:** `Age estimation unavailable`")
                        st.caption(f"Reason: {person.age.reason}")

                    # Dress Colour Card
                    if person.dress_color and person.dress_color.is_determined:
                        hex_c = person.dress_color.hex_code or "#888888"
                        st.markdown(
                            f"**Dress Colour:** <span style='display:inline-block;width:14px;height:14px;background-color:{hex_c};border:1px solid #000;border-radius:2px;'></span> "
                            f"`{person.dress_color.color_name}` ({person.dress_color.confidence*100:.1f}% confidence)",
                            unsafe_allow_html=True
                        )
                    else:
                        reason = person.dress_color.reason if person.dress_color else "Unable to locate torso"
                        st.markdown(f"**Dress Colour:** `Unable to determine`")
                        st.caption(f"Reason: {reason}")

                    st.info("ℹ️ Facial Emotion suppressed per Indian profile condition.")

                # Display Condition 2: US -> Age & Emotion
                elif result.active_condition == "US":
                    st.markdown(f"##### 📌 Displayed Fields ({person.person_id} - US Profile)")

                    # Age Card
                    if person.age.is_available:
                        st.markdown(f"**Age:** `{person.age.bracket}` *(Source: {person.age.source})*")
                    else:
                        st.markdown(f"**Age:** `Age estimation unavailable`")
                        st.caption(f"Reason: {person.age.reason}")

                    # Emotion Card
                    if person.emotion:
                        st.markdown(f"**Facial Emotion:** `{person.emotion.dominant_emotion}` ({person.emotion.confidence*100:.1f}% confidence)")
                        st.progress(float(person.emotion.confidence))

                    st.info("ℹ️ Dress Colour suppressed per US profile condition.")

                # Display Condition 3: AFRICAN_REGION -> Emotion & Dress Colour
                elif result.active_condition == "AFRICAN_REGION":
                    st.markdown(f"##### 📌 Displayed Fields ({person.person_id} - African Regional Profile: {result.nationality_info.country_name})")

                    # Emotion Card
                    if person.emotion:
                        st.markdown(f"**Facial Emotion:** `{person.emotion.dominant_emotion}` ({person.emotion.confidence*100:.1f}% confidence)")
                        st.progress(float(person.emotion.confidence))

                    # Dress Colour Card
                    if person.dress_color and person.dress_color.is_determined:
                        hex_c = person.dress_color.hex_code or "#888888"
                        st.markdown(
                            f"**Dress Colour:** <span style='display:inline-block;width:14px;height:14px;background-color:{hex_c};border:1px solid #000;border-radius:2px;'></span> "
                            f"`{person.dress_color.color_name}` ({person.dress_color.confidence*100:.1f}% confidence)",
                            unsafe_allow_html=True
                        )
                    else:
                        reason = person.dress_color.reason if person.dress_color else "Unable to locate torso"
                        st.markdown(f"**Dress Colour:** `Unable to determine`")
                        st.caption(f"Reason: {reason}")

                    st.info("ℹ️ Age suppressed per African regional profile condition. Africa is classified as a regional group.")

                # Display Condition 4: OTHER -> Nationality & Emotion
                else:
                    st.markdown(f"##### 📌 Displayed Fields ({person.person_id} - Other Sovereign Profile)")

                    # Nationality Card
                    st.markdown(f"**Declared Nationality:** `{result.nationality_info.country_name}` ({result.nationality_info.alpha_3})")

                    # Emotion Card
                    if person.emotion:
                        st.markdown(f"**Facial Emotion:** `{person.emotion.dominant_emotion}` ({person.emotion.confidence*100:.1f}% confidence)")
                        st.progress(float(person.emotion.confidence))

                    st.info("ℹ️ Age and Dress Colour suppressed per Other nationalities profile condition.")

            # If multiple people, display tabs
            if result.num_faces > 1:
                tab_labels = [
                    f"👤 {p.person_id} (Selected)" if p.person_id == result.selected_subject else f"👤 {p.person_id}"
                    for p in result.people
                ]
                person_tabs = st.tabs(tab_labels)
                for t_idx, tab in enumerate(person_tabs):
                    with tab:
                        render_person_card(result.people[t_idx], is_multi=True)
            else:
                render_person_card(result.people[0], is_multi=False)

        # Technical Diagnostics Expander
        with st.expander("🔍 Technical Diagnostics & Raw Feature Payloads"):
            st.json(result.to_dict())
    else:
        st.info("👈 Click **Analyse Profile & Conditions** above to run analysis and view conditional outputs.")
