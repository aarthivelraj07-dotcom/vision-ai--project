"""
VISION AI — Professional Computer Vision Suite.
Main Application Entry Point and Enterprise Task Router.

Deploying:
- Overview: Central Computer Vision Workspace
- Task 1: Attendance System
- Task 2: Animal Detection
- Task 3: Drowsiness Detection
- Task 4: Nationality & Conditional Analysis
- Task 5: Sign Language Recognition
- Task 6: Car Colour Detection
"""

import sys
import os
import streamlit as st

# Add root directory to python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Expose run_app for Task 1 backward compatibility with automated tests (Lazy loaded)
def run_app():
    """Lazy proxy to Task 1 attendance entry point to prevent eager startup model initialization."""
    from modules.attendance.ui import run_app as _run_attendance_app
    return _run_attendance_app()

from modules.common.theme import apply_theme

# Canonical Task Definitions, Navigation Metadata, and Direct Task Numbers
TASK_REGISTRY = {
    "overview": {
        "label": "Overview – Dashboard",
        "nav_title": "Overview",
        "icon": "⊞",
        "task_num": "Overview",
        "aliases": ("0", "overview", "dashboard", "home"),
        "accent": "#EFF6FF",
        "description": "Central control workspace to monitor and launch computer vision applications.",
    },
    "task1": {
        "label": "Task 1 – Attendance System",
        "nav_title": "Attendance System",
        "icon": "👤",
        "task_num": "Task 1",
        "aliases": ("1", "task1", "attendance", "attendance_system"),
        "accent": "#EFF6FF",
        "description": "Face-recognition attendance with emotion analysis and CSV/Excel export.",
    },
    "task2": {
        "label": "Task 2 – Animal Detection",
        "nav_title": "Animal Detection",
        "icon": "🐾",
        "task_num": "Task 2",
        "aliases": ("2", "task2", "animal", "animal_detection"),
        "accent": "#EFF6FF",
        "description": "Detect animals in images and video, with highlighted carnivores.",
    },
    "task3": {
        "label": "Task 3 – Drowsiness Detection",
        "nav_title": "Drowsiness Detection",
        "icon": "👁️",
        "task_num": "Task 3",
        "aliases": ("3", "task3", "drowsiness", "drowsiness_detection"),
        "accent": "#EFF6FF",
        "description": "Monitor driver fatigue from eye closure and trigger alerts.",
    },
    "task4": {
        "label": "Task 4 – Nationality Detection",
        "nav_title": "Nationality & Conditional Analysis",
        "icon": "🌐",
        "task_num": "Task 4",
        "aliases": ("4", "task4", "nationality", "nationality_detection", "conditional"),
        "accent": "#EFF6FF",
        "description": "Conditional demographic, dress-colour, and facial emotion analysis.",
    },
    "task5": {
        "label": "Task 5 – Sign Language Detection",
        "nav_title": "Sign Language Recognition",
        "icon": "✋",
        "task_num": "Task 5",
        "aliases": ("5", "task5", "sign_language", "isl", "sign_language_recognition"),
        "accent": "#EFF6FF",
        "description": "Recognize Indian Sign Language gestures from images or webcam.",
    },
    "task6": {
        "label": "Task 6 – Car Colour Detection",
        "nav_title": "Car Colour Detection",
        "icon": "🚗",
        "task_num": "Task 6",
        "aliases": ("6", "task6", "car_color", "car_colour", "traffic", "car_color_detection"),
        "accent": "#EFF6FF",
        "description": "Detect vehicles, classify blue cars, and count pedestrians.",
    },
}


def handle_card_nav(target_task: str):
    """
    Safe navigation callback executed before the script rerun cycle.
    Queues target navigation without directly modifying active widget session state.
    """
    st.session_state["pending_nav_task"] = target_task
    st.query_params["task"] = target_task


def render_sidebar_navigation(active_task_key: str) -> str:
    """
    Renders the clean, compact enterprise sidebar.
    Returns the resolved active_task_key directly, synchronizing URL query parameters
    without triggering a redundant, expensive full-script rerun.
    """
    with st.sidebar:
        # 1. Brand Identity Header
        st.markdown(
            """
            <div class="vision-brand-container">
                <div class="vision-brand-title">
                    <span class="accent-dot"></span>
                    VISION AI
                </div>
                <div class="vision-brand-subtitle">Computer Vision Suite</div>
            </div>
            <div class="sidebar-nav-header">Workspace Modules</div>
            """,
            unsafe_allow_html=True
        )

        # 2. Navigation items
        nav_keys = list(TASK_REGISTRY.keys())
        nav_display_labels = [
            f"{info['icon']}  {info['nav_title']}"
            for info in TASK_REGISTRY.values()
        ]
        label_to_key = {label: key for key, label in zip(nav_keys, nav_display_labels)}

        def on_sidebar_nav_change():
            selected = st.session_state.get("vision_ai_nav_radio")
            if selected in label_to_key:
                new_key = label_to_key[selected]
                st.query_params["task"] = new_key
                st.session_state["_last_active_key"] = new_key

        current_label = f"{TASK_REGISTRY[active_task_key]['icon']}  {TASK_REGISTRY[active_task_key]['nav_title']}"
        current_idx = nav_display_labels.index(current_label) if current_label in nav_display_labels else 0

        radio_index = None if "vision_ai_nav_radio" in st.session_state else current_idx

        selected_label = st.radio(
            "Navigation Menu",
            nav_display_labels,
            index=radio_index,
            key="vision_ai_nav_radio",
            label_visibility="collapsed",
            on_change=on_sidebar_nav_change
        )

        selected_key = label_to_key.get(selected_label, active_task_key)
        if selected_key != active_task_key:
            st.query_params["task"] = selected_key
            active_task_key = selected_key

        # 3. Compact Minimal Status Footer
        st.markdown(
            """
            <div class="vision-sidebar-footer">
                <div class="status-row">
                    <span>SYSTEM STATUS</span>
                    <span class="status-pill">ONLINE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    return active_task_key


def render_overview_dashboard():
    """
    Renders the professional landing dashboard: VISION AI Workspace.
    Features a compact header and 6 unified task cards as the primary focus.
    """
    # 1. Clear workspace heading with a compact system status indicator.
    st.markdown(
        """
        <div class="overview-header-compact">
            <div class="overview-heading-copy">
                <div class="overview-eyebrow">
                    <span class="overview-brand-title">VISION AI</span>
                    <span class="overview-heading-divider">/</span>
                    <span>WORKSPACE</span>
                </div>
                <h1 class="overview-headline">Computer Vision Suite</h1>
                <p class="overview-subtext">
                    A unified workspace for visual detection, recognition, and monitoring.
                </p>
            </div>
            <div class="overview-status-pill">
                <span class="status-indicator-dot"></span> SYSTEM READY
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Six consistent task cards in two desktop rows of three.
    task_keys = [key for key in TASK_REGISTRY if key != "overview"]
    for row_start in range(0, len(task_keys), 3):
        columns = st.columns(3, gap="large")
        for key, col in zip(task_keys[row_start:row_start + 3], columns):
            info = TASK_REGISTRY[key]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="task-card-inner overview-task-card">
                            <div class="task-card-badge-row">
                                <span class="task-num-badge" style="color: #1D4ED8; background: {info['accent']}; border: 1px solid {info['accent']};">
                                    {info['task_num']}
                                </span>
                                <span class="task-icon-glyph" style="background: {info['accent']};">{info['icon']}</span>
                            </div>
                            <div class="task-card-title">{info['nav_title']}</div>
                            <div class="task-card-desc">{info['description']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.button(
                        "Open Task →",
                        key=f"btn_open_{key}",
                        on_click=handle_card_nav,
                        args=(key,),
                        type="primary",
                        use_container_width=True
                    )


def main():
    """
    Main router for VISION AI.
    Renders unified page configuration, theme CSS, clean sidebar navigation, and executes the active module.
    """
    try:
        st.set_page_config(
            page_title="VISION AI — Computer Vision Suite",
            page_icon="👁️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    # Apply Centralized Enterprise Design System
    apply_theme()

    # 1. Resolve pending navigation requested via card click callback
    if "pending_nav_task" in st.session_state:
        pending_task = st.session_state.pop("pending_nav_task")
        active_key = pending_task
        st.query_params["task"] = active_key
        # Pre-populate the radio widget key BEFORE it is instantiated in the sidebar
        target_label = f"{TASK_REGISTRY[active_key]['icon']}  {TASK_REGISTRY[active_key]['nav_title']}"
        st.session_state["vision_ai_nav_radio"] = target_label
    else:
        # 2. Synchronize active_key from query parameters
        selected_task_raw = st.query_params.get("task", os.environ.get("ACTIVE_TASK", "overview")).lower()
        active_key = "overview"
        for key, info in TASK_REGISTRY.items():
            if selected_task_raw in info["aliases"]:
                active_key = key
                break

        # Ensure radio widget's state matches query param before radio is instantiated
        expected_label = f"{TASK_REGISTRY[active_key]['icon']}  {TASK_REGISTRY[active_key]['nav_title']}"
        current_radio_val = st.session_state.get("vision_ai_nav_radio")
        if current_radio_val is None or st.session_state.get("_last_active_key") != active_key:
            st.session_state["vision_ai_nav_radio"] = expected_label

    st.session_state["_last_active_key"] = active_key

    # 3. Render the clean enterprise sidebar and synchronize selection in a single execution pass
    active_key = render_sidebar_navigation(active_key)

    # 4. Route to target workspace module
    if active_key == "task1":
        from modules.attendance.ui import run_app
        run_app()
    elif active_key == "task2":
        from modules.animal_detection.ui import run_animal_app
        run_animal_app()
    elif active_key == "task3":
        from modules.drowsiness.ui import run_drowsiness_app
        run_drowsiness_app()
    elif active_key == "task4":
        from modules.nationality_detection.ui import run_nationality_app
        run_nationality_app()
    elif active_key == "task5":
        from modules.sign_language.ui import run_sign_language_app
        run_sign_language_app()
    elif active_key == "task6":
        from modules.car_color.ui import run_car_color_app
        run_car_color_app()
    else:
        render_overview_dashboard()


if __name__ == "__main__":
    main()
