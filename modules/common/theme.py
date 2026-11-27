"""Shared design system for the VISION AI Streamlit application."""

import streamlit as st


VISION_AI_CSS = """
<style>
    :root {
        --vision-page: #F5F7F9;
        --vision-surface: #FFFFFF;
        --vision-border: #E2E8F0;
        --vision-text: #172033;
        --vision-muted: #667085;
        --vision-sidebar: #111B2E;
        --vision-sidebar-surface: #17243A;
        --vision-blue: #2563EB;
        --vision-blue-hover: #1D4ED8;
        --vision-blue-soft: #EFF6FF;
        --vision-blue-border: #D7E5F8;
        --vision-mint: #EAF5F2;
        --vision-pink: #FEF1F2;
        --vision-peach: #FFF7ED;
        --vision-lavender: #F2F2FC;
        --vision-butter: #FFFAEB;
        --vision-shadow: 0 4px 16px rgba(16, 24, 40, 0.06);
    }

    html, body, [class*="css"] {
        color: var(--vision-text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--vision-page) !important;
        background-color: var(--vision-page) !important;
        color: var(--vision-text) !important;
    }

    [data-testid="stHeader"],
    header[data-testid="stHeader"] {
        background: var(--vision-surface) !important;
        border-bottom: 1px solid var(--vision-border) !important;
        box-shadow: 0 1px 4px rgba(38, 52, 69, 0.04) !important;
    }

    .main .block-container {
        max-width: 1360px !important;
        padding: 1.5rem 2rem 2.75rem !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, li {
        color: var(--vision-text);
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.02em;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: var(--vision-sidebar) !important;
        border-right: 1px solid #24334B !important;
    }

    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: var(--vision-sidebar) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #C7D2E0 !important;
    }

    .vision-brand-container {
        padding: 12px 8px 18px;
        margin-bottom: 12px;
        border-bottom: 1px solid #293852;
    }

    .vision-brand-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #F8FAFC !important;
        font-size: 1.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .vision-brand-title span.accent-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #60A5FA;
        border-radius: 9999px;
    }

    .vision-brand-subtitle,
    .sidebar-nav-header {
        color: #98A8BE !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .vision-brand-subtitle {
        margin-top: 3px;
        padding-left: 20px;
    }

    .sidebar-nav-header {
        margin: 18px 8px 10px;
        font-size: 0.68rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 6px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"],
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label,
    [data-testid="stSidebar"] [data-testid="stRadioOption"],
    [data-testid="stSidebar"] [role="radio"] {
        width: 100% !important;
        margin: 2px 0 !important;
        padding: 10px 12px !important;
        border: 1px solid transparent !important;
        border-radius: 9px !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: background-color 0.18s ease, border-color 0.18s ease !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover,
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover,
    [data-testid="stSidebar"] [role="radio"]:hover {
        background: #1B2A41 !important;
        border-color: #2A3D5A !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] [role="radio"] p {
        margin: 0 !important;
        color: #C7D2E0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [role="radio"][aria-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
        background: #1B2D48 !important;
        border: 1px solid #2D4567 !important;
        border-left: 3px solid #60A5FA !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.14) !important;
    }

    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) * {
        color: #F8FAFC !important;
        font-weight: 800 !important;
    }

    .vision-sidebar-footer {
        margin-top: 30px;
        padding: 10px 12px;
        background: var(--vision-sidebar-surface) !important;
        border: 1px solid #293852;
        border-radius: 10px;
    }

    .vision-sidebar-footer .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #98A8BE !important;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .vision-sidebar-footer .status-pill {
        color: #D7E3F3 !important;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .vision-sidebar-footer .status-pill::before {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 6px;
        border-radius: 50%;
        background: #60A5FA;
        content: "";
    }

    .overview-header-compact,
    .page-header-container,
    .custom-card,
    .vision-card,
    .task-card-container {
        background: var(--vision-surface) !important;
        border: 1px solid var(--vision-border) !important;
        border-radius: 12px !important;
        color: var(--vision-text) !important;
        box-shadow: var(--vision-shadow) !important;
    }

    .overview-header-compact {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 24px;
        padding: 22px 26px;
        margin-bottom: 22px;
    }

    .overview-heading-copy {
        min-width: 0;
    }

    .overview-eyebrow {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 7px;
        color: var(--vision-muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .overview-brand-title {
        color: #475467 !important;
        font-size: inherit;
        font-weight: inherit;
        letter-spacing: inherit;
    }

    .overview-heading-divider {
        color: #B5C0CB;
    }

    .overview-status-pill,
    .task-num-badge {
        color: #1D4ED8 !important;
    }

    .overview-status-pill {
        display: inline-flex;
        flex: 0 0 auto;
        align-items: center;
        gap: 6px;
        padding: 6px 9px;
        margin-top: 2px;
        border: 1px solid var(--vision-blue-border);
        border-radius: 7px;
        background: var(--vision-blue-soft);
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }

    .overview-status-pill .status-indicator-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--vision-blue);
    }

    .overview-header-compact .overview-headline {
        margin: 0 !important;
        color: var(--vision-text) !important;
        font-size: 1.65rem !important;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.2 !important;
    }

    .overview-subtext,
    .page-header-subtitle,
    .task-card-desc {
        color: var(--vision-muted) !important;
    }

    .overview-subtext {
        margin: 6px 0 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="column"] > div {
        display: flex !important;
        flex: 1 1 auto !important;
        flex-direction: column !important;
        height: 100% !important;
    }

    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card) {
        display: flex !important;
        height: 292px !important;
        min-height: 292px !important;
        max-height: 292px !important;
        box-sizing: border-box !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        padding: 20px !important;
        border: 1px solid var(--vision-border) !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        box-shadow: var(--vision-shadow) !important;
        transition: box-shadow 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
    }

    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card) > [data-testid="stElementContainer"]:has(.overview-task-card) {
        display: flex !important;
        flex: 1 1 auto !important;
        flex-direction: column !important;
        min-height: 0 !important;
    }

    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card):hover {
        border-color: #B9D0F3 !important;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.1) !important;
        transform: translateY(-2px);
    }

    .task-card-inner {
        display: flex !important;
        flex: 1 1 auto !important;
        flex-direction: column !important;
        min-height: 0;
        margin-bottom: 0 !important;
    }

    .task-card-badge-row {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 8px !important;
    }

    .task-num-badge {
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .task-icon-glyph {
        display: inline-flex;
        min-width: 34px;
        min-height: 34px;
        align-items: center;
        justify-content: center;
        padding: 6px;
        border-radius: 9px;
        color: #1D4ED8 !important;
        font-size: 1.35rem;
        line-height: 1;
    }

    .task-card-title {
        display: flex;
        min-height: 2.5em;
        align-items: center;
        margin: 2px 0 6px !important;
        color: var(--vision-text) !important;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.25;
    }

    .task-card-desc {
        display: -webkit-box;
        height: 3em;
        overflow: hidden;
        margin: 0;
        font-size: 0.83rem;
        line-height: 1.45;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
    }

    div.stButton > button,
    button[data-testid="stBaseButton"] {
        min-height: 40px;
        border: 1px solid #D0D5DD !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #344054 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06) !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
    }

    div.stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        border-color: var(--vision-blue) !important;
        background: var(--vision-blue) !important;
        background-color: var(--vision-blue) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card) [data-testid="stButton"] {
        width: 100% !important;
        margin-top: auto !important;
        padding-top: 10px !important;
    }

    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card) [data-testid="stButton"] button {
        width: 100% !important;
        height: 38px !important;
        padding: 8px 14px !important;
        font-size: 0.86rem !important;
    }

    div.stButton > button:hover,
    div.stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .overview-task-card) [data-testid="stButton"] button:hover {
        border-color: #B9D0F3 !important;
        background: var(--vision-blue-soft) !important;
        background-color: var(--vision-blue-soft) !important;
        color: #1D4ED8 !important;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.08) !important;
        transform: translateY(-1px);
    }

    @media (max-width: 760px) {
        .overview-header-compact {
            flex-direction: column;
            gap: 14px;
            padding: 20px;
        }

        [data-testid="stHorizontalBlock"]:has(.overview-task-card) {
            flex-direction: column !important;
            gap: 16px !important;
        }

        [data-testid="stHorizontalBlock"]:has(.overview-task-card) > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    div.stButton > button[kind="secondary"] {
        border-color: #D0D5DD !important;
        background: #FFFFFF !important;
        color: #344054 !important;
    }

    .page-header-container {
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    .page-header-container .page-header-title {
        margin: 0;
        color: var(--vision-text) !important;
        font-size: 1.65rem !important;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .page-header-subtitle {
        margin-top: 4px;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .badge-active-window,
    .badge-pipeline {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 5px;
        color: var(--vision-text) !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .badge-active-window {
        border: 1px solid #F3DEC2;
        background: var(--vision-peach);
    }

    .badge-pipeline {
        border: 1px solid #D6E2F5;
        background: var(--vision-blue-soft);
        color: #1D4ED8 !important;
    }

    .badge-locked-window {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border: 1px solid #FECACA;
        border-radius: 5px;
        background: #FEF2F2;
        color: #991B1B;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .lavender-card,
    .coral-card,
    .amber-card {
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
        border: 1px solid var(--vision-border) !important;
        border-left-width: 4px !important;
        border-radius: 12px !important;
        color: var(--vision-text) !important;
    }

    .lavender-card {
        border-color: var(--vision-blue-border) !important;
        border-left-color: var(--vision-blue) !important;
        background: var(--vision-blue-soft) !important;
    }

    .coral-card {
        border-color: #FECACA !important;
        border-left-color: #EF4444 !important;
        background: var(--vision-pink) !important;
    }

    .amber-card {
        border-color: #FDE68A !important;
        border-left-color: #D97706 !important;
        background: var(--vision-butter) !important;
    }

    .metric-card,
    div[data-testid="stMetric"] {
        border: 1px solid var(--vision-border) !important;
        border-left: 3px solid #8FB4F3 !important;
        border-radius: 11px !important;
        background: #FFFFFF !important;
        color: var(--vision-text) !important;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.05) !important;
    }

    .metric-card {
        padding: 12px 16px !important;
    }

    .metric-card .metric-label,
    .metric-card .metric-desc,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label span {
        color: var(--vision-muted) !important;
    }

    .metric-card .metric-value,
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--vision-text) !important;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        padding: 9px 16px !important;
        border-bottom: 3px solid transparent !important;
        color: var(--vision-muted) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        border-bottom-color: var(--vision-blue) !important;
        color: var(--vision-text) !important;
        font-weight: 700 !important;
    }

    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stExpander"] {
        border: 1px solid var(--vision-border) !important;
        border-radius: 11px !important;
        background: #FFFFFF !important;
    }

    section[data-testid="stFileUploadDropzone"] {
        padding: 18px !important;
        border-style: dashed !important;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #B9D0F3 !important;
        background: #F8FAFC !important;
    }

    section[data-testid="stFileUploadDropzone"] *,
    div[data-testid="stExpander"] * {
        color: var(--vision-text) !important;
    }

    div[data-baseweb="select"] > div,
    input[type="text"],
    input[type="number"],
    input[type="password"] {
        border-color: #D7DEE7 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        color: var(--vision-text) !important;
    }

    div[data-baseweb="select"] * {
        color: var(--vision-text) !important;
    }

    div[data-testid="stSlider"] div[data-baseweb="slider"] {
        color: var(--vision-blue) !important;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: var(--vision-blue) !important;
    }

    div[data-testid="stTable"] table {
        border: 1px solid var(--vision-border) !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        color: var(--vision-text) !important;
    }

    div[data-testid="stTable"] th {
        background: #F2F6FC !important;
        color: var(--vision-text) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTable"] td {
        border-bottom: 1px solid var(--vision-border) !important;
        color: var(--vision-text) !important;
    }

    div[data-testid="stAlert"] {
        color: var(--vision-text) !important;
    }

    .legend-card,
    .table-container,
    .isl-card,
    .sign-display-box,
    .metric-card-car,
    .metric-card-blue,
    .metric-card-person,
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border: 1px solid var(--vision-border) !important;
        border-radius: 12px !important;
        background: var(--vision-surface) !important;
        box-shadow: var(--vision-shadow) !important;
    }

    .legend-title,
    .metric-card .metric-label,
    .metric-card .metric-desc,
    .metric-label {
        color: var(--vision-muted) !important;
    }

    .metric-card-car .metric-value,
    .metric-card-blue .metric-value,
    .metric-card-person .metric-value {
        color: var(--vision-text) !important;
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }

    .sign-display-box {
        border-top: 3px solid var(--vision-blue) !important;
        padding: 24px !important;
    }

    .sign-large-text {
        color: var(--vision-blue) !important;
    }

    .sign-subtitle {
        color: var(--vision-text) !important;
    }

    .table-container {
        overflow-x: auto;
    }

    .species-table th {
        background: #F2F6FC !important;
        color: #344054 !important;
        border-bottom: 1px solid var(--vision-border) !important;
    }

    .species-table td {
        color: var(--vision-text) !important;
        border-bottom: 1px solid var(--vision-border) !important;
    }

    .species-table tr:hover {
        background: #F8FAFC !important;
    }

    div[data-testid="stImage"] img {
        border: 1px solid var(--vision-border);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border: 1px solid var(--vision-border) !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04) !important;
    }

    div[data-baseweb="select"]:focus-within > div,
    input[type="text"]:focus,
    input[type="number"]:focus,
    input[type="password"]:focus {
        border-color: #84A9E8 !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }
</style>
"""


def apply_theme():
    """Injects the shared VISION AI design system."""
    st.markdown(VISION_AI_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge_text: str = None, badge_type: str = "pipeline"):
    """Renders a unified, compact header for task workspaces."""
    badge_html = ""
    if badge_text:
        if badge_type == "active" or badge_text == "ACTIVE":
            cls_name = "badge-active-window"
        elif badge_type == "locked":
            cls_name = "badge-locked-window"
        else:
            cls_name = "badge-pipeline"
        badge_html = f"<div style='margin-bottom:6px;'><span class='{cls_name}'>{badge_text}</span></div>"

    html = f"""
    <div class="page-header-container">
        {badge_html}
        <h1 class="page-header-title">{title}</h1>
        <p class="page-header-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
