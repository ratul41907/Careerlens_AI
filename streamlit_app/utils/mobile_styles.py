"""
Mobile Responsive Styles for CareerLens AI - Complete Rewrite
Fixes all mobile UI issues across all pages
"""

def get_mobile_css():
    """Get comprehensive mobile-responsive CSS"""
    return """
    <style>
    /* ============================================================
       MOBILE RESPONSIVENESS - COMPLETE FIX
       ============================================================ */

    /* Viewport meta is injected via st.markdown separately */

    /* ── Base font scaling ────────────────────────────────────── */
    @media (max-width: 768px) {
        html { font-size: 14px; }
        h1   { font-size: 1.6rem !important; }
        h2   { font-size: 1.35rem !important; }
        h3   { font-size: 1.15rem !important; }
        p, div, span, label { font-size: 0.95rem !important; }
    }
    @media (max-width: 375px) {
        html { font-size: 13px; }
        h1   { font-size: 1.4rem !important; }
    }

    /* ── Main container ───────────────────────────────────────── */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.75rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }
    }

    /* ── Columns: stack vertically on mobile ──────────────────── */
    @media (max-width: 768px) {
        /* Force all Streamlit columns to stack */
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 0 0 100% !important;
        }
        /* Remove horizontal flex from column containers */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
    }

    /* ── Buttons ──────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            padding: 0.7rem 1rem !important;
            font-size: 0.95rem !important;
            min-height: 48px !important;
            margin-bottom: 0.4rem !important;
        }
    }

    /* ── Text inputs & textareas ──────────────────────────────── */
    @media (max-width: 768px) {
        .stTextInput > div > div > input {
            font-size: 16px !important; /* prevents iOS zoom */
            min-height: 48px !important;
            padding: 0.6rem 0.75rem !important;
        }
        .stTextArea > div > div > textarea {
            font-size: 16px !important;
            min-height: 130px !important;
        }
        .stNumberInput input {
            font-size: 16px !important;
            min-height: 44px !important;
        }
        .stSelectbox > div > div {
            font-size: 16px !important;
            min-height: 44px !important;
        }
    }

    /* ── File uploader ────────────────────────────────────────── */
    @media (max-width: 768px) {
        [data-testid="stFileUploader"] {
            width: 100% !important;
        }
        [data-testid="stFileUploadDropzone"] {
            min-height: 100px !important;
            padding: 1rem !important;
        }
        [data-testid="stFileUploader"] label {
            font-size: 0.9rem !important;
        }
        /* Instruction text inside uploader */
        [data-testid="stFileUploader"] small {
            font-size: 0.8rem !important;
        }
    }

    /* ── Metrics ──────────────────────────────────────────────── */
    @media (max-width: 768px) {
        [data-testid="stMetric"] {
            padding: 0.6rem 0.75rem !important;
            background: rgba(59, 130, 246, 0.08) !important;
            border-radius: 8px !important;
            margin-bottom: 0.4rem !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.75rem !important;
        }
    }

    /* ── Radio buttons ────────────────────────────────────────── */
    @media (max-width: 768px) {
        /* Horizontal radio → vertical stack */
        .stRadio > div[role="radiogroup"] {
            flex-direction: column !important;
            gap: 0.4rem !important;
        }
        .stRadio > div[role="radiogroup"] > label {
            padding: 0.6rem 0.75rem !important;
            border: 1px solid rgba(96, 165, 250, 0.25) !important;
            border-radius: 8px !important;
            width: 100% !important;
            margin: 0 !important;
            font-size: 0.9rem !important;
        }
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            gap: 0.25rem !important;
            padding-bottom: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 0.5rem 0.75rem !important;
            white-space: nowrap !important;
            min-width: fit-content !important;
        }
    }

    /* ── Expanders ────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
            padding: 0.6rem 0.75rem !important;
        }
        .streamlit-expanderContent {
            padding: 0.75rem !important;
        }
    }

    /* ── Alerts / info boxes ──────────────────────────────────── */
    @media (max-width: 768px) {
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            padding: 0.6rem 0.75rem !important;
            font-size: 0.9rem !important;
            margin: 0.4rem 0 !important;
        }
    }

    /* ── Progress bar ─────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stProgress > div > div {
            height: 6px !important;
        }
    }

    /* ── Sidebar ──────────────────────────────────────────────── */
    @media (max-width: 768px) {
        /* Sidebar overlays on mobile — ensure proper z-index */
        [data-testid="stSidebar"] {
            z-index: 999 !important;
        }
        [data-testid="stSidebarNav"] a {
            padding: 0.75rem 1rem !important;
            font-size: 0.95rem !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: center !important;
        }
        /* Sidebar collapse button larger tap target */
        [data-testid="collapsedControl"] {
            width: 2.5rem !important;
            height: 2.5rem !important;
        }
    }

    /* ── Score card (CV Matcher) ──────────────────────────────── */
    @media (max-width: 768px) {
        .score-card {
            padding: 1.5rem 1rem !important;
            margin: 1rem 0 !important;
            border-radius: 12px !important;
        }
        .score-value {
            font-size: 3.2rem !important;
        }
        .score-label {
            font-size: 1.1rem !important;
        }
    }

    /* ── Feature cards / glass cards (Home) ──────────────────── */
    @media (max-width: 768px) {
        .glass-card, .feature-card {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            border-radius: 10px !important;
        }
        .feature-card:hover {
            transform: none !important; /* disable hover lift on touch */
        }
        .glass-card:hover {
            transform: none !important;
        }
        .feature-icon {
            font-size: 1.75rem !important;
            margin-bottom: 0.4rem !important;
        }
        .feature-title {
            font-size: 1.05rem !important;
        }
        .feature-desc {
            font-size: 0.875rem !important;
            line-height: 1.5 !important;
        }
    }

    /* ── Neon stat boxes (Home) ───────────────────────────────── */
    @media (max-width: 768px) {
        .neon-stat {
            padding: 0.85rem !important;
            margin-bottom: 0.5rem !important;
        }
        .neon-stat:hover {
            transform: none !important;
        }
        .stat-value {
            font-size: 1.75rem !important;
        }
        .stat-label {
            font-size: 0.7rem !important;
        }
    }

    /* ── Gradient heading ─────────────────────────────────────── */
    @media (max-width: 768px) {
        .gradient-text {
            font-size: 2rem !important;
            margin-bottom: 0.5rem !important;
        }
        .subtitle {
            font-size: 1rem !important;
            margin-bottom: 1.5rem !important;
        }
    }

    /* ── Hero section (Home) ──────────────────────────────────── */
    @media (max-width: 768px) {
        .hero-section {
            padding: 1.5rem 0.75rem !important;
            margin: 0.75rem 0 !important;
            border-radius: 12px !important;
        }
        .hero-section h2 {
            font-size: 1.3rem !important;
            margin-bottom: 0.5rem !important;
        }
        .hero-section p {
            font-size: 0.9rem !important;
        }
    }

    /* ── Section headers ──────────────────────────────────────── */
    @media (max-width: 768px) {
        .section-header {
            font-size: 1.25rem !important;
            margin: 1.5rem 0 1rem 0 !important;
        }
    }

    /* ── Tables ───────────────────────────────────────────────── */
    @media (max-width: 768px) {
        table {
            display: block !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.8rem !important;
            width: 100% !important;
        }
        th, td {
            padding: 0.4rem 0.5rem !important;
            white-space: nowrap !important;
        }
    }

    /* ── Code blocks ──────────────────────────────────────────── */
    @media (max-width: 768px) {
        pre, code {
            font-size: 0.78rem !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        pre { padding: 0.6rem !important; }
    }

    /* ── Slider ───────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stSlider {
            padding: 0.25rem 0 !important;
        }
        .stSlider > div > div > div {
            height: 6px !important;
        }
    }

    /* ── Plotly charts ────────────────────────────────────────── */
    @media (max-width: 768px) {
        .js-plotly-plot {
            min-height: 260px !important;
        }
        .js-plotly-plot .plotly {
            width: 100% !important;
        }
    }

    /* ── Download buttons row ─────────────────────────────────── */
    @media (max-width: 768px) {
        /* When 3 download buttons sit in columns, stack them */
        [data-testid="stHorizontalBlock"]:has(.stDownloadButton) {
            flex-direction: column !important;
        }
    }

    /* ── Skill cards (Matcher results) ───────────────────────── */
    @media (max-width: 768px) {
        .skill-card {
            padding: 0.6rem 0.75rem !important;
            margin-bottom: 0.4rem !important;
            font-size: 0.875rem !important;
        }
    }

    /* ── Interview question cards ─────────────────────────────── */
    @media (max-width: 768px) {
        .question-card {
            padding: 0.75rem 0.875rem !important;
            margin-bottom: 0.6rem !important;
        }
        .question-card strong {
            font-size: 0.9rem !important;
        }
    }

    /* ── STAR evaluation score box ────────────────────────────── */
    @media (max-width: 768px) {
        .score-display {
            padding: 1rem !important;
        }
        .score-value {
            font-size: 2.2rem !important;
        }
    }

    /* ── Highlight / stat info boxes ─────────────────────────── */
    @media (max-width: 768px) {
        .stat-box, .highlight-box {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
            font-size: 0.875rem !important;
        }
        .stat-box h4 {
            font-size: 0.95rem !important;
        }
    }

    /* ── Spacing cleanup ──────────────────────────────────────── */
    @media (max-width: 768px) {
        .element-container {
            margin-bottom: 0.6rem !important;
        }
        hr {
            margin: 1rem 0 !important;
        }
        br { display: none; } /* remove excessive <br> spacers */
    }

    /* ── Touch UX ─────────────────────────────────────────────── */
    @media (max-width: 768px) {
        * {
            -webkit-tap-highlight-color: transparent;
            -webkit-touch-callout: none;
        }
        html {
            scroll-behavior: smooth;
            -webkit-overflow-scrolling: touch;
        }
        /* Ensure all interactive elements meet 44px min touch target */
        button, a, input[type="checkbox"], input[type="radio"] {
            min-height: 44px !important;
        }
    }

    /* ── Hide non-essential on very small screens ─────────────── */
    @media (max-width: 400px) {
        [data-testid="stToolbar"] { display: none !important; }
        footer { display: none !important; }
    }

    /* ── Landscape mobile ─────────────────────────────────────── */
    @media (max-width: 768px) and (orientation: landscape) {
        .main .block-container {
            padding-top: 0.4rem !important;
        }
        h1 { font-size: 1.3rem !important; }
        .hero-section { padding: 1rem !important; }
    }

    /* ── Tablet (768px–1024px) ────────────────────────────────── */
    @media (min-width: 768px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 95% !important;
        }
    }
    </style>
    """


def inject_mobile_styles():
    """Inject mobile styles + viewport meta into Streamlit app"""
    import streamlit as st

    # Viewport meta must come first to prevent iOS from zooming
    st.markdown(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, '
        'maximum-scale=5.0, user-scalable=yes">',
        unsafe_allow_html=True
    )
    st.markdown(get_mobile_css(), unsafe_allow_html=True)