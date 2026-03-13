"""
Mobile Responsive Styles for CareerLens AI
"""

def get_mobile_css():
    """Get mobile-responsive CSS styles"""
    return """
    <style>
    /* ============================================================================
       MOBILE RESPONSIVENESS - DAY 25
       ============================================================================ */
    
    /* Base responsive font sizes */
    @media (max-width: 768px) {
        html {
            font-size: 14px;
        }
        
        h1 {
            font-size: 1.75rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.25rem !important;
        }
        
        p, div, span {
            font-size: 0.95rem !important;
        }
    }
    
    /* Mobile container adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Streamlit columns stack on mobile */
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 100% !important;
        }
    }
    
    /* Mobile-friendly buttons */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
            min-height: 44px !important; /* Touch target size */
        }
        
        .stDownloadButton > button {
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            min-height: 44px !important;
        }
    }
    
    /* Mobile-friendly inputs */
    @media (max-width: 768px) {
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            font-size: 16px !important; /* Prevents zoom on iOS */
            min-height: 44px !important;
        }
        
        .stTextArea > div > div > textarea {
            min-height: 120px !important;
        }
    }
    
    /* Mobile file uploader */
    @media (max-width: 768px) {
        [data-testid="stFileUploader"] {
            width: 100% !important;
        }
        
        [data-testid="stFileUploader"] > div {
            padding: 1rem !important;
        }
        
        [data-testid="stFileUploadDropzone"] {
            min-height: 120px !important;
        }
    }
    
    /* Mobile metrics */
    @media (max-width: 768px) {
        [data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.1) !important;
            padding: 1rem !important;
            border-radius: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
    }
    
    /* Mobile expanders */
    @media (max-width: 768px) {
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 0.75rem !important;
        }
        
        .streamlit-expanderContent {
            padding: 0.75rem !important;
        }
    }
    
    /* Mobile tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
            white-space: nowrap !important;
        }
    }
    
    /* Mobile radio buttons */
    @media (max-width: 768px) {
        .stRadio > div {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        
        .stRadio > div > label {
            padding: 0.75rem !important;
            width: 100% !important;
            border: 1px solid rgba(49, 51, 63, 0.2) !important;
            border-radius: 0.5rem !important;
            margin: 0 !important;
        }
    }
    
    /* Mobile sidebar */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
            transform: translateX(-100%) !important;
            transition: transform 0.3s ease !important;
        }
        
        [data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
        }
        
        [data-testid="stSidebarNav"] {
            padding: 1rem !important;
        }
    }
    
    /* Mobile tables */
    @media (max-width: 768px) {
        table {
            display: block !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.85rem !important;
        }
        
        th, td {
            padding: 0.5rem !important;
            white-space: nowrap !important;
        }
    }
    
    /* Mobile code blocks */
    @media (max-width: 768px) {
        pre {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.8rem !important;
            padding: 0.75rem !important;
        }
        
        code {
            font-size: 0.8rem !important;
        }
    }
    
    /* Mobile alerts/info boxes */
    @media (max-width: 768px) {
        .stAlert {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
        }
        
        .stInfo, .stSuccess, .stWarning, .stError {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
    }
    
    /* Mobile progress bars */
    @media (max-width: 768px) {
        .stProgress > div > div {
            height: 8px !important;
        }
    }
    
    /* Hide unnecessary elements on very small screens */
    @media (max-width: 480px) {
        [data-testid="stToolbar"] {
            display: none !important;
        }
        
        footer {
            display: none !important;
        }
    }
    
    /* Touch-friendly spacing */
    @media (max-width: 768px) {
        .element-container {
            margin-bottom: 1rem !important;
        }
        
        hr {
            margin: 1.5rem 0 !important;
        }
    }
    
    /* Mobile match score display */
    @media (max-width: 768px) {
        .score-container {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }
        
        .score-circle {
            width: 120px !important;
            height: 120px !important;
            font-size: 2rem !important;
        }
    }
    
    /* Mobile skills badges */
    @media (max-width: 768px) {
        .skill-badge {
            font-size: 0.8rem !important;
            padding: 0.25rem 0.5rem !important;
            margin: 0.25rem !important;
        }
    }
    
    /* Landscape orientation adjustments */
    @media (max-width: 768px) and (orientation: landscape) {
        .main .block-container {
            padding-top: 0.5rem !important;
        }
        
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* Very small screens (iPhone SE, etc) */
    @media (max-width: 375px) {
        html {
            font-size: 13px;
        }
        
        .stButton > button {
            font-size: 0.9rem !important;
            padding: 0.6rem 0.8rem !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
    }
    
    /* Tablet portrait */
    @media (min-width: 768px) and (max-width: 1024px) {
        .main .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        [data-testid="column"] {
            min-width: 48% !important;
        }
    }
    
    /* Smooth scrolling for mobile */
    @media (max-width: 768px) {
        html {
            scroll-behavior: smooth;
            -webkit-overflow-scrolling: touch;
        }
        
        * {
            -webkit-tap-highlight-color: transparent;
        }
    }
    
    /* Custom highlight boxes - mobile */
    @media (max-width: 768px) {
        .highlight-box {
            padding: 1rem !important;
            margin: 1rem 0 !important;
            border-radius: 0.5rem !important;
        }
        
        .highlight-box h3 {
            font-size: 1.1rem !important;
        }
    }
    
    /* Mobile navigation improvements */
    @media (max-width: 768px) {
        [data-testid="stSidebarNav"] > ul {
            padding: 0 !important;
        }
        
        [data-testid="stSidebarNav"] a {
            padding: 1rem !important;
            font-size: 1rem !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: center !important;
        }
    }
    </style>
    """


def inject_mobile_styles():
    """Inject mobile styles into Streamlit app"""
    import streamlit as st
    st.markdown(get_mobile_css(), unsafe_allow_html=True)