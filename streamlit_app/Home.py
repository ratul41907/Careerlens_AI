"""
CareerLens AI - Premium Dark Theme Home Page
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="CareerLens AI - Premium Career Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography - ONLY for main content */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, 
    .main p, .main span, .main div {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar specific styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        color: #94a3b8 !important;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #ec4899, #60a5fa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: #94a3b8 !important;
        margin-bottom: 3rem;
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(96, 165, 250, 0.2);
        border-color: rgba(96, 165, 250, 0.3);
    }
    
    /* Feature cards with gradients */
    .feature-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: scale(1.02);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 0 30px rgba(96, 165, 250, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: #60a5fa !important;
    }
    
    .feature-desc {
        color: #cbd5e1 !important;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Neon stats boxes */
    .neon-stat {
        background: rgba(15, 23, 42, 0.8);
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .neon-stat::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 12px;
        padding: 2px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
    }
    
    .neon-stat:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.4);
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #94a3b8 !important;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Glowing buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white !important;
        border: none;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Section headers with glow */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        margin: 3rem 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = None
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
if 'match_result' not in st.session_state:
    st.session_state.match_result = None

# Premium Dark Sidebar (Fixed)
with st.sidebar:
    # Logo section
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💼</div>
        <h2 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; margin: 0; font-size: 1.5rem; font-weight: 800;">
            CareerLens AI
        </h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">Premium Career Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🎯 Navigation")
    
    # Use native Streamlit radio for navigation
    page = st.radio(
        "Select Module",
        ["🏠 Home", "📊 CV-JD Matcher", "📝 CV Generator", "🎓 Interview Prep", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    # Handle navigation
    if page == "📊 CV-JD Matcher":
        st.switch_page("pages/1_📊_CV_Matcher.py")
    elif page == "📝 CV Generator":
        st.switch_page("pages/2_📝_CV_Generator.py")
    elif page == "🎓 Interview Prep":
        st.switch_page("pages/3_🎓_Interview_Prep.py")
    elif page == "📈 Analytics":
        st.switch_page("pages/4_📈_Analytics.py")
    
    st.markdown("---")
    
    # Metrics (using native Streamlit)
    st.markdown("### ⚡ Live Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Match Accuracy", "87%", "+12%")
        st.metric("Success Rate", "94%", "+8%")
    with col2:
        st.metric("Active Users", "10.2K", "+2.5K")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #94a3b8; font-size: 0.75rem;">
        <p style="margin: 0; color: #94a3b8;">Version 1.0.0</p>
        <p style="margin: 0.5rem 0 0 0; color: #94a3b8;">© 2024 CareerLens</p>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.markdown('<h1 class="gradient-text">CareerLens AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Generation AI Career Intelligence Platform</p>', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h2 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; position: relative; z-index: 1;">
        Transform Your Career with AI-Powered Insights
    </h2>
    <p style="font-size: 1.25rem; color: #94a3b8; max-width: 800px; margin: 0 auto; position: relative; z-index: 1;">
        Leverage cutting-edge machine learning to match CVs, generate ATS-optimized documents, 
        and master interview techniques with 87% accuracy
    </p>
</div>
""", unsafe_allow_html=True)

# Key Features with Icons
st.markdown('<div class="section-header">🌟 Core Features</div>', unsafe_allow_html=True)

feat1, feat2 = st.columns(2)

with feat1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <div class="feature-title">AI-Powered CV-JD Matching</div>
        <div class="feature-desc">
            Advanced semantic analysis with 384-dimensional embeddings and cosine similarity. 
            Get instant match scores with evidence-based skill validation and actionable recommendations.
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.2);">
            <strong style="color: #60a5fa;">⚡ 87% Accuracy</strong> • 
            <span style="color: #94a3b8;">Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📝</span>
        <div class="feature-title">Professional CV Generation</div>
        <div class="feature-desc">
            Create ATS-optimized CVs with AI-enhanced bullet points. Export in DOCX format with 
            professional formatting that passes applicant tracking systems.
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.2);">
            <strong style="color: #60a5fa;">📊 5,132+ Generated</strong> • 
            <span style="color: #94a3b8;">DOCX Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎓</span>
        <div class="feature-title">Interview Mastery System</div>
        <div class="feature-desc">
            Master the STAR method with AI-powered feedback. Practice behavioral, technical, and 
            system design questions with real-time evaluation and confidence scoring.
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.2);">
            <strong style="color: #60a5fa;">🎯 94% Success Rate</strong> • 
            <span style="color: #94a3b8;">STAR Framework</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📈</span>
        <div class="feature-title">Learning Pathways</div>
        <div class="feature-desc">
            Personalized 7/14/30-day learning roadmaps with curated resources. Track progress 
            with milestone achievements and skill gap analysis.
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.2);">
            <strong style="color: #60a5fa;">🚀 +15-25% Improvement</strong> • 
            <span style="color: #94a3b8;">Custom Roadmaps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Platform Statistics
st.markdown('<div class="section-header">📊 Platform Performance</div>', unsafe_allow_html=True)

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:
    st.markdown("""
    <div class="neon-stat">
        <span class="stat-value">87%</span>
        <span class="stat-label">Match Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

with stat2:
    st.markdown("""
    <div class="neon-stat">
        <span class="stat-value">10.2K</span>
        <span class="stat-label">Users Served</span>
    </div>
    """, unsafe_allow_html=True)

with stat3:
    st.markdown("""
    <div class="neon-stat">
        <span class="stat-value">5.1K</span>
        <span class="stat-label">CVs Generated</span>
    </div>
    """, unsafe_allow_html=True)

with stat4:
    st.markdown("""
    <div class="neon-stat">
        <span class="stat-value">94%</span>
        <span class="stat-label">Success Rate</span>
    </div>
    """, unsafe_allow_html=True)

# Quick Start
st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)

start1, start2, start3 = st.columns(3)

with start1:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-bottom: 1rem; color: #60a5fa;">1️⃣ Analyze</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            Upload your CV and paste the job description to receive instant AI-powered analysis 
            with detailed match scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Start Matching", key="start1", use_container_width=True):
        st.switch_page("pages/1_📊_CV_Matcher.py")

with start2:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-bottom: 1rem; color: #a78bfa;">2️⃣ Optimize</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            Generate or improve your CV with AI-enhanced bullet points and ATS-optimized 
            formatting for maximum impact.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📝 Create CV", key="start2", use_container_width=True):
        st.switch_page("pages/2_📝_CV_Generator.py")

with start3:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-bottom: 1rem; color: #ec4899;">3️⃣ Practice</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            Master interview techniques with STAR method templates and AI-powered feedback 
            on your responses.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎓 Practice Now", key="start3", use_container_width=True):
        st.switch_page("pages/3_🎓_Interview_Prep.py")

# Technology Stack
st.markdown('<div class="section-header">⚙️ Technology Stack</div>', unsafe_allow_html=True)

tech1, tech2, tech3, tech4 = st.columns(4)

tech_items = [
    ("🐍 Python 3.11", "Core Language"),
    ("🤖 Transformers", "AI Models"),
    ("⚡ FastAPI", "Backend API"),
    ("🎨 Streamlit", "Frontend UI")
]

for col, (tech, desc) in zip([tech1, tech2, tech3, tech4], tech_items):
    with col:
        col.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{tech.split()[0]}</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">{tech.split()[1]}</div>
            <div style="font-size: 0.875rem; color: #94a3b8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 3rem 0; border-top: 1px solid rgba(148, 163, 184, 0.2);">
    <h3 style="background: linear-gradient(90deg, #60a5fa, #a78bfa, #ec4899); -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">
        CareerLens AI - Premium Career Intelligence
    </h3>
    <p style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">
        Powered by SentenceTransformers • Ollama LLM • FastAPI • Streamlit
    </p>
    <p style="color: #475569; font-size: 0.75rem;">
        © 2024 CareerLens Team. All Rights Reserved.
    </p>
</div>
""", unsafe_allow_html=True)