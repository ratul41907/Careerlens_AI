"""
CV-JD Matcher Page - Streamlit Frontend
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.explainability import ExplainabilityEngine

# Page config
st.set_page_config(
    page_title="CV-JD Matcher - CareerLens AI",
    page_icon="📊",
    layout="wide"
)

# Apply dark theme CSS
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, 
    .main p, .main span, .main div {
        color: #e2e8f0 !important;
    }
    
    /* Score display */
    .score-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border: 2px solid rgba(96, 165, 250, 0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.3);
    }
    
    .score-value {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .score-label {
        font-size: 1.5rem;
        color: #cbd5e1;
        margin-top: 1rem;
    }
    
    /* Skill cards */
    .skill-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .skill-card:hover {
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
    }
    
    .skill-matched {
        border-left: 4px solid #10b981;
    }
    
    .skill-missing {
        border-left: 4px solid #ef4444;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white !important;
        border: none;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed rgba(96, 165, 250, 0.3);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'match_result' not in st.session_state:
    st.session_state.match_result = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# Cache the engines
@st.cache_resource
def load_engines():
    """Load AI engines (cached for performance)"""
    with st.spinner("🔧 Loading AI engines..."):
        embedding_engine = EmbeddingEngine()
        scoring_engine = ScoringEngine(embedding_engine)
        explainability_engine = ExplainabilityEngine(embedding_engine)
    return embedding_engine, scoring_engine, explainability_engine

# Title
st.markdown("""
<h1 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
           -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
    📊 CV-JD Matcher
</h1>
<p style="color: #94a3b8; font-size: 1.25rem; margin-bottom: 2rem;">
    AI-powered semantic matching with evidence-based scoring
</p>
""", unsafe_allow_html=True)

# Load engines
try:
    embedding_engine, scoring_engine, explainability_engine = load_engines()
    st.success("✅ AI engines loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load AI engines: {str(e)}")
    st.stop()

# Two columns: CV upload and JD input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Your CV")
    
    # CV upload
    uploaded_file = st.file_uploader(
        "Upload CV (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload your resume in PDF, DOCX, or TXT format"
    )
    
    # OR paste CV text
    st.markdown("**OR paste CV text:**")
    cv_text = st.text_area(
        "Paste your CV here",
        height=300,
        placeholder="Paste your CV content here...\n\nInclude:\n• Work experience\n• Skills\n• Education\n• Projects",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 💼 Job Description")
    
    jd_text = st.text_area(
        "Paste job description",
        height=460,
        placeholder="""Paste the full job description here...

Example:
Senior Software Engineer

Required Skills:
- 5+ years Python experience
- FastAPI framework
- Docker & Kubernetes
- AWS cloud services
- PostgreSQL database

Preferred Skills:
- React/TypeScript
- CI/CD pipelines
- Microservices architecture

Experience: 5+ years""",
        label_visibility="collapsed"
    )

# Match button
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    analyze_button = st.button("🎯 Analyze Match", type="primary", use_container_width=True)

if analyze_button:
    # Validate inputs
    if not jd_text:
        st.error("❌ Please provide a job description")
        st.stop()
    
    if not uploaded_file and not cv_text:
        st.error("❌ Please upload a CV or paste CV text")
        st.stop()
    
    # Parse CV
    with st.spinner("📄 Parsing CV..."):
        cv_parser = CVParser()
        
        try:
            if uploaded_file:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                cv_data = cv_parser.parse(tmp_path)
            else:
                cv_data = {
                    'text': cv_text,
                    'sections': cv_parser._segment_sections(cv_text)
                }
            
            st.success("✅ CV parsed successfully!")
        except Exception as e:
            st.error(f"❌ Failed to parse CV: {str(e)}")
            st.stop()
    
    # Parse JD
    with st.spinner("💼 Parsing job description..."):
        try:
            jd_parser = JDParser()
            jd_data = jd_parser.parse(jd_text)
            st.success(f"✅ Found {len(jd_data['required_skills'])} required skills, {len(jd_data.get('preferred_skills', []))} preferred skills")
        except Exception as e:
            st.error(f"❌ Failed to parse JD: {str(e)}")
            st.stop()
    
    # Compute match
    with st.spinner("🧠 Computing match score (this may take 10-15 seconds)..."):
        try:
            match_result = scoring_engine.compute_match_score(cv_data, jd_data)
            explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
            
            # Store in session state
            st.session_state.match_result = match_result
            st.session_state.explanation = explanation
            
            st.success("✅ Match analysis complete!")
        except Exception as e:
            st.error(f"❌ Failed to compute match: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.match_result:
    match_result = st.session_state.match_result
    explanation = st.session_state.explanation
    
    st.markdown("---")
    st.markdown("## 🎯 Match Results")
    
    # Overall score
    score = match_result['overall_score']
    percentage = match_result['overall_percentage']
    
    # Color based on score
    if score >= 0.75:
        emoji = "🟢"
        level_color = "#10b981"
    elif score >= 0.60:
        emoji = "🔵"
        level_color = "#3b82f6"
    elif score >= 0.45:
        emoji = "🟡"
        level_color = "#f59e0b"
    else:
        emoji = "🔴"
        level_color = "#ef4444"
    
    # Score display
    st.markdown(f"""
    <div class="score-card">
        <div class="score-value">{emoji} {percentage}</div>
        <div class="score-label" style="color: {level_color};">{match_result['interpretation']['level']}</div>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 1rem; max-width: 600px; margin-left: auto; margin-right: auto;">
            {match_result['interpretation']['recommendation']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Breakdown
    st.markdown("### 📊 Score Breakdown")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        req_score = match_result['breakdown']['required_skills']['percentage']
        req_matched = match_result['breakdown']['required_skills']['details']['match_rate']
        st.metric(
            "Required Skills",
            req_score,
            req_matched,
            help="Skills that MUST be on your CV"
        )
    
    with breakdown_col2:
        pref_score = match_result['breakdown']['preferred_skills']['percentage']
        st.metric(
            "Preferred Skills",
            pref_score,
            help="Nice-to-have skills (bonus points)"
        )
    
    with breakdown_col3:
        exp_score = match_result['breakdown']['experience']['percentage']
        st.metric(
            "Experience",
            exp_score,
            help="Years of experience match"
        )
    
    # Detailed skills analysis
    st.markdown("### 🎯 Skills Analysis")
    
    required_skills = match_result['breakdown']['required_skills']['details']['skills']
    
    # Separate matched and missing
    matched_skills = [s for s in required_skills if s['matched']]
    missing_skills = [s for s in required_skills if not s['matched']]
    
    skill_col1, skill_col2 = st.columns(2)
    
    with skill_col1:
        st.markdown(f"#### ✅ Matched Skills ({len(matched_skills)})")
        
        for skill_info in matched_skills[:5]:
            skill = skill_info['skill']
            score = skill_info['score']
            percentage = skill_info['percentage']
            strength = skill_info['strength']
            
            st.markdown(f"""
            <div class="skill-card skill-matched">
                <strong style="color: #10b981; font-size: 1.1rem;">{skill}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">
                    {percentage} • {strength}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)
    
    with skill_col2:
        st.markdown(f"#### ❌ Missing Skills ({len(missing_skills)})")
        
        if missing_skills:
            for skill_info in missing_skills[:5]:
                skill = skill_info['skill']
                score = skill_info['score']
                percentage = skill_info['percentage']
                
                st.markdown(f"""
                <div class="skill-card skill-missing">
                    <strong style="color: #ef4444; font-size: 1.1rem;">{skill}</strong>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">
                        {percentage} • Not found in CV
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(score)
        else:
            st.success("🎉 All required skills matched!")
    
    # Evidence section
    if 'evidence' in explanation and explanation['evidence']:
        st.markdown("### 📝 Evidence")
        
        with st.expander("🔍 See detailed evidence for each skill", expanded=False):
            for skill, evidence_list in list(explanation['evidence'].items())[:5]:
                st.markdown(f"**{skill}:**")
                for ev in evidence_list[:2]:
                    st.info(f"💡 \"{ev['sentence']}\" (Confidence: {ev['score']:.0%})")
    
    # Recommendations
    if 'recommendations' in explanation and explanation['recommendations']:
        st.markdown("### 💡 Recommendations")
        
        for i, rec in enumerate(explanation['recommendations'][:3], 1):
            st.warning(f"{i}. {rec}")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        report = {
            'match_result': match_result,
            'explanation': explanation
        }
        st.download_button(
            "📄 Download JSON Report",
            data=json.dumps(report, indent=2, default=str),
            file_name="match_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col2:
        if st.button("🎯 See Learning Path", use_container_width=True):
            st.info("💡 Learning pathway feature coming soon!")
    
    with export_col3:
        if st.button("📝 Improve CV", use_container_width=True):
            st.switch_page("pages/2_📝_CV_Generator.py")

# Help section
with st.expander("ℹ️ How does matching work?"):
    st.markdown("""
    ### AI-Powered Semantic Matching
    
    **Our system uses:**
    
    1. **Semantic Embeddings** 🧠
       - Converts skills to 384-dimensional vectors
       - Captures meaning, not just keywords
       - Example: "Python" ≈ "Python programming" ≈ "Python development"
    
    2. **Cosine Similarity** 📐
       - Measures angle between vectors (0-1 scale)
       - Threshold: 0.70 for "Strong Match"
       - Accounts for synonyms and variations
    
    3. **Weighted Scoring** ⚖️
       - Required skills: 60% weight
       - Preferred skills: 25% weight
       - Experience: 15% weight
    
    4. **Evidence Citation** 📝
       - Links each skill to specific CV sentences
       - Shows confidence scores
       - Helps you verify the match
    
    ### Interpreting Scores:
    
    - 🟢 **75%+** = Excellent Match (Apply immediately!)
    - 🔵 **60-75%** = Good Match (Strong candidate)
    - 🟡 **45-60%** = Moderate Match (Consider upskilling)
    - 🔴 **<45%** = Weak Match (Significant skill gaps)
    
    ### Tips for Better Matches:
    
    - ✅ Use exact skill names from job description
    - ✅ Include synonyms (e.g., "JS" and "JavaScript")
    - ✅ Quantify your experience (years, projects)
    - ✅ Add context to skills (not just lists)
    """)