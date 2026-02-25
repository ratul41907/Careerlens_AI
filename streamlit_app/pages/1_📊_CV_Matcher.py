"""
CV-JD Matcher Page
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.explainability import ExplainabilityEngine

# Page config
st.set_page_config(page_title="CV-JD Matcher", page_icon="📊", layout="wide")

# Title
st.title("📊 CV-JD Matcher")
st.markdown("Upload your CV and paste a job description to get instant match analysis")

# Initialize engines (cached)
@st.cache_resource
def load_engines():
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    explainability_engine = ExplainabilityEngine(embedding_engine)
    return embedding_engine, scoring_engine, explainability_engine

with st.spinner("Loading AI engines..."):
    embedding_engine, scoring_engine, explainability_engine = load_engines()

st.success("✅ AI engines loaded!")

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
        placeholder="Paste your CV content here..."
    )

with col2:
    st.markdown("### 💼 Job Description")
    
    jd_text = st.text_area(
        "Paste job description",
        height=400,
        placeholder="""Paste the full job description here, including:
        
- Job title
- Required skills
- Preferred skills
- Experience requirements
- Education requirements

Example:
Senior Software Engineer

Required:
- 5+ years Python experience
- FastAPI framework
- Docker & Kubernetes
- AWS cloud services"""
    )

# Match button
st.markdown("---")

if st.button("🎯 Analyze Match", type="primary", use_container_width=True):
    
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
        
        if uploaded_file:
            # Save uploaded file temporarily
            import tempfile
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
    
    # Parse JD
    with st.spinner("💼 Parsing job description..."):
        jd_parser = JDParser()
        jd_data = jd_parser.parse(jd_text)
        st.success(f"✅ Found {len(jd_data['required_skills'])} required skills")
    
    # Compute match
    with st.spinner("🧠 Computing match score..."):
        match_result = scoring_engine.compute_match_score(cv_data, jd_data)
        explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
        
        # Store in session state
        st.session_state.match_result = match_result
        st.session_state.explanation = explanation
    
    # Display results
    st.markdown("---")
    st.markdown("## 🎯 Match Results")
    
    # Overall score
    score = match_result['overall_score']
    percentage = match_result['overall_percentage']
    
    # Color based on score
    if score >= 0.75:
        color = "green"
        emoji = "🟢"
    elif score >= 0.60:
        color = "blue"
        emoji = "🔵"
    elif score >= 0.45:
        color = "orange"
        emoji = "🟡"
    else:
        color = "red"
        emoji = "🔴"
    
    # Score display
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h1 style="font-size: 4rem; margin: 0;">{emoji} {percentage}</h1>
        <h3>{match_result['interpretation']['level']}</h3>
        <p style="font-size: 1.2rem; margin-top: 1rem;">{match_result['interpretation']['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Breakdown
    st.markdown("### 📊 Score Breakdown")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        req_score = match_result['breakdown']['required_skills']['percentage']
        st.metric(
            "Required Skills",
            req_score,
            f"{match_result['breakdown']['required_skills']['details']['match_rate']} matched"
        )
    
    with breakdown_col2:
        pref_score = match_result['breakdown']['preferred_skills']['percentage']
        st.metric("Preferred Skills", pref_score)
    
    with breakdown_col3:
        exp_score = match_result['breakdown']['experience']['percentage']
        st.metric("Experience", exp_score)
    
    # Detailed skills
    st.markdown("### 🎯 Skills Analysis")
    
    for skill_info in match_result['breakdown']['required_skills']['details']['skills'][:5]:
        skill = skill_info['skill']
        score = skill_info['score']
        strength = skill_info['strength']
        matched = skill_info['matched']
        
        # Color based on strength
        if matched:
            emoji = "✅"
            color = "green"
        else:
            emoji = "❌"
            color = "red"
        
        st.markdown(f"{emoji} **{skill}**: {skill_info['percentage']} - {strength}")
        st.progress(score)
    
    # Recommendations
    if 'recommendations' in explanation:
        st.markdown("### 💡 Recommendations")
        for rec in explanation['recommendations']:
            st.info(rec)
    
    # Export option
    st.markdown("---")
    if st.button("📥 Download Full Report (JSON)"):
        import json
        report = {
            'match_result': match_result,
            'explanation': explanation
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name="match_report.json",
            mime="application/json"
        )

# Help section
with st.expander("ℹ️ How does matching work?"):
    st.markdown("""
    **Our AI-powered matching uses:**
    
    1. **Semantic Embeddings** - Converts skills to 384-dimensional vectors
    2. **Cosine Similarity** - Measures how close your skills are to requirements
    3. **Weighted Scoring** - 60% required, 25% preferred, 15% experience
    4. **Evidence Citation** - Links each skill to specific CV sentences
    
    **Interpreting Scores:**
    - 🟢 75%+ = Excellent Match (Apply now!)
    - 🔵 60-75% = Good Match (Strong candidate)
    - 🟡 45-60% = Moderate Match (Consider upskilling)
    - 🔴 <45% = Weak Match (Significant gaps)
    """)