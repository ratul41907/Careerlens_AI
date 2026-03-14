"""
CV-JD Matcher Page - Enhanced with Caching, Progress, Error Handling, and Missing Features
"""
import streamlit as st

# ============================================================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(
    page_title="CV-JD Matcher - CareerLens AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS - After page config
# ============================================================================
import sys
from pathlib import Path
import tempfile
import json
import time
import hashlib
import gc
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add utils path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

# Mobile responsiveness - Day 25
from mobile_styles import inject_mobile_styles
inject_mobile_styles()

# Accessibility - Day 26
from accessibility import AccessibilityHelper
accessibility = AccessibilityHelper()
accessibility.inject_accessibility_css()
accessibility.add_skip_link()

# Add caching utility import (Day 23)
try:
    from caching import ModelCache, ComputationCache
    OPTIMIZATION_ENABLED = True
except ImportError:
    OPTIMIZATION_ENABLED = False
    st.warning("⚠️ Caching disabled - install caching utilities for better performance")

# Import backend modules
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.explainability import ExplainabilityEngine

# ============================================================================
# CONFIGURATION
# ============================================================================
OPTIMIZATION_ENABLED = OPTIMIZATION_ENABLED  # From caching import above

# Rest of your code continues here...
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
    
    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'match_result' not in st.session_state:
    st.session_state.match_result = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# Cache the engines with optimization
def load_engines():
    """Load AI engines with optimized caching"""
    if OPTIMIZATION_ENABLED:
        embedding_engine = ModelCache.load_embedding_engine()
        scoring_engine = ModelCache.load_scoring_engine(embedding_engine)
        explainability_engine = ModelCache.load_explainability_engine(embedding_engine)
    else:
        # Fallback to standard loading
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
    with st.expander("🔧 Debug Information"):
        st.code(f"Error: {str(e)}\n\nPlease restart the application.")
    st.stop()

# Two columns: CV upload and JD input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Your CV")
    
    # CV upload
    uploaded_file = st.file_uploader(
        "Upload CV (PDF, DOCX, or TXT)",
        accepted_types=['pdf', 'docx', 'txt'],
        key="cv_upload",
        help_text="Upload your resume in PDF, DOCX, or TXT format"
    )
    
    # OR paste CV text
    st.markdown("**OR paste CV text:**")
    cv_text = accessibility.accessible_text_area(
        "Paste your CV here",
        height=300,
        placeholder="Paste your CV content here...\n\nInclude:\n• Work experience\n• Skills\n• Education\n• Projects",
        help_text="Paste the full text of your CV including experience, skills, and education",
        label_visibility="collapsed"
    )

# DAY 24 FEATURE 1: Academic Eligibility Validation
st.markdown("---")
st.markdown("### 🎓 Academic Eligibility Validation (Optional)")

eligibility_col1, eligibility_col2 = st.columns([1, 1])

with eligibility_col1:
    st.markdown("**Upload Academic Documents:**")
    academic_docs = st.file_uploader(
        "Upload marksheets, certificates, or transcripts",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="We'll verify degree, GPA, and certification requirements",
        key="academic_docs"
    )
    
    if academic_docs:
        st.success(f"✅ {len(academic_docs)} document(s) uploaded")

with eligibility_col2:
    st.markdown("**Job Description for Eligibility:**")
    eligibility_jd = st.text_area(
        "Paste JD here or use the one below",
        height=100,
        placeholder="Paste job description with eligibility requirements...",
        key="eligibility_jd_text",
        label_visibility="collapsed"
    )

if academic_docs and (eligibility_jd or st.session_state.get('jd_for_eligibility')):
    if st.button("🔍 Validate Eligibility", use_container_width=True):
        with st.spinner("📄 Validating academic eligibility..."):
            try:
                from src.validation.eligibility_validator import EligibilityValidator
                
                validator = EligibilityValidator()
                
                # Use provided JD or stored one
                jd_for_check = eligibility_jd if eligibility_jd else st.session_state.get('jd_for_eligibility', '')
                
                if not jd_for_check:
                    st.error("❌ Please provide a job description for eligibility validation")
                    st.stop()
                
                # Parse JD for requirements
                from src.parsers.jd_parser import JDParser
                jd_parser = JDParser()
                jd_data = jd_parser.parse(jd_for_check)
                
                # Extract academic text from all documents
                academic_text = ""
                for doc in academic_docs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.name.split('.')[-1]}") as tmp:
                        tmp.write(doc.read())
                        tmp_path = tmp.name
                    
                    # Extract text based on file type
                    if doc.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        academic_text += validator._extract_text_from_image(tmp_path) + "\n"
                    else:
                        academic_text += validator._extract_text_from_pdf(tmp_path) + "\n"
                    
                    os.unlink(tmp_path)
                
                # Validate eligibility
                is_eligible, reason = validator.validate_eligibility(jd_data, academic_text)
                
                st.markdown("### 📊 Eligibility Result")
                
                if is_eligible:
                    st.success("✅ **PASS** - You meet the eligibility criteria!")
                    st.info(f"**Reason:** {reason}")
                else:
                    st.error("❌ **FAIL** - You do not meet all eligibility requirements")
                    st.warning(f"**Missing:** {reason}")
                    st.info("💡 **Tip:** Focus on acquiring missing qualifications before applying")
                
            except Exception as e:
                st.error(f"❌ Validation failed: {str(e)}")
                st.info("💡 Try uploading clearer document scans or PDFs with extractable text")

st.markdown("---")

with col2:
    st.markdown("### 💼 Job Description")
    
    jd_text = accessibility.accessible_text_area(
        "Paste job description",
        key="jd_textarea",
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

Experience: 5+ years
Education: Bachelor's in Computer Science
GPA: Minimum 3.0""",
        label_visibility="collapsed"
    )

# Store JD for eligibility validation
if jd_text:
    st.session_state.jd_for_eligibility = jd_text

# Match button
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    analyze_button = accessibility.accessible_button(
        "🎯 Analyze Match", 
        key="analyze_match_btn",
        help_text="Start CV-JD match analysis (takes 6-10 seconds)",
    )
if analyze_button:
    # Validation
    if not jd_text:
        st.error("❌ **Please provide a job description**")
        st.info("💡 **Tip:** Paste the complete job posting including required skills.")
        st.stop()
    
    if not uploaded_file and not cv_text:
        st.error("❌ **Please upload a CV or paste CV text**")
        st.info("💡 **Tip:** You can either upload a PDF/DOCX file or paste your CV text.")
        st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Parse CV (20%) - WITH CACHING
        status_text.info("📄 **Step 1/5:** Parsing your CV...")
        progress_bar.progress(20)
        
        if OPTIMIZATION_ENABLED:
            cv_parser = ModelCache.load_cv_parser()
        else:
            cv_parser = CVParser()
        
        try:
            if uploaded_file:
                # Generate file hash for caching
                file_content = uploaded_file.read()
                file_hash = hashlib.md5(file_content).hexdigest()
                
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name
                
                # Use cached parsing
                if OPTIMIZATION_ENABLED:
                    cv_data = ComputationCache.cache_cv_parse(cv_parser, file_hash, tmp_path)
                else:
                    cv_data = cv_parser.parse(tmp_path)
                
                os.unlink(tmp_path)
            else:
                cv_data = {
                    'text': cv_text,
                    'sections': cv_parser._segment_sections(cv_text)
                }
            
            st.success("✅ CV parsed successfully!")
            time.sleep(0.3)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Failed to parse CV**")
            st.warning(f"""
            **Possible causes:**
            - File corrupted or unsupported format
            - File contains only images (no text)
            - File encoding issue
            
            **Solutions:**
            - Try plain text format
            - Copy and paste content manually
            
            **Technical details:** {str(e)}
            """)
            st.stop()
        
        # Step 2: Parse JD (40%) - WITH CACHING
        status_text.info("💼 **Step 2/5:** Analyzing job description...")
        progress_bar.progress(40)
        
        try:
            if OPTIMIZATION_ENABLED:
                jd_parser = ModelCache.load_jd_parser()
                jd_data, jd_hash = ComputationCache.cache_jd_parse(jd_parser, jd_text)
            else:
                jd_parser = JDParser()
                jd_data = jd_parser.parse(jd_text)
                jd_hash = hashlib.md5(jd_text.encode()).hexdigest()
            
            num_required = len(jd_data.get('required_skills', []))
            num_preferred = len(jd_data.get('preferred_skills', []))
            
            st.success(f"✅ Found {num_required} required, {num_preferred} preferred skills")
            time.sleep(0.3)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Failed to parse job description**")
            st.warning(f"""
            **Possible causes:**
            - JD text incomplete
            - No clear skills section
            
            **Solutions:**
            - Include complete job posting
            - Ensure skills are clearly listed
            
            **Technical details:** {str(e)}
            """)
            st.stop()
        
        # Step 3: Embeddings (60%)
        status_text.info("🧠 **Step 3/5:** Generating semantic embeddings (AI)...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        # Step 4: Calculate Match (80%) - WITH CACHING
        status_text.info("🎯 **Step 4/5:** Calculating match score...")
        progress_bar.progress(80)
        
        try:
            # Use cached match score if available
            cv_hash = hashlib.md5(str(cv_data).encode()).hexdigest()
            
            if OPTIMIZATION_ENABLED:
                match_result = ComputationCache.cache_match_score(
                    scoring_engine, cv_hash, jd_hash, cv_data, jd_data
                )
            else:
                match_result = scoring_engine.compute_match_score(cv_data, jd_data)
            
            time.sleep(0.3)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Failed to compute match score**")
            st.warning(f"""
            **AI model error. Please:**
            1. Restart application
            2. Check 8GB+ RAM available
            3. Try shorter CV/JD text
            
            **Technical details:** {str(e)}
            """)
            st.stop()
        
        # Step 5: Generate Evidence (95%)
        status_text.info("📊 **Step 5/5:** Generating insights...")
        progress_bar.progress(95)
        
        try:
            explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.warning(f"""
            ⚠️ **Match calculated but explanations failed.**
            You can still see your match score below.
            """)
            explanation = {'evidence': {}, 'recommendations': []}
        
        # Complete
        progress_bar.progress(100)
        status_text.success("✅ **Match analysis complete!**")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
# Store results AND data
        st.session_state.match_result = match_result
        st.session_state.explanation = explanation
        st.session_state.cv_data = cv_data
        st.session_state.jd_data = jd_data
        
        # Accessibility - Announce match completion (Day 26)
        accessibility.add_aria_live_region(
            "match-result-announcement",
            f"Match analysis complete. Your score is {match_result['overall_percentage']}. Results are now displayed below.",
            politeness="assertive"
        )
        
        # Cleanup memory if optimization enabled
        if OPTIMIZATION_ENABLED:
            gc.collect()
        
        # st.balloons()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ **Unexpected error during matching**")
        st.warning(f"""
        **Something went wrong. Please try:**
        - Refreshing the page
        - Using different CV/JD
        - Restarting application
        
        **Technical details:** {str(e)}
        """)
        with st.expander("🔧 Debug Information"):
            st.code(f"Exception: {type(e).__name__}\nMessage: {str(e)}")
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
            rec_col1, rec_col2, rec_col3 = st.columns([0.5, 8, 1.5])
            
            with rec_col1:
                st.markdown(f"**{i}.**")
            
            with rec_col2:
                st.warning(rec)
            
            with rec_col3:
                st.markdown("**High**" if i == 1 else "**Medium**" if i == 2 else "**Low**")
    
# DAY 24 FEATURE 2: Counterfactual Skill Impact Analysis
    st.markdown("---")
    st.markdown("### 🎯 Skill Impact Simulation")
    st.info("💡 **What if you learned a missing skill?** See predicted score improvements:")
    
    # Check if we have data
    if 'cv_data' not in st.session_state or 'jd_data' not in st.session_state:
        st.warning("⚠️ Please perform a match analysis first to see skill impact predictions.")
    else:
        try:
            from src.scoring.counterfactual import CounterfactualSimulator
            
            simulator = CounterfactualSimulator(scoring_engine)
            
            # Get missing required skills
            missing_required = []
            for skill_detail in match_result['breakdown']['required_skills']['details']['skills']:
                if not skill_detail['matched']:
                    missing_required.append(skill_detail['skill'])
            
            if missing_required:
                # Extract current score as float
                current_percentage = match_result['overall_percentage']
                if isinstance(current_percentage, str):
                    current_percentage = float(current_percentage.replace('%', '').strip())
                else:
                    current_percentage = float(current_percentage)
                
                # Simulate adding each skill
                impact_results = []
                for skill in missing_required[:10]:
                    simulated_score = simulator.simulate_skill_addition(
                        st.session_state.cv_data,
                        st.session_state.jd_data,
                        skill
                    )
                    improvement = simulated_score - current_percentage
                    impact_results.append({
                        'skill': skill,
                        'current': current_percentage,
                        'predicted': simulated_score,
                        'improvement': improvement
                    })
                
                # Sort by improvement (descending)
                impact_results.sort(key=lambda x: x['improvement'], reverse=True)
                
                # Display top 5
                st.markdown("**🏆 Top 5 High-Impact Skills to Learn:**")
                
                for idx, result in enumerate(impact_results[:5], 1):
                    impact_col1, impact_col2, impact_col3 = st.columns([2, 1, 1])
                    
                    with impact_col1:
                        st.markdown(f"**{idx}. {result['skill']}**")
                    
                    with impact_col2:
                        st.metric("Predicted Score", f"{result['predicted']:.1f}%", f"+{result['improvement']:.1f}%")
                    
                    with impact_col3:
                        if result['improvement'] >= 8:
                            st.success("🔥 High Impact")
                        elif result['improvement'] >= 5:
                            st.info("⭐ Medium Impact")
                        else:
                            st.warning("💡 Low Impact")
                
                st.markdown("**💡 Learning Priority:** Focus on high-impact skills first for maximum score improvement!")
            else:
                st.success("✅ You already have all required skills!")
        
        except Exception as e:
            st.warning(f"⚠️ Counterfactual analysis unavailable: {str(e)}")   
            
            # DAY 24 FEATURE 3: Learning Pathways
    st.markdown("---")
    st.markdown("### 📚 Personalized Learning Pathway")
    
    pathway_duration = st.radio(
        "Choose your learning timeline:",
        ["7-Day Quick Start", "14-Day Accelerated", "30-Day Comprehensive"],
        horizontal=True
    )
    
    if st.button("📖 Generate Learning Plan", use_container_width=True):
        with st.spinner("🎓 Creating your personalized learning pathway..."):
            try:
                from src.guidance.learning_pathways import LearningPathwayGenerator
                
                pathway_gen = LearningPathwayGenerator()
                
   # Extract skill gaps
                skill_gaps = []
                for skill_detail in match_result['breakdown']['required_skills']['details']['skills']:
                    if not skill_detail['matched']:
                        skill_gaps.append(skill_detail['skill'])
                
                # Determine days
                days_map = {
                    "7-Day Quick Start": 7,
                    "14-Day Accelerated": 14,
                    "30-Day Comprehensive": 30
                }
                num_days = days_map[pathway_duration]
                
                # Generate pathway (use session state data)
                pathway = pathway_gen.generate_pathway(
                    skill_gaps,
                    st.session_state.jd_data,  # ← Use session state
                    num_days
                )
                
                st.success(f"✅ **{num_days}-Day Learning Plan Created!**")
                
                # Display pathway
                st.markdown("### 📋 Your Learning Roadmap")
                
                for day_plan in pathway['daily_plans'][:5]:  # Show first 5 days
                    with st.expander(f"📅 Day {day_plan['day']}: {day_plan['focus']}", expanded=(day_plan['day']==1)):
                        st.markdown(f"**🎯 Goal:** {day_plan['goal']}")
                        
                        st.markdown("**📚 Tasks:**")
                        for task in day_plan['tasks']:
                            st.markdown(f"- {task}")
                        
                        if day_plan.get('resources'):
                            st.markdown("**🔗 Resources:**")
                            for resource in day_plan['resources'][:3]:
                                st.markdown(f"- {resource}")
                        
                        if day_plan.get('mini_project'):
                            st.info(f"💡 **Mini-Project:** {day_plan['mini_project']}")
                
                # Show summary
                st.markdown("### 📊 Pathway Summary")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.metric("Duration", f"{num_days} Days")
                    st.metric("Skills to Learn", len(skill_gaps))
                
                with summary_col2:
                    st.metric("Daily Time", f"{pathway.get('estimated_daily_hours', 2)} hours")
                    st.metric("Projects", len([d for d in pathway['daily_plans'] if d.get('mini_project')]))
                
                # Download option
                pathway_json = json.dumps(pathway, indent=2)
                st.download_button(
                    "📥 Download Full Pathway (JSON)",
                    pathway_json,
                    file_name=f"learning_pathway_{num_days}day.json",
                    mime="application/json"
                )
            
            except Exception as e:
                st.error(f"❌ Could not generate learning pathway: {str(e)}")
                st.info("💡 Make sure you have skill gaps identified from the match analysis above")
    
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
            label="📄 Download Match Results (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name="match_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col2:
        if st.button("📝 Improve CV", use_container_width=True):
            st.switch_page("pages/2_📝_CV_Generator.py")
    
    with export_col3:
        if st.button("🎓 Interview Prep", use_container_width=True):
            st.switch_page("pages/3_🎓_Interview_Prep.py")

# DAY 23 FEATURE: Cache Control
if st.checkbox("🔧 Show Performance Tools", value=False):
    st.markdown("### 🛠️ Performance Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Caches", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared!")
            st.info("💡 Caches will rebuild on next use")
    
    with col2:
        if st.button("🧹 Cleanup Memory", use_container_width=True):
            gc.collect()
            st.success("✅ Memory cleaned!")
    
    st.markdown("**Cache Status:**")
    st.info(f"Optimization: {'✅ Enabled' if OPTIMIZATION_ENABLED else '❌ Disabled'}")

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
    
    ### Processing Time:
    
    - **6-10 seconds** is normal for AI semantic matching
    - Shows real AI/ML computation is happening
    - Much more accurate than simple keyword matching
    """)

with st.expander("⌨️ Keyboard Shortcuts & Tips"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Navigation:**
        - Use sidebar to switch pages
        - Browser back/forward may break state
        - Refresh (F5) resets session data
        
        **File Upload:**
        - Drag and drop supported
        - Click to browse files
        - Max size: 10MB per file
        """)
    
    with col2:
        st.markdown("""
        **Tips:**
        - Export results before leaving page
        - Use Ctrl+F to search on page
        - Download files immediately
        - Clear cache if issues occur
        """)