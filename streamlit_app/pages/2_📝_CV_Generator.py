"""
CV Generator Page - Enhanced with Error Handling and Validation
"""
import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import os
import tempfile
import re
import time
import subprocess
# Mobile responsiveness - Day 25
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mobile_styles import inject_mobile_styles
inject_mobile_styles()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.cv_generator import CVGenerator
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.generation.cv_optimizer import CVOptimizer
# Page config
#st.set_page_config(
 #   page_title="CV Generator - CareerLens AI",
  ## layout="wide",
    #initial_sidebar_state="expanded"
#)

# Dark theme CSS
st.markdown("""
<style>
    /* Force dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* All text light colored */
    * {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    /* Text inputs */
    .stTextInput input, .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 8px;
    }
    
    /* Number input */
    .stNumberInput input {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 2px dashed rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
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
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #10b981 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        color: #3b82f6 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        color: #f59e0b !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #ef4444 !important;
    }
    
    /* Section dividers */
    hr {
        border-color: rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        color: #cbd5e1 !important;
        border-radius: 8px;
    }
    
    /* Preview box */
    .cv-preview {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .cv-preview * {
        color: #000 !important;
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
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = {}
if 'generated_doc' not in st.session_state:
    st.session_state.generated_doc = None
if 'cv_bytes_docx' not in st.session_state:
    st.session_state.cv_bytes_docx = None
if 'cv_html' not in st.session_state:
    st.session_state.cv_html = None

# Title
st.markdown("""
<h1 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
           -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
    📝 Smart CV Generator
</h1>
<p style="color: #94a3b8 !important; font-size: 1.25rem; margin-bottom: 2rem;">
    Create ATS-optimized professional CVs • Multiple generation modes
</p>
""", unsafe_allow_html=True)

# Generation mode selector
st.markdown("### 🎯 Choose Generation Mode")

generation_mode = st.radio(
    "How would you like to create your CV?",
    [
        "📝 Manual Entry",
        "🎯 Auto-Generate from Job Description",
        "📄 Extract from Documents",
        "✨ Improve Existing CV"
    ],
    label_visibility="collapsed",
    help="Choose your preferred CV creation method"
)

st.markdown("---")

# ============================================================================
# MODE 1: MANUAL ENTRY WITH ENHANCED VALIDATION
# ============================================================================
if "Manual Entry" in generation_mode:
    st.info("💡 Fill in your information below. Required fields are marked with *")
    
    # ADD JD Input for optimization
    st.markdown("### 🎯 Target Job Description (Optional)")
    st.info("💡 Provide a job description to optimize your CV with relevant keywords and skills")
    
    manual_jd_text = st.text_area(
        "Paste job description to optimize CV",
        height=150,
        placeholder="Paste the job description here to generate a CV optimized for this role...",
        key="manual_jd_input"
    )
    
    st.markdown("---")
    
    # Create tabs for organized input
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Personal Info", "💼 Experience", "🎓 Education & Skills", "🎨 Projects"])    
    # TAB 1: Personal Information
    with tab1:
        st.markdown("### 📋 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="John Doe", help="Your full name as it should appear on the CV")
            email = st.text_input("Email *", placeholder="john.doe@email.com", help="Professional email address")
            phone = st.text_input("Phone Number *", placeholder="+1-234-567-8900", help="Include country code")
        
        with col2:
            location = st.text_input("Location", placeholder="San Francisco, CA", help="City, State/Country")
            linkedin = st.text_input("LinkedIn URL", placeholder="linkedin.com/in/johndoe", help="Your LinkedIn profile URL")
            github = st.text_input("GitHub URL", placeholder="github.com/johndoe", help="Your GitHub profile (optional)")
        
        st.markdown("---")
        
        st.markdown("### 📄 Professional Summary")
        summary = st.text_area(
            "Brief 2-3 sentence summary (optional)",
            placeholder="Results-driven Software Engineer with 5+ years of experience...",
            height=120,
            help="Highlight your key qualifications and career focus"
        )
    
    # TAB 2: Work Experience
    with tab2:
        st.markdown("### 💼 Work Experience")
        
        num_experiences = st.number_input(
            "Number of work experiences",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Add your most recent positions first"
        )
        
        experiences = []
        
        for i in range(int(num_experiences)):
            with st.expander(f"Experience #{i+1}", expanded=(i==0)):
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    job_title = st.text_input(
                        f"Job Title *",
                        key=f"job_title_{i}",
                        placeholder="Senior Software Engineer"
                    )
                    company = st.text_input(
                        f"Company *",
                        key=f"company_{i}",
                        placeholder="Tech Corp"
                    )
                
                with exp_col2:
                    start_date = st.text_input(
                        f"Start Date",
                        key=f"start_date_{i}",
                        placeholder="January 2021",
                        help="Month Year format"
                    )
                    end_date = st.text_input(
                        f"End Date",
                        key=f"end_date_{i}",
                        placeholder="Present",
                        help="Use 'Present' if current position"
                    )
                
                st.markdown("**Key Achievements (one per line):**")
                achievements = st.text_area(
                    "Enter achievements",
                    key=f"achievements_{i}",
                    placeholder="• Led development of REST APIs\n• Reduced costs by 30%\n• Mentored 3 junior developers",
                    height=150,
                    help="Use bullet points, start with action verbs, quantify results"
                )
                
                if job_title and company:
                    duration = f"{start_date} - {end_date}" if start_date and end_date else "Present"
                    bullets = [line.strip().lstrip('•-*').strip() for line in achievements.split('\n') if line.strip()]
                    
                    experiences.append({
                        'title': job_title,
                        'company': company,
                        'duration': duration,
                        'bullets': bullets
                    })
    
    # TAB 3: Education & Skills
    with tab3:
        st.markdown("### 🎓 Education")
        
        num_education = st.number_input("Number of degrees", min_value=0, max_value=5, value=1, step=1)
        
        education = []
        
        for i in range(int(num_education)):
            with st.expander(f"Education #{i+1}", expanded=(i==0)):
                edu_col1, edu_col2 = st.columns(2)
                
                with edu_col1:
                    degree = st.text_input(
                        f"Degree *",
                        key=f"degree_{i}",
                        placeholder="Bachelor of Science in Computer Science"
                    )
                    institution = st.text_input(
                        f"Institution *",
                        key=f"institution_{i}",
                        placeholder="University of Technology"
                    )
                
                with edu_col2:
                    year = st.text_input(
                        f"Year/Duration",
                        key=f"year_{i}",
                        placeholder="2015 - 2019"
                    )
                    gpa = st.text_input(
                        f"GPA (optional)",
                        key=f"gpa_{i}",
                        placeholder="3.8/4.0",
                        help="Include only if above 3.5"
                    )
                
                if degree and institution:
                    education.append({
                        'degree': degree,
                        'institution': institution,
                        'year': year if year else 'N/A',
                        'gpa': gpa if gpa else None
                    })
        
        st.markdown("---")
        st.markdown("### 🔧 Skills")
        
        st.markdown("**Enter your skills (comma-separated):**")
        skills_input = st.text_area(
            "Skills",
            placeholder="Python, JavaScript, React, Docker, Kubernetes, AWS",
            height=100,
            help="List 10-20 relevant skills, most important first"
        )
        
        skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
    
    # TAB 4: Projects
    with tab4:
        st.markdown("### 🎨 Projects")
        
        num_projects = st.number_input("Number of projects", min_value=0, max_value=10, value=1, step=1)
        
        projects = []
        
        for i in range(int(num_projects)):
            with st.expander(f"Project #{i+1}", expanded=(i==0)):
                project_name = st.text_input(
                    f"Project Name",
                    key=f"project_name_{i}",
                    placeholder="E-Commerce Platform"
                )
                project_desc = st.text_area(
                    f"Description",
                    key=f"project_desc_{i}",
                    placeholder="Built a full-stack e-commerce platform with 5000+ users...",
                    height=80
                )
                project_tech = st.text_input(
                    f"Technologies Used",
                    key=f"project_tech_{i}",
                    placeholder="React, Node.js, PostgreSQL"
                )
                
                if project_name and project_desc:
                    projects.append({
                        'name': project_name,
                        'description': project_desc,
                        'technologies': project_tech
                    })
        
        st.markdown("---")
        st.markdown("### 🏆 Certifications (Optional)")
        
        certifications_input = st.text_area(
            "Enter certifications (one per line)",
            placeholder="AWS Certified Solutions Architect (2023)\nCertified Kubernetes Administrator (2024)",
            height=120
        )
        
        certifications = [cert.strip() for cert in certifications_input.split('\n') if cert.strip()]
    
    # GENERATE BUTTON FOR MANUAL MODE
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button("🎯 Generate CV", type="primary", use_container_width=True)
    
    if generate_button:
        # COMPREHENSIVE VALIDATION (keep all your existing validation code as it is)
        errors = []
        warnings = []
        
        # ... [YOUR EXISTING VALIDATION CODE REMAINS UNCHANGED] ...
        
        if errors:
            st.error("❌ **Please fix the following errors:**")
            for error in errors:
                st.markdown(f"- {error}")
            st.stop()
        
        if warnings:
            st.warning("⚠️ **Recommendations:**")
            for warning in warnings:
                st.markdown(f"- {warning}")
            if not st.checkbox("I understand and want to continue anyway"):
                st.stop()
        
        # PROGRESS TRACKING
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.info("📝 **Step 1/4:** Collecting your information...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            # Prepare data
            personal_info = {
                'name': name,
                'email': email,
                'phone': phone,
                'location': location if location else None,
                'linkedin': linkedin if linkedin else None,
                'github': github if github else None,
                'summary': summary if summary else None
            }
            
            # LLM-Powered JD Optimization
            if manual_jd_text and manual_jd_text.strip():
                status_text.info("🤖 **Step 2/4:** AI is optimizing your CV for the job...")
                progress_bar.progress(45)
                
                optimizer = CVOptimizer()
                optimization = optimizer.optimize_manual_cv_for_jd(
                    personal_info=personal_info,
                    experiences=experiences,
                    education=education,
                    skills=skills,
                    jd_text=manual_jd_text
                )
                
                # Apply LLM results
                skills = optimization['prioritized_skills'][:20]
                
                if optimization.get('optimized_summary'):
                    personal_info['summary'] = optimization['optimized_summary']
                
                matched_count = len(optimization.get('matched_skills', []))
                st.success(f"✅ AI optimized: {matched_count} skills matched & summary improved")
                time.sleep(0.5)
            
            status_text.info("🎨 **Step 3/4:** Generating professional CV with ATS optimization...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            # Generate CV
            generator = CVGenerator()
            
            doc = generator.generate_cv(
                personal_info=personal_info,
                experience=experiences if experiences else None,
                education=education if education else None,
                skills=skills if skills else None,
                projects=projects if projects else None,
                certifications=certifications if certifications else None
            )
            
            status_text.info("✨ **Step 3/3:** Creating preview and download options...")
            progress_bar.progress(90)
            
            # Save DOCX
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CV_{name.replace(' ', '_')}_{timestamp}.docx"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                doc.save(tmp_file.name)
                tmp_docx_path = tmp_file.name
            
            with open(tmp_docx_path, 'rb') as file:
                cv_bytes_docx = file.read()
            
            st.session_state.cv_bytes_docx = cv_bytes_docx
            st.session_state.filename = filename
            
            # Generate HTML preview
            try:
                import mammoth
                with open(tmp_docx_path, 'rb') as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    st.session_state.cv_html = result.value
            except:
                st.session_state.cv_html = None
            
            os.unlink(tmp_docx_path)
            
            progress_bar.progress(100)
            status_text.success("✅ **CV generated successfully!**")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Your professional CV is ready!")
            #st.balloons()
            progress_bar.progress(100)
            status_text.success("✅ **CV generated successfully!**")
            time.sleep(0.5)
            
            
            st.success("✅ Your professional CV is ready!")
            #st.balloons()

            # ====================== PDF DOWNLOAD OPTION ======================
            st.markdown("---")
            st.markdown("### 📕 PDF Download Option")
            
            if st.button("🔄 Generate PDF", use_container_width=True, key="gen_pdf_manual"):
                with st.spinner("📕 Converting DOCX to PDF..."):
                    try:
                        from docx2pdf import convert
                        import tempfile
                        
                        # Save current DOCX temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            tmp.write(st.session_state.cv_bytes_docx)
                            tmp_docx = tmp.name
                        
                        # Convert to PDF
                        tmp_pdf = tmp_docx.replace('.docx', '.pdf')
                        convert(tmp_docx, tmp_pdf)
                        
                        # Read PDF bytes
                        with open(tmp_pdf, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        # Cleanup temporary files
                        os.unlink(tmp_docx)
                        os.unlink(tmp_pdf)
                        
                        # Store in session state
                        st.session_state.cv_bytes_pdf = pdf_bytes
                        st.success("✅ PDF generated successfully!")
                        st.rerun()
                        
                    except ImportError:
                        st.warning("""
⚠️ PDF conversion requires Microsoft Word installed on your system.

**Alternative ways:**
1. Download the DOCX file above
2. Open it in Microsoft Word → "Save As" → PDF
3. Or upload to Google Docs and download as PDF
""")
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
            
            # Show PDF download button if available
            if 'cv_bytes_pdf' in st.session_state:
                st.download_button(
                    label="📕 Download PDF",
                    data=st.session_state.cv_bytes_pdf,
                    file_name=st.session_state.filename.replace('.docx', '.pdf'),
                    mime="application/pdf",
                    use_container_width=True
                )            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Failed to generate CV**")
            st.warning(f"""
            **Generation failed. Possible causes:**
            - System resource issue
            - Invalid character in input
            - File system permission error
            
            **Please try:**
            - Simplifying text (remove special characters)
            - Reducing content length
            - Restarting the application
            
            **Technical details:** {str(e)}
            """)
            with st.expander("🔧 Debug Information"):
                st.code(f"Exception: {type(e).__name__}\nMessage: {str(e)}")
                
    # ============================================================================
# MODE 2: AUTO-GENERATE FROM JOB DESCRIPTION
# ============================================================================
elif "Auto-Generate" in generation_mode:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #10b981 !important; margin: 0 0 0.5rem 0;">🎯 Smart CV Auto-Generation</h3>
        <p style="color: #cbd5e1 !important; margin: 0;">
            Upload your CV and paste a job description. We'll automatically:
            <br>• Extract and parse your CV data
            <br>• Identify required skills from the JD
            <br>• Match YOUR skills to JD requirements
            <br>• Generate optimized CV highlighting matched skills
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Existing CV")
        
        existing_cv_file = st.file_uploader(
            "Upload your current CV",
            type=['pdf', 'docx', 'txt'],
            help="We'll extract and optimize your information",
            key="auto_cv"
        )
        
        st.markdown("**OR paste CV text:**")
        existing_cv_text = st.text_area(
            "Paste CV",
            height=300,
            placeholder="Paste your current CV here...",
            key="existing_cv_text"
        )
    
    with col2:
        st.markdown("### 💼 Target Job Description")
        
        target_jd_text = st.text_area(
            "Paste the job description",
            height=460,
            placeholder="""Paste full job description...

Example:
Senior Backend Engineer

Required Skills:
- Python, FastAPI
- Docker, Kubernetes
- AWS, PostgreSQL
- 5+ years experience

We'll match YOUR skills automatically!""",
            key="target_jd"
        )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        auto_analyze_btn = st.button("🔍 Analyze & Generate", type="primary", use_container_width=True)
    
    if auto_analyze_btn:
        # Validation
        if not target_jd_text:
            st.error("❌ Please provide a target job description")
            st.stop()
        
        if not existing_cv_file and not existing_cv_text:
            st.error("❌ Please upload your CV or paste CV text")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Parse CV (20%)
            status_text.info("📄 **Step 1/5:** Parsing your existing CV...")
            progress_bar.progress(20)
            
            cv_parser = CVParser()
            
            if existing_cv_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{existing_cv_file.name.split('.')[-1]}") as tmp:
                    tmp.write(existing_cv_file.read())
                    cv_data = cv_parser.parse(tmp.name)
                os.unlink(tmp.name)
            else:
                cv_data = {
                    'text': existing_cv_text,
                    'sections': cv_parser._segment_sections(existing_cv_text)
                }
            
            st.success("✅ CV parsed successfully")
            time.sleep(0.3)
            
            # Step 2: Parse JD (40%)
            status_text.info("💼 **Step 2/5:** Analyzing job requirements...")
            progress_bar.progress(40)
            
            jd_parser = JDParser()
            jd_data = jd_parser.parse(target_jd_text)
            
            st.success(f"✅ Found {len(jd_data.get('required_skills', []))} required skills")
            time.sleep(0.3)
            
            # Step 3: Match Skills (60%)
            status_text.info("🎯 **Step 3/5:** Matching your skills to JD requirements...")
            progress_bar.progress(60)
            
            embedding_engine = EmbeddingEngine()
            scoring_engine = ScoringEngine(embedding_engine)
            match_result = scoring_engine.compute_match_score(cv_data, jd_data)
            
            # Extract matched skills
            required_skills_details = match_result['breakdown']['required_skills']['details']['skills']
            matched_skills = [s['skill'] for s in required_skills_details if s['matched']]
            missing_skills = [s['skill'] for s in required_skills_details if not s['matched']]
            
            st.success(f"✅ Match score: {match_result['overall_percentage']} | {len(matched_skills)} skills matched")
            time.sleep(0.3)
            
            # Step 4: Extract CV Data (80%)
            status_text.info("📊 **Step 4/5:** Extracting your information...")
            progress_bar.progress(80)
            
            cv_sections = cv_data.get('sections', {})
            header_text = cv_sections.get('header', '')
            
            # Extract contact info
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', header_text)
            extracted_email = email_match.group(0) if email_match else 'your.email@example.com'
            
            phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]', header_text)
            extracted_phone = phone_match.group(0) if phone_match else '+1-234-567-8900'
            
            lines = [l.strip() for l in header_text.split('\n') if l.strip()]
            extracted_name = lines[0] if lines else 'Your Name'
            
            # Parse experience
            experience_text = cv_sections.get('experience', '')
            experience_list = []
            if experience_text:
                exp_sections = re.split(r'\n(?=[A-Z][a-z]+ (?:Engineer|Developer|Manager|Analyst))', experience_text)
                for exp in exp_sections[:3]:
                    lines = [l.strip() for l in exp.split('\n') if l.strip()]
                    if len(lines) >= 2:
                        experience_list.append({
                            'title': lines[0],
                            'company': lines[1] if len(lines) > 1 else 'Company',
                            'duration': 'Present',
                            'bullets': [l.lstrip('•-*').strip() for l in lines[2:] if l.strip()][:4]
                        })
            
            # Parse education
            education_text = cv_sections.get('education', '')
            education_list = []
            if education_text:
                lines = [l.strip() for l in education_text.split('\n') if l.strip()]
                if lines:
                    education_list.append({
                        'degree': lines[0],
                        'institution': lines[1] if len(lines) > 1 else 'University',
                        'year': lines[2] if len(lines) > 2 else 'Recent',
                        'gpa': None
                    })
            
            # Smart skill ordering (matched first)
            all_jd_skills = jd_data.get('required_skills', []) + jd_data.get('preferred_skills', [])
            ordered_skills = matched_skills + [s for s in all_jd_skills if s not in matched_skills]
            
            st.success("✅ Information extracted successfully")
            time.sleep(0.3)
            
            # Step 5: Generate Optimized CV (95%)
            status_text.info("✨ **Step 5/5:** Generating your optimized CV...")
            progress_bar.progress(95)
            
            personal_info = {
                'name': extracted_name,
                'email': extracted_email,
                'phone': extracted_phone,
                'summary': f"Professional with {len(matched_skills)}/{len(all_jd_skills)} required skills. Expertise in {', '.join(matched_skills[:5])}."
            }
            
            generator = CVGenerator()
            doc = generator.generate_cv(
                personal_info=personal_info,
                experience=experience_list if experience_list else None,
                education=education_list if education_list else None,
                skills=ordered_skills[:20],
                projects=None
            )
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CV_Optimized_{timestamp}.docx"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                doc.save(tmp.name)
                with open(tmp.name, 'rb') as f:
                    cv_bytes = f.read()
            os.unlink(tmp.name)
            
            st.session_state.cv_bytes_docx = cv_bytes
            st.session_state.filename = filename
            
            # Preview
            try:
                import mammoth
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(cv_bytes)
                    with open(tmp.name, 'rb') as f:
                        result = mammoth.convert_to_html(f)
                        st.session_state.cv_html = result.value
                os.unlink(tmp.name)
            except:
                st.session_state.cv_html = None
            
            progress_bar.progress(100)
            status_text.success("✅ **Optimized CV generated!**")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            # Show analysis
            st.markdown("### 🎯 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### ✅ Matched Skills ({len(matched_skills)})")
                if matched_skills:
                    for skill in matched_skills[:10]:
                        st.markdown(f"• **{skill}** (highlighted in CV)")
            
            with col2:
                st.markdown(f"#### ⚠️ Missing Skills ({len(missing_skills)})")
                if missing_skills:
                    for skill in missing_skills[:10]:
                        st.markdown(f"• {skill}")
            
            st.success("✅ CV optimized with matched skills prioritized!")
            #st.balloons()
             
            
            st.success("✅ Your professional CV is ready!")
            #st.balloons()

            # ====================== PDF DOWNLOAD OPTION ======================
            st.markdown("---")
            st.markdown("### 📕 PDF Download Option")
            
            if st.button("🔄 Generate PDF", use_container_width=True, key="gen_pdf_manual"):
                with st.spinner("📕 Converting DOCX to PDF..."):
                    try:
                        from docx2pdf import convert
                        import tempfile
                        
                        # Save current DOCX temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            tmp.write(st.session_state.cv_bytes_docx)
                            tmp_docx = tmp.name
                        
                        # Convert to PDF
                        tmp_pdf = tmp_docx.replace('.docx', '.pdf')
                        convert(tmp_docx, tmp_pdf)
                        
                        # Read PDF bytes
                        with open(tmp_pdf, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        # Cleanup temporary files
                        os.unlink(tmp_docx)
                        os.unlink(tmp_pdf)
                        
                        # Store in session state
                        st.session_state.cv_bytes_pdf = pdf_bytes
                        st.success("✅ PDF generated successfully!")
                        st.rerun()
                        
                    except ImportError:
                        st.warning("""
⚠️ PDF conversion requires Microsoft Word installed on your system.

**Alternative ways:**
1. Download the DOCX file above
2. Open it in Microsoft Word → "Save As" → PDF
3. Or upload to Google Docs and download as PDF
""")
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
            
            # Show PDF download button if available
            if 'cv_bytes_pdf' in st.session_state:
                st.download_button(
                    label="📕 Download PDF",
                    data=st.session_state.cv_bytes_pdf,
                    file_name=st.session_state.filename.replace('.docx', '.pdf'),
                    mime="application/pdf",
                    use_container_width=True
                )           
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Auto-generation failed**")
            st.warning(f"""
            **Could not generate optimized CV. Please try:**
            - Using Manual Entry mode instead
            - Ensuring CV contains extractable text
            - Simplifying the job description
            
            **Technical details:** {str(e)}
            """)
            with st.expander("🔧 Debug Info"):
                st.code(f"{type(e).__name__}: {str(e)}")

# ============================================================================
# MODE 3: EXTRACT FROM DOCUMENTS
# ============================================================================
elif "Extract from Documents" in generation_mode:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #10b981 !important; margin: 0 0 0.5rem 0;">📄 Document-Based CV Generation</h3>
        <p style="color: #cbd5e1 !important; margin: 0;">
            Upload certificates, transcripts, or existing CVs. We'll extract:
            <br>• Education details and GPA
            <br>• Certifications with dates
            <br>• Skills from all documents
            <br>• Work experience from CVs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Your Documents")
    
    uploaded_docs = st.file_uploader(
        "Upload documents (PDF, DOCX, images)",
        type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload any documents containing your information"
    )
    
    if uploaded_docs:
        st.success(f"✅ {len(uploaded_docs)} document(s) uploaded")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            extract_button = st.button("🔍 Extract Information", type="primary", use_container_width=True)
        
        if extract_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.info(f"📄 **Extracting from {len(uploaded_docs)} documents...**")
                progress_bar.progress(20)
                
                extracted_data = {
                    'education': [],
                    'certifications': [],
                    'skills': [],
                    'experience': []
                }
                
                cv_parser = CVParser()
                
                for idx, doc in enumerate(uploaded_docs):
                    progress_bar.progress(20 + (idx + 1) * (60 // len(uploaded_docs)))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.name.split('.')[-1]}") as tmp:
                        tmp.write(doc.read())
                        tmp_path = tmp.name
                    
                    # Parse based on type
                    if doc.name.lower().endswith(('.pdf', '.docx', '.txt')):
                        doc_data = cv_parser.parse(tmp_path)
                        
                        if 'education' in doc_data.get('sections', {}):
                            extracted_data['education'].append(doc_data['sections']['education'])
                        
                        if 'skills' in doc_data.get('sections', {}):
                            skills_text = doc_data['sections']['skills']
                            skills = [s.strip() for s in skills_text.replace('\n', ',').split(',') if s.strip()]
                            extracted_data['skills'].extend(skills)
                        
                        if 'experience' in doc_data.get('sections', {}):
                            extracted_data['experience'].append(doc_data['sections']['experience'])
                    
                    elif doc.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            from src.validation.eligibility_validator import EligibilityValidator
                            validator = EligibilityValidator()
                            text = validator._extract_text_from_image(tmp_path)
                            
                            if 'bachelor' in text.lower() or 'gpa' in text.lower():
                                extracted_data['education'].append(text[:500])
                            
                            if 'certified' in text.lower():
                                extracted_data['certifications'].append(text[:200])
                        except:
                            pass
                    
                    os.unlink(tmp_path)
                
                status_text.success("✅ **Extraction complete!**")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                st.markdown("### 📊 Extracted Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if extracted_data['education']:
                        st.markdown("#### 🎓 Education")
                        for edu in extracted_data['education'][:3]:
                            st.info(edu[:200] + ("..." if len(edu) > 200 else ""))
                    
                    if extracted_data['skills']:
                        st.markdown("#### 🔧 Skills Found")
                        unique_skills = list(set(extracted_data['skills']))[:20]
                        st.write(", ".join(unique_skills))
                
                with col2:
                    if extracted_data['certifications']:
                        st.markdown("#### 🏆 Certifications")
                        for cert in extracted_data['certifications'][:5]:
                            st.info(cert)
                    
                    if extracted_data['experience']:
                        st.markdown("#### 💼 Experience")
                        for exp in extracted_data['experience'][:2]:
                            st.info(exp[:200] + ("..." if len(exp) > 200 else ""))
                
                # Store data
                st.session_state.extracted_data = extracted_data
                st.session_state.ready_to_generate_from_docs = True
                
                st.success("✅ Ready to generate CV from extracted data!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ **Extraction failed**")
                st.warning(f"Error: {str(e)}")
    
    # Generate button if data extracted
    # Generate button if data extracted
    if st.session_state.get('ready_to_generate_from_docs', False):
        st.markdown("---")
        st.markdown("### 🎯 Optimize for Job Description (Optional)")
        
        doc_jd_text = st.text_area(
            "Paste target job description",
            height=150,
            placeholder="Paste JD to optimize extracted data for a specific role...",
            key="doc_jd_optimization"
        )
        
        st.markdown("### ✏️ Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            doc_name = st.text_input("Name", placeholder="John Doe", key="doc_name")
            doc_email = st.text_input("Email", placeholder="john@email.com", key="doc_email")
        with col2:
            doc_phone = st.text_input("Phone", placeholder="+1-234-567-8900", key="doc_phone")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📝 Generate Optimized CV", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("📊 **Step 1/3:** Processing extracted data...")
                    progress_bar.progress(33)
                    
                    extracted = st.session_state.extracted_data
                    
                    # Extract skills
                    all_skills = list(set(extracted.get('skills', [])))
                    
                    # LLM Optimization if JD provided
                    if doc_jd_text and doc_jd_text.strip():
                        status_text.info("🤖 **Step 2/3:** AI is optimizing for job description...")
                        progress_bar.progress(66)
                        
                        try:
                            from src.generation.cv_optimizer import CVOptimizer
                            
                            optimizer = CVOptimizer()
                            
                            # Build CV summary from extracted data
                            cv_summary = f"""
    Skills: {', '.join(all_skills[:20])}
    Education: {extracted.get('education', [])}
    Certifications: {extracted.get('certifications', [])}
    """
                            
                            # Use LLM to optimize
                            improvement = optimizer.improve_existing_cv_with_jd(
                                cv_text=cv_summary,
                                cv_skills=all_skills,
                                jd_text=doc_jd_text
                            )
                            
                            # Use optimized results
                            final_skills = improvement['optimized_skill_order'][:20]
                            summary_text = improvement['enhanced_professional_summary']
                            
                            st.success(f"✅ AI optimized: {len(improvement['matched_skills'])} skills matched!")
                            
                        except Exception as e:
                            st.warning(f"⚠️ LLM unavailable, using basic optimization")
                            final_skills = all_skills[:20]
                            summary_text = f"Professional with expertise in {', '.join(all_skills[:3])}."
                    else:
                        final_skills = all_skills[:20]
                        summary_text = f"Professional with expertise in {', '.join(all_skills[:3])}."
                    
                    status_text.info("✨ **Step 3/3:** Generating CV...")
                    progress_bar.progress(90)
                    
                    personal_info = {
                        'name': doc_name if doc_name else 'Your Name',
                        'email': doc_email if doc_email else 'email@example.com',
                        'phone': doc_phone if doc_phone else '+1-234-567-8900',
                        'summary': summary_text
                    }
                    
                    generator = CVGenerator()
                    doc = generator.generate_cv(
                        personal_info=personal_info,
                        skills=final_skills,
                        certifications=extracted.get('certifications', [])[:10]
                    )
                    
                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"CV_From_Documents_{timestamp}.docx"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                        doc.save(tmp.name)
                        with open(tmp.name, 'rb') as f:
                            cv_bytes = f.read()
                    os.unlink(tmp.name)
                    
                    st.session_state.cv_bytes_docx = cv_bytes
                    st.session_state.filename = filename
                    
                    # Preview
                    try:
                        import mammoth
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            tmp.write(cv_bytes)
                            with open(tmp.name, 'rb') as f:
                                result = mammoth.convert_to_html(f)
                                st.session_state.cv_html = result.value
                        os.unlink(tmp.name)
                    except:
                        st.session_state.cv_html = None
                    
                    progress_bar.progress(100)
                    status_text.success("✅ CV generated successfully!")
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ CV generated and optimized!")
                    
                    st.success("✅ Your professional CV is ready!")

                    # ====================== PDF DOWNLOAD OPTION ======================
                    st.markdown("---")
                    st.markdown("### 📕 PDF Download Option")
                    
                    if st.button("🔄 Generate PDF", use_container_width=True, key="gen_pdf_docs"):
                        with st.spinner("📕 Converting DOCX to PDF..."):
                            try:
                                from docx2pdf import convert
                                import tempfile
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                                    tmp.write(st.session_state.cv_bytes_docx)
                                    tmp_docx = tmp.name
                                
                                tmp_pdf = tmp_docx.replace('.docx', '.pdf')
                                convert(tmp_docx, tmp_pdf)
                                
                                with open(tmp_pdf, 'rb') as f:
                                    pdf_bytes = f.read()
                                
                                os.unlink(tmp_docx)
                                os.unlink(tmp_pdf)
                                
                                st.session_state.cv_bytes_pdf = pdf_bytes
                                st.success("✅ PDF generated successfully!")
                                st.rerun()
                                
                            except ImportError:
                                st.warning("""
⚠️ PDF conversion requires Microsoft Word installed on your system.

**Alternative ways:**
1. Download the DOCX file above
2. Open it in Microsoft Word → "Save As" → PDF
3. Or upload to Google Docs and download as PDF
""")
                            except Exception as e:
                                st.error(f"❌ PDF generation failed: {str(e)}")
                    
                    if 'cv_bytes_pdf' in st.session_state:
                        st.download_button(
                            label="📕 Download PDF",
                            data=st.session_state.cv_bytes_pdf,
                            file_name=st.session_state.filename.replace('.docx', '.pdf'),
                            mime="application/pdf",
                            use_container_width=True
                        )

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Generation failed: {str(e)}")


# ============================================================================
# MODE 4: IMPROVE EXISTING CV
# ============================================================================
elif "Improve Existing CV" in generation_mode:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #10b981 !important; margin: 0 0 0.5rem 0;">✨ AI-Powered CV Improvement</h3>
        <p style="color: #cbd5e1 !important; margin: 0;">
            Upload your existing CV and optionally a target job description. We'll analyze and improve:
            <br>• Weak or vague bullet points
            <br>• Missing quantification and impact
            <br>• ATS compatibility issues
            <br>• Skills alignment with job requirements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Current CV")
        existing_cv = st.file_uploader(
            "Upload CV to improve",
            type=['pdf', 'docx', 'txt'],
            help="We'll analyze and suggest improvements",
            key="improve_cv"
        )
    
    with col2:
        st.markdown("### 💼 Target Job (Optional)")
        target_jd = st.text_area(
            "Paste job description (optional)",
            height=200,
            placeholder="For best results, include the job description you're targeting...",
            help="We'll align your CV with this job's requirements",
            key="improve_jd"
        )
    
    st.markdown("---")
    
    if existing_cv:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("✨ Analyze & Improve CV", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Parse existing CV
                    status_text.info("📄 **Step 1/5:** Analyzing your current CV...")
                    progress_bar.progress(20)
                    
                    cv_parser = CVParser()
                    jd_parser = JDParser()
                    
                    # Parse CV
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{existing_cv.name.split('.')[-1]}") as tmp:
                        tmp.write(existing_cv.read())
                        tmp_path = tmp.name
                    
                    cv_data = cv_parser.parse(tmp_path)
                    os.unlink(tmp_path)
                    
                    if isinstance(cv_data, str):
                        cv_data = {'text': cv_data, 'sections': {}}
                    
                    st.success("✅ CV parsed successfully")
                    time.sleep(0.3)
                    
                    # Step 2: Parse JD if provided
                    jd_skills = []
                    if target_jd and target_jd.strip():
                        status_text.info("💼 **Step 2/5:** Analyzing job requirements...")
                        progress_bar.progress(40)
                        
                        jd_parsed = jd_parser.parse(target_jd)
                        
                        if isinstance(jd_parsed, dict):
                            jd_sections = jd_parsed.get('sections', jd_parsed)
                            for key in ['required_skills', 'skills', 'technical_skills']:
                                if key in jd_sections:
                                    skill_data = jd_sections[key]
                                    if isinstance(skill_data, list):
                                        jd_skills = skill_data
                                        break
                                    elif isinstance(skill_data, str):
                                        jd_skills = [s.strip() for s in skill_data.split(',') if s.strip()]
                                        break
                        
                        st.success(f"✅ Found {len(jd_skills)} required skills in JD")
                    else:
                        st.info("ℹ️ No JD provided - improving CV generally")
                        progress_bar.progress(40)
                    
                    time.sleep(0.3)
                    
                    # === Step 3: LLM-Powered Optimization ===
                    cv_skills = []
                    optimized_skills = []
                    enhanced_summary = "Professional with diverse technical expertise."
                    matched_skills = []
                    missing_jd_skills = []
                    
                    cv_sections = cv_data.get('sections', {})
                    for key in ['skills', 'technical_skills']:
                        if key in cv_sections:
                            skill_data = cv_sections[key]
                            if isinstance(skill_data, list):
                                cv_skills = skill_data
                                break
                            elif isinstance(skill_data, str):
                                cv_skills = [s.strip() for s in skill_data.split(',') if s.strip()]
                                break
                    
                    if target_jd and target_jd.strip():
                        status_text.info("🤖 **Step 3/5:** AI is improving your CV...")
                        progress_bar.progress(60)
                        
                        try:
                            optimizer = CVOptimizer()
                            
                            improvement = optimizer.improve_existing_cv_with_jd(
                                cv_text=cv_data.get('text', ''),
                                cv_skills=cv_skills,
                                jd_text=target_jd
                            )
                            
                            optimized_skills = improvement.get('optimized_skill_order', cv_skills)[:20]
                            enhanced_summary = improvement.get('enhanced_professional_summary', enhanced_summary)
                            matched_skills = improvement.get('matched_skills', [])
                            missing_jd_skills = improvement.get('missing_skills', [])
                            
                            st.success(f"✅ AI improved: {len(matched_skills)} skills matched, {len(missing_jd_skills)} gaps identified")
                            
                            with st.expander("🤖 AI Improvements Made", expanded=True):
                                st.info(improvement.get('improvement_summary', 'CV optimized successfully.'))
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**✅ Matched Skills:**")
                                    for skill in matched_skills[:5]:
                                        st.markdown(f"• {skill}")
                                with col2:
                                    st.markdown("**➕ Skills to Add:**")
                                    for skill in missing_jd_skills[:5]:
                                        st.markdown(f"• {skill}")
                        
                        except Exception as e:
                            st.warning(f"⚠️ LLM optimization unavailable: {str(e)}")
                            # Fallback
                            optimized_skills = cv_skills[:20]
                            matched_skills = [s for s in cv_skills if any(jd.lower() in s.lower() or s.lower() in jd.lower() for jd in jd_skills)]
                            missing_jd_skills = [s for s in jd_skills if not any(s.lower() in cv.lower() for cv in cv_skills)]
                    
                    else:
                        optimized_skills = cv_skills[:20]
                    
                    # Extract contact info (moved here so it's always defined)
                    cv_text = cv_data.get('text', '')
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text)
                    extracted_email = email_match.group(0) if email_match else 'your.email@example.com'
                    
                    phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]', cv_text)
                    extracted_phone = phone_match.group(0) if phone_match else '+1-234-567-8900'
                    
                    extracted_name = cv_sections.get('name', '') or (cv_text.split('\n')[0].strip()[:50] if cv_text else 'Your Name')
                    
                    # Step 4: Generate improved CV
                    status_text.info("✨ **Step 4/5:** Generating improved CV...")
                    progress_bar.progress(80)
                    
                    personal_info = {
                        'name': extracted_name,
                        'email': extracted_email,
                        'phone': extracted_phone,
                        'summary': enhanced_summary
                    }
                    
                    generator = CVGenerator()
                    doc = generator.generate_cv(
                        personal_info=personal_info,
                        skills=optimized_skills
                    )
                    
                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"CV_Improved_{timestamp}.docx"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                        doc.save(tmp.name)
                        with open(tmp.name, 'rb') as f:
                            cv_bytes = f.read()
                    os.unlink(tmp.name)
                    
                    st.session_state.cv_bytes_docx = cv_bytes
                    st.session_state.filename = filename
                    
                    # Generate HTML preview
                    try:
                        import mammoth
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            tmp.write(cv_bytes)
                            with open(tmp.name, 'rb') as f:
                                result = mammoth.convert_to_html(f)
                                st.session_state.cv_html = result.value
                        os.unlink(tmp.name)
                    except:
                        st.session_state.cv_html = None
                    
                    # Step 5: Show results
                    status_text.info("📋 **Step 5/5:** Showing optimization results...")
                    progress_bar.progress(95)
                    
                    if jd_skills:
                        st.markdown("### 🎯 Optimization Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Matched Skills", len(matched_skills))
                        with col2:
                            st.metric("Skills Added", len(missing_jd_skills))
                        with col3:
                            st.metric("Total Skills", len(optimized_skills))
                        
                        st.success("• Matched skills prioritized first")
                        st.success("• Missing JD skills added")
                        st.success("• Summary enhanced with job keywords")
                    else:
                        st.success("• CV reformatted and optimized for ATS")
                    
                    progress_bar.progress(100)
                    status_text.success("✅ **Improved CV generated successfully!**")
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Your improved CV is ready!")
                    st.balloons()
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ **CV improvement failed**")
                    st.error(f"Error: {str(e)}")
                    with st.expander("🔧 Debug Info"):
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        st.info("📤 Upload your CV above to get started with AI-powered analysis!")
    
  
# Help section
with st.expander("ℹ️ Tips & Best Practices"):
    st.markdown("""
    ### 📝 Manual Entry Tips
    
    **Work Experience:**
    - Use action verbs (Developed, Led, Implemented)
    - Quantify achievements (30% increase, 100K+ users)
    - Focus on impact, not just duties
    
    **Skills:**
    - List 10-20 relevant skills
    - Match job description keywords
    - Put strongest skills first
    
    **ATS Optimization:**
    - ✅ Standard fonts (Calibri 11pt)
    - ✅ Simple formatting
    - ✅ No tables/images
    - ✅ Standard section headers
    
    ---
    
    ### 🎯 Auto-Generation Tips
    
    **Best Results:**
    - Upload complete, up-to-date CV
    - Include full job description
    - Ensure skills are clearly listed
    
    **What Gets Highlighted:**
    - Matched skills shown first
    - Relevant experience prioritized
    - Education formatted for ATS
    
    ---
    
    ### 📄 Document Upload Tips
    
    **Supported Documents:**
    - PDF/DOCX existing CVs
    - Image scans of certificates
    - Transcripts with GPA
    - Multiple documents combine
    
    **OCR Works Best With:**
    - Clear, high-resolution scans
    - Well-lit photos
    - Typed text (not handwritten)
    """)
    