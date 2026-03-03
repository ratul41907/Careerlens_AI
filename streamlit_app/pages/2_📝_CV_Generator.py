"""
CV Generator Page - Create ATS-Optimized CVs
UPDATE_01: Auto-generate from JD + Document upload
"""
import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import os
import tempfile
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.cv_generator import CVGenerator
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine

# Page config
st.set_page_config(
    page_title="CV Generator - CareerLens AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if 'jd_skills' not in st.session_state:
    st.session_state.jd_skills = []
if 'target_jd' not in st.session_state:
    st.session_state.target_jd = None

# Title
st.markdown("""
<h1 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
           -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
    📝 Smart CV Generator
</h1>
<p style="color: #94a3b8 !important; font-size: 1.25rem; margin-bottom: 2rem;">
    🆕 Auto-optimize for job descriptions • Upload documents • ATS-friendly formatting
</p>
""", unsafe_allow_html=True)

# NEW: Generation mode selector
st.markdown("### 🎯 Choose Generation Mode")

generation_mode = st.radio(
    "How would you like to create your CV?",
    [
        "📝 Manual Entry (Fill forms manually)",
        "🎯 Auto-Generate from Job Description (Smart matching)",
        "📄 Extract from Documents (Upload certificates/transcripts)"
    ],
    label_visibility="collapsed"
)

st.markdown("---")

# ============================================================================
# MODE 1: MANUAL ENTRY (ORIGINAL - KEEPING EXACTLY AS BEFORE)
# ============================================================================
if "Manual Entry" in generation_mode:
    st.info("💡 Fill in your information below. The more detail you provide, the better your CV will be!")
    
    # Create tabs for organized input
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Personal Info", "💼 Experience", "🎓 Education & Skills", "🎨 Projects & Certifications"])
    
    # TAB 1: Personal Information
    with tab1:
        st.markdown("### 📋 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="John Doe")
            email = st.text_input("Email *", placeholder="john.doe@email.com")
            phone = st.text_input("Phone Number *", placeholder="+1-234-567-8900")
        
        with col2:
            location = st.text_input("Location", placeholder="San Francisco, CA")
            linkedin = st.text_input("LinkedIn URL", placeholder="linkedin.com/in/johndoe")
            github = st.text_input("GitHub URL (optional)", placeholder="github.com/johndoe")
        
        st.markdown("---")
        
        st.markdown("### 📄 Professional Summary")
        summary = st.text_area(
            "Write a brief 2-3 sentence summary highlighting your key qualifications",
            placeholder="Results-driven Software Engineer with 5+ years of experience in building scalable applications...",
            height=120
        )
    
    # TAB 2: Work Experience
    with tab2:
        st.markdown("### 💼 Work Experience")
        
        num_experiences = st.number_input("Number of work experiences to add", min_value=0, max_value=10, value=2, step=1)
        
        experiences = []
        
        for i in range(int(num_experiences)):
            with st.expander(f"Experience #{i+1}", expanded=(i==0)):
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    job_title = st.text_input(f"Job Title *", key=f"job_title_{i}", placeholder="Senior Software Engineer")
                    company = st.text_input(f"Company *", key=f"company_{i}", placeholder="Tech Corp")
                
                with exp_col2:
                    start_date = st.text_input(f"Start Date", key=f"start_date_{i}", placeholder="January 2021")
                    end_date = st.text_input(f"End Date", key=f"end_date_{i}", placeholder="Present")
                
                st.markdown("**Key Achievements (one per line):**")
                achievements = st.text_area(
                    "Enter your achievements/responsibilities",
                    key=f"achievements_{i}",
                    placeholder="• Developed REST APIs using FastAPI\n• Reduced costs by 30%\n• Mentored 3 developers",
                    height=150
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
                    degree = st.text_input(f"Degree *", key=f"degree_{i}", placeholder="Bachelor of Science in Computer Science")
                    institution = st.text_input(f"Institution *", key=f"institution_{i}", placeholder="University of Technology")
                
                with edu_col2:
                    year = st.text_input(f"Year/Duration", key=f"year_{i}", placeholder="2015 - 2019")
                    gpa = st.text_input(f"GPA (optional)", key=f"gpa_{i}", placeholder="3.8/4.0")
                
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
            placeholder="Python, JavaScript, React, FastAPI, Docker, Kubernetes, AWS",
            height=100
        )
        
        skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
    
    # TAB 4: Projects & Certifications
    with tab4:
        st.markdown("### 🎨 Projects")
        
        num_projects = st.number_input("Number of projects", min_value=0, max_value=10, value=2, step=1)
        
        projects = []
        
        for i in range(int(num_projects)):
            with st.expander(f"Project #{i+1}", expanded=(i==0)):
                project_name = st.text_input(f"Project Name", key=f"project_name_{i}", placeholder="E-Commerce Platform")
                project_desc = st.text_area(
                    f"Description",
                    key=f"project_desc_{i}",
                    placeholder="Built a full-stack e-commerce platform...",
                    height=80
                )
                project_tech = st.text_input(f"Technologies Used", key=f"project_tech_{i}", placeholder="FastAPI, React, PostgreSQL")
                
                if project_name and project_desc:
                    projects.append({
                        'name': project_name,
                        'description': project_desc,
                        'technologies': project_tech
                    })
        
        st.markdown("---")
        st.markdown("### 🏆 Certifications")
        
        certifications_input = st.text_area(
            "Enter certifications (one per line)",
            placeholder="AWS Certified Solutions Architect (2022)\nCertified Kubernetes Administrator (2023)",
            height=120
        )
        
        certifications = [cert.strip() for cert in certifications_input.split('\n') if cert.strip()]

# ============================================================================
# MODE 2: AUTO-GENERATE FROM JOB DESCRIPTION (NEW)
# ============================================================================
elif "Auto-Generate" in generation_mode:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #10b981 !important; margin: 0 0 0.5rem 0;">🎯 Smart CV Auto-Generation</h3>
        <p style="color: #cbd5e1 !important; margin: 0;">
            Paste a job description and your existing CV. We'll automatically:
            <br>• Extract required skills from the JD
            <br>• Highlight YOUR matching skills prominently
            <br>• Re-order sections to emphasize relevant experience
            <br>• Generate ATS-optimized CV tailored to the role
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Existing CV")
        
        existing_cv_file = st.file_uploader(
            "Upload your current CV",
            type=['pdf', 'docx', 'txt'],
            help="Upload your existing CV - we'll extract and optimize it"
        )
        
        st.markdown("**OR paste your CV text:**")
        existing_cv_text = st.text_area(
            "Paste CV",
            height=300,
            placeholder="Paste your current CV here...",
            key="existing_cv"
        )
    
    with col2:
        st.markdown("### 💼 Target Job Description")
        
        target_jd_text = st.text_area(
            "Paste the job description you're applying for",
            height=460,
            placeholder="""Paste full job description here...

Example:
Senior Software Engineer

Required Skills:
- Python, FastAPI
- Docker, Kubernetes
- AWS cloud services
- 5+ years experience

We'll automatically highlight YOUR matching skills!""",
            key="target_jd"
        )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        auto_generate_button = st.button("🤖 Auto-Generate Optimized CV", type="primary", use_container_width=True)
    
    if auto_generate_button:
        if not target_jd_text:
            st.error("❌ Please provide a target job description")
            st.stop()
        
        if not existing_cv_file and not existing_cv_text:
            st.error("❌ Please upload your CV or paste CV text")
            st.stop()
        
        with st.spinner("🧠 Analyzing job description and your CV..."):
            try:
                # Parse existing CV
                cv_parser = CVParser()
                if existing_cv_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{existing_cv_file.name.split('.')[-1]}") as tmp:
                        tmp.write(existing_cv_file.read())
                        cv_data = cv_parser.parse(tmp.name)
                    os.unlink(tmp.name)
                else:
                    cv_data = {'text': existing_cv_text, 'sections': cv_parser._segment_sections(existing_cv_text)}
                
                # Parse JD
                jd_parser = JDParser()
                jd_data = jd_parser.parse(target_jd_text)
                
                # Match CV to JD
                embedding_engine = EmbeddingEngine()
                scoring_engine = ScoringEngine(embedding_engine)
                match_result = scoring_engine.compute_match_score(cv_data, jd_data)
                
                st.success(f"✅ Analysis complete! Match score: {match_result['overall_percentage']}")
                
                # Extract matched and missing skills
                required_skills_details = match_result['breakdown']['required_skills']['details']['skills']
                matched_skills = [s['skill'] for s in required_skills_details if s['matched']]
                missing_skills_list = [s['skill'] for s in required_skills_details if not s['matched']]
                
                # Show what we found
                st.markdown("### 🎯 Analysis Results")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown(f"#### ✅ Your Matching Skills ({len(matched_skills)})")
                    if matched_skills:
                        for skill in matched_skills[:10]:
                            st.markdown(f"• **{skill}** (will be highlighted)")
                
                with analysis_col2:
                    st.markdown(f"#### ⚠️ Missing Skills ({len(missing_skills_list)})")
                    if missing_skills_list:
                        for skill in missing_skills_list[:10]:
                            st.markdown(f"• {skill} (add if you have it)")
                
                st.info("💡 We'll generate a CV that prominently highlights your matching skills at the top!")
                
                # PROPERLY EXTRACT from parsed CV
                cv_sections = cv_data.get('sections', {})
                
                # Extract personal info
                header_text = cv_sections.get('header', '')
                
                # Extract email
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', header_text)
                extracted_email = email_match.group(0) if email_match else None
                
                # Extract phone
                phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]', header_text)
                extracted_phone = phone_match.group(0) if phone_match else None
                
                # Extract name (first non-empty line)
                lines = [l.strip() for l in header_text.split('\n') if l.strip()]
                extracted_name = lines[0] if lines else None
                
                # Store extracted data
                st.session_state.extracted_name = extracted_name
                st.session_state.extracted_email = extracted_email
                st.session_state.extracted_phone = extracted_phone
                st.session_state.extracted_experience = cv_sections.get('experience', '')
                st.session_state.extracted_education = cv_sections.get('education', '')
                st.session_state.extracted_skills = cv_sections.get('skills', '')
                st.session_state.extracted_projects = cv_sections.get('projects', '')
                
                # Store for generation
                st.session_state.jd_skills = jd_data.get('required_skills', []) + jd_data.get('preferred_skills', [])
                st.session_state.matched_skills = matched_skills
                st.session_state.target_jd = target_jd_text
                st.session_state.parsed_cv = cv_data
                st.session_state.jd_data = jd_data
                
                # Set flag to generate
                st.session_state.ready_to_generate_auto = True
                
                st.success("✅ Extracted complete CV data successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to analyze: {str(e)}")
                st.exception(e)
    
    # If ready, show generate button
    if st.session_state.get('ready_to_generate_auto', False):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Generate Optimized CV Now", type="primary", use_container_width=True):
                with st.spinner("📝 Generating your optimized CV..."):
                    try:
                        matched_skills = st.session_state.matched_skills
                        
                        # Create skills list with matched ones first
                        all_skills = matched_skills + [s for s in st.session_state.jd_skills if s not in matched_skills]
                        
                        # Use EXTRACTED data from CV
                        personal_info = {
                            'name': st.session_state.get('extracted_name', 'Your Name'),
                            'email': st.session_state.get('extracted_email', 'your.email@example.com'),
                            'phone': st.session_state.get('extracted_phone', '+1-234-567-8900'),
                            'summary': f"Professional with proven experience in {', '.join(matched_skills[:5])}. {len(matched_skills)}/{len(st.session_state.jd_skills)} required skills matched."
                        }
                        
                        # Parse experience from text
                        experience_text = st.session_state.get('extracted_experience', '')
                        experience_list = []
                        if experience_text:
                            # Split by common patterns
                            exp_sections = re.split(r'\n(?=[A-Z][a-z]+ (?:Engineer|Developer|Manager|Analyst))', experience_text)
                            for exp in exp_sections[:3]:  # Top 3 experiences
                                lines = [l.strip() for l in exp.split('\n') if l.strip()]
                                if len(lines) >= 2:
                                    experience_list.append({
                                        'title': lines[0],
                                        'company': lines[1] if len(lines) > 1 else 'Company',
                                        'duration': 'Present',
                                        'bullets': [l.lstrip('•-*').strip() for l in lines[2:] if l.strip()][:4]
                                    })
                        
                        # Parse education
                        education_text = st.session_state.get('extracted_education', '')
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
                        
                        # Parse projects
                        projects_text = st.session_state.get('extracted_projects', '')
                        projects_list = []
                        if projects_text:
                            project_sections = re.split(r'\n(?=[A-Z])', projects_text)
                            for proj in project_sections[:3]:
                                lines = [l.strip() for l in proj.split('\n') if l.strip()]
                                if lines:
                                    projects_list.append({
                                        'name': lines[0],
                                        'description': lines[1] if len(lines) > 1 else '',
                                        'technologies': ', '.join(matched_skills[:5])
                                    })
                        
                        # Generate CV with ALL extracted data
                        generator = CVGenerator()
                        doc = generator.generate_cv(
                            personal_info=personal_info,
                            experience=experience_list if experience_list else None,
                            education=education_list if education_list else None,
                            skills=all_skills[:20],  # Matched skills first
                            projects=projects_list if projects_list else None
                        )
                        
                        # Save
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"CV_Optimized_For_Role_{timestamp}.docx"
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            doc.save(tmp.name)
                            with open(tmp.name, 'rb') as f:
                                cv_bytes = f.read()
                        
                        os.unlink(tmp.name)
                        
                        st.session_state.cv_bytes_docx = cv_bytes
                        st.session_state.filename = filename
                        
                        # Generate preview
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
                        
                        st.success("✅ Optimized CV generated!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
                        st.exception(e)

# ============================================================================
# MODE 3: EXTRACT FROM DOCUMENTS (NEW)
# ============================================================================
elif "Extract from Documents" in generation_mode:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #10b981 !important; margin: 0 0 0.5rem 0;">📄 Document-Based CV Generation</h3>
        <p style="color: #cbd5e1 !important; margin: 0;">
            Upload your documents and we'll automatically extract information:
            <br>• 📜 Certificates → Certifications section
            <br>• 🎓 Transcripts/Marksheets → Education + GPA
            <br>• 📄 Existing CV → All sections
            <br>• 🏆 Award letters → Achievements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Your Documents")
    
    uploaded_docs = st.file_uploader(
        "Upload multiple documents (certificates, transcripts, existing CV, etc.)",
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
            with st.spinner("📄 Extracting information from documents..."):
                try:
                    extracted_data = {
                        'education': [],
                        'certifications': [],
                        'skills': [],
                        'experience': []
                    }
                    
                    cv_parser = CVParser()
                    
                    for doc in uploaded_docs:
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.name.split('.')[-1]}") as tmp:
                            tmp.write(doc.read())
                            tmp_path = tmp.name
                        
                        # Parse based on file type
                        if doc.name.lower().endswith(('.pdf', '.docx', '.txt')):
                            doc_data = cv_parser.parse(tmp_path)
                            
                            # Extract education
                            if 'education' in doc_data.get('sections', {}):
                                extracted_data['education'].append(doc_data['sections']['education'])
                            
                            # Extract skills
                            if 'skills' in doc_data.get('sections', {}):
                                skills_text = doc_data['sections']['skills']
                                skills = [s.strip() for s in skills_text.replace('\n', ',').split(',') if s.strip()]
                                extracted_data['skills'].extend(skills)
                        
                        # OCR for images (certificates/transcripts)
                        elif doc.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            try:
                                from src.validation.eligibility_validator import EligibilityValidator
                                validator = EligibilityValidator()
                                text = validator._extract_text_from_image(tmp_path)
                                
                                # Look for degree/GPA
                                if 'bachelor' in text.lower() or 'master' in text.lower() or 'gpa' in text.lower():
                                    extracted_data['education'].append(text[:500])
                                
                                # Look for certifications
                                if 'certified' in text.lower() or 'certificate' in text.lower():
                                    extracted_data['certifications'].append(text[:200])
                            except:
                                pass
                        
                        os.unlink(tmp_path)
                    
                    # Show extracted info
                    st.success("✅ Extraction complete!")
                    
                    st.markdown("### 📊 Extracted Information")
                    
                    if extracted_data['education']:
                        st.markdown("#### 🎓 Education Found")
                        for edu in extracted_data['education'][:3]:
                            st.info(edu[:200] + "...")
                    
                    if extracted_data['certifications']:
                        st.markdown("#### 🏆 Certifications Found")
                        for cert in extracted_data['certifications'][:5]:
                            st.info(cert)
                    
                    if extracted_data['skills']:
                        st.markdown("#### 🔧 Skills Found")
                        unique_skills = list(set(extracted_data['skills']))[:20]
                        st.write(", ".join(unique_skills))
                    
                    st.info("💡 Review the extracted information above. You can now generate a CV or manually edit before generating.")
                    
                    # Store extracted data
                    st.session_state.extracted_data = extracted_data
                    st.session_state.ready_to_generate_from_docs = True
                    
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
                    st.exception(e)
    
    # If extracted, show generate button
    if st.session_state.get('ready_to_generate_from_docs', False):
        st.markdown("---")
        
        # Allow minor edits
        st.markdown("### ✏️ Quick Edits (Optional)")
        
        col1, col2 = st.columns(2)
        with col1:
            doc_name = st.text_input("Your Name", placeholder="John Doe")
            doc_email = st.text_input("Email", placeholder="john@email.com")
        with col2:
            doc_phone = st.text_input("Phone", placeholder="+1-234-567-8900")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📝 Generate CV from Extracted Data", type="primary", use_container_width=True):
                with st.spinner("📝 Generating your CV..."):
                    try:
                        extracted = st.session_state.extracted_data
                        
                        personal_info = {
                            'name': doc_name if doc_name else 'Your Name',
                            'email': doc_email if doc_email else 'email@example.com',
                            'phone': doc_phone if doc_phone else '+1-234-567-8900'
                        }
                        
                        generator = CVGenerator()
                        doc = generator.generate_cv(
                            personal_info=personal_info,
                            skills=extracted.get('skills', [])[:20],
                            certifications=extracted.get('certifications', [])[:10]
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"CV_From_Documents_{timestamp}.docx"
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            doc.save(tmp.name)
                            with open(tmp.name, 'rb') as f:
                                cv_bytes = f.read()
                        
                        os.unlink(tmp.name)
                        
                        st.session_state.cv_bytes_docx = cv_bytes
                        st.session_state.filename = filename
                        
                        # Generate preview
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
                        
                        st.success("✅ CV generated from your documents!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
                        st.exception(e)

# ============================================================================
# GENERATE CV BUTTON (FOR MANUAL MODE)
# ============================================================================
if "Manual Entry" in generation_mode:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button("🎯 Generate CV", type="primary", use_container_width=True)
    
    if generate_button:
        # Validate
        if not name or not email or not phone:
            st.error("❌ Please fill in all required fields (Name, Email, Phone)")
            st.stop()
        
        if not experiences and not education:
            st.error("❌ Please add at least one work experience or education entry")
            st.stop()
        
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
        
        # Generate
        with st.spinner("📝 Generating your ATS-optimized CV..."):
            try:
                generator = CVGenerator()
                
                doc = generator.generate_cv(
                    personal_info=personal_info,
                    experience=experiences if experiences else None,
                    education=education if education else None,
                    skills=skills if skills else None,
                    projects=projects if projects else None,
                    certifications=certifications if certifications else None
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"CV_{name.replace(' ', '_')}_{timestamp}.docx"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    doc.save(tmp_file.name)
                    tmp_docx_path = tmp_file.name
                
                with open(tmp_docx_path, 'rb') as file:
                    cv_bytes_docx = file.read()
                
                st.session_state.generated_doc = doc
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
                
                st.success("✅ CV generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to generate CV: {str(e)}")
                st.exception(e)

# ============================================================================
# DISPLAY RESULTS (ALL MODES)
# ============================================================================
if st.session_state.cv_bytes_docx:
    st.markdown("---")
    st.markdown("## 📊 Your Generated CV")
    
    # Summary
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Generation Mode", "✅ Complete")
    with summary_col2:
        st.metric("Format", "DOCX")
    with summary_col3:
        st.metric("ATS-Optimized", "Yes")
    with summary_col4:
        st.metric("File Size", f"{len(st.session_state.cv_bytes_docx) // 1024} KB")
    
    st.markdown("---")
    
    # Preview and Download
    preview_col, download_col = st.columns([2, 1])
    
    with preview_col:
        st.markdown("### 👁️ CV Preview")
        
        if st.session_state.cv_html:
            st.markdown(f"""
            <div class="cv-preview">
                {st.session_state.cv_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Preview not available. Download the CV to view it in Word or Google Docs.")
    
    with download_col:
        st.markdown("### 📥 Download Options")
        
        # DOCX Download
        st.download_button(
            label="📄 Download DOCX",
            data=st.session_state.cv_bytes_docx,
            file_name=st.session_state.filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
        st.markdown("---")
        
        st.info(f"💾 **{st.session_state.filename}**")
        
        # Quick actions
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Generate Another CV", use_container_width=True):
            # Clear session state
            for key in ['cv_bytes_docx', 'cv_html', 'filename', 'ready_to_generate_auto', 'ready_to_generate_from_docs']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("📊 Test with CV Matcher", use_container_width=True):
            st.switch_page("pages/1_📊_CV_Matcher.py")

# Help section
with st.expander("ℹ️ CV Generation Tips & Best Practices"):
    st.markdown("""
    ### 🎯 Auto-Generation from JD (NEW!)
    
    **How it works:**
    1. Upload your existing CV
    2. Paste target job description
    3. AI analyzes and matches skills
    4. Generated CV highlights YOUR matching skills first
    5. Employers can easily spot what they're looking for!
    
    **Benefits:**
    - ✅ Saves time (no manual editing)
    - ✅ Highlights relevant skills prominently
    - ✅ Better ATS matching
    - ✅ Customized for each job application
    
    ---
    
    ### 📄 Document Upload (NEW!)
    
    **Supported documents:**
    - 📜 Certificates (PDF, images)
    - 🎓 Transcripts/Marksheets (PDF, images)
    - 📄 Existing CVs (DOCX, PDF, TXT)
    - 🏆 Award letters
    
    **What we extract:**
    - Education details + GPA
    - Certifications with dates
    - Skills from all documents
    - Work experience from existing CVs
    
    ---
    
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
    """)