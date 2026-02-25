"""
CV Generator Page - Create ATS-Optimized CVs
"""
import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import os
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.cv_generator import CVGenerator

# Page config
st.set_page_config(
    page_title="CV Generator - CareerLens AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS (EXACTLY AS BEFORE)
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
    
    /* Selectbox */
    .stSelectbox select {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
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
    
    /* NEW: Preview box */
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

# Title (EXACTLY AS BEFORE)
st.markdown("""
<h1 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
           -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
    📝 CV Generator
</h1>
<p style="color: #94a3b8 !important; font-size: 1.25rem; margin-bottom: 2rem;">
    Create ATS-optimized CVs in minutes with professional formatting
</p>
""", unsafe_allow_html=True)

# Info box (EXACTLY AS BEFORE)
st.info("💡 Fill in your information below. The more detail you provide, the better your CV will be!")

# Create tabs for organized input (EXACTLY AS BEFORE)
tab1, tab2, tab3, tab4 = st.tabs(["📋 Personal Info", "💼 Experience", "🎓 Education & Skills", "🎨 Projects & Certifications"])

# TAB 1: Personal Information (EXACTLY AS BEFORE)
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
        placeholder="Results-driven Software Engineer with 5+ years of experience in building scalable applications. Expertise in Python, React, and cloud technologies. Proven track record of delivering high-quality solutions...",
        height=120
    )

# TAB 2: Work Experience (EXACTLY AS BEFORE)
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
                placeholder="• Developed REST APIs using FastAPI, serving 100K+ daily requests\n• Reduced infrastructure costs by 30% through optimization\n• Mentored 3 junior developers",
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

# TAB 3: Education & Skills (EXACTLY AS BEFORE)
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
        placeholder="Python, JavaScript, React, FastAPI, Docker, Kubernetes, AWS, PostgreSQL, MongoDB, Git, CI/CD, Agile",
        height=100
    )
    
    skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]

# TAB 4: Projects & Certifications (EXACTLY AS BEFORE)
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
                placeholder="Built a full-stack e-commerce platform serving 10K+ users...",
                height=80
            )
            project_tech = st.text_input(f"Technologies Used", key=f"project_tech_{i}", placeholder="FastAPI, React, PostgreSQL, Redis")
            
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
        placeholder="AWS Certified Solutions Architect - Associate (2022)\nGoogle Cloud Professional Cloud Developer (2021)\nCertified Kubernetes Administrator (CKA) (2023)",
        height=120
    )
    
    certifications = [cert.strip() for cert in certifications_input.split('\n') if cert.strip()]

# Generate CV Button (EXACTLY AS BEFORE)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    generate_button = st.button("🎯 Generate CV", type="primary", use_container_width=True)

if generate_button:
    # Validate required fields (EXACTLY AS BEFORE)
    if not name or not email or not phone:
        st.error("❌ Please fill in all required fields (Name, Email, Phone)")
        st.stop()
    
    if not experiences and not education:
        st.error("❌ Please add at least one work experience or education entry")
        st.stop()
    
    # Prepare personal info (EXACTLY AS BEFORE)
    personal_info = {
        'name': name,
        'email': email,
        'phone': phone,
        'location': location if location else None,
        'linkedin': linkedin if linkedin else None,
        'github': github if github else None,
        'summary': summary if summary else None
    }
    
    # Generate CV
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
            
            # Save DOCX to bytes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CV_{name.replace(' ', '_')}_{timestamp}.docx"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                doc.save(tmp_file.name)
                tmp_docx_path = tmp_file.name
            
            with open(tmp_docx_path, 'rb') as file:
                cv_bytes_docx = file.read()
            
            # Store in session state
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
            
            # Clean up temp DOCX
            os.unlink(tmp_docx_path)
            
            st.success("✅ CV generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate CV: {str(e)}")
            st.exception(e)

# Display results if generated
if st.session_state.cv_bytes_docx:
    st.markdown("---")
    st.markdown("## 📊 Your Generated CV")
    
    # Summary (EXACTLY AS BEFORE)
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Experience Entries", len(experiences))
    with summary_col2:
        st.metric("Education Entries", len(education))
    with summary_col3:
        st.metric("Skills Listed", len(skills))
    with summary_col4:
        st.metric("Projects", len(projects))
    
    st.markdown("---")
    
    # NEW: Preview and Download Section
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
        
        # DOCX Download (EXISTING FEATURE)
        st.download_button(
            label="📄 Download DOCX",
            data=st.session_state.cv_bytes_docx,
            file_name=st.session_state.filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
        # NEW: PDF Download
        pdf_button = st.button("🔄 Generate PDF", use_container_width=True)
        
        if pdf_button:
            with st.spinner("Converting to PDF..."):
                try:
                    from docx2pdf import convert
                    
                    # Save DOCX temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
                        tmp_docx.write(st.session_state.cv_bytes_docx)
                        tmp_docx_path = tmp_docx.name
                    
                    # Convert to PDF
                    pdf_path = tmp_docx_path.replace('.docx', '.pdf')
                    convert(tmp_docx_path, pdf_path)
                    
                    # Read PDF
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    # Clean up
                    os.unlink(tmp_docx_path)
                    os.unlink(pdf_path)
                    
                    # Store PDF
                    st.session_state.cv_bytes_pdf = pdf_bytes
                    st.session_state.pdf_filename = st.session_state.filename.replace('.docx', '.pdf')
                    
                    st.success("✅ PDF generated!")
                    st.rerun()
                    
                except ImportError:
                    st.error("❌ PDF conversion not available on your system. Please download DOCX instead.")
                except Exception as e:
                    st.error(f"❌ PDF conversion failed: {str(e)}")
        
        # Show PDF download if available
        if 'cv_bytes_pdf' in st.session_state and st.session_state.cv_bytes_pdf:
            st.download_button(
                label="📕 Download PDF",
                data=st.session_state.cv_bytes_pdf,
                file_name=st.session_state.pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )
        
        st.markdown("---")
        
        st.info(f"💾 **{st.session_state.filename}**")
    
    # Tips (EXACTLY AS BEFORE)
    with st.expander("💡 Tips for using your new CV"):
        st.markdown("""
        ### Next Steps:
        
        1. **Review Your CV** 📄
           - Open the downloaded DOCX file
           - Check for any typos or formatting issues
           - Ensure all information is accurate
        
        2. **Customize for Each Job** 🎯
           - Tailor your summary to the specific role
           - Reorder skills to match job requirements
           - Highlight relevant experience
        
        3. **ATS-Friendly Format** ✅
           - Uses standard Calibri font (11pt)
           - Simple, clean formatting
           - No graphics or tables that confuse ATS
           - Proper section headers
        
        4. **Test Your CV** 🧪
           - Use our CV-JD Matcher to test against job descriptions
           - Aim for 75%+ match score
           - Address any missing skills
        
        5. **Keep It Updated** 🔄
           - Update regularly with new achievements
           - Add new skills as you learn them
           - Remove outdated information
        """)

# Help section (EXACTLY AS BEFORE)
with st.expander("ℹ️ CV Generation Tips"):
    st.markdown("""
    ### Creating an Effective CV
    
    **Personal Information:**
    - Use a professional email address
    - Include LinkedIn profile (increases credibility by 40%)
    - Keep phone number format consistent
    
    **Professional Summary:**
    - 2-3 sentences maximum
    - Highlight key achievements with numbers
    - Tailor to your target role
    
    **Work Experience:**
    - Use action verbs (Developed, Implemented, Led)
    - Quantify achievements (percentages, numbers)
    - Focus on impact, not just responsibilities
    - Order: Most recent first
    
    **Skills:**
    - List 10-20 relevant skills
    - Mix of technical and soft skills
    - Match job description keywords
    - Group related skills together
    
    **Projects:**
    - Include 2-3 impressive projects
    - Show real-world impact
    - Mention technologies used
    - Link to GitHub if available
    
    **Certifications:**
    - List relevant certifications only
    - Include year obtained
    - Show expiration dates if applicable
    
    ### ATS Optimization:
    
    ✅ **DO:**
    - Use standard section headers
    - Use simple formatting
    - Include keywords from job description
    - Use standard fonts (Calibri, Arial)
    - Save as .docx format
    
    ❌ **DON'T:**
    - Use headers/footers
    - Insert tables or text boxes
    - Use images or graphics
    - Use unusual fonts
    - Use columns (in some ATS systems)
    """)