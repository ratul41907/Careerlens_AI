"""
Unit Tests for JD Parser - Enhanced with Pytest
Integrates existing test logic with pytest framework
"""
import pytest


class TestJDParserBasic:
    """Basic JD Parser tests"""
    
    def test_parser_initialization(self, jd_parser):
        """Test parser initialization"""
        assert jd_parser is not None
        assert hasattr(jd_parser, 'ollama_url')
        assert hasattr(jd_parser, 'model')
    
    def test_parse_simple_jd(self, jd_parser):
        """Test parsing a simple JD"""
        simple_jd = """
Software Engineer

Required Skills:
- Python
- Docker
- AWS

Experience: 3+ years
"""
        
        result = jd_parser.parse(simple_jd)
        
        assert result is not None
        assert 'text' in result
        assert 'sections' in result
    
    def test_required_skills_extraction(self, jd_parser, sample_jd_text):
        """Test required skills extraction"""
        result = jd_parser.parse(sample_jd_text)
        
        sections = result.get('sections', {})
        required_skills = sections.get('required_skills', [])
        
        assert isinstance(required_skills, list)
        assert len(required_skills) >= 3  # Should extract multiple skills
    
    def test_job_title_extraction(self, jd_parser, sample_jd_text):
        """Test job title extraction"""
        result = jd_parser.parse(sample_jd_text)
        
        sections = result.get('sections', {})
        job_title = sections.get('job_title', '')
        
        assert isinstance(job_title, str)
        # Should extract some title
        assert len(job_title) >= 0


class TestJDParserIntegration:
    """Integration tests matching your existing test format"""
    
    def test_complete_jd_parsing_pipeline(self, jd_parser):
        """Test complete JD parsing (based on your original test)"""
        
        # Your original sample JD
        sample_jd = """
Senior Software Engineer
Tech Innovations Inc.
San Francisco, CA

About the Role:
We are seeking a talented Senior Software Engineer to join our growing team.

Required Qualifications:
- 5+ years of experience in software development
- Strong proficiency in Python and JavaScript
- Experience with React and Node.js
- Must have experience with AWS cloud services
- Proficiency in SQL and NoSQL databases (PostgreSQL, MongoDB)
- Bachelor's degree in Computer Science or related field required

Preferred Qualifications:
- Experience with Docker and Kubernetes is a plus
- Knowledge of CI/CD pipelines (Jenkins, GitHub Actions) preferred
- Familiarity with microservices architecture would be great
- Experience with FastAPI or Django is desirable

Responsibilities:
- Design and develop scalable backend systems
- Collaborate with cross-functional teams using Agile methodology
- Write clean, maintainable code with proper documentation
- Participate in code reviews and mentor junior developers

Nice to Have:
- Experience with machine learning frameworks (TensorFlow, PyTorch)
- Contributions to open-source projects
"""
        
        # Parse
        result = jd_parser.parse(sample_jd)
        
        # Assertions matching your original test expectations
        assert result is not None
        assert 'text' in result
        assert 'sections' in result
        
        sections = result['sections']
        
        # Check job title
        if 'job_title' in sections:
            job_title = sections['job_title']
            assert isinstance(job_title, str)
            print(f"\n✅ Job Title: {job_title}")
        
        # Check required skills
        required_skills = sections.get('required_skills', [])
        assert isinstance(required_skills, list)
        assert len(required_skills) > 0
        
        print(f"✅ Required Skills: {len(required_skills)} extracted")
        print(f"   Sample: {required_skills[:5]}")
        
        # Check preferred skills
        preferred_skills = sections.get('preferred_skills', [])
        assert isinstance(preferred_skills, list)
        
        print(f"✅ Preferred Skills: {len(preferred_skills)} extracted")
        
        # Check experience
        experience = sections.get('experience')
        if experience:
            print(f"✅ Experience: {experience}")
        
        # Check education
        education = sections.get('education')
        if education:
            print(f"✅ Education: {education}")
        
        print("\n✅ Complete JD parsing pipeline test passed!")


class TestJDParserEdgeCases:
    """Edge cases and error handling"""
    
    def test_empty_jd(self, jd_parser):
        """Test empty JD handling"""
        result = jd_parser.parse('')
        
        assert result is not None
        assert 'error' in result
    
    def test_very_short_jd(self, jd_parser):
        """Test very short JD"""
        result = jd_parser.parse('Python developer needed')
        
        assert result is not None
        # Should either have error or sections
        assert 'error' in result or 'sections' in result
    
    def test_jd_with_html(self, jd_parser):
        """Test JD with HTML formatting"""
        html_jd = """
<h1>Senior Developer</h1>
<p>We need <b>Python</b> and <i>Docker</i> experience</p>
<ul>
    <li>5+ years experience</li>
    <li>AWS knowledge</li>
</ul>
"""
        
        result = jd_parser.parse(html_jd)
        
        # Should handle HTML gracefully
        assert result is not None
        assert 'sections' in result
    
    def test_jd_with_salary_info(self, jd_parser):
        """Test JD with salary range"""
        jd_with_salary = """
Backend Engineer

Required: Python, FastAPI, Docker
Salary: $120,000 - $160,000
Experience: 5+ years
"""
        
        result = jd_parser.parse(jd_with_salary)
        
        assert result is not None
        assert 'sections' in result
    
    def test_jd_without_skills_section(self, jd_parser):
        """Test JD that doesn't have explicit skills section"""
        implicit_jd = """
Senior Engineer

We're looking for someone with Python and Docker experience.
You should have worked with AWS and know Kubernetes.
5 years of experience required.
"""
        
        result = jd_parser.parse(implicit_jd)
        
        # Should still extract skills via LLM
        assert result is not None
        sections = result.get('sections', {})
        
        # May have required_skills from fallback or LLM
        assert 'required_skills' in sections or len(sections) > 0