"""
Unit Tests for CV Parser - Enhanced with Pytest
Integrates existing test logic with pytest framework
"""
import pytest
from pathlib import Path
import tempfile
import os


class TestCVParserBasic:
    """Basic CV Parser tests"""
    
    def test_parser_initialization(self, cv_parser):
        """Test that parser initializes correctly"""
        assert cv_parser is not None
        assert hasattr(cv_parser, 'ollama_url')
        assert hasattr(cv_parser, 'model')
    
    def test_parse_txt_file(self, cv_parser, sample_cv_text):
        """Test parsing a TXT file"""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_text)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            
            assert result is not None
            assert 'text' in result
            assert 'sections' in result
            assert len(result['text']) > 0
        finally:
            os.remove(temp_path)
    
    def test_section_extraction(self, cv_parser, sample_cv_text):
        """Test that sections are extracted"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_text)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            sections = result.get('sections', {})
            
            # Should extract multiple sections
            assert len(sections) > 0
            
            # Common sections that should be present
            common_sections = ['skills', 'experience', 'education']
            extracted_sections = [s for s in common_sections if s in sections]
            
            # At least some sections should be extracted
            assert len(extracted_sections) >= 1
        finally:
            os.remove(temp_path)
    
    def test_email_extraction(self, cv_parser, sample_cv_text):
        """Test email extraction from CV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_text)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            sections = result.get('sections', {})
            email = sections.get('email', '')
            
            # Email should be extracted or be empty string
            assert isinstance(email, str)
            if email:
                assert '@' in email
        finally:
            os.remove(temp_path)
    
    def test_skills_extraction(self, cv_parser, sample_cv_text):
        """Test skills extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_text)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            sections = result.get('sections', {})
            skills = sections.get('skills', [])
            
            assert isinstance(skills, list)
            # Should extract at least some skills from the sample CV
            # (Python, FastAPI, Docker are clearly mentioned)
            assert len(skills) >= 3
        finally:
            os.remove(temp_path)


class TestCVParserIntegration:
    """Integration tests matching your existing test format"""
    
    def test_complete_cv_parsing_pipeline(self, cv_parser):
        """Test complete parsing pipeline (based on your original test)"""
        
        # Your original sample CV
        sample_cv_content = """
JOHN DOE
Software Engineer
john.doe@email.com | +1-234-567-8900 | linkedin.com/in/johndoe

SUMMARY
Experienced software engineer with 5 years in full-stack development.
Proficient in Python, JavaScript, and cloud technologies.

EXPERIENCE

Senior Software Engineer | Tech Corp | 2021 - Present
- Built RESTful APIs using FastAPI and PostgreSQL
- Deployed microservices on AWS using Docker and Kubernetes
- Improved system performance by 40% through optimization

Software Engineer | StartupXYZ | 2019 - 2021
- Developed React-based dashboard for data visualization
- Implemented CI/CD pipelines with GitHub Actions
- Collaborated with team of 8 developers using Agile methodology

EDUCATION

Bachelor of Science in Computer Science
University of Technology | 2015 - 2019
GPA: 3.8/4.0

SKILLS

Languages: Python, JavaScript, TypeScript, SQL
Frameworks: FastAPI, React, Node.js, Django
Tools: Docker, Kubernetes, AWS, Git, PostgreSQL

PROJECTS

E-commerce Platform
- Built scalable backend handling 10K requests/second
- Tech stack: FastAPI, Redis, PostgreSQL, Docker

CERTIFICATIONS

AWS Certified Solutions Architect
Issued: 2022
"""
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_content.strip())
            temp_path = f.name
        
        try:
            # Parse
            result = cv_parser.parse(temp_path)
            
            # Assertions matching your original test
            assert result is not None
            assert 'text' in result
            assert 'sections' in result
            assert len(result['text']) > 0
            
            sections = result['sections']
            
            # Check sections were found
            assert len(sections) > 0
            
            # Check specific content
            if 'email' in sections:
                assert 'john.doe@email.com' in sections['email'] or '@' in sections['email']
            
            if 'skills' in sections:
                skills = sections['skills']
                assert isinstance(skills, list)
                assert len(skills) > 0
                
                # Check for key skills
                skills_str = ' '.join([str(s).lower() for s in skills])
                assert any(skill in skills_str for skill in ['python', 'fastapi', 'docker'])
            
            print("\n✅ Complete CV parsing pipeline test passed!")
            print(f"   Sections extracted: {list(sections.keys())}")
            print(f"   Total text length: {len(result['text'])} characters")
            
        finally:
            os.remove(temp_path)


class TestCVParserEdgeCases:
    """Edge cases and error handling"""
    
    def test_empty_cv(self, cv_parser):
        """Test handling of empty CV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('')
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            
            # Should handle gracefully
            assert result is not None
            assert 'error' in result or result['text'] == ''
        finally:
            os.remove(temp_path)
    
    def test_unsupported_file_type(self, cv_parser):
        """Test unsupported file extension"""
        result = cv_parser.parse('test.xyz')
        
        assert result is not None
        assert 'error' in result
        assert 'unsupported' in result['error'].lower()
    
    def test_malformed_cv(self, cv_parser):
        """Test CV with unusual formatting"""
        malformed_cv = "NoProperSections\nJustRandomText\n" * 20
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(malformed_cv)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            
            # Should not crash
            assert result is not None
            assert 'text' in result
        finally:
            os.remove(temp_path)
    
    def test_very_long_cv(self, cv_parser):
        """Test handling of very long CV"""
        long_cv = "Experience:\n" + ("- Worked on project\n" * 500)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(long_cv)
            temp_path = f.name
        
        try:
            result = cv_parser.parse(temp_path)
            
            # Should handle without crashing
            assert result is not None
            assert len(result['text']) > 0
        finally:
            os.remove(temp_path)