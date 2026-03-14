"""
CV Parser - 100% LLM-Based Intelligent CV Parsing
Uses Ollama to extract experience, skills, education from CVs
"""
import requests
from typing import Dict, List, Optional
import re
import PyPDF2
import docx
from pathlib import Path


class CVParser:
    """
    Parse CVs using LLM for intelligent section extraction
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize CV parser
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma2:2b"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call Ollama LLM API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.5
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"TXT extraction error: {e}")
            return ""
    
    def _parse_list_from_text(self, text: str, section_marker: str) -> List[str]:
        """Extract bulleted list from text"""
        items = []
        in_section = False
        
        for line in text.split('\n'):
            line = line.strip()
            
            if section_marker.lower() in line.lower():
                in_section = True
                continue
            
            if in_section and line and line[0].isupper() and ':' in line:
                break
            
            if in_section and line:
                if line.startswith(('-', '•', '*', '–')) or (len(line) > 2 and line[0].isdigit()):
                    clean_item = line.lstrip('-•*–►0123456789.)]: ').strip()
                    if clean_item and len(clean_item) > 2:
                        items.append(clean_item)
        
        return items
    
    def parse(self, file_path: str) -> Dict:
        """
        Parse CV file using LLM
        
        Args:
            file_path: Path to CV file (PDF, DOCX, or TXT)
            
        Returns:
            Parsed CV data with sections
        """
        # Extract text based on file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            cv_text = self._extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            cv_text = self._extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            cv_text = self._extract_text_from_txt(file_path)
        else:
            return {
                'text': '',
                'sections': {},
                'error': f'Unsupported file type: {file_ext}'
            }
        
        if not cv_text or len(cv_text.strip()) < 50:
            return {
                'text': cv_text,
                'sections': {},
                'error': 'CV text too short or extraction failed'
            }
        
        # Use LLM to parse CV
        prompt = f"""You are an expert CV parser extracting structured information from resumes.

CV TEXT:
{cv_text[:4000]}

Extract the following information and format with clear headers:

NAME:
[Extract candidate's full name]

EMAIL:
[Extract email address]

PHONE:
[Extract phone number]

SKILLS:
[List all technical skills, tools, languages mentioned]
- Skill 1
- Skill 2
...

EXPERIENCE:
[Extract total years of professional experience, e.g., "5 years" or "3-5 years"]

WORK HISTORY:
[List job positions with company and duration]
- Position 1: Company (Year-Year)
- Position 2: Company (Year-Year)
...

EDUCATION:
[List degrees with institution and graduation year]
- Degree 1: University (Year)
- Degree 2: University (Year)
...

CERTIFICATIONS:
[List any certifications or courses]
- Certification 1
- Certification 2
...

PROJECTS:
[List notable projects mentioned]
- Project 1
- Project 2
...

Extract all information exactly as stated. If a section is not found, write "Not specified".
Use bullet points with "-" for lists."""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=1500)
        
        # Parse LLM response
        sections = {}
        
        # Extract name
        name_match = re.search(r'NAME:?\s*\n?\s*(.+?)(?:\n\n|\nEMAIL|\nPHONE|$)', llm_response, re.IGNORECASE)
        if name_match:
            sections['name'] = name_match.group(1).strip()
        
        # Extract email
        email_match = re.search(r'EMAIL:?\s*\n?\s*(.+?)(?:\n\n|\nPHONE|\nSKILLS|$)', llm_response, re.IGNORECASE)
        if email_match:
            sections['email'] = email_match.group(1).strip()
        
        # Extract phone
        phone_match = re.search(r'PHONE:?\s*\n?\s*(.+?)(?:\n\n|\nSKILLS|\nEXPERIENCE|$)', llm_response, re.IGNORECASE)
        if phone_match:
            sections['phone'] = phone_match.group(1).strip()
        
        # Extract skills
        skills = self._parse_list_from_text(llm_response, 'SKILLS')
        if skills:
            sections['skills'] = skills
        
        # Extract experience
        exp_match = re.search(r'EXPERIENCE:?\s*\n?\s*(.+?)(?:\n\n|\nWORK HISTORY|\nEDUCATION|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if exp_match:
            sections['experience'] = exp_match.group(1).strip()
        
        # Extract work history
        work_history = self._parse_list_from_text(llm_response, 'WORK HISTORY')
        if work_history:
            sections['work_history'] = work_history
        
        # Extract education
        education = self._parse_list_from_text(llm_response, 'EDUCATION')
        if education:
            sections['education'] = education
        
        # Extract certifications
        certifications = self._parse_list_from_text(llm_response, 'CERTIFICATIONS')
        if certifications:
            sections['certifications'] = certifications
        
        # Extract projects
        projects = self._parse_list_from_text(llm_response, 'PROJECTS')
        if projects:
            sections['projects'] = projects
        
        # Fallback if LLM failed
        if not sections or len(sections) < 3:
            sections = self._fallback_parse(cv_text)
        
        return {
            'text': cv_text,
            'sections': sections,
            'llm_response': llm_response
        }
    
    def _fallback_parse(self, cv_text: str) -> Dict:
        """Fallback parsing if LLM fails"""
        sections = {}
        
        # Extract email with regex
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text)
        if email_match:
            sections['email'] = email_match.group(0)
        
        # Extract phone
        phone_match = re.search(r'[\+\d][\d\-\(\)\s]{8,}', cv_text)
        if phone_match:
            sections['phone'] = phone_match.group(0).strip()
        
        # Common skills
        common_skills = [
            'Python', 'JavaScript', 'Java', 'C++', 'React', 'Node', 'Django',
            'FastAPI', 'SQL', 'MongoDB', 'Docker', 'Kubernetes', 'AWS', 'Git'
        ]
        
        found_skills = [skill for skill in common_skills if skill.lower() in cv_text.lower()]
        if found_skills:
            sections['skills'] = found_skills
        
        # Extract years of experience
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', cv_text, re.IGNORECASE)
        if exp_match:
            sections['experience'] = f"{exp_match.group(1)} years"
        
        return sections


# Test
if __name__ == "__main__":
    parser = CVParser()
    
    # Test with sample text (normally you'd pass a file path)
    import tempfile
    
    test_cv = """John Doe
john.doe@email.com | +1-234-567-8900

EXPERIENCE
Senior Software Engineer | Tech Corp (2020-Present)
- Led development of microservices architecture
- Managed team of 5 developers

Software Engineer | StartupXYZ (2018-2020)
- Built REST APIs with Python/FastAPI
- Implemented CI/CD pipelines

SKILLS
Python, JavaScript, React, Node.js, Docker, Kubernetes, AWS, PostgreSQL, MongoDB

EDUCATION
Bachelor of Science in Computer Science | MIT (2018)

CERTIFICATIONS
- AWS Certified Solutions Architect
- Docker Certified Associate"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_cv)
        temp_path = f.name
    
    result = parser.parse(temp_path)
    print(f"✅ Parsed CV")
    print(f"Name: {result['sections'].get('name', 'N/A')}")
    print(f"Email: {result['sections'].get('email', 'N/A')}")
    print(f"Skills: {len(result['sections'].get('skills', []))}")
    print(f"Experience: {result['sections'].get('experience', 'N/A')}")