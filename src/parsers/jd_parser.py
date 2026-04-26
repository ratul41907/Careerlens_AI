"""
Job Description Parser - 100% LLM-Based Intelligent Parsing
Uses Ollama to extract requirements, skills, and metadata from job descriptions
"""
import requests
from typing import Dict, List, Optional
import re


class JDParser:
    """
    Parse job descriptions using LLM for intelligent extraction
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize JD parser
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma:latest"
    
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
                        "temperature": 0.5  # Lower temperature for more consistent extraction
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
    
    def _parse_list_from_text(self, text: str, section_marker: str) -> List[str]:
        """
        Extract bulleted/numbered list from text after a section marker
        
        Args:
            text: Text to parse
            section_marker: Section header to look for
            
        Returns:
            List of items
        """
        items = []
        in_section = False
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Check if we've entered the section
            if section_marker.lower() in line.lower():
                in_section = True
                continue
            
            # Check if we've left the section (new header)
            if in_section and line and line[0].isupper() and ':' in line and not line.startswith(('-', '•', '*', '1', '2')):
                break
            
            # Extract items from this section
            if in_section and line:
                if line.startswith(('-', '•', '*', '–', '►')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.)'):
                    clean_item = line.lstrip('-•*–►0123456789.)]: ').strip()
                    if clean_item and len(clean_item) > 2:
                        items.append(clean_item)
        
        return items
    
    def parse(self, jd_text: str) -> Dict:
        """
        Parse job description using LLM
        
        Args:
            jd_text: Job description text
            
        Returns:
            Parsed JD data with sections
        """
        if not jd_text or len(jd_text.strip()) < 50:
            return {
                'text': jd_text,
                'sections': {},
                'error': 'Job description too short or empty'
            }
        
        # Build LLM prompt for intelligent extraction
        prompt = f"""You are an expert HR parser extracting structured information from job descriptions.

JOB DESCRIPTION:
{jd_text[:4000]}

Extract the following information and format with clear headers:

JOB TITLE:
[Extract the job title/position name]

REQUIRED SKILLS:
[List all REQUIRED technical skills, tools, languages, frameworks]
- Skill 1
- Skill 2
...

PREFERRED SKILLS:
[List all PREFERRED/nice-to-have skills]
- Skill 1
- Skill 2
...

EXPERIENCE:
[Extract years of experience required, e.g., "5 years" or "3-5 years"]

EDUCATION:
[Extract degree requirements, e.g., "Bachelor's in Computer Science" or "Master's preferred"]

RESPONSIBILITIES:
[List main job responsibilities/duties]
- Responsibility 1
- Responsibility 2
...

QUALIFICATIONS:
[List qualifications beyond skills, e.g., "Strong communication skills"]
- Qualification 1
- Qualification 2
...

BENEFITS:
[List any mentioned benefits, perks, or compensation info]
- Benefit 1
- Benefit 2
...

Extract all information exactly as stated. If a section is not mentioned, write "Not specified".
Use bullet points with "-" for lists."""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=1500)
        
        # Parse LLM response into structured sections
        sections = {}
        
        # Extract job title
        job_title_match = re.search(r'JOB TITLE:?\s*\n?\s*(.+?)(?:\n\n|\nREQUIRED|\nPREFERRED|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if job_title_match:
            sections['job_title'] = job_title_match.group(1).strip()
        
        # Extract required skills
        required_skills = self._parse_list_from_text(llm_response, 'REQUIRED SKILLS')
        if required_skills:
            sections['required_skills'] = required_skills
        
        # Extract preferred skills
        preferred_skills = self._parse_list_from_text(llm_response, 'PREFERRED SKILLS')
        if preferred_skills:
            sections['preferred_skills'] = preferred_skills
        
        # Extract experience
        exp_match = re.search(r'EXPERIENCE:?\s*\n?\s*(.+?)(?:\n\n|\nEDUCATION|\nRESPONSIBILITIES|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1).strip()
            # Extract years
            years_match = re.search(r'(\d+)[\s-]*(?:to|-)?\s*(\d+)?\s*(?:years?|yrs?)', exp_text, re.IGNORECASE)
            if years_match:
                min_years = int(years_match.group(1))
                max_years = int(years_match.group(2)) if years_match.group(2) else min_years
                sections['experience'] = {
                    'text': exp_text,
                    'years': f"{min_years}-{max_years}" if max_years != min_years else str(min_years),
                    'min_years': min_years,
                    'max_years': max_years
                }
            else:
                sections['experience'] = {'text': exp_text, 'years': exp_text}
        
        # Extract education
        edu_match = re.search(r'EDUCATION:?\s*\n?\s*(.+?)(?:\n\n|\nRESPONSIBILITIES|\nQUALIFICATIONS|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if edu_match:
            sections['education'] = edu_match.group(1).strip()
        
        # Extract responsibilities
        responsibilities = self._parse_list_from_text(llm_response, 'RESPONSIBILITIES')
        if responsibilities:
            sections['responsibilities'] = responsibilities
        
        # Extract qualifications
        qualifications = self._parse_list_from_text(llm_response, 'QUALIFICATIONS')
        if qualifications:
            sections['qualifications'] = qualifications
        
        # Extract benefits
        benefits = self._parse_list_from_text(llm_response, 'BENEFITS')
        if benefits:
            sections['benefits'] = benefits
        
        # Fallback: If LLM completely failed, do basic keyword extraction
        if not sections or len(sections) < 2:
            sections = self._fallback_parse(jd_text)
        
        return {
            'text': jd_text,
            'sections': sections,
            'llm_response': llm_response
        }
    
    def _fallback_parse(self, jd_text: str) -> Dict:
        """
        Fallback parsing if LLM fails
        
        Args:
            jd_text: Job description text
            
        Returns:
            Basic parsed sections
        """
        sections = {}
        
        # Try to extract job title (first line or before requirements)
        lines = jd_text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100 and not line.lower().startswith(('we are', 'about', 'description')):
                sections['job_title'] = line
                break
        
        # Common technical skills to look for
        common_skills = [
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Go', 'Rust', 'Swift',
            'React', 'Angular', 'Vue', 'Node', 'Django', 'Flask', 'Spring', 'FastAPI',
            'SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'CI/CD', 'Git', 'Linux',
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'NLP',
            'REST API', 'GraphQL', 'Microservices', 'Agile', 'Scrum'
        ]
        
        found_skills = []
        jd_lower = jd_text.lower()
        
        for skill in common_skills:
            if skill.lower() in jd_lower:
                found_skills.append(skill)
        
        if found_skills:
            sections['required_skills'] = found_skills[:15]
        
        # Try to extract years of experience
        exp_match = re.search(r'(\d+)[\s-]*(?:to|-)?\s*(\d+)?\s*(?:years?|yrs?)', jd_text, re.IGNORECASE)
        if exp_match:
            min_years = int(exp_match.group(1))
            max_years = int(exp_match.group(2)) if exp_match.group(2) else min_years
            sections['experience'] = {
                'years': f"{min_years}-{max_years}" if max_years != min_years else str(min_years),
                'min_years': min_years,
                'max_years': max_years
            }
        
        return sections


# Test
if __name__ == "__main__":
    parser = JDParser()
    
    test_jd = """Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience.

Required Skills:
- Python (Django/FastAPI)
- PostgreSQL
- Docker & Kubernetes
- AWS cloud services
- REST API design

Preferred Skills:
- React.js
- Redis caching
- CI/CD pipelines

Responsibilities:
- Design and implement scalable backend systems
- Mentor junior developers
- Participate in architecture decisions

Education: Bachelor's in Computer Science or equivalent

Benefits:
- Competitive salary
- Remote work options
- Health insurance"""
    
    result = parser.parse(test_jd)
    print(f"✅ Parsed JD")
    print(f"Job Title: {result['sections'].get('job_title', 'N/A')}")
    print(f"Required Skills: {len(result['sections'].get('required_skills', []))}")
    print(f"Experience: {result['sections'].get('experience', {}).get('years', 'N/A')}")