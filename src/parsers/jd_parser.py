"""
Job Description Parser - Extracts structured information from job postings
"""
import re
import spacy
from typing import Dict, List, Optional, Set
from loguru import logger


class JDParser:
    """Parse job descriptions and extract structured requirements"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Keywords for detecting requirement types
        self.required_keywords = [
            'must have', 'required', 'requires', 'necessary',
            'mandatory', 'essential', 'need', 'should have'
        ]
        
        self.preferred_keywords = [
            'preferred', 'desirable', 'nice to have', 'bonus',
            'plus', 'advantage', 'beneficial', 'would be great'
        ]
        
        # Common technical skills (expanded list)
        self.tech_skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go',
            'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r',
            
            # Web Frameworks
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi',
            'express', 'node.js', 'spring', 'laravel', 'rails',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
            'dynamodb', 'oracle', 'sql server', 'sqlite',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'ci/cd', 'git', 'github', 'gitlab',
            
            # Data & ML
            'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn',
            'spark', 'hadoop', 'kafka', 'airflow',
            
            # Other
            'rest api', 'graphql', 'microservices', 'agile', 'scrum',
            'jira', 'linux', 'bash', 'powershell'
        }
    
    def parse(self, jd_text: str) -> Dict:
        """
        Main entry point - parse job description
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            Dict with structured job requirements
        """
        if not jd_text or not jd_text.strip():
            logger.error("Empty job description provided")
            return {
                'success': False,
                'error': 'Empty job description',
                'required_skills': [],
                'preferred_skills': [],
                'experience_years': None
            }
        
        try:
            # Clean text
            clean_text = self._clean_text(jd_text)
            
            # Extract job metadata
            metadata = self._extract_metadata(clean_text)
            
            # Extract skills (required vs preferred)
            required_skills, preferred_skills = self._extract_skills(clean_text)
            
            # Extract experience requirements
            experience = self._extract_experience(clean_text)
            
            # Extract education requirements
            education = self._extract_education(clean_text)
            
            result = {
                'success': True,
                'error': None,
                'job_title': metadata.get('title'),
                'company': metadata.get('company'),
                'location': metadata.get('location'),
                'required_skills': list(required_skills),
                'preferred_skills': list(preferred_skills),
                'experience_years': experience,
                'education': education,
                'raw_text': clean_text[:500] + '...' if len(clean_text) > 500 else clean_text
            }
            
            logger.info(f"Parsed JD: {len(required_skills)} required, {len(preferred_skills)} preferred skills")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing JD: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'required_skills': [],
                'preferred_skills': []
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize job description text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove bullet points (we'll detect structure differently)
        text = re.sub(r'^[•●○■□▪▫–-]\s*', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract job title, company, location"""
        metadata = {
            'title': None,
            'company': None,
            'location': None
        }
        
        # Use spaCy NER to find organizations and locations
        doc = self.nlp(text[:1000])  # First 1000 chars usually have metadata
        
        orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        locs = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        
        if orgs:
            metadata['company'] = orgs[0]
        if locs:
            metadata['location'] = locs[0]
        
        # Try to extract job title from first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            # Job titles are often short and at the start
            if 5 < len(line) < 60 and not any(kw in line.lower() for kw in ['http', 'www', '@']):
                # Check if it contains common job title words
                job_words = ['engineer', 'developer', 'manager', 'analyst', 'specialist', 
                           'architect', 'consultant', 'designer', 'lead', 'senior', 'junior']
                if any(word in line.lower() for word in job_words):
                    metadata['title'] = line.strip()
                    break
        
        return metadata
    
    def _extract_skills(self, text: str) -> tuple[Set[str], Set[str]]:
        """Extract required and preferred skills"""
        required_skills = set()
        preferred_skills = set()
        
        # Split into sentences for context analysis
        doc = self.nlp(text)
        sentences = [sent.text.lower() for sent in doc.sents]
        
        for sentence in sentences:
            # Determine if sentence describes required or preferred skills
            is_required = any(kw in sentence for kw in self.required_keywords)
            is_preferred = any(kw in sentence for kw in self.preferred_keywords)
            
            # Extract technical skills from sentence
            found_skills = self._find_skills_in_text(sentence)
            
            if found_skills:
                if is_required:
                    required_skills.update(found_skills)
                elif is_preferred:
                    preferred_skills.update(found_skills)
                else:
                    # Default to required if no context
                    required_skills.update(found_skills)
        
        # Remove skills that appear in both (keep in required)
        preferred_skills -= required_skills
        
        return required_skills, preferred_skills
    
    def _find_skills_in_text(self, text: str) -> Set[str]:
        """Find technical skills in a piece of text"""
        found_skills = set()
        text_lower = text.lower()
        
        # Check for known technical skills
        for skill in self.tech_skills:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        # Also extract capitalized technical terms (likely skills)
        # e.g., "Experience with TensorFlow and PyTorch"
        words = text.split()
        for word in words:
            # Clean word
            word_clean = re.sub(r'[^\w\s.-]', '', word)
            
            # Check if it looks like a tech skill
            # (contains capital letters in middle, or common patterns)
            if len(word_clean) > 2:
                # CamelCase or has numbers (e.g., React, ES6, C++)
                if (word_clean[0].isupper() and any(c.isupper() for c in word_clean[1:]) or
                    any(c.isdigit() for c in word_clean) or
                    '++' in word_clean or '#' in word_clean):
                    found_skills.add(word_clean.lower())
        
        return found_skills
    
    def _extract_experience(self, text: str) -> Optional[Dict]:
        """Extract experience requirements"""
        # Patterns for experience mentions
        patterns = [
            r'(\d+)\+?\s*(?:to|\-)?\s*(\d+)?\s*years?\s+(?:of\s+)?experience',
            r'(?:minimum|at least)\s+(\d+)\s*years?',
            r'(\d+)\s*years?\s+(?:in|of|with)',
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Extract minimum years
                if isinstance(matches[0], tuple):
                    min_years = int(matches[0][0]) if matches[0][0] else None
                    max_years = int(matches[0][1]) if len(matches[0]) > 1 and matches[0][1] else None
                else:
                    min_years = int(matches[0])
                    max_years = None
                
                return {
                    'min_years': min_years,
                    'max_years': max_years,
                    'required': 'required' in text_lower or 'must' in text_lower
                }
        
        return None
    
    def _extract_education(self, text: str) -> Optional[Dict]:
        """Extract education requirements"""
        education = {
            'degree_level': None,
            'field': None,
            'required': False
        }
        
        text_lower = text.lower()
        
        # Detect degree level
        degree_patterns = {
            'phd': r'\b(?:phd|ph\.d\.|doctorate)\b',
            'masters': r'\b(?:masters?|m\.s\.|msc|master\'s)\b',
            'bachelors': r'\b(?:bachelors?|b\.s\.|bsc|bachelor\'s|undergraduate)\b',
            'associates': r'\b(?:associates?|a\.s\.)\b'
        }
        
        for level, pattern in degree_patterns.items():
            if re.search(pattern, text_lower):
                education['degree_level'] = level
                break
        
        # Detect field of study
        fields = ['computer science', 'engineering', 'mathematics', 'statistics', 
                 'information technology', 'data science', 'physics']
        
        for field in fields:
            if field in text_lower:
                education['field'] = field
                break
        
        # Check if required
        if education['degree_level']:
            # Look for requirement indicators near degree mention
            context_window = 200  # chars around degree mention
            for pattern in degree_patterns.values():
                match = re.search(pattern, text_lower)
                if match:
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text_lower), match.end() + context_window)
                    context = text_lower[context_start:context_end]
                    
                    if any(kw in context for kw in ['required', 'must', 'necessary', 'mandatory']):
                        education['required'] = True
                    break
        
        return education if education['degree_level'] else None


# Convenience function
def parse_jd(jd_text: str) -> Dict:
    """Quick parse function"""
    parser = JDParser()
    return parser.parse(jd_text)