"""
CV Optimizer - 100% LLM-Based CV Enhancement
Uses Ollama to intelligently optimize CVs for specific job descriptions
"""
import requests
from typing import Dict, List, Optional
import json


class CVOptimizer:
    """
    Use LLM to optimize CVs for specific job descriptions
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "gemma:latest"):
        self.ollama_url = ollama_url
        self.model = model
    
    def _call_ollama(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Ollama LLM"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""
    
    def optimize_manual_cv_for_jd(
        self,
        personal_info: Dict,
        experiences: List[Dict],
        education: List[Dict],
        skills: List[str],
        jd_text: str
    ) -> Dict:
        """
        Optimize manually entered CV data for a specific JD using LLM
        
        Args:
            personal_info: Personal information
            experiences: Work experiences
            education: Education details
            skills: List of skills
            jd_text: Job description text
            
        Returns:
            Optimized CV data
        """
        # Build current CV summary
        cv_summary = f"""
Personal Info: {personal_info.get('name', 'N/A')}
Skills: {', '.join(skills) if skills else 'None listed'}
Experience: {len(experiences)} positions
Education: {len(education)} degrees
"""
        
        prompt = f"""You are a professional CV writer. Optimize this CV for the following job description.

JOB DESCRIPTION:
{jd_text[:2000]}

CURRENT CV:
{cv_summary}

Your task:
1. Analyze which skills from the CV match the JD
2. Identify missing skills from JD that should be highlighted
3. Suggest skill reordering (matched skills first)
4. Create an optimized professional summary

Return ONLY a JSON object:
{{
  "matched_skills": ["skill1", "skill2"],
  "prioritized_skills": ["ordered list of all skills, matched first"],
  "skills_to_add": ["missing JD skills to add"],
  "optimized_summary": "2-3 sentence professional summary highlighting matched skills"
}}

JSON:"""

        response = self._call_ollama(prompt, max_tokens=1000)
        
        try:
            # Clean and parse
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            if '{' in response:
                response = response[response.index('{'):response.rindex('}')+1]
            
            result = json.loads(response)
            
            # Validate structure
            optimized = {
                'matched_skills': result.get('matched_skills', []),
                'prioritized_skills': result.get('prioritized_skills', skills),
                'skills_to_add': result.get('skills_to_add', []),
                'optimized_summary': result.get('optimized_summary', '')
            }
            
            return optimized
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            # Fallback
            return {
                'matched_skills': skills[:5],
                'prioritized_skills': skills,
                'skills_to_add': [],
                'optimized_summary': f"Professional with expertise in {', '.join(skills[:3])}."
            }
    
    def improve_existing_cv_with_jd(
        self,
        cv_text: str,
        cv_skills: List[str],
        jd_text: str
    ) -> Dict:
        """
        Improve existing CV by optimizing it for a JD using LLM
        
        Args:
            cv_text: Original CV text
            cv_skills: Extracted CV skills
            jd_text: Job description text
            
        Returns:
            Improved CV components
        """
        prompt = f"""You are a professional CV consultant. Improve this CV to better match the job description.

JOB DESCRIPTION:
{jd_text[:2000]}

CURRENT CV SKILLS:
{', '.join(cv_skills[:30])}

CURRENT CV (EXCERPT):
{cv_text[:1500]}

Analyze and improve:
1. Which current CV skills match JD requirements?
2. What key skills are missing from the CV?
3. How should skills be reordered to prioritize JD matches?
4. What professional summary would highlight relevant experience?

Return ONLY a JSON object:
{{
  "matched_skills": ["skills from CV that match JD"],
  "missing_skills": ["important JD skills not in CV"],
  "optimized_skill_order": ["complete reordered skill list, matched first"],
  "improvement_summary": "brief explanation of changes made",
  "enhanced_professional_summary": "2-3 sentence summary emphasizing JD-relevant experience"
}}

JSON:"""

        response = self._call_ollama(prompt, max_tokens=1500)
        
        try:
            # Parse response
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            if '{' in response:
                response = response[response.index('{'):response.rindex('}')+1]
            
            result = json.loads(response)
            
            return {
                'matched_skills': result.get('matched_skills', []),
                'missing_skills': result.get('missing_skills', []),
                'optimized_skill_order': result.get('optimized_skill_order', cv_skills),
                'improvement_summary': result.get('improvement_summary', ''),
                'enhanced_professional_summary': result.get('enhanced_professional_summary', '')
            }
            
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            return {
                'matched_skills': cv_skills[:10],
                'missing_skills': [],
                'optimized_skill_order': cv_skills,
                'improvement_summary': 'CV optimized with basic improvements',
                'enhanced_professional_summary': f"Professional with experience in {', '.join(cv_skills[:3])}."
            }
    
    def enhance_bullet_points(
        self,
        bullet_points: List[str],
        jd_text: str
    ) -> List[str]:
        """
        Enhance work experience bullet points using LLM
        
        Args:
            bullet_points: Original bullet points
            jd_text: Job description for context
            
        Returns:
            Enhanced bullet points
        """
        bullets_text = '\n'.join([f"- {b}" for b in bullet_points[:10]])
        
        prompt = f"""You are a professional CV writer. Improve these work experience bullet points to better align with this job description.

JOB DESCRIPTION:
{jd_text[:1500]}

CURRENT BULLET POINTS:
{bullets_text}

Improve each bullet point by:
1. Using stronger action verbs
2. Adding quantifiable metrics where possible
3. Highlighting skills relevant to the JD
4. Making achievements more impactful

Return ONLY a JSON array of improved bullet points:
["improved bullet 1", "improved bullet 2", ...]

JSON array:"""

        response = self._call_ollama(prompt, max_tokens=800)
        
        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            if '[' in response:
                response = response[response.index('['):response.rindex(']')+1]
            
            enhanced = json.loads(response)
            
            if isinstance(enhanced, list) and len(enhanced) > 0:
                return enhanced[:10]
            
        except Exception as e:
            print(f"Bullet enhancement failed: {e}")
        
        # Fallback: return originals
        return bullet_points


# Test
if __name__ == "__main__":
    optimizer = CVOptimizer()
    
    # Test skill optimization
    skills = ["Python", "JavaScript", "Docker"]
    jd = "Looking for developer with Python, FastAPI, Docker, Kubernetes"
    
    result = optimizer.optimize_manual_cv_for_jd(
        personal_info={'name': 'Test User'},
        experiences=[],
        education=[],
        skills=skills,
        jd_text=jd
    )
    
    print("✅ Optimization result:")
    print(f"   Matched: {result['matched_skills']}")
    print(f"   Summary: {result['optimized_summary'][:80]}...")