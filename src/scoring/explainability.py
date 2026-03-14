"""
Explainability Engine - 100% LLM-Based Match Explanations
Generates natural language explanations for CV-JD matches using Ollama
"""
import requests
from typing import Dict, List, Optional


class ExplainabilityEngine:
    """
    Generate natural language explanations for match scores using LLM
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize explainability engine
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma2:2b"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1000) -> str:
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
                        "temperature": 0.7
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def explain_match(
        self,
        match_result: Dict,
        cv_data: Dict,
        jd_data: Dict
    ) -> Dict:
        """
        Generate comprehensive explanation for match result
        
        Args:
            match_result: Match result dict with scores
            cv_data: CV data
            jd_data: Job description data
            
        Returns:
            Explanation dict with summary, strengths, gaps, recommendations
        """
        # Extract key information
        overall_score = match_result.get('overall_score', 0) * 100
        
        breakdown = match_result.get('breakdown', {})
        req_skills = breakdown.get('required_skills', {}).get('details', {})
        pref_skills = breakdown.get('preferred_skills', {}).get('details', {})
        experience = breakdown.get('experience', {})
        
        matched_skills = req_skills.get('matched_skills', [])
        missing_skills = req_skills.get('missing_skills', [])
        
        # Build prompt for LLM
        prompt = f"""You are a career advisor explaining a CV-JD match analysis to a job seeker.

MATCH SCORE: {overall_score:.1f}%

MATCHED SKILLS: {', '.join(matched_skills) if matched_skills else 'None identified'}
MISSING SKILLS: {', '.join(missing_skills) if missing_skills else 'None identified'}

JOB TITLE: {jd_data.get('sections', {}).get('job_title', 'Not specified')}
EXPERIENCE REQUIRED: {jd_data.get('sections', {}).get('experience', {}).get('years', 'Not specified')} years
CANDIDATE EXPERIENCE: {cv_data.get('sections', {}).get('experience', 'Not specified')}

Write a helpful, encouraging explanation with these sections:

1. OVERALL ASSESSMENT (2-3 sentences)
Provide a clear summary of the match quality and what it means for the candidate.

2. KEY STRENGTHS (3-5 bullet points)
Highlight what makes this candidate a good fit. Be specific about matched skills and relevant experience.

3. AREAS FOR IMPROVEMENT (3-5 bullet points)
Identify the gaps and missing skills. Be constructive and specific.

4. ACTIONABLE RECOMMENDATIONS (5-7 bullet points)
Provide concrete steps the candidate can take to improve their chances:
- Which skills to learn first (prioritize high-impact)
- How to highlight existing relevant skills better
- Experience gaps to address
- CV improvements to make

Keep the tone professional but encouraging. Focus on actionable advice."""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=1200)
        
        if not llm_response:
            # Fallback explanation
            if overall_score >= 80:
                summary = f"Excellent match! Your CV scores {overall_score:.1f}%, indicating strong alignment with the job requirements."
            elif overall_score >= 65:
                summary = f"Good match at {overall_score:.1f}%. You have most of the required skills with some room for improvement."
            elif overall_score >= 50:
                summary = f"Moderate match at {overall_score:.1f}%. Several key skills match, but significant gaps exist."
            else:
                summary = f"Limited match at {overall_score:.1f}%. Consider developing the missing skills before applying."
            
            strengths = [
                f"You have {len(matched_skills)} of the required skills",
                "Your experience aligns with the role level"
            ] if matched_skills else ["Consider highlighting transferable skills"]
            
            gaps = [
                f"Missing {len(missing_skills)} key skills: {', '.join(missing_skills[:3])}"
            ] if missing_skills else ["No significant skill gaps identified"]
            
            recommendations = [
                f"Focus on learning: {', '.join(missing_skills[:3])}" if missing_skills else "Continue building expertise",
                "Tailor your CV to match job keywords",
                "Quantify your achievements with metrics",
                "Highlight relevant projects prominently"
            ]
            
            return {
                'summary': summary,
                'detailed_explanation': llm_response or summary,
                'strengths': strengths,
                'gaps': gaps,
                'recommendations': recommendations,
                'match_level': 'High' if overall_score >= 75 else 'Medium' if overall_score >= 50 else 'Low'
            }
        
        # Parse LLM response into sections
        sections = {
            'summary': '',
            'strengths': [],
            'gaps': [],
            'recommendations': []
        }
        
        current_section = None
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Detect section headers
            if 'overall' in line_lower or 'assessment' in line_lower or 'summary' in line_lower:
                current_section = 'summary'
                continue
            elif 'strength' in line_lower or 'fit' in line_lower or 'match' in line_lower:
                current_section = 'strengths'
                continue
            elif 'gap' in line_lower or 'improvement' in line_lower or 'weakness' in line_lower or 'area' in line_lower:
                current_section = 'gaps'
                continue
            elif 'recommend' in line_lower or 'action' in line_lower or 'step' in line_lower or 'suggest' in line_lower:
                current_section = 'recommendations'
                continue
            
            # Add content to appropriate section
            if current_section == 'summary':
                if not line.startswith(('-', '•', '*', '1', '2', '3')):
                    sections['summary'] += line + ' '
            elif current_section in ['strengths', 'gaps', 'recommendations']:
                if line.startswith(('-', '•', '*')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.)'):
                    clean_line = line.lstrip('-•*0123456789.)]: ').strip()
                    if clean_line and len(clean_line) > 10:
                        sections[current_section].append(clean_line)
        
        # Ensure we have content
        if not sections['summary']:
            sections['summary'] = llm_response[:200]
        
        if not sections['strengths']:
            sections['strengths'] = [f"You have {len(matched_skills)} required skills"] if matched_skills else ["Review your transferable skills"]
        
        if not sections['gaps']:
            sections['gaps'] = [f"Missing: {', '.join(missing_skills[:3])}"] if missing_skills else ["No major gaps identified"]
        
        if not sections['recommendations']:
            sections['recommendations'] = [
                "Tailor CV to job keywords",
                "Quantify achievements",
                "Highlight relevant projects"
            ]
        
        # Determine match level
        if overall_score >= 75:
            match_level = 'High'
        elif overall_score >= 50:
            match_level = 'Medium'
        else:
            match_level = 'Low'
        
        return {
            'summary': sections['summary'].strip(),
            'detailed_explanation': llm_response,
            'strengths': sections['strengths'][:5],
            'gaps': sections['gaps'][:5],
            'recommendations': sections['recommendations'][:7],
            'match_level': match_level
        }


# Test
if __name__ == "__main__":
    engine = ExplainabilityEngine()
    
    test_match = {
        'overall_score': 0.72,
        'breakdown': {
            'required_skills': {
                'details': {
                    'matched_skills': ['Python', 'FastAPI', 'Docker'],
                    'missing_skills': ['Kubernetes', 'AWS']
                }
            }
        }
    }
    
    test_cv = {'sections': {'experience': '5 years'}}
    test_jd = {'sections': {'job_title': 'Senior Backend Engineer', 'experience': {'years': '5+'}}}
    
    explanation = engine.explain_match(test_match, test_cv, test_jd)
    print(f"Summary: {explanation['summary']}")
    print(f"\nStrengths: {len(explanation['strengths'])}")
    print(f"Gaps: {len(explanation['gaps'])}")
    print(f"Recommendations: {len(explanation['recommendations'])}")