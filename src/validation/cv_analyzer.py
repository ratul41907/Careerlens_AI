"""
CV Analyzer - 100% LLM-Based Quality Analysis
Uses Ollama to provide intelligent CV improvement suggestions
"""
import requests
from typing import Dict, List, Optional, Union
import json


class CVAnalyzer:
    """
    Analyze CV quality and provide improvement suggestions using LLM
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize CV analyzer
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma2:2b"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 2000) -> str:
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
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _parse_llm_analysis(self, llm_response: str) -> Dict:
        """
        Parse LLM analysis into structured format
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Structured analysis dict
        """
        # Try to extract sections from LLM response
        sections = {
            'weak_bullets': [],
            'missing_metrics': [],
            'vague_descriptions': [],
            'ats_compatibility': [],
            'formatting': [],
            'grammar_spelling': []
        }
        
        improvements = []
        
        # Parse the response line by line
        current_category = None
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect category headers
            line_lower = line.lower()
            if 'weak' in line_lower and ('verb' in line_lower or 'action' in line_lower):
                current_category = 'weak_bullets'
            elif 'metric' in line_lower or 'number' in line_lower or 'quantif' in line_lower:
                current_category = 'missing_metrics'
            elif 'vague' in line_lower or 'unclear' in line_lower:
                current_category = 'vague_descriptions'
            elif 'ats' in line_lower or 'keyword' in line_lower:
                current_category = 'ats_compatibility'
            elif 'format' in line_lower or 'layout' in line_lower:
                current_category = 'formatting'
            elif 'grammar' in line_lower or 'spelling' in line_lower:
                current_category = 'grammar_spelling'
            elif 'improvement' in line_lower or 'suggestion' in line_lower:
                current_category = 'improvements'
            
            # Extract bullet points
            if line.startswith(('-', '•', '*', '–', '►')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
                clean_line = line.lstrip('-•*–►0123456789.)]: ').strip()
                if clean_line and len(clean_line) > 10:
                    if current_category == 'improvements':
                        improvements.append(clean_line)
                    elif current_category and current_category in sections:
                        sections[current_category].append(clean_line)
        
        # Count total issues
        total_issues = sum(len(v) for v in sections.values())
        
        # Calculate score based on issues (fewer issues = higher score)
        if total_issues == 0:
            score = 95
        elif total_issues <= 3:
            score = 85
        elif total_issues <= 6:
            score = 75
        elif total_issues <= 10:
            score = 65
        elif total_issues <= 15:
            score = 55
        else:
            score = 45
        
        # Determine grade
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': score,
            'grade': grade,
            'total_issues': total_issues,
            'issues': sections,
            'improvements': improvements if improvements else [
                "Overall structure is good",
                "Continue to tailor CV for each job application",
                "Keep updating with new achievements"
            ],
            'summary': llm_response[:500] if llm_response else "Analysis completed"
        }
    
    def analyze_cv(
        self,
        cv_data: Union[Dict, str],
        jd_data: Optional[Union[Dict, str]] = None
    ) -> Dict:
        """
        Analyze CV quality using LLM
        
        Args:
            cv_data: CV data (dict or string)
            jd_data: Optional job description for context
            
        Returns:
            Analysis results with score, issues, and improvements
        """
        # Convert to dict if string
        if isinstance(cv_data, str):
            cv_data = {'text': cv_data, 'sections': {}}
        
        if isinstance(jd_data, str):
            jd_data = {'text': jd_data}
        
        cv_text = cv_data.get('text', '')
        if not cv_text:
            return {
                'score': 0,
                'grade': 'F',
                'total_issues': 0,
                'issues': {},
                'improvements': [],
                'summary': 'No CV text provided'
            }
        
        # Build LLM prompt
        if jd_data and jd_data.get('text'):
            prompt = f"""You are an expert CV reviewer. Analyze this CV against the job description and provide detailed feedback.

CV:
{cv_text[:3000]}

Job Description:
{jd_data['text'][:1500]}

Analyze the CV and identify issues in these categories:

1. WEAK ACTION VERBS
List specific bullet points using weak verbs (e.g., "responsible for", "worked on", "helped with"). Suggest stronger alternatives.

2. MISSING METRICS
Identify accomplishments that lack quantification. Which bullets need numbers, percentages, or measurable outcomes?

3. VAGUE DESCRIPTIONS
Find unclear or generic statements that don't demonstrate actual skills or impact.

4. ATS COMPATIBILITY
Check for keyword gaps between CV and job description. Are important job requirements missing from the CV?

5. FORMATTING ISSUES
Note any layout, structure, or consistency problems.

6. GRAMMAR/SPELLING
Flag any language errors.

7. IMPROVEMENTS
Provide 5-10 specific, actionable suggestions to improve this CV for this role.

Format your response with clear headers and bullet points for each category."""
        else:
            prompt = f"""You are an expert CV reviewer. Analyze this CV and provide detailed quality feedback.

CV:
{cv_text[:3000]}

Analyze the CV and identify issues in these categories:

1. WEAK ACTION VERBS
List specific bullet points using weak verbs (e.g., "responsible for", "worked on", "helped with"). Suggest stronger alternatives.

2. MISSING METRICS
Identify accomplishments that lack quantification. Which bullets need numbers, percentages, or measurable outcomes?

3. VAGUE DESCRIPTIONS
Find unclear or generic statements that don't demonstrate actual skills or impact.

4. ATS COMPATIBILITY
Check if CV follows ATS-friendly formatting (clear sections, standard fonts, no tables/images in text areas).

5. FORMATTING ISSUES
Note any layout, structure, or consistency problems.

6. GRAMMAR/SPELLING
Flag any language errors.

7. IMPROVEMENTS
Provide 5-10 specific, actionable suggestions to improve this CV.

Format your response with clear headers and bullet points for each category."""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=2000)
        
        if not llm_response:
            # Fallback if LLM fails
            return {
                'score': 70,
                'grade': 'C',
                'total_issues': 3,
                'issues': {
                    'weak_bullets': ["Consider using stronger action verbs"],
                    'missing_metrics': ["Add quantifiable achievements where possible"],
                    'vague_descriptions': ["Make descriptions more specific and impactful"],
                    'ats_compatibility': [],
                    'formatting': [],
                    'grammar_spelling': []
                },
                'improvements': [
                    "Use stronger action verbs (e.g., 'Led', 'Developed', 'Increased')",
                    "Quantify achievements with numbers and percentages",
                    "Make descriptions more specific and results-focused",
                    "Ensure ATS-friendly formatting",
                    "Tailor CV to match job requirements"
                ],
                'summary': 'CV analysis completed. LLM unavailable, using fallback analysis.'
            }
        
        # Parse LLM response into structured format
        return self._parse_llm_analysis(llm_response)


# Test
if __name__ == "__main__":
    analyzer = CVAnalyzer()
    
    test_cv = {
        'text': """John Doe
Software Engineer

Experience:
- Worked on backend development
- Responsible for API design
- Helped with database optimization
- Participated in code reviews

Skills: Python, JavaScript, SQL"""
    }
    
    result = analyzer.analyze_cv(test_cv)
    print(f"Score: {result['score']}/100 (Grade: {result['grade']})")
    print(f"Total Issues: {result['total_issues']}")
    print(f"\nImprovements:")
    for imp in result['improvements']:
        print(f"  - {imp}")