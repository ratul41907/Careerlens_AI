"""
CV Analyzer - Identifies issues and improvement opportunities in CVs
"""
from typing import Dict, List, Optional, Union
from loguru import logger
import re


class CVAnalyzer:
    """
    Analyze CVs for common issues and improvement opportunities
    """
    
    def __init__(self):
        """Initialize CV analyzer"""
        self.issue_categories = [
            'weak_bullets',
            'missing_metrics',
            'vague_descriptions',
            'spelling_grammar',
            'formatting',
            'ats_compatibility'
        ]
        logger.info("CVAnalyzer initialized")
    
    def analyze_cv(self, cv_data: Union[Dict, str], jd_data: Optional[Union[Dict, str]] = None) -> Dict:
        """
        Analyze CV for issues and improvements
        
        Args:
            cv_data: Parsed CV data (dict or string)
            jd_data: Optional job description (dict or string)
            
        Returns:
            Dict with issues found and improvement suggestions
        """
        logger.info("Analyzing CV for improvement opportunities")
        
        # CRITICAL FIX: Ensure cv_data is a dict
        if isinstance(cv_data, str):
            cv_data = {
                'text': cv_data,
                'sections': {
                    'experience': cv_data,
                    'skills': '',
                    'education': ''
                }
            }
        
        # CRITICAL FIX: Ensure jd_data is a dict if provided
        if jd_data and isinstance(jd_data, str):
            jd_data = {
                'text': jd_data,
                'required_skills': [],
                'preferred_skills': []
            }
        
        issues = {
            'weak_bullets': [],
            'missing_metrics': [],
            'vague_descriptions': [],
            'spelling_grammar': [],
            'formatting': [],
            'ats_compatibility': []
        }
        
        cv_text = cv_data.get('text', '')
        
        # Analyze bullet points
        if 'sections' in cv_data and 'experience' in cv_data['sections']:
            experience_text = cv_data['sections']['experience']
            if experience_text:  # Only analyze if not empty
                issues['weak_bullets'] = self._check_weak_bullets(experience_text)
                issues['missing_metrics'] = self._check_missing_metrics(experience_text)
                issues['vague_descriptions'] = self._check_vague_descriptions(experience_text)
        
        # Check ATS compatibility
        if cv_text:
            issues['ats_compatibility'] = self._check_ats_issues(cv_text)
            issues['formatting'] = self._check_formatting_issues(cv_text)
        
        # Generate improvements
        improvements = self._generate_improvements(issues, cv_data, jd_data)
        
        # Calculate overall score
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        score = max(0, 100 - (total_issues * 5))  # Each issue reduces score by 5
        
        result = {
            'score': score,
            'grade': self._get_grade(score),
            'total_issues': total_issues,
            'issues': issues,
            'improvements': improvements,
            'summary': self._generate_summary(score, total_issues)
        }
        
        logger.info(f"CV analysis complete: Score {score}/100, {total_issues} issues found")
        return result
    
    def _check_weak_bullets(self, text: str) -> List[Dict]:
        """Check for weak bullet points"""
        weak_bullets = []
        
        # Weak action verbs
        weak_verbs = ['helped', 'worked on', 'responsible for', 'assisted', 'involved in']
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('-') or line_lower.startswith('•'):
                for weak_verb in weak_verbs:
                    if weak_verb in line_lower:
                        weak_bullets.append({
                            'text': line.strip(),
                            'issue': f"Weak action verb: '{weak_verb}'",
                            'suggestion': f"Use stronger verbs like: Led, Developed, Implemented, Achieved"
                        })
                        break
        
        return weak_bullets[:5]  # Limit to top 5
    
    def _check_missing_metrics(self, text: str) -> List[Dict]:
        """Check for missing quantifiable metrics"""
        missing_metrics = []
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('-') or line_lower.startswith('•'):
                # Check if line has numbers/percentages
                if not re.search(r'\d+', line):
                    missing_metrics.append({
                        'text': line.strip(),
                        'issue': 'No quantifiable metrics',
                        'suggestion': 'Add numbers: how many? how much? by what percentage?'
                    })
        
        return missing_metrics[:5]
    
    def _check_vague_descriptions(self, text: str) -> List[Dict]:
        """Check for vague descriptions"""
        vague_issues = []
        
        vague_phrases = ['various', 'several', 'multiple', 'many', 'some', 'handled']
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for vague_phrase in vague_phrases:
                if vague_phrase in line_lower:
                    vague_issues.append({
                        'text': line.strip(),
                        'issue': f"Vague phrase: '{vague_phrase}'",
                        'suggestion': 'Be specific: name technologies, tools, exact numbers'
                    })
                    break
        
        return vague_issues[:5]
    
    def _check_ats_issues(self, text: str) -> List[Dict]:
        """Check ATS compatibility issues"""
        ats_issues = []
        
        # Check for tables (problematic for ATS)
        if '|' in text or text.count('\t') > 10:
            ats_issues.append({
                'issue': 'Tables or complex formatting detected',
                'suggestion': 'Use simple bullet points instead of tables'
            })
        
        # Check for images/graphics mentions
        if any(word in text.lower() for word in ['[image]', '[graphic]', '[chart]']):
            ats_issues.append({
                'issue': 'Images or graphics detected',
                'suggestion': 'Remove images - ATS systems cannot read them'
            })
        
        # Check for uncommon fonts/special characters
        special_chars = set(text) - set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?\'-\n\t()/@#&+')
        if len(special_chars) > 5:
            ats_issues.append({
                'issue': 'Special characters detected',
                'suggestion': 'Use standard characters for better ATS compatibility'
            })
        
        return ats_issues
    
    def _check_formatting_issues(self, text: str) -> List[Dict]:
        """Check formatting issues"""
        formatting_issues = []
        
        # Check for inconsistent date formats
        date_patterns = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{4}\b|\b\w+ \d{4}\b', text)
        if len(set(date_patterns)) > 2:
            formatting_issues.append({
                'issue': 'Inconsistent date formats',
                'suggestion': 'Use consistent format: Jan 2024 or 01/2024'
            })
        
        # Check for missing section headers
        if 'experience' not in text.lower() and 'work' not in text.lower():
            formatting_issues.append({
                'issue': 'Missing Experience section header',
                'suggestion': 'Add clear section: EXPERIENCE or WORK HISTORY'
            })
        
        if 'education' not in text.lower():
            formatting_issues.append({
                'issue': 'Missing Education section header',
                'suggestion': 'Add clear section: EDUCATION'
            })
        
        return formatting_issues
    
    def _generate_improvements(self, issues: Dict, cv_data: Dict, jd_data: Optional[Dict]) -> List[Dict]:
        """Generate specific improvement suggestions"""
        improvements = []
        
        # Prioritize improvements
        if issues.get('weak_bullets', []):
            improvements.append({
                'priority': 'High',
                'category': 'Impact',
                'title': 'Strengthen Action Verbs',
                'description': f"Replace {len(issues['weak_bullets'])} weak verbs with strong action verbs",
                'examples': ['Led → Spearheaded', 'Helped → Facilitated', 'Worked on → Engineered']
            })
        
        if issues.get('missing_metrics', []):
            improvements.append({
                'priority': 'High',
                'category': 'Quantification',
                'title': 'Add Quantifiable Results',
                'description': f"{len(issues['missing_metrics'])} bullets lack metrics",
                'examples': ['Add: "Improved performance by 40%"', 'Add: "Managed team of 5"', 'Add: "Reduced costs by $50K"']
            })
        
        if issues.get('ats_compatibility', []):
            improvements.append({
                'priority': 'Critical',
                'category': 'ATS',
                'title': 'Fix ATS Compatibility',
                'description': 'Make CV readable by Applicant Tracking Systems',
                'examples': ['Remove tables', 'Use standard fonts', 'Avoid images']
            })
        
        if jd_data:
            # Add job-specific improvements
            jd_skills = jd_data.get('required_skills', []) + jd_data.get('preferred_skills', [])
            cv_skills = cv_data.get('sections', {}).get('skills', '').lower()
            
            missing_keywords = [skill for skill in jd_skills if skill.lower() not in cv_skills]
            if missing_keywords:
                improvements.append({
                    'priority': 'High',
                    'category': 'Keywords',
                    'title': 'Add Missing Job Keywords',
                    'description': f"Add {len(missing_keywords[:5])} key skills from job description",
                    'examples': missing_keywords[:5]
                })
        
        return improvements
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A - Excellent'
        elif score >= 80:
            return 'B - Good'
        elif score >= 70:
            return 'C - Fair'
        elif score >= 60:
            return 'D - Needs Work'
        else:
            return 'F - Poor'
    
    def _generate_summary(self, score: int, total_issues: int) -> str:
        """Generate summary message"""
        if score >= 90:
            return f"✅ Excellent CV! Only {total_issues} minor improvements possible."
        elif score >= 70:
            return f"👍 Good CV with {total_issues} areas for improvement."
        else:
            return f"⚠️ CV needs work. {total_issues} issues found that could hurt your chances."


def analyze_cv(cv_data: Union[Dict, str], jd_data: Optional[Union[Dict, str]] = None) -> Dict:
    """Convenience function"""
    analyzer = CVAnalyzer()
    return analyzer.analyze_cv(cv_data, jd_data)