"""
CV Improver - Analyzes and suggests improvements for existing CVs
"""
from typing import Dict, List, Optional, Tuple
from loguru import logger
import re

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.scoring.scoring_engine import ScoringEngine
from src.embeddings.embedding_engine import EmbeddingEngine
from src.llm.cv_rewriter import CVRewriter


class CVImprover:
    """
    Analyze CV and suggest improvements
    """
    
    def __init__(self,
                 embedding_engine: Optional[EmbeddingEngine] = None,
                 scoring_engine: Optional[ScoringEngine] = None,
                 cv_rewriter: Optional[CVRewriter] = None):
        """
        Initialize CV improver
        
        Args:
            embedding_engine: Pre-initialized embedding engine
            scoring_engine: Pre-initialized scoring engine
            cv_rewriter: Pre-initialized CV rewriter
        """
        self.cv_parser = CVParser()
        self.jd_parser = JDParser()
        
        if embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine
        
        if scoring_engine is None:
            self.scoring_engine = ScoringEngine(self.embedding_engine)
        else:
            self.scoring_engine = scoring_engine
        
        if cv_rewriter is None:
            self.cv_rewriter = CVRewriter()
        else:
            self.cv_rewriter = cv_rewriter
        
        logger.info("CVImprover initialized")
    
    def analyze_and_improve(self,
                           cv_path_or_text: str,
                           jd_text: str,
                           num_suggestions: int = 5) -> Dict:
        """
        Analyze CV against JD and suggest improvements
        
        Args:
            cv_path_or_text: Path to CV file or CV text
            jd_text: Job description text
            num_suggestions: Number of improvement suggestions
            
        Returns:
            Dict with analysis and suggestions
        """
        logger.info("Analyzing CV for improvement opportunities...")
        
        # Parse CV
        if cv_path_or_text.endswith(('.pdf', '.docx', '.doc')):
            cv_data = self.cv_parser.parse(cv_path_or_text)
        else:
            cv_data = {
                'text': cv_path_or_text,
                'sections': self.cv_parser._segment_sections(cv_path_or_text)
            }
        
        # Parse JD
        jd_data = self.jd_parser.parse(jd_text)
        
        # Get match score
        match_result = self.scoring_engine.compute_match_score(cv_data, jd_data)
        
        # Analyze issues
        issues = self._identify_issues(cv_data, jd_data, match_result)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            cv_data, jd_data, match_result, issues, num_suggestions
        )
        
        result = {
            'current_score': match_result['overall_score'],
            'current_percentage': match_result['overall_percentage'],
            'issues_found': len(issues),
            'issues': issues,
            'suggestions': suggestions,
            'estimated_improvement': self._estimate_improvement(suggestions),
            'priority_actions': self._get_priority_actions(suggestions)
        }
        
        logger.info(f"Analysis complete: {len(issues)} issues, {len(suggestions)} suggestions")
        return result
    
    def _identify_issues(self,
                        cv_data: Dict,
                        jd_data: Dict,
                        match_result: Dict) -> List[Dict]:
        """
        Identify issues in CV
        
        Returns:
            List of issues with severity and description
        """
        issues = []
        
        # Issue 1: Missing required skills
        required_details = match_result['breakdown']['required_skills']['details']
        missing_skills = [s for s in required_details['skills'] if not s['matched']]
        
        if missing_skills:
            issues.append({
                'type': 'missing_skills',
                'severity': 'High',
                'title': f"{len(missing_skills)} Required Skills Missing",
                'description': f"CV lacks evidence for: {', '.join([s['skill'] for s in missing_skills[:3]])}",
                'impact': 'Major negative impact on ATS scoring'
            })
        
        # Issue 2: Weak bullets (no quantification)
        weak_bullets = self._find_weak_bullets(cv_data)
        if weak_bullets:
            issues.append({
                'type': 'weak_bullets',
                'severity': 'Medium',
                'title': f"{len(weak_bullets)} Bullets Lack Quantification",
                'description': "Experience bullets missing metrics, percentages, or numbers",
                'impact': 'Reduced credibility and impact'
            })
        
        # Issue 3: Experience mismatch
        exp_details = match_result['breakdown']['experience']['details']
        if not exp_details.get('meets_requirement', True):
            issues.append({
                'type': 'experience_gap',
                'severity': 'High',
                'title': "Experience Below Requirement",
                'description': exp_details.get('note', 'Experience years not sufficient'),
                'impact': 'May not pass initial screening'
            })
        
        # Issue 4: Poor section ordering
        if not self._has_optimal_section_order(cv_data):
            issues.append({
                'type': 'section_order',
                'severity': 'Low',
                'title': "Non-Optimal Section Ordering",
                'description': "Experience should come before Education for experienced professionals",
                'impact': 'Minor - affects readability'
            })
        
        # Issue 5: Missing summary
        if 'summary' not in cv_data['sections'] or not cv_data['sections']['summary']:
            issues.append({
                'type': 'missing_summary',
                'severity': 'Medium',
                'title': "No Professional Summary",
                'description': "CV lacks an impactful opening summary",
                'impact': 'Missed opportunity to highlight key qualifications'
            })
        
        # Issue 6: Generic bullets
        generic_bullets = self._find_generic_bullets(cv_data)
        if generic_bullets:
            issues.append({
                'type': 'generic_bullets',
                'severity': 'Medium',
                'title': f"{len(generic_bullets)} Generic Bullets",
                'description': "Bullets use vague language like 'worked on', 'helped with'",
                'impact': 'Weakens perceived contribution'
            })
        
        return issues
    
    def _find_weak_bullets(self, cv_data: Dict) -> List[str]:
        """Find bullets without quantification"""
        weak_bullets = []
        
        if 'experience' in cv_data['sections']:
            exp_text = cv_data['sections']['experience']
            
            # Split into bullets
            bullets = re.split(r'[•\-\*]\s*', exp_text)
            
            for bullet in bullets:
                bullet = bullet.strip()
                if len(bullet) < 20:  # Too short
                    continue
                
                # Check if has numbers/metrics
                has_number = bool(re.search(r'\d+', bullet))
                has_percent = '%' in bullet
                has_metric = any(word in bullet.lower() for word in ['increased', 'reduced', 'improved', 'achieved'])
                
                if not (has_number or has_percent or has_metric):
                    weak_bullets.append(bullet[:60] + '...' if len(bullet) > 60 else bullet)
        
        return weak_bullets[:5]  # Return max 5
    
    def _find_generic_bullets(self, cv_data: Dict) -> List[str]:
        """Find bullets with generic/weak verbs"""
        generic_verbs = [
            'worked on', 'helped with', 'assisted in', 'responsible for',
            'involved in', 'participated in', 'contributed to', 'used'
        ]
        
        generic_bullets = []
        
        if 'experience' in cv_data['sections']:
            exp_text = cv_data['sections']['experience']
            bullets = re.split(r'[•\-\*]\s*', exp_text)
            
            for bullet in bullets:
                bullet_lower = bullet.lower().strip()
                if any(verb in bullet_lower for verb in generic_verbs):
                    generic_bullets.append(bullet[:60] + '...' if len(bullet) > 60 else bullet)
        
        return generic_bullets[:5]
    
    def _has_optimal_section_order(self, cv_data: Dict) -> bool:
        """Check if sections are in optimal order for ATS"""
        sections = list(cv_data['sections'].keys())
        
        # Optimal order: header → summary → skills → experience → education
        if 'experience' in sections and 'education' in sections:
            exp_idx = sections.index('experience')
            edu_idx = sections.index('education')
            
            # Experience should come before education (for experienced professionals)
            return exp_idx < edu_idx
        
        return True
    
    def _generate_suggestions(self,
                             cv_data: Dict,
                             jd_data: Dict,
                             match_result: Dict,
                             issues: List[Dict],
                             num_suggestions: int) -> List[Dict]:
        """
        Generate actionable improvement suggestions
        
        Returns:
            List of suggestions with before/after examples
        """
        suggestions = []
        
        # Suggestion 1: Add missing skills
        missing_skills = [s for s in match_result['breakdown']['required_skills']['details']['skills'] 
                         if not s['matched']]
        
        if missing_skills and len(suggestions) < num_suggestions:
            top_missing = missing_skills[0]
            
            # Use template instead of LLM (to avoid memory issues)
            template_example = f"Developed and deployed {top_missing['skill']}-based solutions that improved system performance by 30% and reduced latency by 200ms"
            
            suggestions.append({
                'priority': 1,
                'type': 'add_skill',
                'title': f"Add Evidence for '{top_missing['skill']}'",
                'description': f"This required skill is missing from your CV. Current score: {top_missing['percentage']}",
                'action': f"Add a bullet point demonstrating {top_missing['skill']} experience",
                'example': template_example,
                'estimated_impact': "+3-5% overall score"
            })
        
        # Suggestion 2: Improve weak bullets
        weak_bullets = self._find_weak_bullets(cv_data)
        
        if weak_bullets and len(suggestions) < num_suggestions:
            original_bullet = weak_bullets[0]
            
            # Use template instead of LLM
            improved_bullet = "Architected scalable backend systems serving 100K+ daily requests with 99.9% uptime, reducing infrastructure costs by 30%"
            
            suggestions.append({
                'priority': 2,
                'type': 'improve_bullet',
                'title': "Quantify Achievement Bullet",
                'description': "Transform vague bullet into quantified achievement",
                'action': "Add specific metrics, percentages, or numbers",
                'before': original_bullet,
                'after': improved_bullet,
                'estimated_impact': "+2-3% overall score"
            })
        
        # Suggestion 3: Add professional summary
        if 'summary' not in cv_data['sections'] and len(suggestions) < num_suggestions:
            suggestions.append({
                'priority': 3,
                'type': 'add_summary',
                'title': "Add Professional Summary",
                'description': "Start with a strong 2-3 sentence summary highlighting key qualifications",
                'action': "Write a compelling summary targeting this role",
                'example': self._generate_summary_example(cv_data, jd_data),
                'estimated_impact': "+1-2% overall score"
            })
        
        # Suggestion 4: Improve generic bullets
        generic_bullets = self._find_generic_bullets(cv_data)
        
        if generic_bullets and len(suggestions) < num_suggestions:
            original_bullet = generic_bullets[0]
            
            # Use template instead of LLM
            improved_bullet = "Designed and implemented high-performance solutions that increased efficiency by 40% and processed 50K+ transactions daily"
            
            suggestions.append({
                'priority': 4,
                'type': 'strengthen_verb',
                'title': "Replace Weak Action Verb",
                'description': "Use stronger, more specific action verbs",
                'action': "Replace 'worked on' with 'developed', 'architected', etc.",
                'before': original_bullet,
                'after': improved_bullet,
                'estimated_impact': "+1-2% overall score"
            })
        
        # Suggestion 5: Re-order sections
        if not self._has_optimal_section_order(cv_data) and len(suggestions) < num_suggestions:
            suggestions.append({
                'priority': 5,
                'type': 'reorder_sections',
                'title': "Optimize Section Order",
                'description': "Place Experience before Education for better ATS parsing",
                'action': "Reorder: Summary → Skills → Experience → Education → Projects",
                'estimated_impact': "+0.5-1% overall score"
            })
        
        return suggestions[:num_suggestions]
    
    def _generate_summary_example(self, cv_data: Dict, jd_data: Dict) -> str:
        """Generate example professional summary"""
        # Simple template-based approach
        job_title = jd_data.get('job_title', 'Software Engineer')
        required_skills = jd_data.get('required_skills', [])[:3]
        
        summary = f"Results-driven {job_title} with expertise in {', '.join(required_skills)}. "
        summary += "Proven track record of delivering high-quality solutions that improve system performance. "
        summary += "Seeking to leverage technical skills and experience to drive innovation."
        
        return summary
    
    def _estimate_improvement(self, suggestions: List[Dict]) -> str:
        """Estimate total potential improvement"""
        # Parse impact strings and sum
        total_min = 0
        total_max = 0
        
        for sugg in suggestions:
            impact = sugg.get('estimated_impact', '+0%')
            # Parse "+3-5%" → (3, 5)
            match = re.search(r'\+(\d+)-(\d+)%', impact)
            if match:
                total_min += int(match.group(1))
                total_max += int(match.group(2))
        
        if total_max > 0:
            return f"+{total_min}-{total_max}% potential improvement"
        else:
            return "+5-10% potential improvement (estimated)"
    
    def _get_priority_actions(self, suggestions: List[Dict]) -> List[str]:
        """Get top 3 priority actions"""
        actions = []
        for sugg in suggestions[:3]:
            actions.append(f"{sugg['priority']}. {sugg['title']}")
        return actions


# Convenience function
def analyze_cv(cv_path_or_text: str, jd_text: str) -> Dict:
    """Quick CV analysis function"""
    improver = CVImprover()
    return improver.analyze_and_improve(cv_path_or_text, jd_text)