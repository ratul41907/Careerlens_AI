"""
Scoring Engine - Match CV against JD
"""
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class ScoringEngine:
    """Score CV-JD match using embeddings and semantic similarity"""
    
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
    
    def compute_match_score(self, cv_data: Dict, jd_data: Dict) -> Dict:
        """
        Compute match score between CV and JD
        
        Args:
            cv_data: Parsed CV data
            jd_data: Parsed JD data
            
        Returns:
            Dictionary with scores and breakdown
        """
        try:
            # Extract sections - handle both dict with 'sections' key and direct dict
            cv_sections = cv_data.get('sections', cv_data) if isinstance(cv_data, dict) else {}
            jd_sections = jd_data.get('sections', jd_data) if isinstance(jd_data, dict) else {}
            
            # Get CV text - handle multiple formats
            cv_text = cv_data.get('text', '')
            if not cv_text and isinstance(cv_sections, dict):
                # Reconstruct text from sections if needed
                cv_text = ' '.join([
                    str(v) if isinstance(v, str) else ' '.join(v) if isinstance(v, list) else ''
                    for v in cv_sections.values()
                ])
            
            # Get JD text - handle multiple formats
            jd_text = jd_data.get('text', '')
            if not jd_text and isinstance(jd_sections, dict):
                jd_text = ' '.join([
                    str(v) if isinstance(v, str) else ' '.join(v) if isinstance(v, list) else ''
                    for v in jd_sections.values()
                ])
            
            # Extract skills - convert to lists if needed
            cv_skills = cv_sections.get('skills', [])
            if isinstance(cv_skills, str):
                cv_skills = [s.strip() for s in cv_skills.split(',') if s.strip()]
            elif not isinstance(cv_skills, list):
                cv_skills = []
            
            jd_required_skills = jd_sections.get('required_skills', [])
            if isinstance(jd_required_skills, str):
                jd_required_skills = [s.strip() for s in jd_required_skills.split(',') if s.strip()]
            elif not isinstance(jd_required_skills, list):
                jd_required_skills = []
            
            jd_preferred_skills = jd_sections.get('preferred_skills', [])
            if isinstance(jd_preferred_skills, str):
                jd_preferred_skills = [s.strip() for s in jd_preferred_skills.split(',') if s.strip()]
            elif not isinstance(jd_preferred_skills, list):
                jd_preferred_skills = []
            
            # Score required skills (60% weight)
            required_score = self._score_skills(cv_skills, jd_required_skills, cv_text, jd_text)
            
            # Score preferred skills (25% weight)
            preferred_score = self._score_skills(cv_skills, jd_preferred_skills, cv_text, jd_text)
            
            # Score experience (15% weight)
            experience_score = self._score_experience(cv_sections, jd_sections)
            
            # Calculate weighted overall score
            overall_score = (
                required_score * 0.60 +
                preferred_score * 0.25 +
                experience_score * 0.15
            )
            
            # Create breakdown
            breakdown = {
                'required_skills': {
                    'score': required_score,
                    'weight': 0.60,
                    'percentage': f"{required_score*100:.1f}%",
                    'contribution': f"{required_score*0.60*100:.1f}%",
                    'details': self._get_skill_details(cv_skills, jd_required_skills, cv_text, jd_text)
                },
                'preferred_skills': {
                    'score': preferred_score,
                    'weight': 0.25,
                    'percentage': f"{preferred_score*100:.1f}%",
                    'contribution': f"{preferred_score*0.25*100:.1f}%",
                    'details': self._get_skill_details(cv_skills, jd_preferred_skills, cv_text, jd_text)
                },
                'experience': {
                    'score': experience_score,
                    'weight': 0.15,
                    'percentage': f"{experience_score*100:.1f}%",
                    'contribution': f"{experience_score*0.15*100:.1f}%",
                    'details': self._get_experience_details(cv_sections, jd_sections)
                }
            }
            
            return {
                'overall_score': overall_score,
                'overall_percentage': f"{overall_score*100:.1f}%",
                'breakdown': breakdown,
                'interpretation': self._interpret_score(overall_score)
            }
            
        except Exception as e:
            logger.error(f"Error computing match score: {e}")
            import traceback
            traceback.print_exc()
            return {
                'overall_score': 0.0,
                'overall_percentage': "0.0%",
                'breakdown': {},
                'error': str(e)
            }
    
    def _score_skills(self, cv_skills: List[str], jd_skills: List[str], 
                     cv_text: str, jd_text: str) -> float:
        """Score skill match using semantic similarity"""
        try:
            # Ensure we have lists
            if not isinstance(cv_skills, list):
                cv_skills = []
            if not isinstance(jd_skills, list):
                jd_skills = []
            
            if not jd_skills:
                return 1.0  # No required skills = perfect match
            
            if not cv_skills:
                return 0.0  # No CV skills but JD has requirements = no match
            
            # Convert to strings
            cv_skills = [str(s) for s in cv_skills]
            jd_skills = [str(s) for s in jd_skills]
            
            # Create skill text
            cv_skill_text = ' '.join(cv_skills)
            jd_skill_text = ' '.join(jd_skills)
            
            # Get embeddings
            #cv_embedding = self.embedding_engine.get_embedding(cv_skill_text)
            #jd_embedding = self.embedding_engine.get_embedding(jd_skill_text)
            
            cv_embedding = self.embedding_engine.encode(cv_skill_text)
            jd_embedding = self.embedding_engine.encode(jd_skill_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                cv_embedding.reshape(1, -1),
                jd_embedding.reshape(1, -1)
            )[0][0]
            
            # Normalize to 0-1 range
            score = max(0.0, min(1.0, similarity))
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring skills: {e}")
            return 0.0
    
    def _score_experience(self, cv_sections: Dict, jd_sections: Dict) -> float:
        """Score experience match"""
        try:
            # Get experience from CV
            cv_exp = cv_sections.get('experience', '')
            cv_exp_str = str(cv_exp) if cv_exp else '0'
            
            # Extract years from CV
            cv_years = 0
            if 'year' in cv_exp_str.lower():
                import re
                numbers = re.findall(r'\d+', cv_exp_str)
                if numbers:
                    cv_years = int(numbers[0])
            
            # Get experience requirement from JD
            jd_exp = jd_sections.get('experience', {})
            if isinstance(jd_exp, dict):
                jd_years = jd_exp.get('min_years', 0)
            elif isinstance(jd_exp, str):
                import re
                numbers = re.findall(r'\d+', jd_exp)
                jd_years = int(numbers[0]) if numbers else 0
            else:
                jd_years = 0
            
            # Score based on experience match
            if jd_years == 0:
                return 1.0  # No requirement
            
            if cv_years >= jd_years:
                return 1.0  # Meets or exceeds
            elif cv_years >= jd_years * 0.75:
                return 0.8  # Close enough
            elif cv_years >= jd_years * 0.5:
                return 0.6  # Partial match
            else:
                return 0.3  # Significant gap
                
        except Exception as e:
            logger.error(f"Error scoring experience: {e}")
            return 0.5
    
    def _get_skill_details(self, cv_skills: List[str], jd_skills: List[str], 
                          cv_text: str, jd_text: str) -> Dict:
        """Get detailed skill matching information"""
        try:
            # Ensure we have lists
            if not isinstance(cv_skills, list):
                cv_skills = []
            if not isinstance(jd_skills, list):
                jd_skills = []
            
            # Ensure we have strings for text
            cv_text = str(cv_text) if cv_text else ""
            jd_text = str(jd_text) if jd_text else ""
            
            matched_skills = []
            missing_skills = []
            
            for jd_skill in jd_skills:
                jd_skill_str = str(jd_skill).strip()
                if not jd_skill_str:
                    continue
                
                # Check if skill is in CV
                is_matched = False
                
                # Check in CV skills list
                for cv_skill in cv_skills:
                    cv_skill_str = str(cv_skill).strip()
                    if self._skills_match(cv_skill_str, jd_skill_str):
                        matched_skills.append(jd_skill_str)
                        is_matched = True
                        break
                
                # If not matched in list, check in CV text
                if not is_matched and jd_skill_str.lower() in cv_text.lower():
                    matched_skills.append(jd_skill_str)
                    is_matched = True
                
                # If still not matched, it's missing
                if not is_matched:
                    missing_skills.append(jd_skill_str)
            
            return {
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'match_rate': f"{len(matched_skills)}/{len(jd_skills)}" if jd_skills else "0/0"
            }
            
        except Exception as e:
            logger.error(f"Error in _get_skill_details: {e}")
            return {
                'matched_skills': [],
                'missing_skills': [],
                'match_rate': "0/0"
            }
    
    def _skills_match(self, cv_skill: str, jd_skill: str) -> bool:
        """Check if two skills match (case-insensitive, fuzzy)"""
        cv_lower = cv_skill.lower().strip()
        jd_lower = jd_skill.lower().strip()
        
        # Exact match
        if cv_lower == jd_lower:
            return True
        
        # One contains the other
        if cv_lower in jd_lower or jd_lower in cv_lower:
            return True
        
        # Common synonyms
        synonyms = {
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'k8s': 'kubernetes',
            'docker': 'containerization',
            'ci/cd': 'continuous integration'
        }
        
        cv_normalized = synonyms.get(cv_lower, cv_lower)
        jd_normalized = synonyms.get(jd_lower, jd_lower)
        
        return cv_normalized == jd_normalized
    
    def _get_experience_details(self, cv_sections: Dict, jd_sections: Dict) -> Dict:
        """Get experience matching details"""
        try:
            cv_exp = str(cv_sections.get('experience', 'Not specified'))
            jd_exp = jd_sections.get('experience', {})
            
            if isinstance(jd_exp, dict):
                jd_exp_text = jd_exp.get('text', 'Not specified')
            else:
                jd_exp_text = str(jd_exp)
            
            return {
                'cv_experience': cv_exp,
                'jd_requirement': jd_exp_text
            }
        except:
            return {
                'cv_experience': 'Not specified',
                'jd_requirement': 'Not specified'
            }
    
    def _interpret_score(self, score: float) -> Dict:
        """Interpret overall match score"""
        if score >= 0.80:
            return {
                'level': 'Excellent Match',
                'recommendation': 'Highly recommended - proceed with application',
                'color': 'green'
            }
        elif score >= 0.65:
            return {
                'level': 'Good Match',
                'recommendation': 'Good candidate - consider applying',
                'color': 'blue'
            }
        elif score >= 0.50:
            return {
                'level': 'Moderate Match',
                'recommendation': 'Some gaps exist - address in cover letter',
                'color': 'orange'
            }
        else:
            return {
                'level': 'Low Match',
                'recommendation': 'Significant gaps - consider upskilling first',
                'color': 'red'
            }