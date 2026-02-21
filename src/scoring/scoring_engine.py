"""
Scoring Engine - Computes weighted CV-JD match scores
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from src.embeddings.embedding_engine import EmbeddingEngine


class ScoringEngine:
    """
    Compute weighted match scores between CV and Job Description
    
    Scoring Formula:
    Overall Score = 0.60 × Required Skills Score
                  + 0.25 × Preferred Skills Score  
                  + 0.15 × Experience Alignment Score
    """
    
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        """
        Initialize scoring engine
        
        Args:
            embedding_engine: Pre-initialized embedding engine (optional)
        """
        # Weights for different components
        self.weights = {
            'required_skills': 0.60,
            'preferred_skills': 0.25,
            'experience': 0.15
        }
        
        # Threshold for considering a skill as "matched"
        self.match_threshold = 0.60
        
        # Initialize or use provided embedding engine
        if embedding_engine is None:
            logger.info("Creating new embedding engine for scoring")
            self.embedding_engine = EmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine
            logger.info("Using provided embedding engine")
    
    def compute_match_score(self,
                           cv_data: Dict,
                           jd_data: Dict) -> Dict:
        """
        Compute overall match score between CV and JD
        
        Args:
            cv_data: Parsed CV data from cv_parser
            jd_data: Parsed JD data from jd_parser
            
        Returns:
            Dict with overall score, breakdown, and details
        """
        logger.info("Computing match score...")
        
        # Extract CV text/sections
        cv_text = cv_data.get('text', '')
        cv_sections = cv_data.get('sections', {})
        
        # Extract JD requirements
        required_skills = jd_data.get('required_skills', [])
        preferred_skills = jd_data.get('preferred_skills', [])
        jd_experience = jd_data.get('experience_years', {})
        
        # Compute required skills score
        required_score, required_details = self._score_skills(
            cv_text, 
            cv_sections,
            required_skills,
            skill_type='required'
        )
        
        # Compute preferred skills score
        preferred_score, preferred_details = self._score_skills(
            cv_text,
            cv_sections, 
            preferred_skills,
            skill_type='preferred'
        )
        
        # Compute experience alignment score
        experience_score, experience_details = self._score_experience(
            cv_text,
            cv_sections,
            jd_experience
        )
        
        # Calculate weighted overall score
        overall_score = (
            self.weights['required_skills'] * required_score +
            self.weights['preferred_skills'] * preferred_score +
            self.weights['experience'] * experience_score
        )
        
        # Round to 2 decimal places
        overall_score = round(overall_score, 4)
        required_score = round(required_score, 4)
        preferred_score = round(preferred_score, 4)
        experience_score = round(experience_score, 4)
        
        result = {
            'overall_score': overall_score,
            'overall_percentage': f"{overall_score * 100:.1f}%",
            'breakdown': {
                'required_skills': {
                    'score': required_score,
                    'percentage': f"{required_score * 100:.1f}%",
                    'weight': f"{self.weights['required_skills'] * 100:.0f}%",
                    'contribution': f"{required_score * self.weights['required_skills'] * 100:.1f}%",
                    'details': required_details
                },
                'preferred_skills': {
                    'score': preferred_score,
                    'percentage': f"{preferred_score * 100:.1f}%",
                    'weight': f"{self.weights['preferred_skills'] * 100:.0f}%",
                    'contribution': f"{preferred_score * self.weights['preferred_skills'] * 100:.1f}%",
                    'details': preferred_details
                },
                'experience': {
                    'score': experience_score,
                    'percentage': f"{experience_score * 100:.1f}%",
                    'weight': f"{self.weights['experience'] * 100:.0f}%",
                    'contribution': f"{experience_score * self.weights['experience'] * 100:.1f}%",
                    'details': experience_details
                }
            },
            'interpretation': self._interpret_score(overall_score)
        }
        
        logger.info(f"Match score computed: {overall_score * 100:.1f}%")
        return result
    
    def _score_skills(self,
                     cv_text: str,
                     cv_sections: Dict,
                     skills: List[str],
                     skill_type: str = 'required') -> Tuple[float, Dict]:
        """
        Score how well CV matches a list of skills
        
        Args:
            cv_text: Full CV text
            cv_sections: Parsed CV sections
            skills: List of skills to match
            skill_type: 'required' or 'preferred'
            
        Returns:
            (average_score, details_dict)
        """
        if not skills:
            logger.debug(f"No {skill_type} skills to score")
            return 1.0, {'matched': 0, 'total': 0, 'skills': []}
        
        # Extract relevant CV sections for skill matching
        skill_text = self._extract_skill_context(cv_sections)
        
        # If no skill context found, use full text
        if not skill_text:
            skill_text = cv_text
        
        # Encode all skills
        skill_embeddings = self.embedding_engine.encode(skills)
        
        # Encode CV skill context
        cv_embedding = self.embedding_engine.encode(skill_text)
        
        # Compute similarities
        skill_scores = []
        skill_details = []
        
        for i, skill in enumerate(skills):
            similarity = self.embedding_engine.compute_similarity(
                cv_embedding,
                skill_embeddings[i]
            )
            
            # Classify match strength
            if similarity >= 0.80:
                strength = "Strong"
            elif similarity >= self.match_threshold:
                strength = "Partial"
            else:
                strength = "Weak/Missing"
            
            skill_scores.append(similarity)
            skill_details.append({
                'skill': skill,
                'score': round(float(similarity), 4),
                'percentage': f"{similarity * 100:.1f}%",
                'strength': strength,
                'matched': similarity >= self.match_threshold
            })
        
        # Calculate average score
        avg_score = np.mean(skill_scores)
        
        # Count matched skills
        matched_count = sum(1 for s in skill_scores if s >= self.match_threshold)
        
        details = {
            'matched': matched_count,
            'total': len(skills),
            'match_rate': f"{matched_count}/{len(skills)}",
            'skills': skill_details
        }
        
        logger.debug(f"{skill_type.title()} skills: {matched_count}/{len(skills)} matched")
        
        return float(avg_score), details
    
    def _extract_skill_context(self, cv_sections: Dict) -> str:
        """
        Extract text from CV sections most relevant for skill matching
        
        Args:
            cv_sections: Parsed CV sections
            
        Returns:
            Combined text from relevant sections
        """
        relevant_sections = ['skills', 'experience', 'projects', 'certifications']
        
        context_parts = []
        for section_name in relevant_sections:
            if section_name in cv_sections:
                context_parts.append(cv_sections[section_name])
        
        return "\n\n".join(context_parts)
    
    def _score_experience(self,
                         cv_text: str,
                         cv_sections: Dict,
                         jd_experience: Optional[Dict]) -> Tuple[float, Dict]:
        """
        Score experience alignment using sigmoid normalization
        
        Args:
            cv_text: Full CV text
            cv_sections: Parsed CV sections
            jd_experience: JD experience requirements
            
        Returns:
            (experience_score, details_dict)
        """
        if not jd_experience or jd_experience.get('min_years') is None:
            logger.debug("No experience requirement in JD")
            return 1.0, {
                'cv_years': 'Not extracted',
                'jd_required': 'Not specified',
                'score': 1.0,
                'note': 'No experience requirement specified'
            }
        
        # Extract years from CV
        cv_years = self._extract_cv_years(cv_text, cv_sections)
        jd_min_years = jd_experience['min_years']
        
        if cv_years is None:
            logger.warning("Could not extract years of experience from CV")
            # Give benefit of doubt - assume they meet requirement
            return 0.70, {
                'cv_years': 'Not found in CV',
                'jd_required': f"{jd_min_years}+ years",
                'score': 0.70,
                'note': 'Could not determine experience from CV'
            }
        
        # Compute score using sigmoid function
        # This avoids cliff-edge behavior (e.g., 2.9 years vs 3.0 years)
        score = self._sigmoid_experience_score(cv_years, jd_min_years)
        
        details = {
            'cv_years': cv_years,
            'jd_required': f"{jd_min_years}+ years",
            'score': round(score, 4),
            'meets_requirement': cv_years >= jd_min_years,
            'note': self._experience_interpretation(cv_years, jd_min_years)
        }
        
        logger.debug(f"Experience: CV has {cv_years} years, JD requires {jd_min_years}+ years")
        
        return score, details
    
    def _extract_cv_years(self, cv_text: str, cv_sections: Dict) -> Optional[float]:
        """
        Extract years of experience from CV
        
        Args:
            cv_text: Full CV text
            cv_sections: Parsed CV sections
            
        Returns:
            Years of experience (float) or None
        """
        import re
        
        # Look for explicit mentions
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:\s*(\d+)\+?\s*years?',
        ]
        
        text = cv_text.lower()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        
        # If not found explicitly, try to count from experience section
        if 'experience' in cv_sections:
            # Simple heuristic: count year ranges mentioned
            exp_text = cv_sections['experience']
            year_matches = re.findall(r'20\d{2}', exp_text)
            
            if len(year_matches) >= 2:
                # Calculate rough experience from first to last year
                years = [int(y) for y in year_matches]
                experience_years = max(years) - min(years)
                
                # Cap at reasonable maximum
                if 0 < experience_years <= 50:
                    return float(experience_years)
        
        return None
    
    def _sigmoid_experience_score(self, cv_years: float, required_years: float) -> float:
        """
        Compute experience score using sigmoid function
        
        This creates smooth transitions:
        - Well below requirement: ~0.3-0.5
        - Slightly below: ~0.7-0.8
        - Meets requirement: ~0.9
        - Exceeds: ~1.0
        
        Args:
            cv_years: Years from CV
            required_years: Years required by JD
            
        Returns:
            Score between 0 and 1
        """
        # Difference from requirement
        diff = cv_years - required_years
        
        # Sigmoid with smooth curve
        # k controls steepness (lower = more forgiving)
        k = 0.5
        score = 1 / (1 + np.exp(-k * diff))
        
        # Ensure minimum score of 0.3 if they have any experience
        if cv_years > 0:
            score = max(score, 0.30)
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        return float(score)
    
    def _experience_interpretation(self, cv_years: float, required_years: float) -> str:
        """Generate human-readable experience interpretation"""
        diff = cv_years - required_years
        
        if diff >= 2:
            return f"Exceeds requirement by {diff:.0f} years"
        elif diff >= 0:
            return "Meets requirement"
        elif diff >= -1:
            return f"Close to requirement (short by {abs(diff):.0f} year)"
        else:
            return f"Below requirement (short by {abs(diff):.0f} years)"
    
    def _interpret_score(self, score: float) -> Dict:
        """
        Provide interpretation of overall match score
        
        Args:
            score: Overall match score (0-1)
            
        Returns:
            Dict with interpretation details
        """
        percentage = score * 100
        
        if percentage >= 85:
            level = "Excellent Match"
            recommendation = "Strongly recommended - candidate is highly qualified"
            color = "green"
        elif percentage >= 70:
            level = "Good Match"
            recommendation = "Recommended - candidate meets most requirements"
            color = "blue"
        elif percentage >= 55:
            level = "Moderate Match"
            recommendation = "Consider with caution - some gaps in qualifications"
            color = "yellow"
        else:
            level = "Weak Match"
            recommendation = "Not recommended - significant gaps in requirements"
            color = "red"
        
        return {
            'level': level,
            'recommendation': recommendation,
            'color': color
        }


# Convenience function
def compute_match(cv_data: Dict, jd_data: Dict, 
                 embedding_engine: Optional[EmbeddingEngine] = None) -> Dict:
    """
    Quick function to compute match score
    
    Args:
        cv_data: Parsed CV data
        jd_data: Parsed JD data
        embedding_engine: Optional pre-initialized engine
        
    Returns:
        Match score result dict
    """
    engine = ScoringEngine(embedding_engine)
    return engine.compute_match_score(cv_data, jd_data)