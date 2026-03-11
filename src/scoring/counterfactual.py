"""
Counterfactual Simulator - Quantifies skill impact on match scores
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from copy import deepcopy


class CounterfactualSimulator:
    """
    Simulate "what-if" scenarios: What if candidate had skill X?
    """
    
    def __init__(self, scoring_engine):
        """
        Initialize counterfactual simulator
        
        Args:
            scoring_engine: ScoringEngine instance
        """
        self.scoring_engine = scoring_engine
        logger.info("CounterfactualSimulator initialized")
    
    def simulate_skill_addition(self, cv_data: Dict, jd_data: Dict, skill_to_add: str) -> float:
        """
        Simulate adding a skill to CV and return new match score
        
        Args:
            cv_data: Current CV data
            jd_data: Job description data
            skill_to_add: Skill to simulate adding
            
        Returns:
            Predicted match score percentage after adding skill
        """
        try:
            # Create modified CV with new skill
            modified_cv = deepcopy(cv_data)
            
            # Add skill to text
            if 'text' in modified_cv:
                modified_cv['text'] = modified_cv['text'] + f"\n\nSkills: {skill_to_add}"
            
            # Add to skills section
            if 'sections' in modified_cv:
                if 'skills' in modified_cv['sections']:
                    modified_cv['sections']['skills'] = modified_cv['sections']['skills'] + f", {skill_to_add}"
                else:
                    modified_cv['sections']['skills'] = skill_to_add
            else:
                modified_cv['sections'] = {'skills': skill_to_add}
            
            # Compute new match score
            new_result = self.scoring_engine.compute_match_score(modified_cv, jd_data)
            
            return new_result['overall_percentage']
        
        except Exception as e:
            logger.error(f"Error simulating {skill_to_add}: {str(e)}")
            # Return baseline score as fallback
            baseline = self.scoring_engine.compute_match_score(cv_data, jd_data)
            return baseline['overall_percentage']