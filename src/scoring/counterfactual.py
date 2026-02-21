"""
Counterfactual Simulator - Quantifies skill impact on match scores
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from copy import deepcopy

from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine


class CounterfactualSimulator:
    """
    Simulate "what-if" scenarios: What if candidate had skill X?
    Quantifies the match score improvement for each missing skill.
    """
    
    def __init__(self,
                 embedding_engine: Optional[EmbeddingEngine] = None,
                 scoring_engine: Optional[ScoringEngine] = None):
        """
        Initialize counterfactual simulator
        
        Args:
            embedding_engine: Pre-initialized embedding engine
            scoring_engine: Pre-initialized scoring engine
        """
        if embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine
        
        if scoring_engine is None:
            self.scoring_engine = ScoringEngine(self.embedding_engine)
        else:
            self.scoring_engine = scoring_engine
        
        logger.info("CounterfactualSimulator initialized")
    
    def simulate_skill_impact(self,
                             cv_data: Dict,
                             jd_data: Dict,
                             baseline_result: Dict) -> Dict:
        """
        Simulate adding each missing/weak skill and compute score delta
        
        Args:
            cv_data: Parsed CV data
            jd_data: Parsed JD data
            baseline_result: Current match result (baseline)
            
        Returns:
            Dict with skill impact analysis
        """
        logger.info("Starting counterfactual skill impact simulation...")
        
        baseline_score = baseline_result['overall_score']
        
        # Identify skills to simulate
        skills_to_simulate = self._identify_simulation_targets(
            jd_data,
            baseline_result
        )
        
        if not skills_to_simulate:
            logger.info("No skills to simulate - all requirements met")
            return {
                'baseline_score': baseline_score,
                'baseline_percentage': baseline_result['overall_percentage'],
                'simulations': [],
                'recommendations': ["✅ All required skills present - no improvements needed"]
            }
        
        logger.info(f"Simulating {len(skills_to_simulate)} skills...")
        
        # Run simulations
        simulations = []
        for skill_info in skills_to_simulate:
            skill = skill_info['skill']
            current_score = skill_info['current_score']
            
            # Simulate adding this skill
            delta_score, new_score = self._simulate_adding_skill(
                skill,
                cv_data,
                jd_data,
                baseline_score
            )
            
            simulations.append({
                'skill': skill,
                'current_score': round(current_score, 4),
                'current_percentage': f"{current_score * 100:.1f}%",
                'baseline_overall': baseline_score,
                'simulated_overall': new_score,
                'delta_score': round(delta_score, 4),
                'delta_percentage': f"{delta_score * 100:.1f}%",
                'improvement': f"+{delta_score * 100:.1f}%",
                'new_overall_percentage': f"{new_score * 100:.1f}%",
                'priority': self._calculate_priority(delta_score, current_score)
            })
        
        # Sort by impact (highest delta first)
        simulations.sort(key=lambda x: x['delta_score'], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_simulation_recommendations(
            simulations,
            baseline_score
        )
        
        result = {
            'baseline_score': baseline_score,
            'baseline_percentage': baseline_result['overall_percentage'],
            'simulations': simulations,
            'top_skill': simulations[0]['skill'] if simulations else None,
            'top_impact': simulations[0]['delta_percentage'] if simulations else None,
            'recommendations': recommendations,
            'total_potential_gain': sum(s['delta_score'] for s in simulations[:3])  # Top 3
        }
        
        logger.info(f"Simulation complete. Top skill: {result['top_skill']} ({result['top_impact']})")
        return result
    
    def _identify_simulation_targets(self,
                                    jd_data: Dict,
                                    baseline_result: Dict) -> List[Dict]:
        """
        Identify which skills to simulate adding
        
        Args:
            jd_data: Job description data
            baseline_result: Current match result
            
        Returns:
            List of skills to simulate with current scores
        """
        targets = []
        
        # Get required skills that are weak or missing
        required_details = baseline_result['breakdown']['required_skills']['details']
        
        for skill_info in required_details['skills']:
            # Simulate if weak or missing (score < 0.80)
            if skill_info['score'] < 0.80:
                targets.append({
                    'skill': skill_info['skill'],
                    'current_score': skill_info['score'],
                    'strength': skill_info['strength']
                })
        
        # Also check preferred skills if they exist
        preferred_details = baseline_result['breakdown']['preferred_skills']['details']
        if preferred_details.get('skills'):
            for skill_info in preferred_details['skills']:
                if skill_info['score'] < 0.80:
                    targets.append({
                        'skill': skill_info['skill'],
                        'current_score': skill_info['score'],
                        'strength': skill_info['strength']
                    })
        
        return targets
    
    def _simulate_adding_skill(self,
                              skill: str,
                              cv_data: Dict,
                              jd_data: Dict,
                              baseline_score: float) -> Tuple[float, float]:
        """
        Simulate adding a skill to the CV and recompute score
        
        Args:
            skill: Skill to add
            cv_data: CV data
            jd_data: JD data
            baseline_score: Current score
            
        Returns:
            (delta_score, new_overall_score)
        """
        # Create synthetic CV with added skill
        synthetic_cv = self._create_synthetic_cv(cv_data, skill)
        
        # Recompute match score with synthetic CV
        new_result = self.scoring_engine.compute_match_score(synthetic_cv, jd_data)
        new_score = new_result['overall_score']
        
        # Calculate delta
        delta = new_score - baseline_score
        
        return delta, new_score
    
    def _create_synthetic_cv(self, cv_data: Dict, skill: str) -> Dict:
        """
        Create a synthetic CV with added skill
        
        Args:
            cv_data: Original CV data
            skill: Skill to add
            
        Returns:
            Modified CV data
        """
        # Deep copy to avoid modifying original
        synthetic_cv = deepcopy(cv_data)
        
        # Generate a plausible skill mention
        skill_mention = self._generate_skill_mention(skill)
        
        # Add to skills section
        if 'skills' in synthetic_cv['sections']:
            synthetic_cv['sections']['skills'] += f"\n{skill_mention}"
        else:
            synthetic_cv['sections']['skills'] = skill_mention
        
        # Also add to full text
        synthetic_cv['text'] += f"\n{skill_mention}"
        
        return synthetic_cv
    
    def _generate_skill_mention(self, skill: str) -> str:
        """
        Generate a plausible skill mention for synthetic CV
        
        Args:
            skill: Skill name
            
        Returns:
            Skill mention text
        """
        # Templates for different skill types
        templates = [
            f"Proficient in {skill}",
            f"Experience with {skill}",
            f"Strong {skill} skills",
            f"{skill} expertise",
            f"Used {skill} in production",
        ]
        
        # Choose template based on skill name hash (deterministic)
        idx = hash(skill) % len(templates)
        return templates[idx]
    
    def _calculate_priority(self, delta_score: float, current_score: float) -> str:
        """
        Calculate learning priority based on impact
        
        Args:
            delta_score: Score improvement
            current_score: Current skill score
            
        Returns:
            Priority level: "Critical", "High", "Medium", "Low"
        """
        # High impact (>5% improvement) = Critical/High
        # Current score very low (<0.50) = Higher priority
        
        impact_percentage = delta_score * 100
        
        if impact_percentage >= 8.0:
            return "Critical"
        elif impact_percentage >= 5.0:
            return "High"
        elif impact_percentage >= 2.0:
            return "Medium"
        else:
            return "Low"
    
    def _generate_simulation_recommendations(self,
                                           simulations: List[Dict],
                                           baseline_score: float) -> List[str]:
        """
        Generate actionable recommendations from simulations
        
        Args:
            simulations: Simulation results
            baseline_score: Current score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not simulations:
            return ["✅ All skills present at strong levels"]
        
        # Overall assessment
        top_3_gain = sum(s['delta_score'] for s in simulations[:3])
        potential_score = baseline_score + top_3_gain
        
        recommendations.append(
            f"🎯 Learning top 3 skills could increase score to {potential_score * 100:.1f}% "
            f"(+{top_3_gain * 100:.1f}%)"
        )
        
        # Top skill recommendation
        top = simulations[0]
        recommendations.append(
            f"🔥 PRIORITY #1: Learn {top['skill']} → +{top['delta_percentage']} improvement"
        )
        
        # Second priority
        if len(simulations) > 1:
            second = simulations[1]
            recommendations.append(
                f"📚 PRIORITY #2: Learn {second['skill']} → +{second['delta_percentage']} improvement"
            )
        
        # Critical skills (high impact)
        critical = [s for s in simulations if s['priority'] == 'Critical']
        if critical and len(critical) > 1:
            skills_list = ", ".join([s['skill'] for s in critical[:3]])
            recommendations.append(
                f"⚠️ Critical skills needed: {skills_list}"
            )
        
        # Efficiency recommendation
        high_impact = [s for s in simulations if s['delta_score'] >= 0.05]
        if len(high_impact) >= 2:
            recommendations.append(
                f"💡 Focus on {len(high_impact)} high-impact skills for maximum ROI"
            )
        
        return recommendations


# Convenience function
def simulate_skill_impact(cv_data: Dict,
                         jd_data: Dict,
                         baseline_result: Dict,
                         embedding_engine: Optional[EmbeddingEngine] = None) -> Dict:
    """Quick simulation function"""
    simulator = CounterfactualSimulator(embedding_engine)
    return simulator.simulate_skill_impact(cv_data, jd_data, baseline_result)