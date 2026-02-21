"""
Explainability Layer - Provides detailed evidence for match scores
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from src.embeddings.embedding_engine import EmbeddingEngine


class ExplainabilityEngine:
    """
    Generate human-readable explanations for CV-JD match scores
    """
    
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        """
        Initialize explainability engine
        
        Args:
            embedding_engine: Pre-initialized embedding engine
        """
        if embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine
        
        self.evidence_threshold = 0.60  # Minimum score to cite as evidence
    
    def explain_match(self,
                     cv_data: Dict,
                     jd_data: Dict,
                     match_result: Dict) -> Dict:
        """
        Generate detailed explanations for match result
        
        Args:
            cv_data: Parsed CV data
            jd_data: Parsed JD data
            match_result: Result from scoring engine
            
        Returns:
            Dict with detailed explanations
        """
        logger.info("Generating explainability report...")
        
        cv_text = cv_data.get('text', '')
        cv_sections = cv_data.get('sections', {})
        
        required_skills = jd_data.get('required_skills', [])
        preferred_skills = jd_data.get('preferred_skills', [])
        
        # Extract evidence for each skill
        required_explanations = self._explain_skills(
            cv_text,
            cv_sections,
            required_skills,
            match_result['breakdown']['required_skills']['details']['skills']
        )
        
        preferred_explanations = []
        if preferred_skills:
            preferred_explanations = self._explain_skills(
                cv_text,
                cv_sections,
                preferred_skills,
                match_result['breakdown']['preferred_skills']['details'].get('skills', [])
            )
        
        # Identify missing skills
        missing_skills = self._identify_missing_skills(
            required_skills,
            match_result['breakdown']['required_skills']['details']['skills']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            match_result,
            missing_skills
        )
        
        explanation = {
            'overall_assessment': self._overall_assessment(match_result),
            'required_skills_evidence': required_explanations,
            'preferred_skills_evidence': preferred_explanations,
            'missing_skills': missing_skills,
            'recommendations': recommendations,
            'summary': self._generate_summary(match_result, missing_skills)
        }
        
        logger.info("Explainability report generated")
        return explanation
    
    def _explain_skills(self,
                       cv_text: str,
                       cv_sections: Dict,
                       skills: List[str],
                       skill_scores: List[Dict]) -> List[Dict]:
        """
        Generate evidence citations for each skill
        
        Args:
            cv_text: Full CV text
            cv_sections: Parsed CV sections
            skills: List of skills
            skill_scores: Scores from matching
            
        Returns:
            List of skill explanations with evidence
        """
        explanations = []
        
        # Extract evidence sentences from CV
        evidence_sentences = self._extract_evidence_sentences(cv_sections)
        
        if not evidence_sentences:
            evidence_sentences = cv_text.split('.')
        
        # Clean sentences
        evidence_sentences = [s.strip() for s in evidence_sentences if len(s.strip()) > 20]
        
        for skill_info in skill_scores:
            skill = skill_info['skill']
            score = skill_info['score']
            strength = skill_info['strength']
            
            # Find best matching evidence
            evidence = self._find_best_evidence(skill, evidence_sentences)
            
            explanation = {
                'skill': skill,
                'score': score,
                'percentage': skill_info['percentage'],
                'strength': strength,
                'matched': skill_info['matched'],
                'evidence': evidence,
                'explanation': self._skill_explanation(skill, score, strength, evidence)
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _extract_evidence_sentences(self, cv_sections: Dict) -> List[str]:
        """
        Extract individual sentences from relevant CV sections
        
        Args:
            cv_sections: Parsed CV sections
            
        Returns:
            List of evidence sentences
        """
        relevant_sections = ['experience', 'projects', 'skills', 'certifications']
        
        sentences = []
        for section_name in relevant_sections:
            if section_name in cv_sections:
                section_text = cv_sections[section_name]
                
                # Split by bullet points and newlines
                lines = section_text.replace('•', '\n').split('\n')
                
                for line in lines:
                    line = line.strip()
                    if len(line) > 20:  # Meaningful content
                        # Further split by periods if needed
                        sub_sentences = line.split('.')
                        sentences.extend([s.strip() for s in sub_sentences if len(s.strip()) > 20])
        
        return sentences
    
    def _find_best_evidence(self,
                           skill: str,
                           evidence_sentences: List[str]) -> Optional[Dict]:
        """
        Find the best evidence sentence for a skill
        
        Args:
            skill: Skill to find evidence for
            evidence_sentences: List of candidate sentences
            
        Returns:
            Dict with evidence text and score, or None
        """
        if not evidence_sentences:
            return None
        
        # Encode skill and all evidence sentences
        skill_embedding = self.embedding_engine.encode(skill)
        evidence_embeddings = self.embedding_engine.encode(evidence_sentences)
        
        # Compute similarities
        similarities = []
        for evidence_emb in evidence_embeddings:
            sim = self.embedding_engine.compute_similarity(skill_embedding, evidence_emb)
            similarities.append(sim)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.evidence_threshold:
            return {
                'text': evidence_sentences[best_idx],
                'similarity': round(float(best_score), 4),
                'confidence': 'High' if best_score >= 0.75 else 'Medium'
            }
        
        return None
    
    def _skill_explanation(self,
                          skill: str,
                          score: float,
                          strength: str,
                          evidence: Optional[Dict]) -> str:
        """
        Generate natural language explanation for a skill
        
        Args:
            skill: Skill name
            score: Match score
            strength: Strength classification
            evidence: Evidence dict or None
            
        Returns:
            Human-readable explanation
        """
        if strength == "Strong":
            base = f"Strong evidence of {skill} expertise."
        elif strength == "Partial":
            base = f"Some evidence of {skill} experience."
        else:
            base = f"Little to no evidence of {skill} in CV."
        
        if evidence:
            return f"{base} Found in CV: \"{evidence['text'][:100]}...\""
        else:
            return f"{base} No clear evidence found in CV."
    
    def _identify_missing_skills(self,
                                required_skills: List[str],
                                skill_scores: List[Dict]) -> List[Dict]:
        """
        Identify skills that are missing or weak
        
        Args:
            required_skills: List of required skills from JD
            skill_scores: Match scores for each skill
            
        Returns:
            List of missing/weak skills with details
        """
        missing = []
        
        for skill_info in skill_scores:
            if skill_info['strength'] in ['Weak/Missing']:
                missing.append({
                    'skill': skill_info['skill'],
                    'current_score': skill_info['percentage'],
                    'gap': f"{(0.80 - skill_info['score']) * 100:.1f}% below strong match",
                    'priority': 'High' if skill_info['score'] < 0.50 else 'Medium'
                })
        
        # Sort by priority
        missing.sort(key=lambda x: x['current_score'])
        
        return missing
    
    def _generate_recommendations(self,
                                 match_result: Dict,
                                 missing_skills: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            match_result: Match result from scoring
            missing_skills: List of missing skills
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        overall_score = match_result['overall_score']
        
        # Overall assessment
        if overall_score >= 0.85:
            recommendations.append("✅ Excellent match - proceed with application confidently")
        elif overall_score >= 0.70:
            recommendations.append("✅ Good match - application recommended")
            recommendations.append("💡 Consider highlighting relevant experience in cover letter")
        elif overall_score >= 0.55:
            recommendations.append("⚠️ Moderate match - address skill gaps before applying")
        else:
            recommendations.append("❌ Weak match - significant skill development needed")
        
        # Skill-specific recommendations
        required_details = match_result['breakdown']['required_skills']['details']
        matched_count = required_details['matched']
        total_count = required_details['total']
        
        if matched_count < total_count:
            gap = total_count - matched_count
            recommendations.append(f"📚 Learn {gap} missing required skill(s) to improve match")
        
        # Top 3 missing skills to learn
        if missing_skills:
            top_missing = missing_skills[:3]
            skills_list = ", ".join([s['skill'] for s in top_missing])
            recommendations.append(f"🎯 Priority skills to learn: {skills_list}")
        
        # Experience recommendation
        exp_details = match_result['breakdown']['experience']['details']
        if not exp_details.get('meets_requirement', True):
            cv_years = exp_details.get('cv_years', 0)
            jd_years = exp_details.get('jd_required', '').split('+')[0]
            
            try:
                gap_years = float(jd_years) - cv_years
                if gap_years > 0:
                    recommendations.append(f"⏳ Gain {gap_years:.0f} more year(s) of experience to meet requirement")
            except:
                pass
        
        return recommendations
    
    def _overall_assessment(self, match_result: Dict) -> str:
        """
        Generate overall assessment narrative
        
        Args:
            match_result: Match result from scoring
            
        Returns:
            Assessment text
        """
        score = match_result['overall_score']
        interpretation = match_result['interpretation']
        
        required_pct = match_result['breakdown']['required_skills']['percentage']
        exp_pct = match_result['breakdown']['experience']['percentage']
        
        assessment = f"{interpretation['level']} ({score * 100:.1f}%). "
        assessment += f"Required skills match: {required_pct}. "
        assessment += f"Experience alignment: {exp_pct}. "
        assessment += interpretation['recommendation']
        
        return assessment
    
    def _generate_summary(self,
                         match_result: Dict,
                         missing_skills: List[Dict]) -> Dict:
        """
        Generate executive summary
        
        Args:
            match_result: Match result
            missing_skills: Missing skills list
            
        Returns:
            Summary dict
        """
        required_details = match_result['breakdown']['required_skills']['details']
        
        return {
            'overall_score': match_result['overall_percentage'],
            'match_level': match_result['interpretation']['level'],
            'skills_matched': f"{required_details['matched']}/{required_details['total']}",
            'missing_count': len(missing_skills),
            'top_strength': self._find_top_strength(required_details['skills']),
            'top_weakness': missing_skills[0]['skill'] if missing_skills else 'None'
        }
    
    def _find_top_strength(self, skill_scores: List[Dict]) -> str:
        """Find the strongest skill"""
        if not skill_scores:
            return 'None'
        
        sorted_skills = sorted(skill_scores, key=lambda x: x['score'], reverse=True)
        return sorted_skills[0]['skill'] if sorted_skills else 'None'


# Convenience function
def explain_match(cv_data: Dict,
                 jd_data: Dict,
                 match_result: Dict,
                 embedding_engine: Optional[EmbeddingEngine] = None) -> Dict:
    """Quick explain function"""
    engine = ExplainabilityEngine(embedding_engine)
    return engine.explain_match(cv_data, jd_data, match_result)