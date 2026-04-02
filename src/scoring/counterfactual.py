"""
Counterfactual Simulator - 100% LLM-Based Skill Impact Analysis
Uses Ollama to predict and explain how learning skills would improve match scores
"""
import requests
from typing import Dict, List, Optional
import json
from copy import deepcopy


class CounterfactualSimulator:
    """
    Simulate and explain impact of learning new skills using LLM
    """
    
    def __init__(self, scoring_engine, ollama_url: str = "http://localhost:11434"):
        """
        Initialize counterfactual simulator
        
        Args:
            scoring_engine: ScoringEngine instance for actual score calculation
            ollama_url: Ollama API endpoint
        """
        self.scoring_engine = scoring_engine
        self.ollama_url = ollama_url
        self.model = "gemma2:2b"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 800) -> str:
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
    
    def simulate_skill_addition(
        self,
        cv_data: Dict,
        jd_data: Dict,
        skill_to_add: str
    ) -> float:
        """
        Simulate adding a skill to CV and calculate new match score
        
        Args:
            cv_data: Original CV data
            jd_data: Job description data
            skill_to_add: Skill to simulate adding
            
        Returns:
            New match score after adding skill (as percentage 0-100)
        """
        try:
            # Create a copy of CV data
            modified_cv = deepcopy(cv_data)
            
            # Add skill to CV text
            if 'text' in modified_cv:
                modified_cv['text'] += f"\n{skill_to_add}"
            
            # Add skill to sections
            if 'sections' in modified_cv:
                if 'skills' not in modified_cv['sections']:
                    modified_cv['sections']['skills'] = []
                
                skills = modified_cv['sections']['skills']
                
                if isinstance(skills, list):
                    if skill_to_add not in skills:
                        modified_cv['sections']['skills'].append(skill_to_add)
                elif isinstance(skills, str):
                    if skill_to_add not in skills:
                        modified_cv['sections']['skills'] += f", {skill_to_add}"
                else:
                    modified_cv['sections']['skills'] = [skill_to_add]
            
            # Calculate new match score
            new_result = self.scoring_engine.compute_match_score(modified_cv, jd_data)
            
            # Extract score as percentage (0-100)
            new_score = new_result.get('overall_score', 0)
            
            if isinstance(new_score, str):
                # Remove % sign and convert
                new_score = float(new_score.replace('%', '').strip())
            elif isinstance(new_score, float):
                # Convert 0-1 range to 0-100
                if new_score <= 1.0:
                    new_score = new_score * 100
            
            return float(new_score)
            
        except Exception as e:
            print(f"Error simulating skill addition: {e}")
            return 0.0
    
    def analyze_skill_impact(
        self,
        skill: str,
        current_score: float,
        potential_score: float,
        jd_data: Dict
    ) -> Dict:
        """
        Use LLM to explain WHY learning this skill would improve the match
        
        Args:
            skill: Skill being analyzed
            current_score: Current match score (0-100)
            potential_score: Score after adding skill (0-100)
            jd_data: Job description data
            
        Returns:
            Analysis dict with explanation and priority
        """
        score_increase = potential_score - current_score
        
        # Determine priority based on score increase
        if score_increase >= 10:
            priority = "High"
            priority_explanation = "Critical skill - major impact on match score"
        elif score_increase >= 5:
            priority = "Medium"
            priority_explanation = "Important skill - moderate impact on match"
        elif score_increase >= 2:
            priority = "Low"
            priority_explanation = "Nice-to-have skill - minor impact"
        else:
            priority = "Very Low"
            priority_explanation = "Optional skill - minimal impact"
        
        # Get JD text
        jd_text = jd_data.get('text', '') if isinstance(jd_data, dict) else str(jd_data)
        jd_text = jd_text[:1000]  # Limit length
        
        # Build LLM prompt
        prompt = f"""Explain in 2-3 sentences WHY learning "{skill}" would help a candidate for this role.

Job Requirements:
{jd_text}

Current Match: {current_score:.1f}%
If {skill} learned: {potential_score:.1f}%
Improvement: +{score_increase:.1f}%

Explain the practical value of this skill for the role. Focus on job relevance:"""

        # Call LLM
        llm_explanation = self._call_ollama(prompt, max_tokens=200)
        
        # Fallback explanation if LLM fails
        if not llm_explanation or len(llm_explanation) < 30:
            if score_increase >= 8:
                llm_explanation = f"{skill} is a key requirement for this role and would significantly strengthen your application. This skill is essential for core responsibilities."
            elif score_increase >= 4:
                llm_explanation = f"{skill} is mentioned in the job requirements and would improve your profile. Learning this skill would help you meet qualifications."
            else:
                llm_explanation = f"{skill} is relevant to this role and would add value. While not critical, it demonstrates broader capability."
        
        return {
            'skill': skill,
            'current_score': current_score,
            'potential_score': potential_score,
            'score_increase': score_increase,
            'priority': priority,
            'priority_explanation': priority_explanation,
            'explanation': llm_explanation.strip(),
            'impact_percentage': f"+{score_increase:.1f}%"
        }
    
    def analyze_all_missing_skills(
        self,
        cv_data: Dict,
        jd_data: Dict,
        missing_skills: List[str],
        current_score: float
    ) -> List[Dict]:
        """
        Analyze impact of learning each missing skill
        
        Args:
            cv_data: CV data
            jd_data: Job description data
            missing_skills: List of missing skills
            current_score: Current match score (0-100 or 0-1, will be normalized)
            
        Returns:
            List of skill impact analyses, sorted by impact
        """
        if not missing_skills:
            return []
        
        # Normalize current score to 0-100 range
        if isinstance(current_score, float) and current_score <= 1.0:
            current_score = current_score * 100
        elif isinstance(current_score, str):
            current_score = float(current_score.replace('%', '').strip())
        
        analyses = []
        
        # Limit to top 10 to avoid too many API calls
        for skill in missing_skills[:10]:
            try:
                # Simulate adding this skill
                potential_score = self.simulate_skill_addition(cv_data, jd_data, skill)
                
                # Get LLM explanation of impact
                analysis = self.analyze_skill_impact(
                    skill,
                    current_score,
                    potential_score,
                    jd_data
                )
                
                analyses.append(analysis)
                
            except Exception as e:
                print(f"Error analyzing skill {skill}: {e}")
                continue
        
        # Sort by score increase (descending)
        analyses.sort(key=lambda x: x['score_increase'], reverse=True)
        
        return analyses
    
    def generate_learning_priority_plan(
        self,
        skill_analyses: List[Dict]
    ) -> Dict:
        """
        Use LLM to create a learning priority plan
        
        Args:
            skill_analyses: List of skill impact analyses
            
        Returns:
            Priority plan dict
        """
        if not skill_analyses:
            return {
                'success': False,
                'error': 'No skills to analyze'
            }
        
        # Build skill summary for LLM
        skills_summary = []
        for analysis in skill_analyses[:5]:  # Top 5 skills
            skills_summary.append(
                f"- {analysis['skill']}: +{analysis['score_increase']:.1f}% impact ({analysis['priority']} priority)"
            )
        
        prompt = f"""Create a learning priority plan for these skills:

{chr(10).join(skills_summary)}

Return ONLY a JSON object:
{{
  "immediate_focus": ["skill1", "skill2"],
  "short_term_goals": ["skill3", "skill4"],
  "long_term_development": ["skill5"],
  "learning_strategy": ["tip1", "tip2", "tip3"]
}}

Immediate: Start in next 2 weeks
Short-term: Next 1-2 months
Long-term: Next 3-6 months

Return JSON only:"""

        llm_plan = self._call_ollama(prompt, max_tokens=500)
        
        try:
            # Parse JSON
            if '```json' in llm_plan:
                llm_plan = llm_plan.split('```json')[1].split('```')[0]
            elif '```' in llm_plan:
                llm_plan = llm_plan.split('```')[1].split('```')[0]
            
            if '{' in llm_plan:
                llm_plan = llm_plan[llm_plan.index('{'):llm_plan.rindex('}')+1]
            
            plan = json.loads(llm_plan)
            
            return {
                'success': True,
                'immediate_focus': plan.get('immediate_focus', [])[:2],
                'short_term_goals': plan.get('short_term_goals', [])[:3],
                'long_term_development': plan.get('long_term_development', [])[:3],
                'learning_strategy': plan.get('learning_strategy', [])[:4],
                'total_skills_analyzed': len(skill_analyses)
            }
            
        except Exception as e:
            print(f"Error parsing learning plan: {e}")
            
            # Fallback plan
            return {
                'success': True,
                'immediate_focus': [skill_analyses[0]['skill']] if skill_analyses else [],
                'short_term_goals': [a['skill'] for a in skill_analyses[1:3]],
                'long_term_development': [a['skill'] for a in skill_analyses[3:5]],
                'learning_strategy': [
                    "Take online courses (Udemy, Coursera)",
                    "Build practical projects",
                    "Contribute to open source",
                    "Join developer communities"
                ],
                'total_skills_analyzed': len(skill_analyses)
            }


# Test
if __name__ == "__main__":
    from src.embeddings.embedding_engine import EmbeddingEngine
    from src.scoring.scoring_engine import ScoringEngine
    
    # Initialize engines
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    simulator = CounterfactualSimulator(scoring_engine)
    
    # Test data
    cv_data = {
        'text': 'Python, FastAPI, Docker experience',
        'sections': {'skills': ['Python', 'FastAPI', 'Docker']}
    }
    
    jd_data = {
        'text': 'Looking for developer with Python, FastAPI, Docker, Kubernetes, AWS'
    }
    
    missing_skills = ['Kubernetes', 'AWS']
    
    # Analyze impact
    analyses = simulator.analyze_all_missing_skills(cv_data, jd_data, missing_skills, 65.0)
    
    print("✅ Skill Impact Analysis:")
    for analysis in analyses:
        print(f"\n{analysis['skill']}: {analysis['impact_percentage']} ({analysis['priority']} priority)")
        print(f"  {analysis['explanation'][:100]}...")