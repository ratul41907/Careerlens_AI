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
            New match score after adding skill
        """
        # Create a copy of CV data
        modified_cv = deepcopy(cv_data)
        
        # Add skill to CV text
        if 'text' in modified_cv:
            modified_cv['text'] += f"\n{skill_to_add}"
        
        # Add skill to sections if they exist
        if 'sections' in modified_cv and 'skills' in modified_cv['sections']:
            if isinstance(modified_cv['sections']['skills'], list):
                modified_cv['sections']['skills'].append(skill_to_add)
            else:
                modified_cv['sections']['skills'] += f", {skill_to_add}"
        
        # Calculate new match score
        new_result = self.scoring_engine.compute_match_score(modified_cv, jd_data)
        
        # Extract score as float
        new_score = new_result.get('overall_score', 0)
        if isinstance(new_score, str):
            new_score = float(new_score.strip('%')) / 100
        
        return new_score
    
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
            current_score: Current match score
            potential_score: Score after adding skill
            jd_data: Job description data
            
        Returns:
            Analysis dict with explanation and priority
        """
        score_increase = (potential_score - current_score) * 100
        
        # Build LLM prompt
        prompt = f"""You are a career advisor explaining skill importance to a job seeker.

JOB REQUIREMENTS:
{jd_data.get('text', '')[:1500]}

SKILL TO LEARN: {skill}
CURRENT MATCH SCORE: {current_score*100:.1f}%
SCORE IF LEARNED: {potential_score*100:.1f}%
SCORE INCREASE: +{score_increase:.1f}%

Explain in 2-3 sentences WHY learning {skill} would improve this candidate's chances for this role.

Consider:
1. Is this skill explicitly required or preferred in the job description?
2. How critical is this skill for the role's responsibilities?
3. What specific job tasks or projects would this skill enable?

Provide a clear, practical explanation focused on job relevance:"""
        
        # Call LLM
        llm_explanation = self._call_ollama(prompt, max_tokens=300)
        
        # Determine priority based on score increase
        if score_increase >= 15:
            priority = "High"
            priority_explanation = "Critical skill - major impact on match score"
        elif score_increase >= 8:
            priority = "Medium"
            priority_explanation = "Important skill - moderate impact on match"
        elif score_increase >= 3:
            priority = "Low"
            priority_explanation = "Nice-to-have skill - minor impact"
        else:
            priority = "Very Low"
            priority_explanation = "Optional skill - minimal impact"
        
        # Fallback explanation if LLM fails
        if not llm_explanation or len(llm_explanation) < 50:
            if score_increase >= 10:
                llm_explanation = f"{skill} is a key requirement for this role and would significantly strengthen your application. This skill appears prominently in the job description and is essential for the core responsibilities."
            elif score_increase >= 5:
                llm_explanation = f"{skill} is mentioned in the job requirements and would improve your profile. Learning this skill would help you better meet the qualifications listed."
            else:
                llm_explanation = f"{skill} is relevant to this role and would add value to your application. While not critical, it would demonstrate broader capability."
        
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
            current_score: Current match score
            
        Returns:
            List of skill impact analyses, sorted by impact
        """
        if not missing_skills:
            return []
        
        analyses = []
        
        for skill in missing_skills[:10]:  # Limit to top 10 to avoid too many API calls
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
        
        prompt = f"""You are a career advisor creating a learning plan for a job seeker.

The candidate is missing these skills for their target role:

{chr(10).join(skills_summary)}

Create a practical learning priority plan with these sections:

1. IMMEDIATE FOCUS (Next 2 weeks)
Which 1-2 skills should they start learning NOW? Choose based on highest impact and learning difficulty.

2. SHORT-TERM GOALS (Next 1-2 months)
Which 2-3 skills should they tackle next?

3. LONG-TERM DEVELOPMENT (Next 3-6 months)
Which skills can wait but are still valuable?

4. LEARNING STRATEGY
Provide 3-4 specific tips for learning these skills efficiently (online courses, projects, practice, etc.)

Keep it practical, specific, and encouraging. Format with clear headers and bullet points:"""
        
        llm_plan = self._call_ollama(prompt, max_tokens=800)
        
        # Parse the plan
        sections = {
            'immediate': [],
            'short_term': [],
            'long_term': [],
            'strategy': []
        }
        
        current_section = None
        lines = llm_plan.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Detect sections
            if 'immediate' in line_lower or 'next 2 week' in line_lower or 'start now' in line_lower:
                current_section = 'immediate'
                continue
            elif 'short' in line_lower and 'term' in line_lower or 'next 1-2 month' in line_lower:
                current_section = 'short_term'
                continue
            elif 'long' in line_lower and 'term' in line_lower or 'next 3-6 month' in line_lower:
                current_section = 'long_term'
                continue
            elif 'strategy' in line_lower or 'tip' in line_lower or 'how to' in line_lower:
                current_section = 'strategy'
                continue
            
            # Add bullet points to current section
            if current_section and (line.startswith(('-', '•', '*')) or (len(line) > 2 and line[0].isdigit())):
                clean_line = line.lstrip('-•*0123456789.)]: ').strip()
                if clean_line and len(clean_line) > 5:
                    sections[current_section].append(clean_line)
        
        # Fallback if parsing fails
        if not any(sections.values()):
            sections = {
                'immediate': [skill_analyses[0]['skill']] if skill_analyses else [],
                'short_term': [a['skill'] for a in skill_analyses[1:3]],
                'long_term': [a['skill'] for a in skill_analyses[3:5]],
                'strategy': [
                    "Take online courses on Udemy/Coursera",
                    "Build practical projects to apply new skills",
                    "Contribute to open source projects",
                    "Join developer communities for support"
                ]
            }
        
        return {
            'success': True,
            'immediate_focus': sections['immediate'][:2],
            'short_term_goals': sections['short_term'][:3],
            'long_term_development': sections['long_term'][:3],
            'learning_strategy': sections['strategy'][:4],
            'full_plan': llm_plan,
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
    analyses = simulator.analyze_all_missing_skills(cv_data, jd_data, missing_skills, 0.65)
    
    print("✅ Skill Impact Analysis:")
    for analysis in analyses:
        print(f"\n{analysis['skill']}: {analysis['impact_percentage']} ({analysis['priority']} priority)")
        print(f"  {analysis['explanation'][:100]}...")