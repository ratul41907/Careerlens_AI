"""
Learning Pathway Generator - Creates personalized learning roadmaps
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger


class LearningPathwayGenerator:
    """
    Generate personalized learning pathways based on skill gaps
    """
    
    def __init__(self):
        """Initialize learning pathway generator"""
        self.skill_resources = {
            'python': ['Python.org Tutorial', 'Real Python', 'Automate the Boring Stuff'],
            'javascript': ['MDN Web Docs', 'JavaScript.info', 'freeCodeCamp'],
            'docker': ['Docker Official Docs', 'Docker Deep Dive', 'Play with Docker'],
            'kubernetes': ['Kubernetes Basics', 'K8s by Example', 'KodeKloud'],
            'react': ['React Official Tutorial', 'React for Beginners', 'freeCodeCamp React'],
            'aws': ['AWS Free Tier Guide', 'AWS Solutions Architect', 'Cloud Practitioner'],
            'fastapi': ['FastAPI Official Docs', 'TestDriven.io FastAPI', 'FastAPI Course'],
            'nodejs': ['Node.js Docs', 'Node.js Complete Guide', 'Node.js Best Practices'],
            'sql': ['SQL Tutorial', 'SQLZoo', 'Mode Analytics SQL'],
            'git': ['Git Official Tutorial', 'Pro Git Book', 'GitHub Learning Lab']
        }
        logger.info("LearningPathwayGenerator initialized")
    
    def generate_pathway(self, skill_gaps: List[str], jd_data: Dict, num_days: int = 30) -> Dict:
        """Generate personalized learning pathway"""
        logger.info(f"Generating {num_days}-day pathway for {len(skill_gaps)} skills")
        
        if not skill_gaps:
            return {'success': False, 'message': 'No skill gaps', 'daily_plans': []}
        
        num_skills = 2 if num_days == 7 else 3 if num_days == 14 else 4
        focus_skills = skill_gaps[:num_skills]
        daily_plans = self._generate_daily_plans(focus_skills, num_days)
        
        return {
            'success': True,
            'timeline_days': num_days,
            'focus_skills': focus_skills,
            'total_skills': len(skill_gaps),
            'daily_plans': daily_plans,
            'estimated_daily_hours': 2 if num_days == 7 else 1.5 if num_days == 14 else 1,
            'completion_date': (datetime.now() + timedelta(days=num_days)).strftime('%Y-%m-%d')
        }
    
    def _generate_daily_plans(self, skills: List[str], num_days: int) -> List[Dict]:
        """Generate day-by-day learning plan"""
        plans = []
        skill1 = skills[0] if len(skills) > 0 else 'Python'
        skill2 = skills[1] if len(skills) > 1 else skill1
        
        if num_days == 7:
            plans = [
                {'day': 1, 'focus': f'{skill1} - Setup & Basics', 'goal': 'Environment setup', 'tasks': [f'Install {skill1}', 'Watch intro tutorial', 'First example'], 'resources': self._get_resources(skill1), 'mini_project': f'Setup {skill1}'},
                {'day': 2, 'focus': f'{skill1} - Core Concepts', 'goal': 'Learn fundamentals', 'tasks': ['Study core concepts', 'Practice exercises', 'Build demo'], 'resources': self._get_resources(skill1), 'mini_project': 'Simple demo'},
                {'day': 3, 'focus': f'{skill1} - Practice', 'goal': 'Hands-on coding', 'tasks': ['Work on exercises', 'Debug issues', 'Build project'], 'resources': self._get_resources(skill1), 'mini_project': 'Mini project'},
                {'day': 4, 'focus': f'{skill2} - Introduction', 'goal': 'Second skill basics', 'tasks': [f'Learn {skill2}', 'Setup environment', 'First tutorial'], 'resources': self._get_resources(skill2), 'mini_project': f'{skill2} setup'},
                {'day': 5, 'focus': 'Integration', 'goal': 'Combine skills', 'tasks': ['Practice both skills', 'Integration examples', 'Work on project'], 'resources': ['Integration guides'], 'mini_project': 'Combined demo'},
                {'day': 6, 'focus': 'Project Development', 'goal': 'Build portfolio', 'tasks': ['Build complete project', 'Add documentation', 'Testing'], 'resources': ['Project ideas'], 'mini_project': 'Portfolio piece'},
                {'day': 7, 'focus': 'Polish & Showcase', 'goal': 'Finalize work', 'tasks': ['Polish code', 'Write README', 'Update CV'], 'resources': ['GitHub guide'], 'mini_project': 'Publish on GitHub'}
            ]
        elif num_days == 14:
            skill3 = skills[2] if len(skills) > 2 else 'Git'
            plans = [
                {'day': 1, 'focus': f'{skill1} - Fundamentals', 'goal': 'Setup and basics', 'tasks': [f'Learn {skill1}', 'Setup', 'Tutorial'], 'resources': self._get_resources(skill1), 'mini_project': f'Hello World {skill1}'},
                {'day': 2, 'focus': f'{skill1} - Core', 'goal': 'Deep dive', 'tasks': ['Study concepts', 'Practice'], 'resources': self._get_resources(skill1), 'mini_project': 'Demo app'},
                {'day': 3, 'focus': f'{skill1} - Practice', 'goal': 'Hands-on', 'tasks': ['Code challenges', 'Debug'], 'resources': self._get_resources(skill1), 'mini_project': 'Exercise'},
                {'day': 4, 'focus': f'{skill1} - Project', 'goal': 'Build', 'tasks': ['Plan project', 'Start coding'], 'resources': self._get_resources(skill1), 'mini_project': 'Main project'},
                {'day': 5, 'focus': f'{skill2} - Intro', 'goal': 'New skill', 'tasks': [f'Learn {skill2}', 'Setup'], 'resources': self._get_resources(skill2), 'mini_project': 'Setup'},
                {'day': 6, 'focus': f'{skill2} - Deep Dive', 'goal': 'Master basics', 'tasks': ['Study', 'Practice'], 'resources': self._get_resources(skill2), 'mini_project': 'Demo'},
                {'day': 7, 'focus': 'Review', 'goal': 'Consolidate', 'tasks': ['Review skills', 'Continue projects'], 'resources': ['Review materials'], 'mini_project': 'Milestone'},
                {'day': 8, 'focus': f'{skill3} - Basics', 'goal': 'Third skill', 'tasks': [f'Intro to {skill3}'], 'resources': self._get_resources(skill3), 'mini_project': 'Quick demo'},
                {'day': 9, 'focus': 'Integration', 'goal': 'Combine', 'tasks': ['Integrate skills', 'Project work'], 'resources': ['Integration tutorials'], 'mini_project': 'Integration'},
                {'day': 10, 'focus': 'Advanced', 'goal': 'Level up', 'tasks': ['Advanced tutorials', 'Complex examples'], 'resources': ['Advanced guides'], 'mini_project': 'Advanced feature'},
                {'day': 11, 'focus': 'Project', 'goal': 'Build', 'tasks': ['Focus on project', 'Add features'], 'resources': ['Testing guides'], 'mini_project': 'Feature complete'},
                {'day': 12, 'focus': 'Polish', 'goal': 'Professional', 'tasks': ['Code cleanup', 'Write docs'], 'resources': ['Docs templates'], 'mini_project': 'Add README'},
                {'day': 13, 'focus': 'Deployment', 'goal': 'Go live', 'tasks': ['Deploy project', 'CI/CD'], 'resources': ['Deployment guides'], 'mini_project': 'Live deployment'},
                {'day': 14, 'focus': 'Portfolio', 'goal': 'Showcase', 'tasks': ['Update CV', 'GitHub portfolio'], 'resources': ['Portfolio examples'], 'mini_project': 'Complete portfolio'}
            ]
        else:  # 30 days
            skill3 = skills[2] if len(skills) > 2 else 'React'
            skill4 = skills[3] if len(skills) > 3 else 'AWS'
            all_skills = [skill1, skill2, skill3, skill4]
            
            for day in range(1, 31):
                week = (day - 1) // 7
                skill_index = min(week, len(all_skills) - 1)
                current_skill = all_skills[skill_index]
                day_in_week = ((day - 1) % 7) + 1
                
                if day_in_week == 1:
                    focus = f'{current_skill} - Introduction'
                    goal = 'Setup and fundamentals'
                elif day_in_week <= 3:
                    focus = f'{current_skill} - Core Learning'
                    goal = 'Master key concepts'
                elif day_in_week <= 5:
                    focus = f'{current_skill} - Practice'
                    goal = 'Hands-on exercises'
                else:
                    focus = f'{current_skill} - Project Work'
                    goal = 'Build something real'
                
                plans.append({
                    'day': day,
                    'focus': focus,
                    'goal': goal,
                    'tasks': [f'Study {current_skill}', 'Practice exercises', 'Work on projects'],
                    'resources': self._get_resources(current_skill),
                    'mini_project': f'Day {day} milestone'
                })
        
        return plans
    
    def _get_resources(self, skill: str) -> List[str]:
        """Get learning resources for a skill"""
        skill_lower = skill.lower()
        if skill_lower in self.skill_resources:
            return self.skill_resources[skill_lower]
        return [f'Official {skill} Documentation', f'{skill} Tutorial', f'{skill} Course']


def generate_learning_path(skill_gaps: List[str], jd_data: Dict, num_days: int = 30) -> Dict:
    """Convenience function"""
    generator = LearningPathwayGenerator()
    return generator.generate_pathway(skill_gaps, jd_data, num_days)