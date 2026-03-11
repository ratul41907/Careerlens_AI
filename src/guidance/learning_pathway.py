"""
Learning Pathway Generator - Creates personalized learning roadmaps
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import json


class LearningPathwayGenerator:
    """
    Generate personalized learning pathways based on skill gaps
    """
    
    def __init__(self):
        """Initialize learning pathway generator"""
        
        # Skill learning resources database
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
        """
        Generate personalized learning pathway
        
        Args:
            skill_gaps: List of missing skill names (strings)
            jd_data: Job description data
            num_days: Timeline (7, 14, or 30 days)
            
        Returns:
            Dict with daily plans and resources
        """
        logger.info(f"Generating {num_days}-day pathway for {len(skill_gaps)} skills")
        
        if not skill_gaps:
            return {
                'success': False,
                'message': 'No skill gaps to address',
                'daily_plans': []
            }
        
        # Take top skills based on timeline
        num_skills = 2 if num_days == 7 else 3 if num_days == 14 else 4
        focus_skills = skill_gaps[:num_skills]
        
        # Generate daily plans
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
        
        if num_days == 7:
            # 7-day intensive
            skill1 = skills[0] if len(skills) > 0 else 'Python'
            skill2 = skills[1] if len(skills) > 1 else skill1
            
            plans = [
                {
                    'day': 1,
                    'focus': f'{skill1} - Setup & Basics',
                    'goal': 'Environment setup and understand fundamentals',
                    'tasks': [
                        f'Install {skill1} tools and dependencies',
                        f'Watch "Introduction to {skill1}" tutorial (1-2 hours)',
                        'Read official documentation overview',
                        'Complete first "Hello World" example'
                    ],
                    'resources': self._get_resources(skill1),
                    'mini_project': f'Setup {skill1} development environment'
                },
                {
                    'day': 2,
                    'focus': f'{skill1} - Core Concepts',
                    'goal': 'Master fundamental concepts',
                    'tasks': [
                        f'Study {skill1} core concepts (2 hours)',
                        'Complete 3-5 beginner exercises',
                        'Build simple demo application',
                        'Review common patterns'
                    ],
                    'resources': self._get_resources(skill1),
                    'mini_project': f'Build a simple {skill1} demo'
                },
                {
                    'day': 3,
                    'focus': f'{skill1} - Hands-on Practice',
                    'goal': 'Apply knowledge through practice',
                    'tasks': [
                        'Work on practical exercises (2-3 hours)',
                        'Debug and troubleshoot issues',
                        'Read best practices guide',
                        'Start building a mini-project'
                    ],
                    'resources': self._get_resources(skill1),
                    'mini_project': f'Start portfolio project using {skill1}'
                },
                {
                    'day': 4,
                    'focus': f'{skill2} - Introduction',
                    'goal': 'Learn second skill basics',
                    'tasks': [
                        f'Introduction to {skill2}',
                        'Setup and configuration',
                        'Complete beginner tutorial',
                        'Build simple example'
                    ],
                    'resources': self._get_resources(skill2),
                    'mini_project': f'Hello World in {skill2}'
                },
                {
                    'day': 5,
                    'focus': 'Integration & Practice',
                    'goal': 'Combine learned skills',
                    'tasks': [
                        'Continue both skill practices',
                        'Look for integration opportunities',
                        'Work on mini-project',
                        'Review progress'
                    ],
                    'resources': self._get_resources(skill1) + self._get_resources(skill2),
                    'mini_project': 'Combine skills in one project'
                },
                {
                    'day': 6,
                    'focus': 'Project Development',
                    'goal': 'Build portfolio piece',
                    'tasks': [
                        'Focus on building complete project',
                        'Apply best practices',
                        'Add documentation',
                        'Test thoroughly'
                    ],
                    'resources': ['Project templates', 'Code review guidelines'],
                    'mini_project': 'Complete portfolio project'
                },
                {
                    'day': 7,
                    'focus': 'Polish & Showcase',
                    'goal': 'Finalize and document',
                    'tasks': [
                        'Polish project code',
                        'Write project README',
                        'Deploy/publish project',
                        'Update CV with new skills'
                    ],
                    'resources': ['GitHub portfolio guide', 'CV writing tips'],
                    'mini_project': 'Publish project on GitHub'
                }
            ]
        
        elif num_days == 14:
            # 14-day balanced
            skill1 = skills[0] if len(skills) > 0 else 'Python'
            skill2 = skills[1] if len(skills) > 1 else 'Docker'
            skill3 = skills[2] if len(skills) > 2 else 'Git'
            
            plans = [
                {'day': 1, 'focus': f'{skill1} - Fundamentals', 'goal': 'Setup and basics', 'tasks': [f'Learn {skill1} basics', 'Setup environment', 'First tutorial'], 'resources': self._get_resources(skill1), 'mini_project': f'Hello World {skill1}'},
                {'day': 2, 'focus': f'{skill1} - Core Concepts', 'goal': 'Deep dive', 'tasks': ['Study core features', 'Practice exercises'], 'resources': self._get_resources(skill1), 'mini_project': 'Simple demo app'},
                {'day': 3, 'focus': f'{skill1} - Practice', 'goal': 'Hands-on', 'tasks': ['Code challenges', 'Debug issues'], 'resources': self._get_resources(skill1), 'mini_project': 'Practical exercise'},
                {'day': 4, 'focus': f'{skill1} - Project Start', 'goal': 'Begin building', 'tasks': ['Plan project', 'Start coding'], 'resources': self._get_resources(skill1), 'mini_project': 'Start main project'},
                {'day': 5, 'focus': f'{skill2} - Introduction', 'goal': 'New skill basics', 'tasks': [f'Learn {skill2}', 'Setup'], 'resources': self._get_resources(skill2), 'mini_project': 'Basic setup'},
                {'day': 6, 'focus': f'{skill2} - Deep Dive', 'goal': 'Master fundamentals', 'tasks': ['Study concepts', 'Practice'], 'resources': self._get_resources(skill2), 'mini_project': 'Demo application'},
                {'day': 7, 'focus': 'Week 1 Review', 'goal': 'Consolidate', 'tasks': ['Review both skills', 'Continue projects'], 'resources': ['Review materials'], 'mini_project': 'Weekly milestone'},
                {'day': 8, 'focus': f'{skill3} - Basics', 'goal': 'Third skill intro', 'tasks': [f'Introduction to {skill3}'], 'resources': self._get_resources(skill3), 'mini_project': 'Quick demo'},
                {'day': 9, 'focus': 'Integration', 'goal': 'Combine skills', 'tasks': ['Integrate skills', 'Work on project'], 'resources': ['Integration tutorials'], 'mini_project': 'Integration demo'},
                {'day': 10, 'focus': 'Advanced Concepts', 'goal': 'Level up', 'tasks': ['Advanced tutorials', 'Complex examples'], 'resources': ['Advanced guides'], 'mini_project': 'Advanced feature'},
                {'day': 11, 'focus': 'Project Development', 'goal': 'Build portfolio', 'tasks': ['Focus on main project', 'Add features'], 'resources': ['Testing guides'], 'mini_project': 'Feature complete'},
                {'day': 12, 'focus': 'Polish & Documentation', 'goal': 'Professional finish', 'tasks': ['Code cleanup', 'Write docs'], 'resources': ['Documentation templates'], 'mini_project': 'Add README'},
                {'day': 13, 'focus': 'Deployment', 'goal': 'Make it live', 'tasks': ['Deploy project', 'Setup CI/CD'], 'resources': ['Deployment guides'], 'mini_project': 'Live deployment'},
                {'day': 14, 'focus': 'Portfolio Update', 'goal': 'Showcase work', 'tasks': ['Update CV', 'GitHub portfolio'], 'resources': ['Portfolio examples'], 'mini_project': 'Complete portfolio'}
            ]
        
        else:  # 30 days
            # Generate 30-day comprehensive plan
            skill1 = skills[0] if len(skills) > 0 else 'Python'
            skill2 = skills[1] if len(skills) > 1 else 'Docker'
            skill3 = skills[2] if len(skills) > 2 else 'React'
            skill4 = skills[3] if len(skills) > 3 else 'AWS'
            
            all_skills = [skill1, skill2, skill3, skill4]
            
            for day in range(1, 31):
                # Determine focus skill (rotate through skills)
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
                    'tasks': [
                        f'Study {current_skill}',
                        'Practice exercises',
                        'Work on projects'
                    ],
                    'resources': self._get_resources(current_skill),
                    'mini_project': f'Day {day} milestone'
                })
        
        return plans
    
    def _get_resources(self, skill: str) -> List[str]:
        """Get learning resources for a skill"""
        skill_lower = skill.lower()
        
        if skill_lower in self.skill_resources:
            return self.skill_resources[skill_lower]
        
        # Default resources
        return [
            f'Official {skill} Documentation',
            f'{skill} Tutorial on YouTube',
            f'{skill} Course on Udemy/Coursera'
        ]


def generate_learning_path(skill_gaps: List[str], jd_data: Dict, num_days: int = 30) -> Dict:
    """Convenience function for generating learning pathways"""
    generator = LearningPathwayGenerator()
    return generator.generate_pathway(skill_gaps, jd_data, num_days)