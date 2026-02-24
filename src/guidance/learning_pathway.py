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
            'python': {
                'beginner': {
                    'courses': [
                        {'name': 'Python for Everybody (Coursera)', 'duration': '8 weeks', 'url': 'https://coursera.org/python'},
                        {'name': 'Learn Python - Codecademy', 'duration': '4 weeks', 'url': 'https://codecademy.com/python'}
                    ],
                    'tutorials': [
                        {'name': 'Official Python Tutorial', 'url': 'https://docs.python.org/tutorial'},
                        {'name': 'Real Python Tutorials', 'url': 'https://realpython.com'}
                    ],
                    'practice': [
                        {'name': 'LeetCode Easy Problems', 'url': 'https://leetcode.com'},
                        {'name': 'HackerRank Python', 'url': 'https://hackerrank.com/python'}
                    ]
                },
                'intermediate': {
                    'courses': [
                        {'name': 'Python 3 Programming (Coursera)', 'duration': '5 weeks'},
                        {'name': 'Advanced Python - Udemy', 'duration': '6 weeks'}
                    ],
                    'projects': [
                        'Build a REST API with Flask',
                        'Create a web scraper',
                        'Develop a CLI tool'
                    ]
                }
            },
            'fastapi': {
                'beginner': {
                    'courses': [
                        {'name': 'FastAPI - The Complete Course (Udemy)', 'duration': '3 weeks'},
                        {'name': 'Building APIs with FastAPI', 'duration': '2 weeks'}
                    ],
                    'tutorials': [
                        {'name': 'Official FastAPI Tutorial', 'url': 'https://fastapi.tiangolo.com/tutorial'},
                        {'name': 'FastAPI for Flask Users', 'url': 'https://testdriven.io/fastapi-flask'}
                    ],
                    'projects': [
                        'Build a Todo API',
                        'Create a user authentication system',
                        'Develop a simple CRUD application'
                    ]
                }
            },
            'docker': {
                'beginner': {
                    'courses': [
                        {'name': 'Docker Mastery (Udemy)', 'duration': '4 weeks'},
                        {'name': 'Docker for Developers', 'duration': '3 weeks'}
                    ],
                    'tutorials': [
                        {'name': 'Docker Getting Started', 'url': 'https://docs.docker.com/get-started'},
                        {'name': 'Play with Docker', 'url': 'https://labs.play-with-docker.com'}
                    ],
                    'practice': [
                        {'name': 'Containerize a Python app', 'type': 'hands-on'},
                        {'name': 'Docker Compose multi-container setup', 'type': 'hands-on'}
                    ]
                }
            },
            'kubernetes': {
                'beginner': {
                    'courses': [
                        {'name': 'Kubernetes for Beginners (KodeKloud)', 'duration': '4 weeks'},
                        {'name': 'Introduction to Kubernetes (edX)', 'duration': '5 weeks'}
                    ],
                    'tutorials': [
                        {'name': 'Kubernetes Basics', 'url': 'https://kubernetes.io/docs/tutorials/kubernetes-basics'}
                    ]
                }
            },
            'aws': {
                'beginner': {
                    'courses': [
                        {'name': 'AWS Certified Cloud Practitioner', 'duration': '6 weeks'},
                        {'name': 'AWS Fundamentals (Coursera)', 'duration': '4 weeks'}
                    ],
                    'certifications': [
                        {'name': 'AWS Certified Solutions Architect - Associate', 'recommended': True}
                    ]
                }
            },
            'react': {
                'beginner': {
                    'courses': [
                        {'name': 'React - The Complete Guide (Udemy)', 'duration': '8 weeks'},
                        {'name': 'React Basics (Scrimba)', 'duration': '4 weeks'}
                    ],
                    'tutorials': [
                        {'name': 'Official React Tutorial', 'url': 'https://react.dev/learn'}
                    ],
                    'projects': [
                        'Build a Todo app',
                        'Create a weather dashboard',
                        'Develop a personal portfolio'
                    ]
                }
            },
            'javascript': {
                'beginner': {
                    'courses': [
                        {'name': 'JavaScript - The Complete Guide', 'duration': '10 weeks'},
                        {'name': 'Modern JavaScript (ES6+)', 'duration': '6 weeks'}
                    ]
                }
            },
            'nodejs': {
                'beginner': {
                    'courses': [
                        {'name': 'Node.js - The Complete Guide', 'duration': '8 weeks'},
                        {'name': 'Node.js API Development', 'duration': '5 weeks'}
                    ]
                }
            }
        }
        
        # Default resources for skills not in database
        self.default_resources = {
            'courses': [
                {'name': 'Udemy - {skill} for Beginners', 'duration': '4 weeks'},
                {'name': 'Coursera - {skill} Specialization', 'duration': '6 weeks'}
            ],
            'tutorials': [
                {'name': 'Official {skill} Documentation', 'type': 'reading'},
                {'name': 'YouTube - {skill} Tutorial Playlist', 'type': 'video'}
            ],
            'practice': [
                {'name': 'LeetCode {skill} Problems', 'type': 'coding'},
                {'name': 'Build a {skill} project', 'type': 'hands-on'}
            ]
        }
        
        logger.info("LearningPathwayGenerator initialized")
    
    def generate_pathway(self,
                        missing_skills: List[Dict],
                        timeline_days: int = 30,
                        skill_priorities: Optional[Dict] = None) -> Dict:
        """
        Generate personalized learning pathway
        
        Args:
            missing_skills: List of skills with scores (from counterfactual or match result)
            timeline_days: Learning timeline (7, 14, or 30 days)
            skill_priorities: Optional custom priorities
            
        Returns:
            Dict with learning pathway, milestones, and timeline
        """
        logger.info(f"Generating {timeline_days}-day learning pathway for {len(missing_skills)} skills")
        
        # Prioritize skills
        prioritized_skills = self._prioritize_skills(missing_skills, skill_priorities)
        
        # Create timeline
        timeline = self._create_timeline(prioritized_skills, timeline_days)
        
        # Generate detailed plan for each skill
        detailed_plan = self._create_detailed_plan(prioritized_skills, timeline_days)
        
        # Create milestones
        milestones = self._create_milestones(timeline_days)
        
        pathway = {
            'timeline_days': timeline_days,
            'total_skills': len(prioritized_skills),
            'prioritized_skills': prioritized_skills,
            'timeline': timeline,
            'detailed_plan': detailed_plan,
            'milestones': milestones,
            'estimated_improvement': self._estimate_improvement(prioritized_skills),
            'success_tips': self._generate_success_tips(timeline_days)
        }
        
        logger.info(f"Pathway generated: {len(timeline)} phases, {len(milestones)} milestones")
        return pathway
    
    def _prioritize_skills(self,
                          missing_skills: List[Dict],
                          custom_priorities: Optional[Dict] = None) -> List[Dict]:
        """
        Prioritize skills based on impact and difficulty
        
        Returns:
            Sorted list of skills with priority levels
        """
        prioritized = []
        
        for skill_info in missing_skills:
            skill = skill_info.get('skill', '')
            current_score = skill_info.get('current_score', 0)
            
            # Calculate priority score
            # Lower current score = higher priority
            impact_score = 1.0 - current_score
            
            # Add custom priority if provided
            if custom_priorities and skill in custom_priorities:
                impact_score *= custom_priorities[skill]
            
            priority_level = 'Critical' if impact_score > 0.5 else 'High' if impact_score > 0.3 else 'Medium'
            
            prioritized.append({
                'skill': skill,
                'current_score': current_score,
                'impact_score': round(impact_score, 3),
                'priority': priority_level,
                'estimated_weeks': self._estimate_learning_time(skill)
            })
        
        # Sort by impact score (highest first)
        prioritized.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return prioritized
    
    def _estimate_learning_time(self, skill: str) -> int:
        """Estimate weeks needed to learn skill"""
        # Simple heuristic (can be improved)
        complex_skills = ['kubernetes', 'aws', 'react', 'machine learning', 'tensorflow']
        medium_skills = ['docker', 'fastapi', 'nodejs', 'mongodb']
        
        if any(s in skill.lower() for s in complex_skills):
            return 6
        elif any(s in skill.lower() for s in medium_skills):
            return 4
        else:
            return 3
    
    def _create_timeline(self,
                        prioritized_skills: List[Dict],
                        timeline_days: int) -> List[Dict]:
        """
        Create learning timeline with phases
        
        Returns:
            List of phases with skills and dates
        """
        phases = []
        
        if timeline_days == 7:
            # 7-day intensive: Focus on 1-2 top skills
            top_skills = prioritized_skills[:2]
            phases.append({
                'phase': 1,
                'days': '1-3',
                'duration': '3 days',
                'focus': 'Fundamentals',
                'skills': [top_skills[0]['skill']],
                'activities': ['Watch tutorials', 'Read documentation', 'Setup environment']
            })
            phases.append({
                'phase': 2,
                'days': '4-7',
                'duration': '4 days',
                'focus': 'Practice & Build',
                'skills': [s['skill'] for s in top_skills],
                'activities': ['Hands-on exercises', 'Build mini-project', 'Code review']
            })
        
        elif timeline_days == 14:
            # 14-day balanced: 2-3 skills
            top_skills = prioritized_skills[:3]
            phases.append({
                'phase': 1,
                'days': '1-4',
                'duration': '4 days',
                'focus': 'Skill 1 - Fundamentals',
                'skills': [top_skills[0]['skill']],
                'activities': ['Complete beginner course', 'Daily coding practice']
            })
            phases.append({
                'phase': 2,
                'days': '5-10',
                'duration': '6 days',
                'focus': 'Skill 1 & 2 - Deep Dive',
                'skills': [top_skills[0]['skill'], top_skills[1]['skill'] if len(top_skills) > 1 else ''],
                'activities': ['Build projects', 'Advanced tutorials', 'Real-world practice']
            })
            phases.append({
                'phase': 3,
                'days': '11-14',
                'duration': '4 days',
                'focus': 'Integration & Polish',
                'skills': [s['skill'] for s in top_skills],
                'activities': ['Combine skills in project', 'CV update', 'Portfolio piece']
            })
        
        else:  # 30 days
            # 30-day comprehensive: 3-4 skills
            top_skills = prioritized_skills[:4]
            phases.append({
                'phase': 1,
                'days': '1-7',
                'duration': '1 week',
                'focus': 'Foundation Week',
                'skills': [top_skills[0]['skill']],
                'activities': ['Complete online course', 'Daily practice (2 hrs)', 'Build simple project']
            })
            phases.append({
                'phase': 2,
                'days': '8-14',
                'duration': '1 week',
                'focus': 'Skill Expansion',
                'skills': [top_skills[1]['skill'] if len(top_skills) > 1 else ''],
                'activities': ['New course start', 'Continue practicing first skill', 'Build intermediate project']
            })
            phases.append({
                'phase': 3,
                'days': '15-21',
                'duration': '1 week',
                'focus': 'Advanced Techniques',
                'skills': [s['skill'] for s in top_skills[:3]],
                'activities': ['Advanced tutorials', 'Real-world scenarios', 'Code review & refactoring']
            })
            phases.append({
                'phase': 4,
                'days': '22-30',
                'duration': '9 days',
                'focus': 'Portfolio & Integration',
                'skills': [s['skill'] for s in top_skills],
                'activities': ['Build showcase project', 'Update CV & portfolio', 'Mock interviews']
            })
        
        return phases
    
    def _create_detailed_plan(self,
                             prioritized_skills: List[Dict],
                             timeline_days: int) -> List[Dict]:
        """
        Create detailed learning plan for each skill
        
        Returns:
            List of skill plans with resources
        """
        detailed_plans = []
        
        # Focus on top 3-4 skills based on timeline
        num_skills = 2 if timeline_days == 7 else 3 if timeline_days == 14 else 4
        top_skills = prioritized_skills[:num_skills]
        
        for i, skill_info in enumerate(top_skills, 1):
            skill = skill_info['skill']
            
            # Get resources for this skill
            resources = self._get_skill_resources(skill)
            
            plan = {
                'rank': i,
                'skill': skill,
                'priority': skill_info['priority'],
                'current_level': 'Beginner',  # Assume beginner
                'target_level': 'Intermediate' if timeline_days >= 14 else 'Beginner+',
                'estimated_weeks': skill_info['estimated_weeks'],
                'resources': resources,
                'daily_commitment': '2-3 hours' if timeline_days == 7 else '1-2 hours',
                'learning_path': self._create_skill_learning_path(skill, timeline_days)
            }
            
            detailed_plans.append(plan)
        
        return detailed_plans
    
    def _get_skill_resources(self, skill: str) -> Dict:
        """Get learning resources for a skill"""
        skill_lower = skill.lower()
        
        # Check if we have specific resources
        if skill_lower in self.skill_resources:
            return self.skill_resources[skill_lower].get('beginner', {})
        
        # Use default resources
        return {
            'courses': [
                {'name': f'{skill.title()} for Beginners', 'duration': '4 weeks', 'platform': 'Udemy/Coursera'},
                {'name': f'Complete {skill.title()} Guide', 'duration': '6 weeks', 'platform': 'Various'}
            ],
            'tutorials': [
                {'name': f'Official {skill.title()} Documentation', 'type': 'reading'},
                {'name': f'{skill.title()} YouTube Tutorial Playlist', 'type': 'video'}
            ],
            'practice': [
                {'name': f'Build a {skill} project', 'type': 'hands-on'},
                {'name': f'{skill} coding challenges', 'type': 'problem-solving'}
            ]
        }
    
    def _create_skill_learning_path(self, skill: str, timeline_days: int) -> List[str]:
        """Create step-by-step learning path for a skill"""
        if timeline_days == 7:
            return [
                f'Day 1-2: Learn {skill} fundamentals',
                f'Day 3-4: Hands-on practice',
                f'Day 5-7: Build mini-project'
            ]
        elif timeline_days == 14:
            return [
                f'Week 1: Complete {skill} beginner course',
                f'Week 2: Build practical project',
                f'Bonus: Add to portfolio'
            ]
        else:  # 30 days
            return [
                f'Week 1: Fundamentals & setup',
                f'Week 2: Intermediate concepts',
                f'Week 3: Advanced techniques',
                f'Week 4: Real-world project'
            ]
    
    def _create_milestones(self, timeline_days: int) -> List[Dict]:
        """Create achievement milestones"""
        milestones = []
        
        if timeline_days == 7:
            milestones = [
                {'day': 3, 'milestone': 'Complete first tutorial', 'reward': '🎖️ Foundation Badge'},
                {'day': 7, 'milestone': 'Build first project', 'reward': '🏆 Builder Badge'}
            ]
        elif timeline_days == 14:
            milestones = [
                {'day': 4, 'milestone': 'Finish first course', 'reward': '📚 Learner Badge'},
                {'day': 10, 'milestone': 'Complete hands-on project', 'reward': '💻 Developer Badge'},
                {'day': 14, 'milestone': 'Update CV with new skills', 'reward': '🌟 Career Ready Badge'}
            ]
        else:  # 30 days
            milestones = [
                {'day': 7, 'milestone': 'First skill foundation complete', 'reward': '🎖️ Week 1 Champion'},
                {'day': 14, 'milestone': 'Second skill acquired', 'reward': '📚 Dual-Skill Badge'},
                {'day': 21, 'milestone': 'Advanced project built', 'reward': '💻 Advanced Developer'},
                {'day': 30, 'milestone': 'Portfolio updated & ready', 'reward': '🏆 Career Transformer'}
            ]
        
        return milestones
    
    def _estimate_improvement(self, prioritized_skills: List[Dict]) -> str:
        """Estimate match score improvement"""
        # Simple estimation: sum impact scores
        total_impact = sum(s['impact_score'] for s in prioritized_skills[:3])
        
        # Convert to percentage improvement
        improvement_pct = total_impact * 10  # Rough heuristic
        
        return f"+{improvement_pct:.0f}-{improvement_pct+5:.0f}% estimated score improvement"
    
    def _generate_success_tips(self, timeline_days: int) -> List[str]:
        """Generate success tips based on timeline"""
        base_tips = [
            "📅 Set aside dedicated learning time each day",
            "🎯 Focus on one concept at a time - depth over breadth",
            "💻 Practice by building real projects, not just tutorials",
            "📝 Document your learning journey in a blog or GitHub",
            "👥 Join online communities for support and accountability"
        ]
        
        if timeline_days == 7:
            base_tips.insert(0, "⚡ Intensive week - stay focused and avoid distractions")
        elif timeline_days == 14:
            base_tips.insert(0, "🎯 Balanced approach - consistency is key")
        else:
            base_tips.insert(0, "🚀 Long-term success - build sustainable learning habits")
        
        return base_tips


# Convenience function
def generate_learning_path(missing_skills: List[Dict], timeline_days: int = 30) -> Dict:
    """Quick pathway generation"""
    generator = LearningPathwayGenerator()
    return generator.generate_pathway(missing_skills, timeline_days)