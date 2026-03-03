"""
Interview Guidance System - Helps candidates prepare for technical interviews
"""
from typing import Dict, List, Optional
from loguru import logger
import random


class InterviewGuidanceSystem:
    """
    Generate interview questions and STAR method answers
    """
    
    def __init__(self):
        """Initialize interview guidance system"""
        
        # Common technical interview questions by category
        self.question_database = {
            'behavioral': [
                {
                    'question': 'Tell me about a time when you faced a challenging deadline. How did you handle it?',
                    'category': 'Time Management',
                    'difficulty': 'Medium',
                    'star_example': {
                        'situation': 'Product launch deadline moved up by 2 weeks',
                        'task': 'Complete API integration and testing ahead of schedule',
                        'action': 'Prioritized critical features, automated testing, worked with team to parallelize work',
                        'result': 'Delivered 3 days early, 100% test coverage, zero critical bugs in production'
                    }
                },
                {
                    'question': 'Describe a situation where you had to work with a difficult team member.',
                    'category': 'Teamwork',
                    'difficulty': 'Hard',
                    'star_example': {
                        'situation': 'Team member consistently missed code review deadlines',
                        'task': 'Maintain code quality while keeping project on track',
                        'action': 'Had 1-on-1 conversation, understood blockers, paired programming sessions, adjusted review process',
                        'result': 'Review turnaround improved from 3 days to same-day, team velocity increased 25%'
                    }
                },
                {
                    'question': 'Give an example of when you took initiative on a project.',
                    'category': 'Leadership',
                    'difficulty': 'Medium',
                    'star_example': {
                        'situation': 'Noticed repeated customer complaints about slow dashboard load times',
                        'task': 'Improve performance without disrupting ongoing features',
                        'action': 'Profiled application, implemented Redis caching, optimized database queries',
                        'result': 'Load time reduced from 8s to 1.2s (85% improvement), customer satisfaction +30%'
                    }
                }
            ],
            'technical': [
                {
                    'question': 'Explain the difference between a list and a tuple in Python.',
                    'skill': 'python',
                    'difficulty': 'Easy',
                    'answer_points': [
                        'Lists are mutable, tuples are immutable',
                        'Lists use [], tuples use ()',
                        'Tuples are faster for iteration',
                        'Tuples can be used as dictionary keys'
                    ]
                },
                {
                    'question': 'What is a RESTful API? What are its key principles?',
                    'skill': 'api',
                    'difficulty': 'Medium',
                    'answer_points': [
                        'REST = Representational State Transfer',
                        'Stateless communication',
                        'HTTP methods: GET, POST, PUT, DELETE',
                        'Resource-based URLs',
                        'JSON response format'
                    ]
                },
                {
                    'question': 'How does Docker work? What problem does it solve?',
                    'skill': 'docker',
                    'difficulty': 'Medium',
                    'answer_points': [
                        'Containerization platform',
                        'Packages application + dependencies',
                        'Solves "works on my machine" problem',
                        'Lightweight vs VMs',
                        'Image → Container lifecycle'
                    ]
                },
                {
                    'question': 'Explain the difference between SQL and NoSQL databases.',
                    'skill': 'database',
                    'difficulty': 'Medium',
                    'answer_points': [
                        'SQL: Structured, relational, ACID compliant',
                        'NoSQL: Flexible schema, horizontal scaling',
                        'Use cases: SQL for transactions, NoSQL for big data',
                        'Examples: PostgreSQL (SQL) vs MongoDB (NoSQL)'
                    ]
                }
            ],
            'coding': [
                {
                    'question': 'Write a function to reverse a string.',
                    'skill': 'python',
                    'difficulty': 'Easy',
                    'solution': 'def reverse_string(s):\n    return s[::-1]',
                    'concepts': ['String slicing', 'Pythonic syntax']
                },
                {
                    'question': 'Implement a function to check if a string is a palindrome.',
                    'skill': 'python',
                    'difficulty': 'Easy',
                    'solution': 'def is_palindrome(s):\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]',
                    'concepts': ['String manipulation', 'Comparison']
                },
                {
                    'question': 'Write a function to find the nth Fibonacci number.',
                    'skill': 'algorithm',
                    'difficulty': 'Medium',
                    'solution': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b',
                    'concepts': ['Iteration', 'Dynamic programming']
                }
            ],
            'system_design': [
                {
                    'question': 'Design a URL shortening service like bit.ly.',
                    'difficulty': 'Hard',
                    'key_components': [
                        'Hash function for URL generation',
                        'Database schema (URL mapping)',
                        'Caching layer (Redis)',
                        'Load balancing',
                        'Analytics tracking'
                    ]
                },
                {
                    'question': 'How would you design a real-time chat application?',
                    'difficulty': 'Hard',
                    'key_components': [
                        'WebSocket connections',
                        'Message queue (Kafka/RabbitMQ)',
                        'Database for message persistence',
                        'Notification service',
                        'Scaling considerations'
                    ]
                }
            ]
        }
        
        # STAR method templates
        self.star_templates = {
            'achievement': {
                'situation': 'In my role as {role}, we faced {challenge}',
                'task': 'I was responsible for {responsibility}',
                'action': 'I {action_verb} by {specific_actions}',
                'result': 'This resulted in {quantified_outcome}'
            },
            'problem_solving': {
                'situation': 'We encountered {problem} that was affecting {impact}',
                'task': 'My goal was to {objective}',
                'action': 'I approached this by {methodology}',
                'result': 'The solution {outcome} and {metric}'
            },
            'collaboration': {
                'situation': 'While working on {project}, we needed to {collaboration_need}',
                'task': 'I took on the role of {role}',
                'action': 'I facilitated {collaborative_action}',
                'result': 'The team {team_outcome} and delivered {deliverable}'
            }
        }
        
        logger.info("InterviewGuidanceSystem initialized")
    
    def get_recommended_questions(self,
                                 skills: List[str],
                                 num_questions: int = 10,
                                 include_behavioral: bool = True) -> Dict:
        """
        Get recommended interview questions based on skills
        
        Args:
            skills: List of skills from CV/JD match
            num_questions: Number of questions to return
            include_behavioral: Include behavioral questions
            
        Returns:
            Dict with categorized questions
        """
        logger.info(f"Generating {num_questions} interview questions for skills: {skills}")
        
        recommended = []
        
        # Add behavioral questions
        if include_behavioral:
            for q in self.question_database['behavioral']:
                recommended.append({
                    'question': q['question'],
                    'category': 'Behavioral',
                    'difficulty': q['difficulty'],
                    'hint': f"Use STAR method: {q['star_example']['situation'][:50]}..."
                })
        
        # Add technical questions matching skills
        for question in self.question_database['technical']:
            if any(skill.lower() in question.get('skill', '').lower() for skill in skills):
                recommended.append({
                    'question': question['question'],
                    'category': 'Technical',
                    'difficulty': question['difficulty'],
                    'hint': question['answer_points'][0] if question.get('answer_points') else None
                })
        
        # Add coding questions
        for question in self.question_database['coding']:
            if any(skill.lower() in question.get('skill', '').lower() for skill in skills):
                recommended.append({
                    'question': question['question'],
                    'category': 'Coding',
                    'difficulty': question['difficulty'],
                    'hint': ', '.join(question.get('concepts', []))
                })
        
        # Add system design for senior roles
        for q in self.question_database['system_design'][:2]:
            recommended.append({
                'question': q['question'],
                'category': 'System Design',
                'difficulty': q['difficulty'],
                'hint': q['key_components'][0] if q.get('key_components') else None
            })
        
        # Group by category
        by_category = {}
        for q in recommended:
            category = q['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)
        
        result = {
            'total_questions': len(recommended),
            'by_category': by_category,
            'questions_by_category': by_category,  # Backward compatibility
            'preparation_tips': self._generate_prep_tips(skills),
            'difficulty_breakdown': self._calculate_difficulty_breakdown(recommended)
        }
        
        logger.info(f"Generated {len(recommended)} questions across {len(by_category)} categories")
        return result
    
    def generate_star_answer(self,
                            question: str,
                            user_context: Optional[Dict] = None) -> Dict:
        """
        Generate STAR method answer template
        
        Args:
            question: Interview question
            user_context: Optional user context (role, company, project)
            
        Returns:
            Dict with STAR framework answer
        """
        logger.info(f"Generating STAR answer for: {question[:50]}...")
        
        # Determine question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['challenging', 'difficult', 'problem']):
            template_type = 'problem_solving'
        elif any(word in question_lower for word in ['team', 'collaborate', 'work with']):
            template_type = 'collaboration'
        else:
            template_type = 'achievement'
        
        template = self.star_templates[template_type]
        
        # Generate answer structure
        answer = {
            'question': question,
            'method': 'STAR (Situation, Task, Action, Result)',
            'framework': {
                'situation': {
                    'prompt': 'Describe the context and challenge',
                    'template': template['situation'],
                    'tips': [
                        'Set the scene briefly (1-2 sentences)',
                        'Mention company/project name if relevant',
                        'Highlight the challenge or stakes'
                    ]
                },
                'task': {
                    'prompt': 'Explain your specific responsibility',
                    'template': template['task'],
                    'tips': [
                        'Focus on YOUR role, not the team',
                        'Be specific about what you were accountable for',
                        'Mention any constraints (time, resources)'
                    ]
                },
                'action': {
                    'prompt': 'Detail the steps you took',
                    'template': template['action'],
                    'tips': [
                        'Use "I" not "we" - highlight your contribution',
                        'Be specific about methodology',
                        'Mention tools/technologies used',
                        'Show your thought process'
                    ]
                },
                'result': {
                    'prompt': 'Share the quantified outcome',
                    'template': template['result'],
                    'tips': [
                        'Quantify with numbers/percentages',
                        'Mention business impact',
                        'Include what you learned',
                        'End on a positive note'
                    ]
                }
            },
            'examples': [
                {
                    'scenario': 'Problem Solving',
                    'full_answer': self._generate_example_answer(question, 'problem_solving')
                },
                {
                    'scenario': 'Team Collaboration',
                    'full_answer': self._generate_example_answer(question, 'collaboration')
                },
                {
                    'scenario': 'Achievement',
                    'full_answer': self._generate_example_answer(question, 'achievement')
                }
            ],
            'common_mistakes': [
                '❌ Being too vague - use specific examples',
                '❌ Focusing on "we" instead of "I"',
                '❌ Forgetting to quantify results',
                '❌ Making up stories - use real experiences'
            ],
            'good_practices': [
                '✅ Use the 2-minute rule (1 min answer)',
                '✅ Practice out loud before the interview',
                '✅ Prepare 3-4 STAR stories covering different scenarios',
                '✅ Tailor examples to the job requirements'
            ]
        }
        
        return answer
    
    def _generate_example_answer(self, question: str, template_type: str) -> str:
        """Generate example STAR answer"""
        examples = {
            'problem_solving': """
**Situation:** In my previous role as a Backend Developer, our API response times suddenly increased to 5+ seconds, affecting 10,000+ daily users.

**Task:** I was responsible for identifying the root cause and implementing a solution within 48 hours before it impacted our SLA.

**Action:** I profiled the application using New Relic, identified N+1 query issues in our user endpoint, implemented eager loading with SQLAlchemy, added Redis caching for frequent queries, and deployed to production after thorough testing.

**Result:** Response times dropped from 5.2s to 340ms (93% improvement), user complaints reduced to zero, and we avoided potential SLA penalties worth $50K. I also documented the fix and presented it to the team to prevent similar issues.
            """,
            'collaboration': """
**Situation:** While building a microservices architecture, our frontend and backend teams were frequently blocked due to API contract mismatches.

**Task:** I took the initiative to improve cross-team collaboration and reduce integration issues.

**Action:** I organized weekly API design reviews, implemented OpenAPI (Swagger) documentation, created a shared Postman collection, and set up contract testing with Pact.

**Result:** Integration bugs decreased by 70%, deployment velocity increased from bi-weekly to weekly releases, and team satisfaction scores improved by 25% in our retrospectives.
            """,
            'achievement': """
**Situation:** Our legacy monolithic application was becoming difficult to scale and deploy, with downtime during every release.

**Task:** I was assigned to lead the migration to a microservices architecture without disrupting existing features.

**Action:** I designed the target architecture, created a phased migration plan, implemented the first three microservices (auth, user, payment) using FastAPI, set up CI/CD pipelines with GitHub Actions, and mentored two junior developers through the process.

**Result:** We achieved zero-downtime deployments, reduced deployment time from 2 hours to 15 minutes, improved system scalability by 300%, and successfully migrated 60% of features within 4 months.
            """
        }
        
        return examples.get(template_type, examples['achievement']).strip()
    
    def _generate_prep_tips(self, skills: List[str]) -> List[str]:
        """Generate preparation tips based on skills"""
        tips = [
            "📚 Review fundamental concepts for each skill on your CV",
            "💻 Practice coding problems on LeetCode/HackerRank (20-30 mins daily)",
            "🎯 Prepare 3-4 STAR stories covering: achievement, challenge, teamwork, failure",
            "🗣️ Practice explaining technical concepts to non-technical audience",
            "❓ Prepare questions to ask the interviewer about team/tech stack"
        ]
        
        # Add skill-specific tips
        if any(s in ['python', 'javascript', 'java'] for s in skills):
            tips.append("🐍 Review language-specific concepts (e.g., Python decorators, JS closures)")
        
        if any(s in ['docker', 'kubernetes', 'aws'] for s in skills):
            tips.append("☁️ Be ready to explain containerization and cloud architecture decisions")
        
        if any(s in ['react', 'angular', 'vue'] for s in skills):
            tips.append("⚛️ Understand component lifecycle and state management")
        
        return tips
    
    def _calculate_difficulty_breakdown(self, questions: List[Dict]) -> Dict:
        """Calculate difficulty distribution"""
        difficulty_count = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        
        for q in questions:
            diff = q.get('difficulty', 'Medium')
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        return difficulty_count
    
    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """
        Evaluate an interview answer based on STAR method
        
        Args:
            question: The interview question
            answer: The candidate's answer
            
        Returns:
            Evaluation results with score and feedback
        """
        # Word count
        word_count = len(answer.split())
        
        # Length score (20 points max)
        if 200 <= word_count <= 500:
            length_score = 20
            length_status = "✅ Good length"
        elif word_count < 200:
            length_score = max(0, int(word_count / 200 * 20))
            length_status = "⚠️ Too short - add more detail"
        else:
            length_score = max(10, 20 - int((word_count - 500) / 100 * 2))
            length_status = "⚠️ Too long - be more concise"
        
        # Check for STAR components (case-insensitive)
        answer_lower = answer.lower()
        
        # Situation (20 points)
        situation_keywords = ['situation', 'context', 'background', 'challenge', 'faced', 'when']
        situation_score = 20 if any(kw in answer_lower for kw in situation_keywords) else 5
        
        # Task (20 points)
        task_keywords = ['task', 'responsibility', 'role', 'goal', 'objective', 'needed to']
        task_score = 20 if any(kw in answer_lower for kw in task_keywords) else 5
        
        # Action (20 points) - should be substantial
        action_keywords = ['i developed', 'i created', 'i implemented', 'i designed', 
                          'i built', 'i led', 'i organized', 'my approach', 'i decided']
        action_count = sum(1 for kw in action_keywords if kw in answer_lower)
        action_score = min(20, action_count * 7)
        
        # Result (20 points)
        result_keywords = ['result', 'outcome', 'achieved', 'increased', 'decreased', 
                          'improved', 'reduced', '%', 'impact', 'delivered']
        result_count = sum(1 for kw in result_keywords if kw in answer_lower)
        result_score = min(20, result_count * 5)
        
        # Calculate overall score
        overall_score = length_score + situation_score + task_score + action_score + result_score
        
        # Rating
        if overall_score >= 80:
            rating = "Excellent"
        elif overall_score >= 60:
            rating = "Good"
        elif overall_score >= 40:
            rating = "Needs Improvement"
        else:
            rating = "Poor"
        
        # Generate feedback
        feedback = []
        
        if length_score < 15:
            if word_count < 200:
                feedback.append("⚠️ Answer too brief - add more details about the situation and your actions")
            else:
                feedback.append("⚠️ Answer too long - focus on the most important points")
        else:
            feedback.append("✅ Good answer length")
        
        if situation_score < 15:
            feedback.append("❌ Missing situation/context - set the scene for your story")
        else:
            feedback.append("✅ Situation described")
        
        if task_score < 15:
            feedback.append("❌ Missing task/responsibility - explain what you needed to do")
        else:
            feedback.append("✅ Task/responsibility clear")
        
        if action_score < 15:
            feedback.append("❌ Actions not detailed enough - explain specific steps YOU took (use 'I' not 'we')")
        else:
            feedback.append("✅ Actions well described")
        
        if result_score < 15:
            feedback.append("❌ Results missing or vague - include quantifiable outcomes")
        else:
            feedback.append("✅ Results included")
        
        # Suggestions for improvement
        suggestions = []
        
        if situation_score < 15:
            suggestions.append("Add context: When and where did this happen? What was the challenge?")
        
        if task_score < 15:
            suggestions.append("Clarify your role: What were you specifically responsible for?")
        
        if action_score < 15:
            suggestions.append("Detail your actions: What specific steps did YOU take? Focus on 'I' statements.")
        
        if result_score < 15:
            suggestions.append("Quantify results: Include numbers, percentages, or measurable outcomes.")
        
        if word_count < 200:
            suggestions.append("Expand your answer with more specific examples and details.")
        
        return {
            'overall_score': overall_score,
            'rating': rating,
            'breakdown': {
                'length': {
                    'score': length_score,
                    'max_score': 20,
                    'word_count': word_count,
                    'status': length_status
                },
                'situation': {
                    'score': situation_score,
                    'max_score': 20
                },
                'task': {
                    'score': task_score,
                    'max_score': 20
                },
                'action': {
                    'score': action_score,
                    'max_score': 20
                },
                'result': {
                    'score': result_score,
                    'max_score': 20
                }
            },
            'feedback': feedback,
            'suggestions': suggestions,
            'question': question,
            'answer_preview': answer[:200] + "..." if len(answer) > 200 else answer
        }


# Convenience function
def get_interview_prep(skills: List[str]) -> Dict:
    """Quick interview prep function"""
    system = InterviewGuidanceSystem()
    return system.get_recommended_questions(skills)