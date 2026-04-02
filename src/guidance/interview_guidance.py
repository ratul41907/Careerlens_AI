"""
Interview Guidance - 100% LLM-Based Question Generation
Generates contextual interview questions using Ollama
"""
import requests
from typing import List, Dict, Optional
import json


class InterviewGuidance:
    """
    Generate interview questions and evaluate answers using LLM
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize interview guidance system
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma2:2b"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1500) -> str:
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
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def generate_questions(
        self,
        skills: Optional[List[str]] = None,
        cv_data: Optional[Dict] = None,
        jd_data: Optional[Dict] = None,
        num_questions: int = 10,
        question_type: str = "mixed"
    ) -> Dict:
        """
        Generate interview questions using LLM
        
        Args:
            skills: List of skills to focus on
            cv_data: CV data for context
            jd_data: Job description data
            num_questions: Number of questions to generate
            question_type: 'behavioral', 'technical', 'coding', 'system_design', or 'mixed'
            
        Returns:
            Dict with questions and metadata
        """
        try:
            # Build skills string
            if skills and isinstance(skills, list):
                skills_str = ', '.join(str(s) for s in skills[:10])
            elif skills:
                skills_str = str(skills)
            else:
                skills_str = "general software development"
            
            # Build prompt
            prompt = f"""Generate {num_questions} interview questions for a candidate with these skills: {skills_str}

Return ONLY a JSON array of questions in this exact format:
[
  {{
    "question": "Tell me about a time when you had to debug a complex production issue",
    "category": "Behavioral",
    "difficulty": "Medium",
    "hints": ["Focus on your problem-solving approach", "Explain the tools you used"]
  }}
]

Requirements:
- Generate exactly {num_questions} questions
- Mix of categories: Behavioral, Technical, System Design
- Each question must have: question, category, difficulty, hints
- Return ONLY valid JSON array, no other text

Generate the JSON array now:"""

            # Call Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                
                # Clean response
                result = result.strip()
                
                # Extract JSON
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0]
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0]
                
                # Find JSON array
                if '[' in result:
                    result = result[result.index('['):result.rindex(']')+1]
                
                # Parse JSON
                questions = json.loads(result)
                
                # Validate and ensure proper structure
                validated_questions = []
                for q in questions:
                    if isinstance(q, dict) and 'question' in q:
                        validated_questions.append({
                            'question': q.get('question', 'No question'),
                            'category': q.get('category', 'General'),
                            'difficulty': q.get('difficulty', 'Medium'),
                            'hints': q.get('hints', [])
                        })
                
                if validated_questions:
                    # Group by category for compatibility
                    by_category = {}
                    for q in validated_questions:
                        category = q['category']
                        if category not in by_category:
                            by_category[category] = []
                        by_category[category].append(q)
                    
                    # Preparation tips
                    preparation_tips = [
                        "Practice the STAR method for behavioral questions",
                        "Review key concepts for your technical skills",
                        "Prepare examples from your real work experience",
                        "Research the company and role thoroughly"
                    ]
                    
                    return {
                        'success': True,
                        'questions': validated_questions,
                        'by_category': by_category,
                        'total_questions': len(validated_questions),
                        'preparation_tips': preparation_tips
                    }
            
            # Fallback if LLM fails
            return self._generate_fallback_questions(skills_str, num_questions)
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._generate_fallback_questions(skills if isinstance(skills, str) else "general", num_questions)
    
    def _generate_fallback_questions(self, skills_context: str, num_questions: int) -> Dict:
        """Generate fallback questions if LLM fails"""
        fallback_questions = [
            {
                'question': f"Tell me about a challenging project involving {skills_context} and how you overcame obstacles.",
                'category': "Behavioral",
                'difficulty': "Medium",
                'hints': ["Use STAR method", "Focus on your specific actions"]
            },
            {
                'question': f"Explain how you would implement {skills_context} in a production environment.",
                'category': "Technical",
                'difficulty': "Medium",
                'hints': ["Consider scalability", "Discuss best practices"]
            },
            {
                'question': "Describe a time when you had to debug a critical production issue under time pressure.",
                'category': "Behavioral",
                'difficulty': "Medium",
                'hints': ["Explain your debugging process", "Show problem-solving skills"]
            },
            {
                'question': "How would you design a scalable system that handles 1 million requests per day?",
                'category': "System Design",
                'difficulty': "Hard",
                'hints': ["Discuss architecture patterns", "Consider bottlenecks"]
            },
            {
                'question': "Describe your approach to writing clean, maintainable code.",
                'category': "Technical",
                'difficulty': "Easy",
                'hints': ["Mention code reviews", "Discuss testing strategies"]
            },
            {
                'question': "Tell me about a time you had to learn a new technology quickly.",
                'category': "Behavioral",
                'difficulty': "Easy",
                'hints': ["Show learning ability", "Explain your process"]
            },
            {
                'question': "How do you handle disagreements with team members about technical decisions?",
                'category': "Behavioral",
                'difficulty': "Medium",
                'hints': ["Show collaboration skills", "Focus on outcomes"]
            },
            {
                'question': "Explain the trade-offs between different architectural patterns you've used.",
                'category': "System Design",
                'difficulty': "Hard",
                'hints': ["Compare microservices vs monolith", "Discuss real examples"]
            },
            {
                'question': "What strategies do you use to ensure code quality in a fast-paced environment?",
                'category': "Technical",
                'difficulty': "Medium",
                'hints': ["Mention CI/CD", "Discuss testing pyramid"]
            },
            {
                'question': "Describe a time when you had to make a difficult trade-off decision.",
                'category': "Behavioral",
                'difficulty': "Medium",
                'hints': ["Explain the constraints", "Show decision-making process"]
            }
        ]
        
        selected_questions = fallback_questions[:num_questions]
        
        # Group by category
        by_category = {}
        for q in selected_questions:
            category = q['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)
        
        preparation_tips = [
            "Practice the STAR method for behavioral questions",
            "Review key concepts for your technical skills",
            "Prepare examples from your real work experience",
            "Research the company and role thoroughly"
        ]
        
        return {
            'success': True,
            'questions': selected_questions,
            'by_category': by_category,
            'total_questions': len(selected_questions),
            'preparation_tips': preparation_tips,
            'metadata': {
                'generated_by': 'fallback',
                'note': 'Using fallback questions - check Ollama connection'
            }
        }
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        category: str = "General"
    ) -> Dict:
        """
        Evaluate an interview answer using LLM
        
        Args:
            question: The interview question
            answer: Candidate's answer
            category: Question category
            
        Returns:
            Evaluation with score and feedback
        """
        try:
            word_count = len(answer.split())
            
            prompt = f"""You are an expert interviewer evaluating a candidate's answer.

Question: {question}
Category: {category}

Candidate's Answer:
{answer}

Evaluate this answer and return ONLY a JSON object:
{{
  "score": 75,
  "rating": "Good",
  "strengths": ["Clear structure", "Good examples"],
  "weaknesses": ["Could add more metrics", "Missing context"],
  "suggestions": ["Quantify results", "Add more technical detail"]
}}

Scoring criteria:
- 90-100: Excellent (complete, clear, with great examples)
- 75-89: Good (solid answer with some examples)
- 60-74: Acceptable (answers question but lacks depth)
- Below 60: Needs improvement (incomplete or unclear)

Return ONLY the JSON object:"""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                
                # Clean and parse
                result = result.strip()
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0]
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0]
                
                if '{' in result:
                    result = result[result.index('{'):result.rindex('}')+1]
                
                evaluation = json.loads(result)
                
                # Ensure proper structure
                score = evaluation.get('score', 70)
                rating = evaluation.get('rating', 'Good')
                
                # Build breakdown
                breakdown = {
                    'situation': {'score': score // 4, 'max_score': 25},
                    'task': {'score': score // 4, 'max_score': 25},
                    'action': {'score': score // 4, 'max_score': 25},
                    'result': {'score': score // 4, 'max_score': 25},
                    'length': {
                        'word_count': word_count,
                        'score': min(20, word_count // 10),
                        'status': '✅ Good length' if 150 <= word_count <= 400 else '⚠️ Check length'
                    }
                }
                
                return {
                    'overall_score': score,
                    'rating': rating,
                    'breakdown': breakdown,
                    'feedback': evaluation.get('strengths', []) + evaluation.get('weaknesses', []),
                    'suggestions': evaluation.get('suggestions', [])
                }
            
            # Fallback evaluation
            return self._fallback_evaluation(word_count, answer)
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return self._fallback_evaluation(len(answer.split()), answer)
    
    def _fallback_evaluation(self, word_count: int, answer: str) -> Dict:
        """Fallback evaluation if LLM fails"""
        if word_count < 50:
            score = 40
            rating = "Needs Improvement"
            feedback = ["Answer is too brief", "Add more detail and examples"]
        elif word_count < 150:
            score = 65
            rating = "Acceptable"
            feedback = ["Good start", "Could add more specific examples"]
        elif word_count < 400:
            score = 80
            rating = "Good"
            feedback = ["Well-structured answer", "Good level of detail"]
        else:
            score = 75
            rating = "Good"
            feedback = ["Comprehensive answer", "Consider being more concise"]
        
        breakdown = {
            'situation': {'score': score // 4, 'max_score': 25},
            'task': {'score': score // 4, 'max_score': 25},
            'action': {'score': score // 4, 'max_score': 25},
            'result': {'score': score // 4, 'max_score': 25},
            'length': {
                'word_count': word_count,
                'score': min(20, word_count // 10),
                'status': '✅ Good length' if 150 <= word_count <= 400 else '⚠️ Check length'
            }
        }
        
        return {
            'overall_score': score,
            'rating': rating,
            'breakdown': breakdown,
            'feedback': feedback,
            'suggestions': ["Add more specific examples", "Quantify your results"]
        }


# Test
if __name__ == "__main__":
    guidance = InterviewGuidance()
    
    # Test question generation
    result = guidance.generate_questions(
        skills=["Python", "FastAPI", "Docker"],
        num_questions=5,
        question_type="mixed"
    )
    
    print(f"✅ Generated {result['total_questions']} questions")
    for q in result['questions']:
        print(f"\n[{q['category']}] {q['question']}")