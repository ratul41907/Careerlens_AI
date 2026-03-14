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
                        "temperature": 0.8
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _parse_questions(self, llm_response: str) -> List[Dict]:
        """
        Parse LLM response into structured questions
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            List of question dicts
        """
        questions = []
        current_question = None
        current_category = "General"
        
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect category headers
            line_lower = line.lower()
            if 'behavioral' in line_lower or 'behavior' in line_lower:
                current_category = "Behavioral"
            elif 'technical' in line_lower:
                current_category = "Technical"
            elif 'coding' in line_lower or 'algorithm' in line_lower:
                current_category = "Coding"
            elif 'system design' in line_lower or 'architecture' in line_lower:
                current_category = "System Design"
            
            # Extract questions (lines starting with number or bullet)
            if line and (line[0].isdigit() or line.startswith(('-', '•', '*', 'Q:'))):
                # Clean the question
                clean_q = line.lstrip('0123456789.)•*-Q: ').strip()
                
                if clean_q and len(clean_q) > 20 and '?' in clean_q:
                    questions.append({
                        'question': clean_q,
                        'category': current_category,
                        'difficulty': 'Medium'
                    })
        
        return questions
    
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
        # Build context
        context_parts = []
        
        if skills:
            context_parts.append(f"Skills to assess: {', '.join(skills[:10])}")
        
        if cv_data and cv_data.get('text'):
            context_parts.append(f"Candidate's CV:\n{cv_data['text'][:1000]}")
        
        if jd_data and jd_data.get('text'):
            context_parts.append(f"Job Description:\n{jd_data['text'][:1000]}")
        
        context = '\n\n'.join(context_parts) if context_parts else "General software engineering role"
        
        # Build prompt based on question type
        if question_type == "behavioral":
            type_instruction = f"Generate {num_questions} behavioral interview questions using the STAR method (Situation, Task, Action, Result). Focus on real scenarios."
        elif question_type == "technical":
            type_instruction = f"Generate {num_questions} technical interview questions about concepts, tools, and best practices."
        elif question_type == "coding":
            type_instruction = f"Generate {num_questions} coding/algorithm questions with clear problem statements."
        elif question_type == "system_design":
            type_instruction = f"Generate {num_questions} system design questions about architecture and scalability."
        else:  # mixed
            type_instruction = f"Generate {num_questions} interview questions: mix of behavioral ({num_questions//3}), technical ({num_questions//3}), and coding/system design ({num_questions//3})."
        
        prompt = f"""You are an expert technical interviewer. {type_instruction}

Context:
{context}

Requirements:
- Make questions specific to the candidate's background and job requirements
- Ensure questions test actual competency, not just theoretical knowledge
- For behavioral: Use STAR framework
- For technical: Focus on practical application
- For coding: Provide clear problem statements
- For system design: Ask about real-world scenarios

Format each question on a new line starting with a number.
Organize by category with headers: BEHAVIORAL, TECHNICAL, CODING, SYSTEM DESIGN

Generate the questions now:"""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=1500)
        
        if not llm_response:
            # Fallback generic questions
            fallback_questions = [
                {
                    'question': "Tell me about a challenging project you worked on and how you overcame obstacles.",
                    'category': "Behavioral",
                    'difficulty': "Medium"
                },
                {
                    'question': f"Explain how you would implement {skills[0] if skills else 'a web application'} in a production environment.",
                    'category': "Technical",
                    'difficulty': "Medium"
                },
                {
                    'question': "Describe a time when you had to debug a critical production issue under time pressure.",
                    'category': "Behavioral",
                    'difficulty': "Medium"
                },
                {
                    'question': "How would you design a scalable API that handles 1 million requests per day?",
                    'category': "System Design",
                    'difficulty': "Hard"
                },
                {
                    'question': "Write a function to find the longest palindromic substring in a given string.",
                    'category': "Coding",
                    'difficulty': "Medium"
                }
            ]
            
            return {
                'success': True,
                'questions': fallback_questions[:num_questions],
                'total_questions': min(len(fallback_questions), num_questions),
                'metadata': {
                    'generated_by': 'fallback',
                    'question_type': question_type
                }
            }
        
        # Parse questions
        questions = self._parse_questions(llm_response)
        
        # Ensure we have enough questions
        if len(questions) < num_questions:
            # Add generic questions to fill
            generic = [
                {
                    'question': "Describe your approach to writing clean, maintainable code.",
                    'category': "Technical",
                    'difficulty': "Medium"
                },
                {
                    'question': "Tell me about a time you had to learn a new technology quickly.",
                    'category': "Behavioral",
                    'difficulty': "Easy"
                }
            ]
            questions.extend(generic[:num_questions - len(questions)])
        
        return {
            'success': True,
            'questions': questions[:num_questions],
            'total_questions': len(questions[:num_questions]),
            'metadata': {
                'generated_by': 'llm',
                'question_type': question_type,
                'skills_assessed': skills[:5] if skills else []
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
        prompt = f"""You are an expert interviewer evaluating a candidate's answer.

Question: {question}
Category: {category}

Candidate's Answer:
{answer}

Evaluate this answer and provide:

1. SCORE (0-100)
Rate the answer quality considering:
- Completeness (did they answer the question?)
- Clarity (is it well-structured and easy to follow?)
- Depth (do they show real understanding?)
- Examples (do they provide concrete examples?)

2. STRENGTHS
What did the candidate do well? (2-3 bullet points)

3. WEAKNESSES
What could be improved? (2-3 bullet points)

4. SUGGESTIONS
How can they improve their answer? (2-3 specific suggestions)

5. STAR ANALYSIS (if behavioral question)
- Situation: Did they set context?
- Task: Was the challenge clear?
- Action: Did they explain what THEY did?
- Result: Did they show measurable outcomes?

Provide your evaluation:"""
        
        llm_response = self._call_ollama(prompt, max_tokens=1000)
        
        if not llm_response:
            # Fallback evaluation
            word_count = len(answer.split())
            
            if word_count < 50:
                score = 40
                feedback = "Answer is too brief. Provide more detail and examples."
            elif word_count < 150:
                score = 60
                feedback = "Good start. Add more specific examples and measurable outcomes."
            elif word_count < 400:
                score = 75
                feedback = "Well-structured answer. Consider adding more depth to key points."
            else:
                score = 85
                feedback = "Comprehensive answer with good detail."
            
            return {
                'score': score,
                'feedback': feedback,
                'word_count': word_count,
                'rating': 'Good' if score >= 70 else 'Needs Improvement'
            }
        
        # Parse score from LLM response
        score = 70  # default
        try:
            for line in llm_response.split('\n'):
                if 'score' in line.lower() and any(c.isdigit() for c in line):
                    # Extract number
                    numbers = ''.join(c for c in line if c.isdigit())
                    if numbers:
                        score = min(100, int(numbers[:2]))  # Take first 2 digits, cap at 100
                        break
        except:
            pass
        
        # Determine rating
        if score >= 90:
            rating = "Excellent"
        elif score >= 75:
            rating = "Good"
        elif score >= 60:
            rating = "Acceptable"
        else:
            rating = "Needs Improvement"
        
        return {
            'score': score,
            'rating': rating,
            'feedback': llm_response,
            'word_count': len(answer.split())
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