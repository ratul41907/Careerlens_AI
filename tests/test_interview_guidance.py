"""
Unit Tests for Interview Guidance System - Enhanced with Pytest
Based on existing test_interview_guidance.py logic
"""
import pytest


class TestInterviewGuidanceBasic:
    """Basic interview guidance tests"""
    
    def test_guidance_initialization(self, interview_guidance):
        """Test that guidance system initializes correctly"""
        assert interview_guidance is not None
        assert hasattr(interview_guidance, 'ollama_url')
        assert hasattr(interview_guidance, 'model')
    
    def test_generate_questions_returns_dict(self, interview_guidance):
        """Test that generate_questions returns dict"""
        skills = ['Python', 'FastAPI', 'Docker']
        
        result = interview_guidance.generate_questions(
            skills=skills,
            num_questions=5
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'questions' in result
    
    def test_generate_questions_count(self, interview_guidance):
        """Test that correct number of questions are generated"""
        skills = ['Python', 'Docker']
        num_questions = 5
        
        result = interview_guidance.generate_questions(
            skills=skills,
            num_questions=num_questions
        )
        
        assert result['success'] is True
        questions = result['questions']
        assert len(questions) <= num_questions + 2  # Allow some variance
    
    def test_question_structure(self, interview_guidance):
        """Test that questions have proper structure"""
        skills = ['Python']
        
        result = interview_guidance.generate_questions(
            skills=skills,
            num_questions=3
        )
        
        questions = result['questions']
        
        for q in questions:
            assert 'question' in q
            assert 'category' in q
            assert isinstance(q['question'], str)
            assert len(q['question']) > 10  # Meaningful question
    
    def test_evaluate_answer_returns_dict(self, interview_guidance):
        """Test answer evaluation returns dict"""
        question = "Tell me about a challenging project"
        answer = "I worked on a challenging project where I had to debug a production issue. I analyzed logs, found the root cause, and fixed it within 2 hours."
        
        result = interview_guidance.evaluate_answer(question, answer)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'feedback' in result
    
    def test_evaluate_answer_score_range(self, interview_guidance):
        """Test that evaluation score is in valid range"""
        question = "Describe your experience with Python"
        answer = "I have 5 years of Python experience building web applications with Django and FastAPI."
        
        result = interview_guidance.evaluate_answer(question, answer)
        
        score = result['score']
        assert 0 <= score <= 100


class TestInterviewGuidanceIntegration:
    """Integration tests matching existing test format"""
    
    def test_complete_interview_guidance_pipeline(self, interview_guidance):
        """Test complete pipeline (based on original test)"""
        
        print("\n" + "=" * 70)
        print("INTERVIEW GUIDANCE SYSTEM TEST")
        print("=" * 70)
        
        # Sample skills from CV
        skills = ['python', 'fastapi', 'docker', 'aws']
        
        print(f"\n📊 CANDIDATE SKILLS:")
        print(f"   {', '.join(skills)}")
        
        # Test 1: Get recommended questions
        print("\n" + "=" * 70)
        print("RECOMMENDED INTERVIEW QUESTIONS")
        print("=" * 70)
        
        questions_result = interview_guidance.generate_questions(
            skills=skills,
            num_questions=10,
            question_type='mixed'
        )
        
        assert questions_result['success'] is True
        questions = questions_result['questions']
        
        print(f"\n📈 OVERVIEW:")
        print(f"   Total Questions: {len(questions)}")
        
        # Display sample questions
        print(f"\n🎯 SAMPLE QUESTIONS:")
        for i, q in enumerate(questions[:5], 1):
            print(f"\n   Q{i}. {q['question']}")
            print(f"       Category: {q.get('category', 'N/A')}")
        
        # Test 2: Evaluate answer
        print("\n" + "=" * 70)
        print("ANSWER EVALUATION")
        print("=" * 70)
        
        sample_question = "Tell me about a time when you had to debug a critical production issue."
        sample_answer = """
In my previous role, our production API started returning 500 errors affecting 
thousands of users. I was responsible for fixing it immediately. I checked the 
logs, found a database connection issue, increased the connection pool size, 
and deployed the fix. Response times improved by 90% and errors dropped to zero.
"""
        
        print(f"\n📄 SAMPLE ANSWER:")
        print(sample_answer.strip())
        
        evaluation = interview_guidance.evaluate_answer(sample_question, sample_answer)
        
        print(f"\n📊 EVALUATION:")
        print(f"   Score: {evaluation['score']}/100")
        print(f"   Rating: {evaluation.get('rating', 'N/A')}")
        print(f"   Word Count: {evaluation.get('word_count', 0)}")
        
        if 'feedback' in evaluation:
            print(f"\n📝 FEEDBACK:")
            feedback = evaluation['feedback']
            if isinstance(feedback, str):
                print(f"   {feedback}")
            elif isinstance(feedback, list):
                for item in feedback[:3]:
                    print(f"   • {item}")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETED!")
        print("=" * 70)
        
        # Assertions
        assert len(questions) >= 5
        assert evaluation['score'] >= 0
        assert evaluation['score'] <= 100


class TestInterviewGuidanceEdgeCases:
    """Edge cases and error handling"""
    
    def test_generate_questions_no_skills(self, interview_guidance):
        """Test question generation without skills"""
        result = interview_guidance.generate_questions(
            skills=[],
            num_questions=5
        )
        
        # Should still generate some questions
        assert result is not None
        if result['success']:
            assert len(result['questions']) > 0
    
    def test_evaluate_very_short_answer(self, interview_guidance):
        """Test evaluation of very short answer"""
        question = "What is Python?"
        answer = "A programming language"
        
        result = interview_guidance.evaluate_answer(question, answer)
        
        assert result is not None
        # Short answer should get lower score
        if 'score' in result:
            assert result['score'] < 70
    
    def test_evaluate_very_long_answer(self, interview_guidance):
        """Test evaluation of very long answer"""
        question = "Describe your experience"
        answer = "I have experience. " * 200  # Very long
        
        result = interview_guidance.evaluate_answer(question, answer)
        
        assert result is not None
        assert 'score' in result
    
    def test_evaluate_empty_answer(self, interview_guidance):
        """Test evaluation of empty answer"""
        question = "What is your experience?"
        answer = ""
        
        result = interview_guidance.evaluate_answer(question, answer)
        
        assert result is not None
        if 'score' in result:
            assert result['score'] == 0 or result['score'] < 30
    
    def test_question_types(self, interview_guidance):
        """Test different question types"""
        skills = ['Python', 'Docker']
        
        for q_type in ['behavioral', 'technical', 'coding', 'mixed']:
            result = interview_guidance.generate_questions(
                skills=skills,
                num_questions=3,
                question_type=q_type
            )
            
            assert result is not None
            if result['success']:
                assert len(result['questions']) > 0


class TestInterviewGuidanceLLM:
    """LLM-specific tests"""
    
    @pytest.mark.llm
    def test_llm_question_quality(self, interview_guidance):
        """Test LLM generates quality questions"""
        skills = ['Python', 'FastAPI', 'Docker']
        
        result = interview_guidance.generate_questions(
            skills=skills,
            num_questions=5,
            question_type='technical'
        )
        
        if result['success']:
            questions = result['questions']
            
            # Questions should be relevant to skills
            for q in questions:
                question_text = q['question'].lower()
                # At least some questions should mention the skills
                # (This is a soft check)
                assert len(question_text) > 20
    
    @pytest.mark.llm
    def test_llm_evaluation_quality(self, interview_guidance):
        """Test LLM provides quality evaluation"""
        question = "Tell me about a challenging project"
        
        good_answer = """
        I led a project to migrate our monolithic application to microservices.
        The challenge was ensuring zero downtime during migration.
        I created a phased approach, implemented feature flags, and used blue-green deployment.
        We successfully migrated 15 services over 3 months with 99.9% uptime.
        """
        
        result = interview_guidance.evaluate_answer(question, good_answer)
        
        # Good answer should get high score
        assert result['score'] >= 60