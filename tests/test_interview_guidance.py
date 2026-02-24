"""
Test Interview Guidance System
"""
from src.guidance.interview_guidance import InterviewGuidanceSystem
import json


def test_interview_guidance():
    """Test interview guidance system"""
    
    print("=" * 70)
    print("INTERVIEW GUIDANCE SYSTEM TEST")
    print("=" * 70)
    
    # Sample skills from CV
    skills = ['python', 'fastapi', 'docker', 'aws']
    
    print(f"\n📊 CANDIDATE SKILLS:")
    print(f"   {', '.join(skills)}")
    
    # Initialize system
    system = InterviewGuidanceSystem()
    
    # Test 1: Get recommended questions
    print("\n" + "=" * 70)
    print("RECOMMENDED INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = system.get_recommended_questions(skills, num_questions=10)
    
    print(f"\n📈 OVERVIEW:")
    print(f"   Total Questions: {questions['total_questions']}")
    print(f"   Difficulty Breakdown: {questions['difficulty_breakdown']}")
    
    print(f"\n🎯 BEHAVIORAL QUESTIONS:")
    for i, q in enumerate(questions['questions_by_category']['behavioral'][:3], 1):
        print(f"\n   Q{i}. {q['question']}")
        print(f"       Category: {q['category']} | Difficulty: {q['difficulty']}")
    
    print(f"\n💻 TECHNICAL QUESTIONS:")
    for i, q in enumerate(questions['questions_by_category']['technical'][:3], 1):
        print(f"\n   Q{i}. {q['question']}")
        print(f"       Skill: {q['skill']} | Difficulty: {q['difficulty']}")
    
    print(f"\n🔨 CODING QUESTIONS:")
    for i, q in enumerate(questions['questions_by_category']['coding'][:2], 1):
        print(f"\n   Q{i}. {q['question']}")
        print(f"       Skill: {q['skill']} | Difficulty: {q['difficulty']}")
    
    print(f"\n💡 PREPARATION TIPS:")
    for tip in questions['preparation_tips']:
        print(f"   {tip}")
    
    # Test 2: Generate STAR answer
    print("\n" + "=" * 70)
    print("STAR METHOD ANSWER GENERATOR")
    print("=" * 70)
    
    sample_question = "Tell me about a time when you had to debug a critical production issue."
    
    print(f"\n❓ QUESTION:")
    print(f"   {sample_question}")
    
    star_answer = system.generate_star_answer(sample_question)
    
    print(f"\n📝 STAR FRAMEWORK:")
    for component, details in star_answer['framework'].items():
        print(f"\n   {component.upper()}:")
        print(f"   Prompt: {details['prompt']}")
        print(f"   Tips:")
        for tip in details['tips'][:2]:
            print(f"      • {tip}")
    
    print(f"\n💡 EXAMPLE ANSWER:")
    print(star_answer['example_answer'])
    
    print(f"\n✅ GOOD PRACTICES:")
    for practice in star_answer['good_practices']:
        print(f"   {practice}")
    
    # Test 3: Evaluate answer
    print("\n" + "=" * 70)
    print("ANSWER EVALUATION")
    print("=" * 70)
    
    sample_answer = """
    In my previous role, our production API started returning 500 errors affecting 
    thousands of users. I was responsible for fixing it immediately. I checked the 
    logs, found a database connection issue, increased the connection pool size, 
    and deployed the fix. Response times improved by 90% and errors dropped to zero.
    """
    
    print(f"\n📄 SAMPLE ANSWER:")
    print(sample_answer.strip())
    
    evaluation = system.evaluate_answer(sample_answer)
    
    print(f"\n📊 EVALUATION:")
    print(f"   Score: {evaluation['score']}/100")
    print(f"   Rating: {evaluation['rating']}")
    print(f"   Word Count: {evaluation['word_count']}")
    
    print(f"\n📝 FEEDBACK:")
    for item in evaluation['feedback']:
        print(f"   {item}")
    
    print(f"\n💡 SUGGESTIONS:")
    for suggestion in evaluation['suggestions']:
        print(f"   • {suggestion}")
    
    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT (QUESTIONS):")
    print("=" * 70)
    print(json.dumps(questions, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    test_interview_guidance()