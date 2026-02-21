"""
Test Explainability Engine
"""
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.explainability import ExplainabilityEngine
import json


def test_explainability():
    """Test complete explainable matching pipeline"""
    
    print("=" * 70)
    print("EXPLAINABILITY ENGINE TEST")
    print("=" * 70)
    
    # Sample CV
    sample_cv = """
    JOHN DOE
    Senior Software Engineer
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2021 - Present
    • Built RESTful APIs using FastAPI and PostgreSQL
    • Deployed microservices on AWS using Docker and Kubernetes
    • Improved system performance by 40% through optimization
    • Led team of 3 junior developers
    
    Software Engineer | StartupXYZ | 2019 - 2021
    • Developed React-based dashboard for data visualization
    • Implemented CI/CD pipelines with GitHub Actions
    
    SKILLS
    Python, JavaScript, React, FastAPI, PostgreSQL, Docker, Kubernetes, AWS
    """
    
    # Sample JD
    sample_jd = """
    Senior Software Engineer Position
    
    Required:
    • 5+ years software development experience
    • Python and JavaScript proficiency required
    • FastAPI framework experience needed
    • AWS cloud services required
    • Docker and Kubernetes required
    
    Preferred:
    • TensorFlow or PyTorch experience
    • Go programming language
    """
    
    print("\n📊 STEP 1: Parsing & Matching")
    print("-" * 70)
    
    # Parse CV
    cv_parser = CVParser()
    cv_data = {
        'text': sample_cv,
        'sections': cv_parser._segment_sections(sample_cv)
    }
    
    # Parse JD
    jd_parser = JDParser()
    jd_data = jd_parser.parse(sample_jd)
    
    # Compute match
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    match_result = scoring_engine.compute_match_score(cv_data, jd_data)
    
    print(f"✅ Match Score: {match_result['overall_percentage']}")
    
    print("\n📝 STEP 2: Generating Explanations")
    print("-" * 70)
    
    # Generate explanations
    explainability_engine = ExplainabilityEngine(embedding_engine)
    explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
    
    # Display results
    print(f"\n{'='*70}")
    print("EXPLAINABLE MATCH REPORT")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"   {explanation['overall_assessment']}")
    
    print(f"\n📈 EXECUTIVE SUMMARY:")
    summary = explanation['summary']
    print(f"   • Overall Score: {summary['overall_score']}")
    print(f"   • Match Level: {summary['match_level']}")
    print(f"   • Skills Matched: {summary['skills_matched']}")
    print(f"   • Top Strength: {summary['top_strength']}")
    print(f"   • Top Weakness: {summary['top_weakness']}")
    
    print(f"\n✅ REQUIRED SKILLS EVIDENCE:")
    print("-" * 70)
    for skill_exp in explanation['required_skills_evidence'][:5]:  # Top 5
        emoji = "🟢" if skill_exp['strength'] == "Strong" else "🟡" if skill_exp['strength'] == "Partial" else "🔴"
        print(f"\n{emoji} {skill_exp['skill'].upper()} - {skill_exp['percentage']}")
        print(f"   Strength: {skill_exp['strength']}")
        
        if skill_exp['evidence']:
            print(f"   Evidence: \"{skill_exp['evidence']['text'][:80]}...\"")
            print(f"   Confidence: {skill_exp['evidence']['confidence']}")
        else:
            print(f"   Evidence: Not found in CV")
    
    print(f"\n❌ MISSING SKILLS:")
    print("-" * 70)
    if explanation['missing_skills']:
        for missing in explanation['missing_skills'][:3]:  # Top 3
            print(f"   • {missing['skill']}: {missing['current_score']} ({missing['priority']} priority)")
            print(f"     Gap: {missing['gap']}")
    else:
        print("   ✅ All required skills present!")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    for i, rec in enumerate(explanation['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n{'='*70}")
    print("FULL JSON OUTPUT:")
    print(f"{'='*70}")
    print(json.dumps(explanation, indent=2))
    
    print(f"\n{'='*70}")
    print("✅ EXPLAINABILITY TEST COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_explainability()