"""
Test Scoring Engine
"""
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
import json


def test_scoring_engine():
    """Test complete CV-JD matching pipeline"""
    
    print("=" * 70)
    print("SCORING ENGINE TEST - COMPLETE CV-JD MATCHING")
    print("=" * 70)
    
    # Sample CV
    sample_cv_text = """
    JOHN DOE
    Senior Software Engineer
    john.doe@email.com | +1-234-567-8900
    
    SUMMARY
    Experienced software engineer with 5 years in full-stack development.
    
    EXPERIENCE
    
    Senior Software Engineer | Tech Corp | 2021 - Present
    • Built RESTful APIs using FastAPI and PostgreSQL
    • Deployed microservices on AWS using Docker and Kubernetes
    • Improved system performance by 40% through optimization
    • Led team of 3 junior developers
    
    Software Engineer | StartupXYZ | 2019 - 2021
    • Developed React-based dashboard for data visualization
    • Implemented CI/CD pipelines with GitHub Actions
    • Worked with MongoDB and Redis for caching
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2015 - 2019
    
    SKILLS
    Python, JavaScript, React, Node.js, FastAPI, Django
    PostgreSQL, MongoDB, Redis
    Docker, Kubernetes, AWS, Git
    """
    
    # Sample JD
    sample_jd_text = """
    Senior Software Engineer
    Tech Innovations Inc.
    
    Required Qualifications:
    • 5+ years of experience in software development
    • Strong proficiency in Python and JavaScript
    • Experience with React and Node.js
    • Must have experience with AWS cloud services
    • Proficiency in SQL databases (PostgreSQL)
    • Bachelor's degree in Computer Science required
    
    Preferred Qualifications:
    • Experience with Docker and Kubernetes is a plus
    • Knowledge of CI/CD pipelines preferred
    • Experience with FastAPI or Django is desirable
    • Experience with TensorFlow or PyTorch
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: PARSING CV")
    print("=" * 70)
    
    cv_parser = CVParser()
    cv_data = {
        'text': sample_cv_text,
        'sections': cv_parser._segment_sections(sample_cv_text)
    }
    
    print(f"✅ CV Parsed")
    print(f"   Sections: {list(cv_data['sections'].keys())}")
    
    print("\n" + "=" * 70)
    print("STEP 2: PARSING JOB DESCRIPTION")
    print("=" * 70)
    
    jd_parser = JDParser()
    jd_data = jd_parser.parse(sample_jd_text)
    
    print(f"✅ JD Parsed")
    print(f"   Required Skills: {len(jd_data['required_skills'])}")
    print(f"   Preferred Skills: {len(jd_data['preferred_skills'])}")
    print(f"   Experience Req: {jd_data['experience_years']}")
    
    print("\n" + "=" * 70)
    print("STEP 3: COMPUTING MATCH SCORE")
    print("=" * 70)
    
    # Initialize engines
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    
    # Compute match
    result = scoring_engine.compute_match_score(cv_data, jd_data)
    
    # Display results
    print(f"\n🎯 OVERALL MATCH SCORE: {result['overall_percentage']}")
    print(f"   Score: {result['overall_score']:.4f}/1.0000")
    print(f"   Interpretation: {result['interpretation']['level']}")
    print(f"   Recommendation: {result['interpretation']['recommendation']}")
    
    print("\n📊 SCORE BREAKDOWN:")
    print("-" * 70)
    
    for component, data in result['breakdown'].items():
        print(f"\n{component.upper().replace('_', ' ')}:")
        print(f"   Score: {data['percentage']} (weight: {data['weight']})")
        print(f"   Contribution to overall: {data['contribution']}")
        
        details = data['details']
        if 'matched' in details and 'match_rate' in details:
            print(f"   Skills matched: {details['match_rate']}")
            
            # Show top 5 skills
            print(f"\n   Top Skills:")
            skills = sorted(details['skills'], key=lambda x: x['score'], reverse=True)[:5]
            for skill_info in skills:
                strength_emoji = "🟢" if skill_info['strength'] == "Strong" else "🟡" if skill_info['strength'] == "Partial" else "🔴"
                print(f"      {strength_emoji} {skill_info['skill']}: {skill_info['percentage']} ({skill_info['strength']})")
    
    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT:")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    test_scoring_engine()