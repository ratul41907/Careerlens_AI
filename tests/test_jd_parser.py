"""
Test Job Description Parser
"""
from src.parsers.jd_parser import JDParser
import json


def test_jd_parser():
    """Test JD parser with sample job description"""
    
    sample_jd = """
    Senior Software Engineer
    Tech Innovations Inc.
    San Francisco, CA
    
    About the Role:
    We are seeking a talented Senior Software Engineer to join our growing team.
    
    Required Qualifications:
    • 5+ years of experience in software development
    • Strong proficiency in Python and JavaScript
    • Experience with React and Node.js
    • Must have experience with AWS cloud services
    • Proficiency in SQL and NoSQL databases (PostgreSQL, MongoDB)
    • Bachelor's degree in Computer Science or related field required
    
    Preferred Qualifications:
    • Experience with Docker and Kubernetes is a plus
    • Knowledge of CI/CD pipelines (Jenkins, GitHub Actions) preferred
    • Familiarity with microservices architecture would be great
    • Experience with FastAPI or Django is desirable
    
    Responsibilities:
    • Design and develop scalable backend systems
    • Collaborate with cross-functional teams using Agile methodology
    • Write clean, maintainable code with proper documentation
    • Participate in code reviews and mentor junior developers
    
    Nice to Have:
    • Experience with machine learning frameworks (TensorFlow, PyTorch)
    • Contributions to open-source projects
    """
    
    # Parse the JD
    parser = JDParser()
    result = parser.parse(sample_jd)
    
    # Display results
    print("=" * 60)
    print("JOB DESCRIPTION PARSER TEST RESULTS")
    print("=" * 60)
    print(f"\n✅ Parse Success: {result['success']}")
    
    if result['success']:
        print(f"\n📋 Job Title: {result.get('job_title', 'Not detected')}")
        print(f"🏢 Company: {result.get('company', 'Not detected')}")
        print(f"📍 Location: {result.get('location', 'Not detected')}")
        
        print(f"\n🎯 Required Skills ({len(result['required_skills'])}):")
        for skill in sorted(result['required_skills'])[:10]:  # Show first 10
            print(f"  • {skill}")
        
        print(f"\n⭐ Preferred Skills ({len(result['preferred_skills'])}):")
        for skill in sorted(result['preferred_skills'])[:10]:
            print(f"  • {skill}")
        
        if result.get('experience_years'):
            exp = result['experience_years']
            print(f"\n💼 Experience Required:")
            print(f"  • Minimum: {exp['min_years']} years")
            if exp.get('max_years'):
                print(f"  • Maximum: {exp['max_years']} years")
            print(f"  • Required: {exp.get('required', False)}")
        
        if result.get('education'):
            edu = result['education']
            print(f"\n🎓 Education:")
            print(f"  • Level: {edu.get('degree_level', 'Not specified')}")
            if edu.get('field'):
                print(f"  • Field: {edu['field']}")
            print(f"  • Required: {edu.get('required', False)}")
        
        print("\n" + "=" * 60)
        print("JSON OUTPUT:")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    print("\n✅ Test completed successfully!")
    return result


if __name__ == "__main__":
    test_jd_parser()