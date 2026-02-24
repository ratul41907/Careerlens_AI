"""
Test CV Improver
"""
from src.generation.cv_improver import CVImprover
import json


def test_cv_improver():
    """Test CV improvement analysis"""
    
    print("=" * 70)
    print("CV IMPROVEMENT ANALYZER TEST")
    print("=" * 70)
    
    # Weak CV (has issues)
    weak_cv = """
    JOHN DOE
    Software Developer
    john@email.com | 555-1234
    
    EXPERIENCE
    
    Software Developer | TechCo | 2021 - Present
    • Worked on backend development
    • Helped with API projects
    • Used Python for various tasks
    • Responsible for bug fixes
    
    Junior Developer | StartupXYZ | 2019 - 2021
    • Assisted in frontend development
    • Participated in team meetings
    • Contributed to codebase
    
    EDUCATION
    Bachelor's in Computer Science | University | 2019
    
    SKILLS
    Python, JavaScript, Git
    """
    
    # Target JD
    target_jd = """
    Senior Software Engineer
    
    Required:
    • 5+ years Python development
    • FastAPI framework expertise
    • Docker and Kubernetes required
    • AWS cloud services
    • Strong problem-solving skills
    
    Preferred:
    • Experience with microservices
    • CI/CD pipeline knowledge
    """
    
    print("\n📊 ANALYZING CV...")
    print("-" * 70)
    
    # Analyze
    improver = CVImprover()
    result = improver.analyze_and_improve(weak_cv, target_jd, num_suggestions=5)
    
    # Display results
    print(f"\n{'='*70}")
    print("CV IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📈 CURRENT STATUS:")
    print(f"   Match Score: {result['current_percentage']}")
    print(f"   Issues Found: {result['issues_found']}")
    print(f"   Potential Gain: {result['estimated_improvement']}")
    
    print(f"\n🔴 ISSUES IDENTIFIED:")
    print("-" * 70)
    for i, issue in enumerate(result['issues'], 1):
        severity_emoji = "🔴" if issue['severity'] == 'High' else "🟡" if issue['severity'] == 'Medium' else "🟢"
        print(f"\n{severity_emoji} Issue #{i}: {issue['title']} ({issue['severity']} Priority)")
        print(f"   Description: {issue['description']}")
        print(f"   Impact: {issue['impact']}")
    
    print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
    print("-" * 70)
    for sugg in result['suggestions']:
        print(f"\n🎯 Priority #{sugg['priority']}: {sugg['title']}")
        print(f"   {sugg['description']}")
        print(f"   Action: {sugg['action']}")
        
        if 'before' in sugg:
            print(f"\n   ❌ Before: {sugg['before']}")
            print(f"   ✅ After:  {sugg['after']}")
        elif 'example' in sugg:
            print(f"\n   Example: {sugg['example']}")
        
        print(f"   Impact: {sugg['estimated_impact']}")
    
    print(f"\n🎯 PRIORITY ACTIONS:")
    print("-" * 70)
    for action in result['priority_actions']:
        print(f"   • {action}")
    
    print(f"\n{'='*70}")
    print("FULL JSON OUTPUT:")
    print(f"{'='*70}")
    print(json.dumps(result, indent=2, default=str))
    
    print(f"\n{'='*70}")
    print("✅ TEST COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_cv_improver()