"""
Test FastAPI Endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_api():
    """Test all API endpoints"""
    
    print("=" * 70)
    print("FASTAPI ENDPOINTS TEST")
    print("=" * 70)
    
    # Test 1: Health check
    print("\n✅ TEST 1: Health Check")
    print("-" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Parse CV
    print("\n✅ TEST 2: Parse CV")
    print("-" * 70)
    
    cv_text = """
    JOHN DOE
    Senior Software Engineer
    
    EXPERIENCE
    • Built REST APIs with FastAPI
    • Deployed on AWS using Docker
    
    SKILLS
    Python, FastAPI, Docker, AWS
    """
    
    response = requests.post(
        f"{BASE_URL}/api/parse-cv",
        json={"text": cv_text}
    )
    
    print(f"Status: {response.status_code}")
    cv_data = response.json()
    print(f"Sections found: {list(cv_data.get('sections', {}).keys())}")
    
    # Test 3: Parse JD
    print("\n✅ TEST 3: Parse Job Description")
    print("-" * 70)
    
    jd_text = """
    Senior Software Engineer
    
    Required:
    • 5+ years experience
    • Python and FastAPI required
    • AWS cloud experience needed
    """
    
    response = requests.post(
        f"{BASE_URL}/api/parse-jd",
        json={"text": jd_text}
    )
    
    print(f"Status: {response.status_code}")
    jd_data = response.json()
    print(f"Required skills: {jd_data.get('required_skills', [])}")
    print(f"Experience: {jd_data.get('experience_years')}")
    
    # Test 4: Complete Match
    print("\n✅ TEST 4: Complete CV-JD Match")
    print("-" * 70)
    
    response = requests.post(
        f"{BASE_URL}/api/match",
        json={
            "cv_text": cv_text,
            "jd_text": jd_text,
            "include_explainability": True
        }
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        match = result['match_result']
        print(f"\n🎯 MATCH SCORE: {match['overall_percentage']}")
        print(f"   Interpretation: {match['interpretation']['level']}")
        print(f"   Recommendation: {match['interpretation']['recommendation']}")
        
        if 'explainability' in match:
            exp = match['explainability']
            print(f"\n📊 SUMMARY:")
            print(f"   Skills matched: {exp['summary']['skills_matched']}")
            print(f"   Top strength: {exp['summary']['top_strength']}")
            print(f"   Top weakness: {exp['summary']['top_weakness']}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in exp['recommendations'][:3]:
                print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("✅ ALL API TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n⚠️  MAKE SURE API IS RUNNING!")
    print("   Run in another terminal: python -m uvicorn src.api.main:app --reload")
    print("\n   Press Enter when ready...")
    input()
    
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Make sure the API is running on http://localhost:8000")