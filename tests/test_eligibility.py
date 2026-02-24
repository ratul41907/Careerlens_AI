"""
Test Eligibility Validator
"""
from src.validation.eligibility_validator import EligibilityValidator
import json


def test_eligibility_validator():
    """Test eligibility validation with sample transcript"""
    
    print("=" * 70)
    print("ELIGIBILITY VALIDATOR TEST")
    print("=" * 70)
    
    print("\n⚠️  NOTE: This test requires:")
    print("   1. Tesseract OCR installed")
    print("   2. Sample transcript image")
    print("\n   We'll create a text-based mock for now")
    print("-" * 70)
    
    # Mock transcript text (simulating OCR output)
    mock_transcript = """
    UNIVERSITY OF TECHNOLOGY
    Official Transcript
    
    Student Name: John Doe
    Student ID: 12345678
    
    Degree: Bachelor of Science in Computer Science
    
    Graduation Date: May 2019
    
    Cumulative GPA: 3.8/4.0
    
    Major: Computer Science
    
    Courses Completed: 120 Credits
    """
    
    # Job requirements
    job_requirements = {
        'degree_level': 'bachelor',
        'min_gpa': 3.0,
        'major': 'Computer Science'
    }
    
    print("\n📝 MOCK TRANSCRIPT:")
    print(mock_transcript)
    
    print("\n📋 JOB REQUIREMENTS:")
    print(f"   Degree: {job_requirements['degree_level']}")
    print(f"   Min GPA: {job_requirements['min_gpa']}/4.0")
    print(f"   Major: {job_requirements['major']}")
    
    # Initialize validator
    validator = EligibilityValidator()
    
    # Parse mock transcript
    print("\n🔍 PARSING TRANSCRIPT...")
    transcript_data = validator._parse_transcript(mock_transcript)
    
    print(f"\n✅ EXTRACTED DATA:")
    print(f"   Degree: {transcript_data['degree']}")
    print(f"   GPA: {transcript_data['gpa']}/{transcript_data['gpa_scale']}")
    print(f"   Institution: {transcript_data['institution']}")
    print(f"   Graduation Year: {transcript_data['graduation_year']}")
    print(f"   Major: {transcript_data['major']}")
    
    # Make decision
    print("\n⚖️  MAKING ELIGIBILITY DECISION...")
    decision = validator._make_decision(transcript_data, job_requirements)
    
    print(f"\n{'='*70}")
    print("ELIGIBILITY DECISION")
    print(f"{'='*70}")
    
    decision_emoji = "✅" if decision['decision'] == 'PASS' else "⚠️" if decision['decision'] == 'REVIEW' else "❌"
    print(f"\n{decision_emoji} DECISION: {decision['decision']}")
    print(f"   Reason: {decision['reason']}")
    print(f"   Confidence: {decision['confidence']}%")
    
    if decision['details']['failures']:
        print(f"\n❌ FAILURES:")
        for failure in decision['details']['failures']:
            print(f"   • {failure}")
    
    if decision['details']['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in decision['details']['warnings']:
            print(f"   • {warning}")
    
    print(f"\n{'='*70}")
    print("FULL JSON OUTPUT:")
    print(f"{'='*70}")
    print(json.dumps(decision, indent=2))
    
    print(f"\n{'='*70}")
    print("✅ TEST COMPLETED!")
    print(f"{'='*70}")
    
    print("\n💡 To test with real images:")
    print("   1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   2. Place transcript image in data/sample_transcripts/")
    print("   3. Run: validator.validate_transcript('path/to/image.png')")


if __name__ == "__main__":
    test_eligibility_validator()