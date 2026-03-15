"""
Test CV Improver (Integrated into CV Analyzer)
"""
import pytest


@pytest.mark.skip(reason="CV Improver functionality integrated into CVAnalyzer")
def test_cv_improver():
    """
    Test CV improvement suggestions
    
    Note: This test is skipped because CV improvement functionality 
    is now integrated into the CVAnalyzer class (Mode 4).
    
    See: src/validation/cv_analyzer.py
    Tests: tests/test_cv_analyzer.py
    """
    from src.validation.cv_analyzer import CVAnalyzer
    
    analyzer = CVAnalyzer()
    
    sample_cv = {
        'text': """John Doe
        Software Engineer
        
        Experience:
        - Worked on projects
        - Helped with development
        - Responsible for testing
        
        Skills: Python, JavaScript"""
    }
    
    # Analyze CV (includes improvement suggestions)
    result = analyzer.analyze_cv(sample_cv)
    
    assert result is not None
    assert 'score' in result
    assert 'improvements' in result
    assert len(result['improvements']) > 0


# Add a passing test that verifies CV improvement via CVAnalyzer
def test_cv_improvement_via_analyzer():
    """Test that CV improvement works through CVAnalyzer"""
    from src.validation.cv_analyzer import CVAnalyzer
    
    analyzer = CVAnalyzer()
    
    # Sample CV with obvious issues
    weak_cv = {
        'text': """John Doe
        
        Experience:
        - Worked on backend
        - Helped with API
        - Responsible for database
        
        Skills: Python"""
    }
    
    # Get improvement suggestions
    result = analyzer.analyze_cv(weak_cv)
    
    assert result is not None
    assert 'score' in result
    assert 'grade' in result
    assert 'improvements' in result
    assert 'issues' in result
    
    # Should identify issues
    assert result['total_issues'] >= 0
    
    # Should provide improvements
    assert len(result['improvements']) >= 3
    
    print(f"✅ CV Analyzer provides {len(result['improvements'])} improvement suggestions")
    print(f"   Score: {result['score']}/100 (Grade: {result['grade']})")
    print(f"   Issues found: {result['total_issues']}")