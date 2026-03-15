"""
Unit Tests for Scoring Engine - Enhanced with Pytest
Integrates existing test logic with pytest framework
"""
import pytest


class TestScoringEngineBasic:
    """Basic scoring engine tests"""
    
    def test_engine_initialization(self, scoring_engine):
        """Test scoring engine initialization"""
        assert scoring_engine is not None
        assert scoring_engine.embedding_engine is not None
    
    def test_compute_match_score_returns_dict(self, scoring_engine, sample_cv_dict, sample_jd_dict):
        """Test that compute_match_score returns expected structure"""
        result = scoring_engine.compute_match_score(sample_cv_dict, sample_jd_dict)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'overall_score' in result
        assert 'breakdown' in result
    
    def test_score_in_valid_range(self, scoring_engine, sample_cv_dict, sample_jd_dict):
        """Test score is between 0 and 1"""
        result = scoring_engine.compute_match_score(sample_cv_dict, sample_jd_dict)
        
        score = result['overall_score']
        assert 0 <= score <= 1
    
    def test_breakdown_components(self, scoring_engine, sample_cv_dict, sample_jd_dict):
        """Test breakdown has all required components"""
        result = scoring_engine.compute_match_score(sample_cv_dict, sample_jd_dict)
        
        breakdown = result['breakdown']
        
        # Should have these three components
        assert 'required_skills' in breakdown
        assert 'preferred_skills' in breakdown
        assert 'experience' in breakdown


class TestScoringEngineIntegration:
    """Integration tests matching your existing test format"""
    
    def test_complete_matching_pipeline(self, scoring_engine):
        """Test complete CV-JD matching pipeline (based on your original test)"""
        
        # Your original sample data
        sample_cv_text = """
JOHN DOE
Senior Software Engineer
john.doe@email.com | +1-234-567-8900

SUMMARY
Experienced software engineer with 5 years in full-stack development.

EXPERIENCE

Senior Software Engineer | Tech Corp | 2021 - Present
- Built RESTful APIs using FastAPI and PostgreSQL
- Deployed microservices on AWS using Docker and Kubernetes
- Improved system performance by 40% through optimization
- Led team of 3 junior developers

Software Engineer | StartupXYZ | 2019 - 2021
- Developed React-based dashboard for data visualization
- Implemented CI/CD pipelines with GitHub Actions
- Worked with MongoDB and Redis for caching

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2015 - 2019

SKILLS
Python, JavaScript, React, Node.js, FastAPI, Django
PostgreSQL, MongoDB, Redis
Docker, Kubernetes, AWS, Git
"""
        
        sample_jd_text = """
Senior Software Engineer
Tech Innovations Inc.

Required Qualifications:
- 5+ years of experience in software development
- Strong proficiency in Python and JavaScript
- Experience with React and Node.js
- Must have experience with AWS cloud services
- Proficiency in SQL databases (PostgreSQL)
- Bachelor's degree in Computer Science required

Preferred Qualifications:
- Experience with Docker and Kubernetes is a plus
- Knowledge of CI/CD pipelines preferred
- Experience with FastAPI or Django is desirable
- Experience with TensorFlow or PyTorch
"""
        
        # Prepare CV data
        cv_data = {
            'text': sample_cv_text,
            'sections': {
                'skills': ['Python', 'JavaScript', 'React', 'Node.js', 'FastAPI', 
                          'Django', 'PostgreSQL', 'MongoDB', 'Redis', 'Docker', 
                          'Kubernetes', 'AWS', 'Git'],
                'experience': '5 years'
            }
        }
        
        # Prepare JD data
        jd_data = {
            'text': sample_jd_text,
            'sections': {
                'required_skills': ['Python', 'JavaScript', 'React', 'Node.js', 
                                   'AWS', 'PostgreSQL', 'Computer Science'],
                'preferred_skills': ['Docker', 'Kubernetes', 'CI/CD', 'FastAPI', 
                                    'Django', 'TensorFlow', 'PyTorch'],
                'experience': {
                    'years': '5',
                    'min_years': 5
                }
            }
        }
        
        # Compute match
        result = scoring_engine.compute_match_score(cv_data, jd_data)
        
        # Display results (matching your original format)
        print("\n" + "=" * 70)
        print("SCORING ENGINE TEST - COMPLETE CV-JD MATCHING")
        print("=" * 70)
        
        print(f"\n🎯 OVERALL MATCH SCORE: {result.get('overall_percentage', 'N/A')}")
        print(f"   Score: {result['overall_score']:.4f}/1.0000")
        
        # Check interpretation if available
        if 'interpretation' in result:
            interp = result['interpretation']
            print(f"   Level: {interp.get('level', 'N/A')}")
            print(f"   Recommendation: {interp.get('recommendation', 'N/A')}")
        
        print("\n📊 SCORE BREAKDOWN:")
        print("-" * 70)
        
        breakdown = result['breakdown']
        
        for component, data in breakdown.items():
            print(f"\n{component.upper().replace('_', ' ')}:")
            
            if 'percentage' in data:
                print(f"   Score: {data['percentage']}")
            elif 'score' in data:
                print(f"   Score: {data['score']:.4f}")
            
            if 'weight' in data:
                print(f"   Weight: {data['weight']}")
            
            # Show details if available
            if 'details' in data:
                details = data['details']
                
                if 'matched_skills' in details:
                    matched = details['matched_skills']
                    print(f"   Matched skills: {len(matched)}")
                    if matched:
                        print(f"   Sample matched: {matched[:3]}")
                
                if 'missing_skills' in details:
                    missing = details['missing_skills']
                    print(f"   Missing skills: {len(missing)}")
                    if missing:
                        print(f"   Sample missing: {missing[:3]}")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Assertions
        assert result['overall_score'] > 0.5  # Should have reasonable match
        assert 'breakdown' in result
        assert len(breakdown) == 3  # All three components


class TestScoringEngineWeights:
    """Test scoring weights (60/25/15)"""
    
    def test_weight_distribution(self, scoring_engine, sample_cv_dict, sample_jd_dict):
        """Test that weights are 60/25/15"""
        result = scoring_engine.compute_match_score(sample_cv_dict, sample_jd_dict)
        
        breakdown = result['breakdown']
        
        # Check weights (if engine provides them)
        if 'weight' in breakdown.get('required_skills', {}):
            req_weight = breakdown['required_skills']['weight']
            pref_weight = breakdown['preferred_skills']['weight']
            exp_weight = breakdown['experience']['weight']
            
            # Should be 60%, 25%, 15%
            assert abs(req_weight - 0.60) < 0.01
            assert abs(pref_weight - 0.25) < 0.01
            assert abs(exp_weight - 0.15) < 0.01


class TestScoringEngineEdgeCases:
    """Edge cases and robustness tests"""
    
    def test_empty_cv(self, scoring_engine, sample_jd_dict):
        """Test with empty CV"""
        empty_cv = {'text': '', 'sections': {}}
        
        result = scoring_engine.compute_match_score(empty_cv, sample_jd_dict)
        
        assert result is not None
        assert result['overall_score'] >= 0
    
    def test_empty_jd(self, scoring_engine, sample_cv_dict):
        """Test with empty JD"""
        empty_jd = {'text': '', 'sections': {}}
        
        result = scoring_engine.compute_match_score(sample_cv_dict, empty_jd)
        
        assert result is not None
        assert result['overall_score'] >= 0
    
    def test_perfect_match(self, scoring_engine):
        """Test perfect match scenario"""
        perfect_data = {
            'text': 'Python FastAPI Docker AWS',
            'sections': {
                'skills': ['Python', 'FastAPI', 'Docker', 'AWS'],
                'experience': '5 years'
            }
        }
        
        perfect_jd = {
            'text': 'Python FastAPI Docker AWS',
            'sections': {
                'required_skills': ['Python', 'FastAPI', 'Docker', 'AWS'],
                'preferred_skills': [],
                'experience': {'years': '5', 'min_years': 5}
            }
        }
        
        result = scoring_engine.compute_match_score(perfect_data, perfect_jd)
        
        # Should have very high score
        assert result['overall_score'] >= 0.75
    
    def test_no_match(self, scoring_engine):
        """Test completely unmatched CV-JD"""
        unmatched_cv = {
            'text': 'Cooking Chef Restaurant',
            'sections': {
                'skills': ['Cooking', 'Food Preparation'],
                'experience': '2 years'
            }
        }
        
        tech_jd = {
            'text': 'Python Docker Kubernetes',
            'sections': {
                'required_skills': ['Python', 'Docker', 'Kubernetes'],
                'preferred_skills': [],
                'experience': {'years': '5', 'min_years': 5}
            }
        }
        
        result = scoring_engine.compute_match_score(unmatched_cv, tech_jd)
        
        # Should have low score
        assert result['overall_score'] < 0.50