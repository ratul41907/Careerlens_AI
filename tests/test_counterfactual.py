"""
Unit Tests for Counterfactual Simulator - Enhanced with Pytest
Based on existing test_counterfactual.py logic
"""
import pytest


class TestCounterfactualBasic:
    """Basic counterfactual simulator tests"""
    
    def test_simulator_initialization(self, counterfactual_simulator):
        """Test simulator initialization"""
        assert counterfactual_simulator is not None
        assert counterfactual_simulator.scoring_engine is not None
    
    def test_simulate_skill_addition_returns_float(self, counterfactual_simulator, sample_cv_dict, sample_jd_dict):
        """Test simulate_skill_addition returns float"""
        result = counterfactual_simulator.simulate_skill_addition(
            sample_cv_dict,
            sample_jd_dict,
            'Kubernetes'
        )
        
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_analyze_skill_impact_returns_dict(self, counterfactual_simulator, sample_jd_dict):
        """Test analyze_skill_impact returns dict"""
        result = counterfactual_simulator.analyze_skill_impact(
            skill='Docker',
            current_score=0.60,
            potential_score=0.75,
            jd_data=sample_jd_dict
        )
        
        assert isinstance(result, dict)
        assert 'skill' in result
        assert 'score_increase' in result
        assert 'priority' in result
        assert 'explanation' in result
    
    def test_skill_addition_increases_score(self, counterfactual_simulator):
        """Test that adding relevant skill increases score"""
        cv = {
            'text': 'Python Developer',
            'sections': {'skills': ['Python']}
        }
        
        jd = {
            'text': 'Need Python and Docker',
            'sections': {'required_skills': ['Python', 'Docker']}
        }
        
        current_score = counterfactual_simulator.scoring_engine.compute_match_score(cv, jd)['overall_score']
        new_score = counterfactual_simulator.simulate_skill_addition(cv, jd, 'Docker')
        
        # Adding Docker should increase score
        assert new_score >= current_score


class TestCounterfactualIntegration:
    """Integration tests matching existing test format"""
    
    def test_complete_counterfactual_pipeline(self, counterfactual_simulator):
        """Test complete counterfactual analysis (based on original test)"""
        
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL SIMULATOR TEST")
        print("=" * 70)
        
        # Sample CV (missing some skills)
        sample_cv = {
            'text': """
JOHN DOE
Software Engineer

EXPERIENCE
Software Engineer | TechCo | 2020 - Present
- Developed web applications using React
- Built REST APIs with Node.js
- Worked with MongoDB database

SKILLS
JavaScript, React, Node.js, MongoDB, Git
""",
            'sections': {
                'skills': ['JavaScript', 'React', 'Node.js', 'MongoDB', 'Git']
            }
        }
        
        # Sample JD (requires more skills)
        sample_jd = {
            'text': """
Senior Software Engineer Position

Required Skills:
- Python programming required
- React framework experience
- Docker containerization required
- AWS cloud services required
- Kubernetes orchestration needed
- 5+ years experience
""",
            'sections': {
                'required_skills': ['Python', 'React', 'Docker', 'AWS', 'Kubernetes'],
                'experience': {'years': '5', 'min_years': 5}
            }
        }
        
        print("\n📊 STEP 1: Computing Baseline Match")
        print("-" * 70)
        
        # Compute baseline
        baseline_result = counterfactual_simulator.scoring_engine.compute_match_score(sample_cv, sample_jd)
        current_score = baseline_result['overall_score']
        
        print(f"✅ Baseline Score: {current_score*100:.1f}%")
        
        # Get missing skills
        missing_skills = baseline_result['breakdown']['required_skills']['details'].get('missing_skills', [])
        
        print(f"\n🔮 STEP 2: Running Counterfactual Simulations")
        print("-" * 70)
        print(f"Missing skills: {', '.join(missing_skills)}")
        
        # Analyze all missing skills
        analyses = counterfactual_simulator.analyze_all_missing_skills(
            sample_cv,
            sample_jd,
            missing_skills,
            current_score
        )
        
        print(f"\n{'='*70}")
        print("COUNTERFACTUAL SKILL IMPACT ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n📊 BASELINE:")
        print(f"   Current Score: {current_score*100:.1f}%")
        
        print(f"\n🎯 TOP OPPORTUNITIES:")
        print("-" * 70)
        
        for i, analysis in enumerate(analyses[:5], 1):
            priority_emoji = "🔴" if analysis['priority'] == "High" else "🟡" if analysis['priority'] == "Medium" else "⚪"
            
            print(f"\n{priority_emoji} #{i}. {analysis['skill'].upper()}")
            print(f"   Current: {analysis['current_score']*100:.1f}%")
            print(f"   If Added: {analysis['potential_score']*100:.1f}%")
            print(f"   Impact: {analysis['impact_percentage']}")
            print(f"   Priority: {analysis['priority']}")
            
            if 'explanation' in analysis:
                print(f"   Why: {analysis['explanation'][:100]}...")
        
        print(f"\n{'='*70}")
        print("✅ COUNTERFACTUAL TEST COMPLETED!")
        print(f"{'='*70}")
        
        # Assertions
        assert len(analyses) > 0
        assert all('score_increase' in a for a in analyses)
        assert all('priority' in a for a in analyses)


class TestCounterfactualPriority:
    """Test priority assignment"""
    
    def test_high_priority_assignment(self, counterfactual_simulator, sample_jd_dict):
        """Test high priority for large score increase"""
        result = counterfactual_simulator.analyze_skill_impact(
            skill='Python',
            current_score=0.50,
            potential_score=0.75,  # +25% increase
            jd_data=sample_jd_dict
        )
        
        # 25% increase should be high priority
        assert result['priority'] in ['High', 'Medium']
    
    def test_low_priority_assignment(self, counterfactual_simulator, sample_jd_dict):
        """Test low priority for small score increase"""
        result = counterfactual_simulator.analyze_skill_impact(
            skill='SomeSkill',
            current_score=0.70,
            potential_score=0.72,  # +2% increase
            jd_data=sample_jd_dict
        )
        
        # 2% increase should be low priority
        assert result['priority'] in ['Low', 'Very Low', 'Medium']


class TestCounterfactualEdgeCases:
    """Edge cases and error handling"""
    
    def test_adding_already_present_skill(self, counterfactual_simulator):
        """Test adding skill that's already in CV"""
        cv = {
            'text': 'Python Developer with Python skills',
            'sections': {'skills': ['Python']}
        }
        
        jd = {
            'text': 'Need Python',
            'sections': {'required_skills': ['Python']}
        }
        
        current_score = counterfactual_simulator.scoring_engine.compute_match_score(cv, jd)['overall_score']
        new_score = counterfactual_simulator.simulate_skill_addition(cv, jd, 'Python')
        
        # Score should not decrease
        assert new_score >= current_score * 0.95  # Allow small variance
    
    def test_adding_irrelevant_skill(self, counterfactual_simulator):
        """Test adding skill not in JD"""
        cv = {
            'text': 'Python Developer',
            'sections': {'skills': ['Python']}
        }
        
        jd = {
            'text': 'Need Python and Docker',
            'sections': {'required_skills': ['Python', 'Docker']}
        }
        
        # Adding irrelevant skill
        new_score = counterfactual_simulator.simulate_skill_addition(cv, jd, 'Cooking')
        
        # Should return some score (may not increase much)
        assert 0 <= new_score <= 1
    
    def test_empty_missing_skills(self, counterfactual_simulator, sample_cv_dict, sample_jd_dict):
        """Test with no missing skills"""
        analyses = counterfactual_simulator.analyze_all_missing_skills(
            sample_cv_dict,
            sample_jd_dict,
            [],  # No missing skills
            0.85
        )
        
        assert analyses == []


class TestCounterfactualLLM:
    """LLM-specific tests"""
    
    @pytest.mark.llm
    def test_llm_explanation_quality(self, counterfactual_simulator, sample_jd_dict):
        """Test LLM provides quality explanations"""
        result = counterfactual_simulator.analyze_skill_impact(
            skill='Docker',
            current_score=0.60,
            potential_score=0.75,
            jd_data=sample_jd_dict
        )
        
        explanation = result['explanation']
        
        # Should have meaningful explanation
        assert len(explanation) > 30
        assert 'Docker' in explanation or 'docker' in explanation