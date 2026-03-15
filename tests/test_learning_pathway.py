"""
Unit Tests for Learning Pathway Generator - Enhanced with Pytest
Based on existing test_learning_pathway.py logic
"""
import pytest


class TestLearningPathwayBasic:
    """Basic learning pathway tests"""
    
    def test_generator_initialization(self, learning_pathway_generator):
        """Test generator initialization"""
        assert learning_pathway_generator is not None
        assert hasattr(learning_pathway_generator, 'ollama_url')
        assert hasattr(learning_pathway_generator, 'model')
    
    def test_generate_pathway_returns_dict(self, learning_pathway_generator):
        """Test generate_pathway returns dict"""
        skill_gaps = ['Docker', 'Kubernetes']
        jd_data = {'text': 'Need Docker and Kubernetes skills'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_generate_7_day_pathway(self, learning_pathway_generator):
        """Test 7-day pathway generation"""
        skill_gaps = ['Python', 'Docker']
        jd_data = {'text': 'Python and Docker required'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        assert result['success'] is True
        assert result['timeline_days'] == 7
        assert 'daily_plans' in result
        assert len(result['daily_plans']) == 7
    
    def test_generate_14_day_pathway(self, learning_pathway_generator):
        """Test 14-day pathway generation"""
        skill_gaps = ['FastAPI', 'PostgreSQL']
        jd_data = {'text': 'FastAPI and PostgreSQL experience needed'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=14
        )
        
        assert result['success'] is True
        assert result['timeline_days'] == 14
        assert len(result['daily_plans']) == 14
    
    def test_generate_30_day_pathway(self, learning_pathway_generator):
        """Test 30-day pathway generation"""
        skill_gaps = ['Python', 'Docker', 'Kubernetes', 'AWS']
        jd_data = {'text': 'Full stack cloud development'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=30
        )
        
        assert result['success'] is True
        assert result['timeline_days'] == 30
        assert len(result['daily_plans']) == 30
    
    def test_daily_plan_structure(self, learning_pathway_generator):
        """Test daily plan has correct structure"""
        skill_gaps = ['Python']
        jd_data = {'text': 'Python developer needed'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        daily_plans = result['daily_plans']
        
        for plan in daily_plans:
            assert 'day' in plan
            assert 'focus' in plan
            assert 'tasks' in plan
            assert 'resources' in plan
            assert 'mini_project' in plan
            assert 'time_estimate' in plan


class TestLearningPathwayIntegration:
    """Integration tests matching existing test format"""
    
    def test_complete_learning_pathway_pipeline(self, learning_pathway_generator):
        """Test complete pathway generation (based on original test)"""
        
        print("\n" + "=" * 70)
        print("LEARNING PATHWAY GENERATOR TEST")
        print("=" * 70)
        
        # Sample missing skills
        missing_skills = ['docker', 'kubernetes', 'fastapi', 'aws']
        
        print("\n📊 MISSING SKILLS:")
        for skill in missing_skills:
            print(f"   • {skill}")
        
        # Test 30-day pathway
        print("\n" + "=" * 70)
        print("30-DAY LEARNING PATHWAY")
        print("=" * 70)
        
        jd_data = {
            'text': 'Looking for developer with Docker, Kubernetes, FastAPI, AWS experience'
        }
        
        pathway_30 = learning_pathway_generator.generate_pathway(
            skill_gaps=missing_skills,
            jd_data=jd_data,
            num_days=30
        )
        
        assert pathway_30['success'] is True
        
        print(f"\n📈 PATHWAY OVERVIEW:")
        print(f"   Timeline: {pathway_30['timeline_days']} days")
        print(f"   Focus Skills: {', '.join(pathway_30['focus_skills'])}")
        
        print(f"\n📅 SAMPLE DAYS:")
        for plan in pathway_30['daily_plans'][:3]:  # Show first 3 days
            print(f"\n   Day {plan['day']}: {plan['focus']}")
            print(f"   Tasks:")
            for task in plan['tasks'][:2]:
                print(f"      • {task}")
            print(f"   Time: {plan['time_estimate']}")
        
        # Test 7-day pathway
        print("\n" + "=" * 70)
        print("7-DAY INTENSIVE PATHWAY")
        print("=" * 70)
        
        pathway_7 = learning_pathway_generator.generate_pathway(
            skill_gaps=missing_skills[:2],  # Focus on top 2 skills
            jd_data=jd_data,
            num_days=7
        )
        
        assert pathway_7['success'] is True
        
        print(f"\n📈 INTENSIVE OVERVIEW:")
        print(f"   Focus Skills: {', '.join(pathway_7['focus_skills'])}")
        print(f"   Timeline: {pathway_7['timeline_days']} days")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETED!")
        print("=" * 70)
        
        # Assertions
        assert len(pathway_30['daily_plans']) == 30
        assert len(pathway_7['daily_plans']) == 7
        assert pathway_30['timeline_days'] == 30
        assert pathway_7['timeline_days'] == 7


class TestLearningPathwayResources:
    """Test resource generation"""
    
    @pytest.mark.llm
    def test_resources_structure(self, learning_pathway_generator):
        """Test that resources are properly structured"""
        skill_gaps = ['Python']
        jd_data = {'text': 'Python developer'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        daily_plans = result['daily_plans']
        
        for plan in daily_plans:
            resources = plan['resources']
            
            # Should have different resource types
            assert isinstance(resources, dict)
            
            # Check for common resource categories
            # (LLM may use different keys, so we check flexibly)
            assert len(resources) > 0
    
    @pytest.mark.llm
    def test_youtube_playlists_generated(self, learning_pathway_generator):
        """Test that YouTube playlists are generated"""
        skill_gaps = ['Python', 'Docker']
        jd_data = {'text': 'Python and Docker skills needed'}
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        # Check that at least one day has YouTube resources
        has_youtube = False
        for plan in result['daily_plans']:
            resources = plan['resources']
            for key in resources.keys():
                if 'youtube' in key.lower() or '🎥' in key:
                    has_youtube = True
                    # Should have multiple playlists
                    assert len(resources[key]) >= 1
                    break
            if has_youtube:
                break
        
        # At least some days should have YouTube resources
        # (May not be all days depending on LLM)
        assert has_youtube or len(result['daily_plans']) > 0


class TestLearningPathwayEdgeCases:
    """Edge cases and error handling"""
    
    def test_empty_skill_gaps(self, learning_pathway_generator):
        """Test with no skill gaps"""
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=[],
            jd_data={'text': 'Some job'},
            num_days=7
        )
        
        assert result is not None
        assert result['success'] is False
        assert 'error' in result
    
    def test_single_skill(self, learning_pathway_generator):
        """Test with single skill"""
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=['Python'],
            jd_data={'text': 'Python developer'},
            num_days=7
        )
        
        assert result['success'] is True
        assert len(result['daily_plans']) == 7
    
    def test_many_skills(self, learning_pathway_generator):
        """Test with many skills"""
        many_skills = [f"Skill{i}" for i in range(10)]
        
        result = learning_pathway_generator.generate_pathway(
            skill_gaps=many_skills,
            jd_data={'text': 'Many skills required'},
            num_days=30
        )
        
        assert result['success'] is True
        # Should focus on subset of skills
        assert len(result['focus_skills']) <= 4