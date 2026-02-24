"""
Test Learning Pathway Generator
"""
from src.guidance.learning_pathway import LearningPathwayGenerator
import json


def test_learning_pathway():
    """Test learning pathway generation"""
    
    print("=" * 70)
    print("LEARNING PATHWAY GENERATOR TEST")
    print("=" * 70)
    
    # Sample missing skills (from counterfactual simulator)
    missing_skills = [
        {'skill': 'docker', 'current_score': 0.57, 'gap': '23% below strong match', 'priority': 'High'},
        {'skill': 'kubernetes', 'current_score': 0.56, 'gap': '24% below strong match', 'priority': 'High'},
        {'skill': 'fastapi', 'current_score': 0.60, 'gap': '20% below strong match', 'priority': 'Medium'},
        {'skill': 'aws', 'current_score': 0.59, 'gap': '21% below strong match', 'priority': 'Medium'}
    ]
    
    print("\n📊 MISSING SKILLS:")
    for skill in missing_skills:
        print(f"   • {skill['skill']}: {skill['current_score']*100:.0f}% ({skill['priority']} priority)")
    
    # Test 30-day pathway
    print("\n" + "=" * 70)
    print("30-DAY LEARNING PATHWAY")
    print("=" * 70)
    
    generator = LearningPathwayGenerator()
    pathway_30 = generator.generate_pathway(missing_skills, timeline_days=30)
    
    print(f"\n📈 PATHWAY OVERVIEW:")
    print(f"   Timeline: {pathway_30['timeline_days']} days")
    print(f"   Skills to Learn: {pathway_30['total_skills']}")
    print(f"   Estimated Improvement: {pathway_30['estimated_improvement']}")
    
    print(f"\n🎯 PRIORITIZED SKILLS:")
    for i, skill in enumerate(pathway_30['prioritized_skills'], 1):
        print(f"   {i}. {skill['skill']} - {skill['priority']} priority")
        print(f"      Current: {skill['current_score']*100:.0f}%, Impact: {skill['impact_score']}, Est. Time: {skill['estimated_weeks']} weeks")
    
    print(f"\n📅 LEARNING TIMELINE:")
    for phase in pathway_30['timeline']:
        print(f"\n   Phase {phase['phase']}: {phase['focus']} (Days {phase['days']})")
        print(f"   Skills: {', '.join(phase['skills'])}")
        print(f"   Activities:")
        for activity in phase['activities']:
            print(f"      • {activity}")
    
    print(f"\n📚 DETAILED LEARNING PLANS:")
    for plan in pathway_30['detailed_plan']:
        print(f"\n   #{plan['rank']} - {plan['skill'].upper()}")
        print(f"   Priority: {plan['priority']} | Level: {plan['current_level']} → {plan['target_level']}")
        print(f"   Daily Commitment: {plan['daily_commitment']}")
        
        if 'courses' in plan['resources']:
            print(f"\n   📖 Recommended Courses:")
            for course in plan['resources']['courses'][:2]:
                print(f"      • {course['name']} ({course.get('duration', 'N/A')})")
        
        if 'projects' in plan['resources']:
            print(f"\n   💻 Practice Projects:")
            for project in plan['resources']['projects'][:2]:
                print(f"      • {project}")
        
        print(f"\n   🗺️ Learning Path:")
        for step in plan['learning_path']:
            print(f"      {step}")
    
    print(f"\n🏆 MILESTONES:")
    for milestone in pathway_30['milestones']:
        print(f"   Day {milestone['day']}: {milestone['milestone']} → {milestone['reward']}")
    
    print(f"\n💡 SUCCESS TIPS:")
    for tip in pathway_30['success_tips']:
        print(f"   {tip}")
    
    # Test 7-day intensive pathway
    print("\n" + "=" * 70)
    print("7-DAY INTENSIVE PATHWAY (QUICK COMPARISON)")
    print("=" * 70)
    
    pathway_7 = generator.generate_pathway(missing_skills, timeline_days=7)
    
    print(f"\n📈 INTENSIVE OVERVIEW:")
    print(f"   Focus Skills: {', '.join([s['skill'] for s in pathway_7['prioritized_skills'][:2]])}")
    print(f"   Timeline: {pathway_7['timeline_days']} days")
    
    print(f"\n📅 PHASES:")
    for phase in pathway_7['timeline']:
        print(f"   Phase {phase['phase']}: {phase['focus']} (Days {phase['days']})")
    
    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT (30-DAY PATHWAY):")
    print("=" * 70)
    print(json.dumps(pathway_30, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    test_learning_pathway()