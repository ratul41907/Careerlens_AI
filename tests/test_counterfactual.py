"""
Test Counterfactual Simulator
"""
from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.counterfactual import CounterfactualSimulator
import json


def test_counterfactual():
    """Test counterfactual skill impact simulation"""
    
    print("=" * 70)
    print("COUNTERFACTUAL SIMULATOR TEST")
    print("=" * 70)
    
    # Sample CV (missing some skills)
    sample_cv = """
    JOHN DOE
    Software Engineer
    
    EXPERIENCE
    Software Engineer | TechCo | 2020 - Present
    • Developed web applications using React
    • Built REST APIs with Node.js
    • Worked with MongoDB database
    
    SKILLS
    JavaScript, React, Node.js, MongoDB, Git
    """
    
    # Sample JD (requires more skills)
    sample_jd = """
    Senior Software Engineer Position
    
    Required Skills:
    • Python programming required
    • React framework experience
    • Docker containerization required
    • AWS cloud services required
    • Kubernetes orchestration needed
    • 5+ years experience
    """
    
    print("\n📊 STEP 1: Computing Baseline Match")
    print("-" * 70)
    
    # Parse
    cv_parser = CVParser()
    cv_data = {
        'text': sample_cv,
        'sections': cv_parser._segment_sections(sample_cv)
    }
    
    jd_parser = JDParser()
    jd_data = jd_parser.parse(sample_jd)
    
    # Compute baseline match
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    baseline_result = scoring_engine.compute_match_score(cv_data, jd_data)
    
    print(f"✅ Baseline Score: {baseline_result['overall_percentage']}")
    print(f"   Interpretation: {baseline_result['interpretation']['level']}")
    
    print("\n🔮 STEP 2: Running Counterfactual Simulations")
    print("-" * 70)
    
    # Run counterfactual simulation
    simulator = CounterfactualSimulator(embedding_engine, scoring_engine)
    simulation_result = simulator.simulate_skill_impact(cv_data, jd_data, baseline_result)
    
    # Display results
    print(f"\n{'='*70}")
    print("COUNTERFACTUAL SKILL IMPACT ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📊 BASELINE:")
    print(f"   Current Score: {simulation_result['baseline_percentage']}")
    
    print(f"\n🎯 TOP OPPORTUNITIES:")
    print("-" * 70)
    
    for i, sim in enumerate(simulation_result['simulations'][:5], 1):
        priority_emoji = "🔴" if sim['priority'] == "Critical" else "🟠" if sim['priority'] == "High" else "🟡"
        
        print(f"\n{priority_emoji} #{i}. {sim['skill'].upper()}")
        print(f"   Current Match: {sim['current_percentage']}")
        print(f"   If Added: {sim['new_overall_percentage']} overall")
        print(f"   Impact: {sim['improvement']} gain")
        print(f"   Priority: {sim['priority']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    for i, rec in enumerate(simulation_result['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📈 POTENTIAL GAIN:")
    print("-" * 70)
    print(f"   Top 3 Skills Combined: +{simulation_result['total_potential_gain'] * 100:.1f}%")
    print(f"   From: {simulation_result['baseline_percentage']}")
    print(f"   To: {(float(simulation_result['baseline_score']) + simulation_result['total_potential_gain']) * 100:.1f}%")
    
    print(f"\n{'='*70}")
    print("FULL JSON OUTPUT:")
    print(f"{'='*70}")
    print(json.dumps(simulation_result, indent=2))
    
    print(f"\n{'='*70}")
    print("✅ COUNTERFACTUAL TEST COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_counterfactual()