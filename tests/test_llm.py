"""
Test Ollama LLM Integration
"""
from src.llm.ollama_client import OllamaClient
from src.llm.cv_rewriter import CVRewriter


def test_llm():
    """Test Ollama LLM functionality"""
    
    print("=" * 70)
    print("OLLAMA LLM INTEGRATION TEST")
    print("=" * 70)
    
    # Test 1: Basic generation
    print("\n✅ TEST 1: Basic Text Generation")
    print("-" * 70)
    
    client = OllamaClient()
    response = client.generate(
        "What are the top 3 skills for a software engineer? Be brief.",
        temperature=0.7
    )
    
    print(f"Response: {response}\n")
    
    # Test 2: CV Bullet Rewriting
    print("\n✅ TEST 2: CV Bullet Point Rewriting")
    print("-" * 70)
    
    rewriter = CVRewriter(client)
    
    weak_bullets = [
        "Worked on API development",
        "Used Python for various tasks",
        "Helped with Docker containers"
    ]
    
    for bullet in weak_bullets:
        improved = rewriter.rewrite_bullet(bullet)
        print(f"\nOriginal: {bullet}")
        print(f"Improved: {improved}")
    
    # Test 3: Skill-Targeted Rewriting
    print("\n✅ TEST 3: Skill-Targeted Rewriting")
    print("-" * 70)
    
    bullet = "Built backend services"
    target_skill = "FastAPI"
    
    improved = rewriter.rewrite_bullet(bullet, target_skill=target_skill)
    
    print(f"\nOriginal: {bullet}")
    print(f"Target: Emphasize {target_skill}")
    print(f"Improved: {improved}")
    
    # Test 4: Generate New Bullet
    print("\n✅ TEST 4: Generate Missing Skill Bullet")
    print("-" * 70)
    
    missing_skill = "Docker"
    context = "Senior Software Engineer at a cloud services company"
    
    new_bullet = rewriter.generate_skill_bullet(missing_skill, context)
    
    print(f"\nSkill: {missing_skill}")
    print(f"Context: {context}")
    print(f"Generated: {new_bullet}")
    
    print("\n" + "=" * 70)
    print("✅ ALL LLM TESTS COMPLETED!")
    print("=" * 70)
    print("\n💡 Note: Responses may vary due to LLM's creative nature")


if __name__ == "__main__":
    print("\n⚠️  MAKE SURE OLLAMA IS RUNNING!")
    print("   Check: ollama list")
    print("   Model: llama3.2:3b should be available")
    print("\n   Press Enter when ready...")
    input()
    
    try:
        test_llm()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? (Check Task Manager)")
        print("2. Is llama3.2:3b downloaded? (Run: ollama pull llama3.2:3b)")
        print("3. Try: ollama run llama3.2:3b")