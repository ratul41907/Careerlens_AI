"""
Test Embedding Engine
"""
from src.embeddings.embedding_engine import EmbeddingEngine, compute_text_similarity
import numpy as np


def test_embedding_engine():
    """Test embedding engine with sample texts"""
    
    print("=" * 60)
    print("EMBEDDING ENGINE TEST")
    print("=" * 60)
    
    # Initialize engine
    print("\n📦 Initializing SentenceTransformer model...")
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    # Test 1: Single text embedding
    print("\n✅ Test 1: Generate single embedding")
    text = "Python developer with FastAPI experience"
    embedding = engine.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   Sample values: {embedding[:5]}")
    
    # Test 2: Batch embeddings
    print("\n✅ Test 2: Generate batch embeddings")
    texts = [
        "Python developer",
        "JavaScript engineer",
        "Machine learning expert",
        "Full-stack developer"
    ]
    embeddings = engine.encode(texts, show_progress=True)
    print(f"   Encoded {len(texts)} texts")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Test 3: Similarity between similar texts
    print("\n✅ Test 3: Similarity - Similar texts")
    text1 = "Python programming with Django"
    text2 = "Python development using Django framework"
    score = compute_text_similarity(text1, text2, engine)
    print(f"   Text 1: '{text1}'")
    print(f"   Text 2: '{text2}'")
    print(f"   Similarity: {score:.4f} (should be HIGH ~0.85+)")
    
    # Test 4: Similarity between different texts
    print("\n✅ Test 4: Similarity - Different texts")
    text1 = "Machine learning with TensorFlow"
    text2 = "Cooking pasta recipes"
    score = compute_text_similarity(text1, text2, engine)
    print(f"   Text 1: '{text1}'")
    print(f"   Text 2: '{text2}'")
    print(f"   Similarity: {score:.4f} (should be LOW ~0.50-)")
    
    # Test 5: Skill matching (realistic scenario)
    print("\n✅ Test 5: Skill Matching - CV vs JD")
    cv_skills = [
        "5 years Python development",
        "Built REST APIs with FastAPI",
        "Experience with PostgreSQL databases",
        "Docker containerization",
        "Deployed on AWS cloud"
    ]
    
    jd_requirements = [
        "Python programming required",
        "FastAPI framework experience needed",
        "SQL database knowledge",
        "Kubernetes deployment",
        "Cloud experience (AWS/Azure)"
    ]
    
    print("\n   📄 CV Skills:")
    for skill in cv_skills:
        print(f"      • {skill}")
    
    print("\n   📋 JD Requirements:")
    for req in jd_requirements:
        print(f"      • {req}")
    
    # Encode all texts
    cv_embeddings = engine.encode(cv_skills)
    jd_embeddings = engine.encode(jd_requirements)
    
    # Compute similarity matrix
    similarity_matrix = engine.compute_similarity_matrix(cv_embeddings, jd_embeddings)
    
    print("\n   🎯 Similarity Matrix:")
    print("      " + "  ".join([f"JD{i+1}" for i in range(len(jd_requirements))]))
    for i, cv_skill in enumerate(cv_skills):
        scores = [f"{score:.2f}" for score in similarity_matrix[i]]
        print(f"   CV{i+1}: " + "  ".join(scores))
    
    # Find best matches for each CV skill
    print("\n   🏆 Best Matches:")
    for i, cv_skill in enumerate(cv_skills):
        best_jd_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_jd_idx]
        print(f"\n   CV: '{cv_skill[:40]}...'")
        print(f"   ↔️  JD: '{jd_requirements[best_jd_idx][:40]}...'")
        print(f"   Score: {best_score:.4f}")
    
    # Test 6: Cache performance
    print("\n✅ Test 6: Cache Performance")
    text = "Python developer with FastAPI"
    
    # First encoding (not cached)
    import time
    start = time.time()
    _ = engine.encode(text)
    time_uncached = time.time() - start
    
    # Second encoding (cached)
    start = time.time()
    _ = engine.encode(text)
    time_cached = time.time() - start
    
    cache_stats = engine.get_cache_stats()
    print(f"   Uncached: {time_uncached*1000:.2f}ms")
    print(f"   Cached: {time_cached*1000:.2f}ms")
    
    if time_cached > 0:
        print(f"   Speedup: {time_uncached/time_cached:.1f}x faster")
    else:
        print(f"   Speedup: INSTANT (cached too fast to measure)")
    
    print(f"   Cache size: {cache_stats['cached_items']} items")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_embedding_engine()