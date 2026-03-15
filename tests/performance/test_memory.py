"""
Memory Profiling Tests
Monitor memory usage and detect leaks
"""
import pytest
import time
import gc
import psutil
import os


class TestMemoryUsage:
    """Test memory consumption of core components"""
    
    def get_memory_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    @pytest.mark.slow
    def test_cv_parsing_memory(self, cv_parser):
        """Test memory usage during CV parsing"""
        
        print("\n" + "=" * 70)
        print("MEMORY TEST: CV Parsing Memory Usage")
        print("=" * 70)
        
        sample_cv = """
LARGE CV CONTENT
""" + ("Experience line\n" * 1000)  # Create larger CV
        
        import tempfile
        
        # Force garbage collection
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = self.get_memory_mb()
        print(f"\n📊 Baseline Memory: {baseline_memory:.2f} MB")
        
        # Parse CV multiple times
        iterations = 50
        print(f"\n⚡ Parsing CV {iterations} times...")
        
        for i in range(iterations):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_cv)
                cv_path = f.name
            
            try:
                cv_parser.parse(cv_path)
            finally:
                os.remove(cv_path)
            
            if (i + 1) % 10 == 0:
                current_memory = self.get_memory_mb()
                print(f"   Iteration {i+1}: {current_memory:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        time.sleep(0.5)
        
        # Measure final memory
        final_memory = self.get_memory_mb()
        memory_increase = final_memory - baseline_memory
        
        print(f"\n📊 MEMORY ANALYSIS:")
        print(f"   Baseline: {baseline_memory:.2f} MB")
        print(f"   Final: {final_memory:.2f} MB")
        print(f"   Increase: {memory_increase:.2f} MB")
        print(f"   Per Iteration: {memory_increase/iterations:.3f} MB")
        
        # Memory should not increase significantly (< 100 MB for 50 iterations)
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.2f} MB increase"
        
        print(f"\n✅ Memory Test Passed: {memory_increase:.2f} MB increase (acceptable)")
    
    @pytest.mark.slow
    def test_scoring_memory(self, scoring_engine):
        """Test memory usage during scoring"""
        
        print("\n" + "=" * 70)
        print("MEMORY TEST: Scoring Engine Memory Usage")
        print("=" * 70)
        
        cv_data = {
            'text': 'Python Developer with extensive experience',
            'sections': {
                'skills': ['Python', 'Docker', 'FastAPI', 'PostgreSQL', 'AWS', 
                          'Kubernetes', 'Redis', 'MongoDB', 'React', 'TypeScript']
            }
        }
        
        jd_data = {
            'text': 'Looking for experienced developer',
            'sections': {
                'required_skills': ['Python', 'Docker', 'FastAPI', 'PostgreSQL'],
                'preferred_skills': ['AWS', 'Kubernetes', 'Redis']
            }
        }
        
        gc.collect()
        baseline_memory = self.get_memory_mb()
        
        print(f"\n📊 Baseline Memory: {baseline_memory:.2f} MB")
        
        # Score multiple times
        iterations = 100
        print(f"\n⚡ Computing {iterations} match scores...")
        
        for i in range(iterations):
            scoring_engine.compute_match_score(cv_data, jd_data)
            
            if (i + 1) % 20 == 0:
                current_memory = self.get_memory_mb()
                print(f"   Iteration {i+1}: {current_memory:.2f} MB")
        
        gc.collect()
        time.sleep(0.5)
        
        final_memory = self.get_memory_mb()
        memory_increase = final_memory - baseline_memory
        
        print(f"\n📊 MEMORY ANALYSIS:")
        print(f"   Baseline: {baseline_memory:.2f} MB")
        print(f"   Final: {final_memory:.2f} MB")
        print(f"   Increase: {memory_increase:.2f} MB")
        print(f"   Per Iteration: {memory_increase/iterations:.3f} MB")
        
        # Should not increase significantly
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.2f} MB"
        
        print(f"\n✅ Memory Test Passed: {memory_increase:.2f} MB increase")
    
    @pytest.mark.slow
    def test_embedding_cache_memory(self, embedding_engine):
        """Test memory usage of embedding cache"""
        
        print("\n" + "=" * 70)
        print("MEMORY TEST: Embedding Cache Memory Usage")
        print("=" * 70)
        
        gc.collect()
        baseline_memory = self.get_memory_mb()
        
        print(f"\n📊 Baseline Memory: {baseline_memory:.2f} MB")
        
        # Generate embeddings for many unique texts
        num_embeddings = 500
        print(f"\n⚡ Generating {num_embeddings} embeddings...")
        
        for i in range(num_embeddings):
            text = f"Unique text number {i} with skills Python Docker AWS"
            embedding_engine.get_embedding(text)
            
            if (i + 1) % 100 == 0:
                current_memory = self.get_memory_mb()
                print(f"   Generated {i+1}: {current_memory:.2f} MB")
        
        final_memory = self.get_memory_mb()
        memory_increase = final_memory - baseline_memory
        
        print(f"\n📊 MEMORY ANALYSIS:")
        print(f"   Baseline: {baseline_memory:.2f} MB")
        print(f"   Final: {final_memory:.2f} MB")
        print(f"   Increase: {memory_increase:.2f} MB")
        print(f"   Cache Size: {len(embedding_engine.cache)} embeddings")
        
        # Cache should not grow unbounded
        assert memory_increase < 200, f"Cache using too much memory: {memory_increase:.2f} MB"
        
        print(f"\n✅ Cache Memory Test Passed: {memory_increase:.2f} MB for {num_embeddings} embeddings")


class TestMemoryLeaks:
    """Test for memory leaks"""
    
    @pytest.mark.slow
    def test_repeated_operations_no_leak(self, cv_parser, jd_parser, scoring_engine):
        """Test that repeated operations don't leak memory"""
        
        print("\n" + "=" * 70)
        print("MEMORY LEAK TEST: Repeated Operations")
        print("=" * 70)
        
        import tempfile
        
        sample_cv = "John Doe\nPython Developer\nSkills: Python, Docker"
        sample_jd = "Looking for Python developer with Docker experience"
        
        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        print(f"\n📊 Initial Memory: {initial_memory:.2f} MB")
        
        # Perform complete workflow 50 times
        iterations = 50
        memory_samples = []
        
        print(f"\n⚡ Running {iterations} complete workflows...")
        
        for i in range(iterations):
            # Parse CV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_cv)
                cv_path = f.name
            
            try:
                cv_data = cv_parser.parse(cv_path)
            finally:
                os.remove(cv_path)
            
            # Parse JD
            jd_data = jd_parser.parse(sample_jd)
            
            # Score
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            
            # Sample memory every 10 iterations
            if (i + 1) % 10 == 0:
                gc.collect()
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                print(f"   Iteration {i+1}: {current_memory:.2f} MB")
        
        # Final garbage collection
        gc.collect()
        time.sleep(1)
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"\n📊 MEMORY LEAK ANALYSIS:")
        print(f"   Initial: {initial_memory:.2f} MB")
        print(f"   Final: {final_memory:.2f} MB")
        print(f"   Total Increase: {total_increase:.2f} MB")
        print(f"   Per Operation: {total_increase/iterations:.3f} MB")
        
        # Check if memory growth is linear (indicates leak)
        if len(memory_samples) > 2:
            growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
            print(f"   Growth Rate: {growth_rate:.3f} MB per sample")
        
        # Total increase should be reasonable (< 80 MB)
        assert total_increase < 80, f"Potential memory leak: {total_increase:.2f} MB increase"
        
        print(f"\n✅ No Memory Leak Detected: {total_increase:.2f} MB increase over {iterations} operations")