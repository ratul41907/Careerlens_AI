"""
Performance Benchmark Tests
Measure response times and throughput
"""
import pytest
import time
import statistics
import tempfile
import os


class TestPerformanceBenchmarks:
    """Benchmark performance of core components"""
    
    @pytest.mark.slow
    def test_cv_parsing_performance(self, cv_parser):
        """Benchmark CV parsing speed"""
        
        print("\n" + "=" * 70)
        print("BENCHMARK: CV Parsing Performance")
        print("=" * 70)
        
        sample_cv = """
JOHN DOE
Software Engineer
john.doe@email.com

EXPERIENCE
Senior Engineer | TechCorp | 2020-Present
- Developed applications
- Led team projects
- Optimized performance

SKILLS
Python, JavaScript, Docker, AWS, PostgreSQL

EDUCATION
BS Computer Science | University | 2020
"""
        
        # Run multiple iterations
        iterations = 10
        times = []
        
        print(f"\n⏱️  Running {iterations} iterations...")
        
        for i in range(iterations):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_cv)
                cv_path = f.name
            
            try:
                start = time.time()
                result = cv_parser.parse(cv_path)
                elapsed = time.time() - start
                
                times.append(elapsed)
                assert result is not None
                
            finally:
                os.remove(cv_path)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        
        print(f"\n📊 RESULTS:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Median:  {median_time:.3f}s")
        print(f"   Min:     {min_time:.3f}s")
        print(f"   Max:     {max_time:.3f}s")
        
        # Performance assertion
        assert avg_time < 3.0  # Should average under 3s
        
        print(f"\n✅ CV Parsing: {avg_time:.3f}s average (target: <3s)")
    
    @pytest.mark.slow
    def test_scoring_performance(self, scoring_engine):
        """Benchmark scoring engine performance"""
        
        print("\n" + "=" * 70)
        print("BENCHMARK: Scoring Engine Performance")
        print("=" * 70)
        
        cv_data = {
            'text': 'Python FastAPI Docker PostgreSQL AWS',
            'sections': {
                'skills': ['Python', 'FastAPI', 'Docker', 'PostgreSQL', 'AWS']
            }
        }
        
        jd_data = {
            'text': 'Python FastAPI Docker PostgreSQL',
            'sections': {
                'required_skills': ['Python', 'FastAPI', 'Docker', 'PostgreSQL']
            }
        }
        
        iterations = 20
        times = []
        
        print(f"\n⏱️  Running {iterations} iterations...")
        
        for i in range(iterations):
            start = time.time()
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            elapsed = time.time() - start
            
            times.append(elapsed)
            assert result is not None
        
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        
        print(f"\n📊 RESULTS:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Median:  {median_time:.3f}s")
        
        assert avg_time < 2.0  # Should average under 2s
        
        print(f"\n✅ Scoring: {avg_time:.3f}s average (target: <2s)")
    
    @pytest.mark.slow
    @pytest.mark.llm
    def test_llm_response_time(self, learning_pathway_generator):
        """Benchmark LLM response times"""
        
        print("\n" + "=" * 70)
        print("BENCHMARK: LLM Response Time")
        print("=" * 70)
        
        skill_gaps = ['Docker', 'Kubernetes']
        jd_data = {'text': 'Need Docker and Kubernetes'}
        
        iterations = 5  # LLM tests are slower
        times = []
        
        print(f"\n⏱️  Running {iterations} LLM calls...")
        
        for i in range(iterations):
            print(f"   Iteration {i+1}/{iterations}...", end='')
            start = time.time()
            
            result = learning_pathway_generator.generate_pathway(
                skill_gaps=skill_gaps,
                jd_data=jd_data,
                num_days=7
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f" {elapsed:.2f}s")
            assert result is not None
        
        avg_time = statistics.mean(times)
        
        print(f"\n📊 RESULTS:")
        print(f"   Average LLM Response: {avg_time:.2f}s")
        
        # LLM calls can be slower
        assert avg_time < 30.0  # Should average under 30s
        
        print(f"\n✅ LLM Response: {avg_time:.2f}s average (target: <30s)")


class TestThroughput:
    """Test system throughput"""
    
    @pytest.mark.slow
    def test_concurrent_scoring(self, scoring_engine):
        """Test scoring multiple CV-JD pairs sequentially"""
        
        print("\n" + "=" * 70)
        print("THROUGHPUT TEST: Sequential Scoring")
        print("=" * 70)
        
        # Create 10 different CV-JD pairs
        num_pairs = 10
        
        cv_jd_pairs = [
            {
                'cv': {
                    'text': f'Developer {i}',
                    'sections': {'skills': ['Python', 'Docker']}
                },
                'jd': {
                    'text': f'Job {i}',
                    'sections': {'required_skills': ['Python', 'Docker']}
                }
            }
            for i in range(num_pairs)
        ]
        
        print(f"\n⏱️  Scoring {num_pairs} CV-JD pairs sequentially...")
        
        start_time = time.time()
        results = []
        
        for i, pair in enumerate(cv_jd_pairs, 1):
            result = scoring_engine.compute_match_score(pair['cv'], pair['jd'])
            results.append(result)
            print(f"   Pair {i}/{num_pairs} completed")
        
        total_time = time.time() - start_time
        throughput = num_pairs / total_time
        
        print(f"\n📊 THROUGHPUT RESULTS:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} pairs/second")
        print(f"   Avg per pair: {total_time/num_pairs:.2f}s")
        
        assert len(results) == num_pairs
        assert throughput > 0.5  # At least 0.5 pairs/second
        
        print(f"\n✅ Throughput: {throughput:.2f} pairs/sec")