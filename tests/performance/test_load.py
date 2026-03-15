"""
Load Testing - Simulate Multiple Concurrent Users
"""
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os


class TestLoadTesting:
    """Simulate load from multiple concurrent users"""
    
    @pytest.mark.slow
    def test_concurrent_cv_parsing(self, cv_parser):
        """Test parsing CVs concurrently"""
        
        print("\n" + "=" * 70)
        print("LOAD TEST: Concurrent CV Parsing")
        print("=" * 70)
        
        sample_cv = """
JOHN DOE
Software Engineer
john.doe@email.com | +1-555-0123

EXPERIENCE
Senior Developer | TechCorp | 2020-Present
- Developed microservices using Python
- Implemented Docker containerization
- Led team of 5 developers

SKILLS
Python, JavaScript, Docker, Kubernetes, AWS

EDUCATION
BS Computer Science | University | 2020
"""
        
        def parse_cv_task(task_id):
            """Single CV parsing task"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_cv)
                cv_path = f.name
            
            try:
                start = time.time()
                result = cv_parser.parse(cv_path)
                elapsed = time.time() - start
                
                return {
                    'task_id': task_id,
                    'success': result is not None,
                    'time': elapsed
                }
            finally:
                if os.path.exists(cv_path):
                    os.remove(cv_path)
        
        # Simulate 10 concurrent users
        num_users = 10
        
        print(f"\n👥 Simulating {num_users} concurrent users...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(parse_cv_task, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        times = [r['time'] for r in successful]
        
        print(f"\n📊 LOAD TEST RESULTS:")
        print(f"   Total Users: {num_users}")
        print(f"   Successful: {len(successful)}/{num_users}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Avg Response Time: {statistics.mean(times):.2f}s")
        print(f"   Max Response Time: {max(times):.2f}s")
        print(f"   Min Response Time: {min(times):.2f}s")
        
        # Assertions
        assert len(successful) == num_users
        assert statistics.mean(times) < 5.0  # Avg under 5s
        
        print(f"\n✅ Load Test Passed: {len(successful)}/{num_users} users handled successfully")
    
    @pytest.mark.slow
    def test_concurrent_scoring(self, scoring_engine):
        """Test scoring multiple CV-JD pairs concurrently"""
        
        print("\n" + "=" * 70)
        print("LOAD TEST: Concurrent Match Scoring")
        print("=" * 70)
        
        def scoring_task(task_id):
            """Single scoring task"""
            cv_data = {
                'text': f'Developer {task_id}',
                'sections': {
                    'skills': ['Python', 'Docker', 'FastAPI', 'PostgreSQL', 'AWS']
                }
            }
            
            jd_data = {
                'text': f'Job {task_id}',
                'sections': {
                    'required_skills': ['Python', 'Docker', 'FastAPI', 'PostgreSQL'],
                    'preferred_skills': ['AWS', 'Kubernetes']
                }
            }
            
            start = time.time()
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            elapsed = time.time() - start
            
            return {
                'task_id': task_id,
                'success': result is not None,
                'score': result['overall_score'] if result else 0,
                'time': elapsed
            }
        
        # Simulate 20 concurrent scoring requests
        num_requests = 20
        
        print(f"\n⚡ Processing {num_requests} concurrent scoring requests...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(scoring_task, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        times = [r['time'] for r in successful]
        scores = [r['score'] for r in successful]
        
        throughput = len(successful) / total_time
        
        print(f"\n📊 LOAD TEST RESULTS:")
        print(f"   Total Requests: {num_requests}")
        print(f"   Successful: {len(successful)}/{num_requests}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} requests/sec")
        print(f"   Avg Response Time: {statistics.mean(times):.2f}s")
        print(f"   Avg Match Score: {statistics.mean(scores)*100:.1f}%")
        
        # Assertions
        assert len(successful) == num_requests
        assert throughput > 1.0  # At least 1 request/sec
        
        print(f"\n✅ Load Test Passed: {throughput:.2f} requests/sec throughput")
    
    @pytest.mark.slow
    @pytest.mark.llm
    def test_sequential_llm_load(self, interview_guidance):
        """Test LLM under sequential load (safer than concurrent)"""
        
        print("\n" + "=" * 70)
        print("LOAD TEST: Sequential LLM Requests")
        print("=" * 70)
        
        skills_variations = [
            ['Python', 'Docker'],
            ['JavaScript', 'React'],
            ['Java', 'Spring'],
            ['Go', 'Kubernetes'],
            ['Ruby', 'Rails']
        ]
        
        print(f"\n📝 Generating questions for {len(skills_variations)} skill sets...")
        
        results = []
        start_time = time.time()
        
        for i, skills in enumerate(skills_variations, 1):
            print(f"   Request {i}/{len(skills_variations)}: {', '.join(skills)}...", end='')
            
            task_start = time.time()
            result = interview_guidance.generate_questions(
                skills=skills,
                num_questions=5,
                question_type='technical'
            )
            elapsed = time.time() - task_start
            
            results.append({
                'success': result is not None and result.get('success', False),
                'time': elapsed,
                'questions': len(result.get('questions', [])) if result else 0
            })
            
            print(f" {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        times = [r['time'] for r in successful]
        
        print(f"\n📊 LOAD TEST RESULTS:")
        print(f"   Total Requests: {len(skills_variations)}")
        print(f"   Successful: {len(successful)}/{len(skills_variations)}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Avg Response Time: {statistics.mean(times):.2f}s")
        print(f"   Total Questions Generated: {sum(r['questions'] for r in successful)}")
        
        assert len(successful) == len(skills_variations)
        
        print(f"\n✅ LLM Load Test Passed: {len(successful)} sequential requests handled")


class TestScalability:
    """Test system scalability"""
    
    @pytest.mark.slow
    def test_increasing_load(self, scoring_engine):
        """Test system behavior under increasing load"""
        
        print("\n" + "=" * 70)
        print("SCALABILITY TEST: Increasing Load")
        print("=" * 70)
        
        def scoring_task():
            """Single scoring task"""
            cv_data = {
                'text': 'Python Developer',
                'sections': {'skills': ['Python', 'FastAPI', 'Docker']}
            }
            jd_data = {
                'text': 'Python Developer needed',
                'sections': {'required_skills': ['Python', 'FastAPI', 'Docker']}
            }
            
            start = time.time()
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            elapsed = time.time() - start
            
            return elapsed
        
        # Test with increasing load: 1, 5, 10, 15 concurrent requests
        load_levels = [1, 5, 10, 15]
        results = {}
        
        for load in load_levels:
            print(f"\n⚡ Testing with {load} concurrent requests...")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=load) as executor:
                futures = [executor.submit(scoring_task) for _ in range(load)]
                times = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            throughput = load / total_time
            
            results[load] = {
                'total_time': total_time,
                'avg_response': statistics.mean(times),
                'throughput': throughput
            }
            
            print(f"   Total Time: {total_time:.2f}s")
            print(f"   Avg Response: {statistics.mean(times):.2f}s")
            print(f"   Throughput: {throughput:.2f} req/sec")
        
        # Display scalability analysis
        print(f"\n📊 SCALABILITY ANALYSIS:")
        print(f"{'Load':<10} {'Total Time':<15} {'Avg Response':<15} {'Throughput':<15}")
        print("-" * 60)
        
        for load, data in results.items():
            print(f"{load:<10} {data['total_time']:<15.2f} {data['avg_response']:<15.2f} {data['throughput']:<15.2f}")
        
        # Check that throughput doesn't degrade too much
        throughput_1 = results[1]['throughput']
        throughput_15 = results[15]['throughput']
        degradation = (throughput_1 - throughput_15) / throughput_1 * 100
        
        print(f"\n📉 Throughput Degradation: {degradation:.1f}%")
        
        # Acceptable degradation under 50%
        assert degradation < 50, f"Throughput degraded too much: {degradation:.1f}%"
        
        print(f"\n✅ Scalability Test Passed: {degradation:.1f}% degradation (acceptable)")