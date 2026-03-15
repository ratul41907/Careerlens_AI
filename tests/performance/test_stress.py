"""
Stress Testing - Push System to Limits
"""
import pytest
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor


class TestStressTesting:
    """Stress test system under extreme conditions"""
    
    @pytest.mark.slow
    def test_large_cv_handling(self, cv_parser):
        """Test handling of very large CVs"""
        
        print("\n" + "=" * 70)
        print("STRESS TEST: Large CV Handling")
        print("=" * 70)
        
        # Create very large CV (10,000+ lines)
        large_cv_lines = ["JOHN DOE\nSenior Software Engineer\njohn@email.com\n"]
        large_cv_lines.append("\nEXPERIENCE:\n")
        
        # Add 5000 experience bullets
        for i in range(5000):
            large_cv_lines.append(f"- Worked on project {i} achieving results\n")
        
        large_cv_lines.append("\nSKILLS:\n")
        # Add 500 skills
        skills = [f"Skill{i}" for i in range(500)]
        large_cv_lines.append(", ".join(skills))
        
        large_cv = "".join(large_cv_lines)
        cv_size_kb = len(large_cv) / 1024
        
        print(f"\n📄 CV Size: {cv_size_kb:.1f} KB ({len(large_cv_lines)} lines)")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(large_cv)
            cv_path = f.name
        
        try:
            print(f"\n⚡ Parsing large CV...")
            start_time = time.time()
            
            result = cv_parser.parse(cv_path)
            
            parse_time = time.time() - start_time
            
            assert result is not None
            assert 'text' in result
            
            print(f"✅ Large CV parsed successfully in {parse_time:.2f}s")
            print(f"   Size: {cv_size_kb:.1f} KB")
            print(f"   Parse time: {parse_time:.2f}s")
            
            # Should complete in reasonable time even for large CV
            assert parse_time < 30, f"Parsing took too long: {parse_time:.2f}s"
            
        finally:
            os.remove(cv_path)
        
        print(f"\n✅ Large CV Stress Test Passed")
    
    @pytest.mark.slow
    def test_many_skills_matching(self, scoring_engine):
        """Test matching with many skills"""
        
        print("\n" + "=" * 70)
        print("STRESS TEST: Many Skills Matching")
        print("=" * 70)
        
        # Create CV with 200 skills
        cv_skills = [f"Skill{i}" for i in range(200)]
        cv_data = {
            'text': ' '.join(cv_skills),
            'sections': {'skills': cv_skills}
        }
        
        # Create JD with 100 required skills
        jd_skills = [f"Skill{i}" for i in range(100)]
        jd_data = {
            'text': ' '.join(jd_skills),
            'sections': {
                'required_skills': jd_skills[:80],
                'preferred_skills': jd_skills[80:]
            }
        }
        
        print(f"\n📊 Test Configuration:")
        print(f"   CV Skills: {len(cv_skills)}")
        print(f"   JD Required: {len(jd_data['sections']['required_skills'])}")
        print(f"   JD Preferred: {len(jd_data['sections']['preferred_skills'])}")
        
        print(f"\n⚡ Computing match with many skills...")
        start_time = time.time()
        
        result = scoring_engine.compute_match_score(cv_data, jd_data)
        
        compute_time = time.time() - start_time
        
        assert result is not None
        assert 'overall_score' in result
        
        print(f"✅ Match computed in {compute_time:.2f}s")
        print(f"   Overall Score: {result['overall_score']*100:.1f}%")
        
        # Should handle many skills efficiently
        assert compute_time < 10, f"Computation took too long: {compute_time:.2f}s"
        
        print(f"\n✅ Many Skills Stress Test Passed")
    
    @pytest.mark.slow
    def test_rapid_sequential_requests(self, scoring_engine):
        """Test handling rapid sequential requests"""
        
        print("\n" + "=" * 70)
        print("STRESS TEST: Rapid Sequential Requests")
        print("=" * 70)
        
        cv_data = {
            'text': 'Python Developer',
            'sections': {'skills': ['Python', 'Docker']}
        }
        
        jd_data = {
            'text': 'Python Developer needed',
            'sections': {'required_skills': ['Python', 'Docker']}
        }
        
        # Send 100 requests as fast as possible
        num_requests = 100
        
        print(f"\n⚡ Sending {num_requests} rapid sequential requests...")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        for i in range(num_requests):
            try:
                result = scoring_engine.compute_match_score(cv_data, jd_data)
                if result is not None:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"   Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        throughput = successful / total_time
        
        print(f"\n📊 RAPID REQUEST RESULTS:")
        print(f"   Total Requests: {num_requests}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} req/sec")
        print(f"   Avg Time: {total_time/num_requests:.3f}s per request")
        
        # Should handle at least 90% successfully
        success_rate = (successful / num_requests) * 100
        assert success_rate >= 90, f"Success rate too low: {success_rate:.1f}%"
        
        print(f"\n✅ Rapid Request Stress Test Passed: {success_rate:.1f}% success rate")
    
    @pytest.mark.slow
    def test_sustained_load(self, scoring_engine):
        """Test sustained load over extended period"""
        
        print("\n" + "=" * 70)
        print("STRESS TEST: Sustained Load (60 seconds)")
        print("=" * 70)
        
        cv_data = {
            'text': 'Python Developer',
            'sections': {'skills': ['Python', 'FastAPI', 'Docker']}
        }
        
        jd_data = {
            'text': 'Python Developer needed',
            'sections': {'required_skills': ['Python', 'FastAPI', 'Docker']}
        }
        
        duration = 60  # 60 seconds
        print(f"\n⚡ Running sustained load for {duration} seconds...")
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while (time.time() - start_time) < duration:
            try:
                result = scoring_engine.compute_match_score(cv_data, jd_data)
                if result is not None:
                    request_count += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
            
            # Brief pause to avoid overwhelming
            time.sleep(0.1)
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        
        print(f"\n📊 SUSTAINED LOAD RESULTS:")
        print(f"   Duration: {actual_duration:.1f}s")
        print(f"   Total Requests: {request_count}")
        print(f"   Errors: {errors}")
        print(f"   Throughput: {throughput:.2f} req/sec")
        print(f"   Success Rate: {(request_count/(request_count+errors))*100:.1f}%")
        
        # Should maintain reasonable throughput
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} req/sec"
        assert errors < request_count * 0.05, f"Too many errors: {errors}"
        
        print(f"\n✅ Sustained Load Test Passed: {throughput:.2f} req/sec sustained")