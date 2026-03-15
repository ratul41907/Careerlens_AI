"""
Load Test Runner Script
Simulate concurrent users
"""
import sys
from pathlib import Path
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine


class LoadTester:
    """Load testing utility"""
    
    def __init__(self):
        self.cv_parser = CVParser()
        self.jd_parser = JDParser()
        self.embedding_engine = EmbeddingEngine()
        self.scoring_engine = ScoringEngine(self.embedding_engine)
    
    def simulate_user_workflow(self, user_id):
        """Simulate a single user's complete workflow"""
        cv_text = f"""User {user_id} CV
Senior Developer
Skills: Python, Docker, FastAPI, AWS
Experience: 5 years"""
        
        jd_text = f"""Job {user_id}
Looking for Python developer
Required: Python, Docker, FastAPI"""
        
        start = time.time()
        
        try:
            # Parse JD
            jd_data = self.jd_parser.parse(jd_text)
            
            # Parse CV (simulate file upload)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(cv_text)
                cv_path = f.name
            
            try:
                cv_data = self.cv_parser.parse(cv_path)
            finally:
                os.remove(cv_path)
            
            # Compute match
            result = self.scoring_engine.compute_match_score(cv_data, jd_data)
            
            elapsed = time.time() - start
            
            return {
                'user_id': user_id,
                'success': True,
                'time': elapsed,
                'score': result['overall_score'] if result else 0
            }
        
        except Exception as e:
            elapsed = time.time() - start
            return {
                'user_id': user_id,
                'success': False,
                'time': elapsed,
                'error': str(e)
            }
    
    def run_load_test(self, num_users=10, max_workers=5):
        """Run load test with concurrent users"""
        print("\n" + "=" * 70)
        print(f"LOAD TEST: {num_users} Concurrent Users")
        print("=" * 70)
        
        print(f"\n⚡ Simulating {num_users} users with {max_workers} max concurrent...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.simulate_user_workflow, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if successful:
            times = [r['time'] for r in successful]
            scores = [r['score'] for r in successful]
            
            print(f"\n📊 LOAD TEST RESULTS:")
            print("-" * 70)
            print(f"Total Users: {num_users}")
            print(f"Successful: {len(successful)}/{num_users} ({len(successful)/num_users*100:.1f}%)")
            print(f"Failed: {len(failed)}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Throughput: {len(successful)/total_time:.2f} users/sec")
            print(f"\nResponse Times:")
            print(f"  Average: {statistics.mean(times):.2f}s")
            print(f"  Median: {statistics.median(times):.2f}s")
            print(f"  Min: {min(times):.2f}s")
            print(f"  Max: {max(times):.2f}s")
            print(f"\nMatch Scores:")
            print(f"  Average: {statistics.mean(scores)*100:.1f}%")
        
        if failed:
            print(f"\n❌ FAILURES:")
            for f in failed:
                print(f"  User {f['user_id']}: {f.get('error', 'Unknown error')}")
        
        return {
            'total_users': num_users,
            'successful': len(successful),
            'failed': len(failed),
            'total_time': total_time,
            'throughput': len(successful) / total_time if total_time > 0 else 0
        }
    
    def run_scalability_test(self):
        """Test scalability with increasing load"""
        print("\n" + "=" * 70)
        print("SCALABILITY TEST: Increasing Load")
        print("=" * 70)
        
        load_levels = [5, 10, 20, 30]
        results = {}
        
        for load in load_levels:
            print(f"\n{'='*70}")
            result = self.run_load_test(num_users=load, max_workers=10)
            results[load] = result
            time.sleep(2)  # Brief pause between tests
        
        # Summary
        print("\n" + "=" * 70)
        print("SCALABILITY ANALYSIS")
        print("=" * 70)
        print(f"\n{'Users':<10} {'Success Rate':<15} {'Throughput':<15} {'Total Time'}")
        print("-" * 60)
        
        for load, data in results.items():
            success_rate = (data['successful'] / data['total_users']) * 100
            print(f"{load:<10} {success_rate:<15.1f}% {data['throughput']:<15.2f} {data['total_time']:.2f}s")
        
        print("\n" + "=" * 70)


def main():
    """Run load tests"""
    print("\n" + "=" * 70)
    print("CAREERLENS AI - LOAD TESTING")
    print("=" * 70)
    
    tester = LoadTester()
    
    # Run individual load test
    print("\n1. Running basic load test (10 users)...")
    tester.run_load_test(num_users=10, max_workers=5)
    
    # Run scalability test
    print("\n2. Running scalability test...")
    tester.run_scalability_test()
    
    print("\n✅ Load testing complete!")


if __name__ == "__main__":
    main()