"""
Benchmark Runner Script
Automated performance benchmarking
"""
import sys
import time
import statistics
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.validation.cv_analyzer import CVAnalyzer
from src.guidance.interview_guidance import InterviewGuidance
from src.guidance.learning_pathways import LearningPathwayGenerator
import tempfile
import os


class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_cv_parsing(self, iterations=10):
        """Benchmark CV parsing"""
        print("\n" + "=" * 70)
        print("BENCHMARK: CV Parsing")
        print("=" * 70)
        
        parser = CVParser()
        
        sample_cv = """
JOHN DOE
Senior Software Engineer
john.doe@email.com | +1-555-0123

PROFESSIONAL SUMMARY
Experienced software engineer with 7+ years in full-stack development.

EXPERIENCE
Senior Software Engineer | TechCorp | 2020-Present
- Developed microservices using Python and FastAPI
- Implemented Docker containerization for 20+ services
- Led team of 5 developers
- Reduced deployment time by 60% with CI/CD

Software Engineer | StartupXYZ | 2018-2020
- Built REST APIs handling 500K+ requests/day
- Optimized database queries improving performance by 75%
- Worked with PostgreSQL, MongoDB, Redis

SKILLS
Languages: Python, JavaScript, TypeScript, SQL
Frameworks: FastAPI, Django, React, Node.js
Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD
Databases: PostgreSQL, MongoDB, Redis

EDUCATION
Bachelor of Science in Computer Science | MIT | 2018
GPA: 3.8/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect (2022)
- Certified Kubernetes Administrator (2023)
"""
        
        times = []
        
        for i in range(iterations):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_cv)
                cv_path = f.name
            
            try:
                start = time.time()
                result = parser.parse(cv_path)
                elapsed = time.time() - start
                times.append(elapsed)
            finally:
                os.remove(cv_path)
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        self.results['cv_parsing'] = {
            'avg': avg_time,
            'min': min_time,
            'max': max_time,
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")
        
        return avg_time
    
    def benchmark_jd_parsing(self, iterations=10):
        """Benchmark JD parsing"""
        print("\n" + "=" * 70)
        print("BENCHMARK: JD Parsing")
        print("=" * 70)
        
        parser = JDParser()
        
        sample_jd = """
Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience.

REQUIRED SKILLS:
- Python (Django/FastAPI)
- PostgreSQL or MySQL
- Docker & Kubernetes
- AWS cloud services
- REST API design
- Microservices architecture

PREFERRED SKILLS:
- React.js
- Redis caching
- CI/CD pipelines
- GraphQL
- Event-driven architecture

RESPONSIBILITIES:
- Design and implement scalable backend systems
- Mentor junior developers
- Participate in architecture decisions
- Code reviews and best practices

REQUIREMENTS:
- 5+ years of backend development
- Bachelor's degree in Computer Science
- Strong problem-solving skills
- Excellent communication

BENEFITS:
- Competitive salary ($140K-$180K)
- Remote work options
- Health insurance
- 401(k) matching
"""
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            result = parser.parse(sample_jd)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        self.results['jd_parsing'] = {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.3f}s")
        
        return avg_time
    
    def benchmark_scoring(self, iterations=20):
        """Benchmark scoring engine"""
        print("\n" + "=" * 70)
        print("BENCHMARK: Scoring Engine")
        print("=" * 70)
        
        embedding_engine = EmbeddingEngine()
        scoring_engine = ScoringEngine(embedding_engine)
        
        cv_data = {
            'text': 'Senior Python Developer with FastAPI, Docker, AWS experience',
            'sections': {
                'skills': ['Python', 'FastAPI', 'Django', 'Docker', 'Kubernetes', 
                          'AWS', 'PostgreSQL', 'Redis', 'React', 'TypeScript'],
                'experience': '7 years'
            }
        }
        
        jd_data = {
            'text': 'Looking for Senior Backend Engineer',
            'sections': {
                'required_skills': ['Python', 'FastAPI', 'Docker', 'PostgreSQL', 'AWS'],
                'preferred_skills': ['Kubernetes', 'Redis', 'React'],
                'experience': {'years': '5', 'min_years': 5}
            }
        }
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        self.results['scoring'] = {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.3f}s")
        
        return avg_time
    
    def benchmark_cv_analysis(self, iterations=5):
        """Benchmark CV analysis"""
        print("\n" + "=" * 70)
        print("BENCHMARK: CV Analysis (LLM)")
        print("=" * 70)
        
        analyzer = CVAnalyzer()
        
        cv_data = {
            'text': """Senior Developer
            Experience: Worked on projects, helped team
            Skills: Python, Docker"""
        }
        
        times = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...", end='')
            start = time.time()
            result = analyzer.analyze_cv(cv_data)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f" {elapsed:.2f}s")
        
        avg_time = statistics.mean(times)
        
        self.results['cv_analysis'] = {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.2f}s")
        
        return avg_time
    
    def benchmark_interview_questions(self, iterations=5):
        """Benchmark interview question generation"""
        print("\n" + "=" * 70)
        print("BENCHMARK: Interview Questions (LLM)")
        print("=" * 70)
        
        guidance = InterviewGuidance()
        
        times = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...", end='')
            start = time.time()
            result = guidance.generate_questions(
                skills=['Python', 'Docker', 'AWS'],
                num_questions=5
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f" {elapsed:.2f}s")
        
        avg_time = statistics.mean(times)
        
        self.results['interview_questions'] = {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.2f}s")
        
        return avg_time
    
    def benchmark_learning_pathway(self, iterations=3):
        """Benchmark learning pathway generation"""
        print("\n" + "=" * 70)
        print("BENCHMARK: Learning Pathway (LLM)")
        print("=" * 70)
        
        generator = LearningPathwayGenerator()
        
        times = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...", end='')
            start = time.time()
            result = generator.generate_pathway(
                skill_gaps=['Docker', 'Kubernetes'],
                jd_data={'text': 'Need Docker and Kubernetes'},
                num_days=7
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f" {elapsed:.2f}s")
        
        avg_time = statistics.mean(times)
        
        self.results['learning_pathway'] = {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'iterations': iterations
        }
        
        print(f"Average: {avg_time:.2f}s")
        
        return avg_time
    
    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 70)
        
        print(f"\n{'Component':<25} {'Avg Time':<15} {'Min':<10} {'Max':<10} {'Iterations'}")
        print("-" * 70)
        
        for component, data in self.results.items():
            comp_name = component.replace('_', ' ').title()
            print(f"{comp_name:<25} {data['avg']:<15.3f} {data['min']:<10.3f} {data['max']:<10.3f} {data['iterations']}")
        
        # Calculate total time for complete workflow
        if 'cv_parsing' in self.results and 'jd_parsing' in self.results and 'scoring' in self.results:
            workflow_time = (
                self.results['cv_parsing']['avg'] +
                self.results['jd_parsing']['avg'] +
                self.results['scoring']['avg']
            )
            
            print(f"\n{'COMPLETE WORKFLOW':<25} {workflow_time:<15.3f}")
        
        print("\n" + "=" * 70)
        
        # Performance grades
        print("\nPERFORMANCE GRADES:")
        print("-" * 70)
        
        grades = {
            'cv_parsing': ('CV Parsing', 3.0),
            'jd_parsing': ('JD Parsing', 2.0),
            'scoring': ('Scoring', 2.0),
            'cv_analysis': ('CV Analysis', 15.0),
            'interview_questions': ('Interview Questions', 15.0),
            'learning_pathway': ('Learning Pathway', 25.0)
        }
        
        for key, (name, target) in grades.items():
            if key in self.results:
                actual = self.results[key]['avg']
                if actual <= target:
                    grade = "✅ EXCELLENT"
                elif actual <= target * 1.5:
                    grade = "🟡 GOOD"
                else:
                    grade = "⚠️  NEEDS OPTIMIZATION"
                
                print(f"{name:<25} {actual:>8.2f}s / {target:>6.1f}s target → {grade}")
        
        print("\n" + "=" * 70)


def main():
    """Run all benchmarks"""
    print("\n" + "=" * 70)
    print("CAREERLENS AI - PERFORMANCE BENCHMARKING")
    print("=" * 70)
    print("\nStarting comprehensive performance benchmarks...")
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    benchmark.benchmark_cv_parsing()
    benchmark.benchmark_jd_parsing()
    benchmark.benchmark_scoring()
    benchmark.benchmark_cv_analysis()
    benchmark.benchmark_interview_questions()
    benchmark.benchmark_learning_pathway()
    
    # Generate report
    benchmark.generate_report()
    
    print("\n✅ Benchmarking complete!")


if __name__ == "__main__":
    main()