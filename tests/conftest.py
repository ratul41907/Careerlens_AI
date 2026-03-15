"""
Pytest Configuration and Shared Fixtures
"""
import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components with error handling
try:
    from src.parsers.cv_parser import CVParser
except ImportError as e:
    print(f"Warning: Could not import CVParser: {e}")
    CVParser = None

try:
    from src.parsers.jd_parser import JDParser
except ImportError as e:
    print(f"Warning: Could not import JDParser: {e}")
    JDParser = None

try:
    from src.embeddings.embedding_engine import EmbeddingEngine
except ImportError as e:
    print(f"Warning: Could not import EmbeddingEngine: {e}")
    EmbeddingEngine = None

try:
    from src.scoring.scoring_engine import ScoringEngine
except ImportError as e:
    print(f"Warning: Could not import ScoringEngine: {e}")
    ScoringEngine = None

try:
    from src.scoring.explainability import ExplainabilityEngine
except ImportError as e:
    print(f"Warning: Could not import ExplainabilityEngine: {e}")
    ExplainabilityEngine = None

try:
    from src.scoring.counterfactual import CounterfactualSimulator
except ImportError as e:
    print(f"Warning: Could not import CounterfactualSimulator: {e}")
    CounterfactualSimulator = None

try:
    from src.validation.cv_analyzer import CVAnalyzer
except ImportError as e:
    print(f"Warning: Could not import CVAnalyzer: {e}")
    CVAnalyzer = None

try:
    # Try both possible class names
    from src.guidance.interview_guidance import InterviewGuidance
except ImportError:
    try:
        from src.guidance.interview_guidance import InterviewGuidanceSystem as InterviewGuidance
    except ImportError as e:
        print(f"Warning: Could not import InterviewGuidance: {e}")
        InterviewGuidance = None

try:
    from src.guidance.learning_pathways import LearningPathwayGenerator
except ImportError as e:
    print(f"Warning: Could not import LearningPathwayGenerator: {e}")
    LearningPathwayGenerator = None


@pytest.fixture
def sample_cv_text():
    """Sample CV text for testing"""
    return """John Doe
john.doe@email.com | +1-234-567-8900

PROFESSIONAL SUMMARY
Senior Software Engineer with 5+ years of experience in Python, FastAPI, and Docker.

EXPERIENCE
Senior Software Engineer | Tech Corp (2020-Present)
- Developed microservices using Python and FastAPI
- Implemented Docker containerization for 20+ services
- Led team of 5 developers

Software Engineer | StartupXYZ (2018-2020)
- Built REST APIs with Python
- Worked with PostgreSQL and MongoDB

SKILLS
Python, JavaScript, FastAPI, Django, Docker, Kubernetes, PostgreSQL, MongoDB, React, AWS

EDUCATION
Bachelor of Science in Computer Science | MIT (2018)
GPA: 3.8/4.0"""


@pytest.fixture
def sample_jd_text():
    """Sample job description for testing"""
    return """Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience.

Required Skills:
- Python (Django/FastAPI)
- PostgreSQL or MySQL
- Docker & Kubernetes
- AWS cloud services
- REST API design

Preferred Skills:
- React.js
- Redis caching
- CI/CD pipelines
- Microservices architecture

Responsibilities:
- Design and implement scalable backend systems
- Mentor junior developers
- Participate in architecture decisions

Requirements:
- 5+ years of backend development
- Bachelor's degree in Computer Science
- Strong problem-solving skills

Benefits:
- Competitive salary ($120K-$160K)
- Remote work options
- Health insurance"""


@pytest.fixture
def sample_cv_dict(sample_cv_text):
    """Sample CV as dictionary"""
    return {
        'text': sample_cv_text,
        'sections': {
            'name': 'John Doe',
            'email': 'john.doe@email.com',
            'phone': '+1-234-567-8900',
            'skills': ['Python', 'JavaScript', 'FastAPI', 'Django', 'Docker', 'Kubernetes', 
                      'PostgreSQL', 'MongoDB', 'React', 'AWS'],
            'experience': '5+ years',
            'work_history': [
                'Senior Software Engineer: Tech Corp (2020-Present)',
                'Software Engineer: StartupXYZ (2018-2020)'
            ],
            'education': ['Bachelor of Science in Computer Science: MIT (2018)']
        }
    }


@pytest.fixture
def sample_jd_dict(sample_jd_text):
    """Sample JD as dictionary"""
    return {
        'text': sample_jd_text,
        'sections': {
            'job_title': 'Senior Backend Engineer',
            'required_skills': ['Python', 'Django', 'FastAPI', 'PostgreSQL', 'MySQL', 
                               'Docker', 'Kubernetes', 'AWS', 'REST API'],
            'preferred_skills': ['React.js', 'Redis', 'CI/CD', 'Microservices'],
            'experience': {
                'text': '5+ years of backend development',
                'years': '5',
                'min_years': 5,
                'max_years': 5
            },
            'education': 'Bachelor\'s degree in Computer Science',
            'responsibilities': [
                'Design and implement scalable backend systems',
                'Mentor junior developers',
                'Participate in architecture decisions'
            ]
        }
    }


@pytest.fixture
def cv_parser():
    """CVParser instance"""
    if CVParser is None:
        pytest.skip("CVParser not available")
    return CVParser()


@pytest.fixture
def jd_parser():
    """JDParser instance"""
    if JDParser is None:
        pytest.skip("JDParser not available")
    return JDParser()


@pytest.fixture
def embedding_engine():
    """EmbeddingEngine instance"""
    if EmbeddingEngine is None:
        pytest.skip("EmbeddingEngine not available")
    return EmbeddingEngine()


@pytest.fixture
def scoring_engine(embedding_engine):
    """ScoringEngine instance"""
    if ScoringEngine is None:
        pytest.skip("ScoringEngine not available")
    return ScoringEngine(embedding_engine)


@pytest.fixture
def explainability_engine():
    """ExplainabilityEngine instance"""
    if ExplainabilityEngine is None:
        pytest.skip("ExplainabilityEngine not available")
    return ExplainabilityEngine()


@pytest.fixture
def counterfactual_simulator(scoring_engine):
    """CounterfactualSimulator instance"""
    if CounterfactualSimulator is None:
        pytest.skip("CounterfactualSimulator not available")
    return CounterfactualSimulator(scoring_engine)


@pytest.fixture
def cv_analyzer():
    """CVAnalyzer instance"""
    if CVAnalyzer is None:
        pytest.skip("CVAnalyzer not available")
    return CVAnalyzer()


@pytest.fixture
def interview_guidance():
    """InterviewGuidance instance"""
    if InterviewGuidance is None:
        pytest.skip("InterviewGuidance not available")
    return InterviewGuidance()


@pytest.fixture
def learning_pathway_generator():
    """LearningPathwayGenerator instance"""
    if LearningPathwayGenerator is None:
        pytest.skip("LearningPathwayGenerator not available")
    return LearningPathwayGenerator()


@pytest.fixture
def temp_cv_file(sample_cv_text):
    """Create temporary CV text file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_cv_text)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "model": "gemma2:2b",
        "response": "This is a mock LLM response for testing purposes."
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require Ollama LLM"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )