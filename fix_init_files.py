"""
Fix all __init__.py files
"""
import os
from pathlib import Path

# Define all __init__.py contents
init_files = {
    'src/__init__.py': '"""\nCareerLens AI - Main package\n"""',
    
    'src/parsers/__init__.py': '''"""
Parsers module - CV and JD parsing
"""
from .cv_parser import CVParser
from .jd_parser import JDParser

__all__ = ['CVParser', 'JDParser']''',
    
    'src/embeddings/__init__.py': '''"""
Embeddings module - Semantic embeddings generation
"""
from .embedding_engine import EmbeddingEngine

__all__ = ['EmbeddingEngine']''',
    
    'src/scoring/__init__.py': '''"""
Scoring module - Match scoring and explainability
"""
from .scoring_engine import ScoringEngine
from .explainability import ExplainabilityEngine
from .counterfactual import CounterfactualSimulator

__all__ = ['ScoringEngine', 'ExplainabilityEngine', 'CounterfactualSimulator']''',
    
    'src/generation/__init__.py': '''"""
Generation module - CV generation
"""
from .cv_generator import CVGenerator

__all__ = ['CVGenerator']''',
    
    'src/validation/__init__.py': '''"""
Validation module - CV analysis and eligibility validation
"""
from .cv_analyzer import CVAnalyzer
from .eligibility_validator import EligibilityValidator

__all__ = ['CVAnalyzer', 'EligibilityValidator']''',
    
    'src/guidance/__init__.py': '''"""
Guidance module - Learning pathways and interview preparation
"""
from .learning_pathways import LearningPathwayGenerator
from .interview_guidance import InterviewGuidanceSystem

__all__ = ['LearningPathwayGenerator', 'InterviewGuidanceSystem']''',
    
    'src/api/__init__.py': '"""\nAPI module - FastAPI backend (optional)\n"""',
    
    'streamlit_app/__init__.py': '"""\nStreamlit application\n"""',
    
    'streamlit_app/utils/__init__.py': '"""\nStreamlit utilities\n"""',
    
    'streamlit_app/pages/__init__.py': '"""\nStreamlit pages\n"""',
}

# Update all files
project_root = Path(__file__).parent

for file_path, content in init_files.items():
    full_path = project_root / file_path
    
    print(f"Updating: {file_path}")
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Updated!")

print("\n🎉 All __init__.py files updated successfully!")