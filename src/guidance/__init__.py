"""
Guidance package - Interview prep and learning pathways
"""
try:
    from .interview_guidance import InterviewGuidance
    __all__ = ['InterviewGuidance', 'LearningPathwayGenerator']
except ImportError:
    try:
        from .interview_guidance import InterviewGuidance
        from .learning_pathways import LearningPathwayGenerator
        __all__ = ['InterviewGuidance', 'LearningPathwayGenerator']
    except ImportError as e:
        print(f"Warning: guidance module import failed: {e}")


try:
    from .learning_pathways import LearningPathwayGenerator
except ImportError:
    pass