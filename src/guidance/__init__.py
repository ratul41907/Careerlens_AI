"""
Guidance package - Interview prep and learning pathways
"""
try:
    from .interview_guidance import InterviewGuidance
    __all__ = ['InterviewGuidance', 'LearningPathwayGenerator']
except ImportError:
    try:
        from .interview_guidance import InterviewGuidanceSystem
        InterviewGuidance = InterviewGuidanceSystem
        __all__ = ['InterviewGuidance', 'LearningPathwayGenerator']
    except ImportError:
        pass

try:
    from .learning_pathways import LearningPathwayGenerator
except ImportError:
    pass