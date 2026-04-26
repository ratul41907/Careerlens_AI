"""
FastAPI Backend Server for CareerLens AI
Provides REST endpoints for all frontend features
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import with CORRECT actual filenames
try:
    from src.parsers.cv_parser import CVParser
    from src.parsers.jd_parser import JDParser
    from src.embeddings.embedding_engine import EmbeddingEngine          # was: SemanticEmbeddings
    from src.scoring.scoring_engine import ScoringEngine                 # was: MatchScorer
    from src.scoring.explainability import ExplainabilityEngine
    from src.generation.cv_generator import CVGenerator
    from src.guidance.interview_guidance import InterviewGuidance        # was: InterviewGuidanceSystem
    modules_available = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    modules_available = False
    
# Initialize FastAPI app
app = FastAPI(
    title="CareerLens AI API",
    description="AI-powered CV-JD matching and career guidance platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components only if modules available
if modules_available:
    try:
        cv_parser = CVParser()
        jd_parser = JDParser()
        embeddings = EmbeddingEngine()                    # was: SemanticEmbeddings()
        scorer = ScoringEngine(embeddings)                # was: MatchScorer(embeddings)
        explainer = ExplainabilityEngine()
        cv_generator = CVGenerator()
        interview_system = InterviewGuidance()            # was: InterviewGuidanceSystem()
        components_initialized = True
    except Exception as e:
        print(f"Warning: Failed to initialize components: {e}")
        components_initialized = False
else:
    components_initialized = False
# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str

class MatchResponse(BaseModel):
    overall_score: float
    breakdown: Dict[str, float]
    matched_skills: List[str]
    missing_skills: List[str]
    evidence: List[Dict]
    recommendations: List[str]

class CVGenerationRequest(BaseModel):
    personal_info: Dict[str, str]
    experience: List[Dict]
    education: List[Dict]
    skills: List[str]
    projects: Optional[List[Dict]] = None
    certifications: Optional[List[Dict]] = None

class InterviewQuestionsRequest(BaseModel):
    skills: List[str]
    num_questions: int = 10

class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "CareerLens AI API",
        "status": "running",
        "version": "1.0.0",
        "modules_loaded": modules_available,
        "components_ready": components_initialized
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if components_initialized else "degraded",
        "services": {
            "cv_parser": "ready" if components_initialized else "unavailable",
            "jd_parser": "ready" if components_initialized else "unavailable",
            "embeddings": "ready" if components_initialized else "unavailable",
            "scorer": "ready" if components_initialized else "unavailable",
            "cv_generator": "ready" if components_initialized else "unavailable",
            "interview_system": "ready" if components_initialized else "unavailable"
        }
    }

# ============================================================================
# CV-JD MATCHING ENDPOINTS
# ============================================================================

@app.post("/api/match", response_model=MatchResponse)
async def match_cv_jd(request: MatchRequest):
    """Match CV against Job Description"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Parse CV
        cv_data = cv_parser.parse_text(request.cv_text)
        
        # Parse JD
        jd_data = jd_parser.parse(request.jd_text)
        
        # Calculate match
        match_result = scorer.calculate_match(cv_data, jd_data)
        
        # Generate explanations
        evidence = explainer.generate_evidence(cv_data, jd_data, match_result)
        
        # Get recommendations
        recommendations = explainer.generate_recommendations(cv_data, jd_data, match_result)
        
        return MatchResponse(
            overall_score=match_result['overall_score'],
            breakdown=match_result['breakdown'],
            matched_skills=match_result.get('matched_skills', []),
            missing_skills=match_result.get('missing_skills', []),
            evidence=evidence,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CV GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/cv/generate")
async def generate_cv(request: CVGenerationRequest):
    """Generate ATS-optimized CV"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        cv_data = {
            'personal_info': request.personal_info,
            'experience': request.experience,
            'education': request.education,
            'skills': request.skills,
            'projects': request.projects or [],
            'certifications': request.certifications or []
        }
        
        cv_path = cv_generator.generate_cv(cv_data)
        
        return {
            "status": "success",
            "cv_path": str(cv_path),
            "message": "CV generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# INTERVIEW PREP ENDPOINTS
# ============================================================================

@app.post("/api/interview/questions")
async def get_interview_questions(request: InterviewQuestionsRequest):
    """Get personalized interview questions"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        questions = interview_system.get_recommended_questions(
            skills=request.skills,
            num_questions=request.num_questions
        )
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview/evaluate")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Evaluate interview answer"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        evaluation = interview_system.evaluate_answer(
            question=request.question,
            answer=request.answer
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary (demo data)"""
    return {
        "total_matches": 47,
        "avg_score": 73.5,
        "cvs_generated": 12,
        "interviews_practiced": 23
    }

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.post("/api/parse/cv")
async def parse_cv(file: UploadFile = File(...)):
    """Parse uploaded CV file"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            cv_data = cv_parser.parse_pdf_bytes(content)
        elif file.filename.endswith('.docx'):
            cv_data = cv_parser.parse_docx_bytes(content)
        else:
            cv_data = cv_parser.parse_text(content.decode('utf-8'))
        
        return cv_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parse/jd")
async def parse_jd(jd_text: str):
    """Parse job description text"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        jd_data = jd_parser.parse(jd_text)
        return jd_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)