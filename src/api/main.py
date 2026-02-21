"""
FastAPI Main Application
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile
import os
from loguru import logger

from src.parsers.cv_parser import CVParser
from src.parsers.jd_parser import JDParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.scoring_engine import ScoringEngine
from src.scoring.explainability import ExplainabilityEngine

# Initialize FastAPI app
app = FastAPI(
    title="CareerLens AI API",
    description="Explainable CV-Job Matching & Career Guidance System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines (singleton pattern - loaded once)
cv_parser = CVParser()
jd_parser = JDParser()
embedding_engine = EmbeddingEngine()
scoring_engine = ScoringEngine(embedding_engine)
explainability_engine = ExplainabilityEngine(embedding_engine)

logger.info("CareerLens AI API initialized successfully")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


class CVParseRequest(BaseModel):
    text: str = Field(..., description="CV text content")


class JDParseRequest(BaseModel):
    text: str = Field(..., description="Job description text")


class MatchRequest(BaseModel):
    cv_text: str = Field(..., description="CV text content")
    jd_text: str = Field(..., description="Job description text")
    include_explainability: bool = Field(
        default=True,
        description="Include detailed explanations and evidence"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API information"""
    return {
        "status": "active",
        "version": "1.0.0",
        "message": "CareerLens AI API is running. Visit /docs for documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "message": "All systems operational"
    }


@app.post("/api/parse-cv")
async def parse_cv_text(request: CVParseRequest):
    """
    Parse CV from text
    
    Args:
        request: CVParseRequest with CV text
        
    Returns:
        Parsed CV data with sections
    """
    try:
        logger.info("Parsing CV from text")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(request.text)
            tmp_path = tmp.name
        
        # Parse (using text content, not file)
        cv_data = {
            'success': True,
            'text': request.text,
            'sections': cv_parser._segment_sections(request.text)
        }
        
        # Cleanup
        os.unlink(tmp_path)
        
        logger.info(f"CV parsed successfully: {len(cv_data['sections'])} sections")
        return cv_data
        
    except Exception as e:
        logger.error(f"Error parsing CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse-cv/file")
async def parse_cv_file(file: UploadFile = File(...)):
    """
    Parse CV from uploaded file (PDF/DOCX)
    
    Args:
        file: Uploaded CV file
        
    Returns:
        Parsed CV data with sections
    """
    try:
        logger.info(f"Parsing CV file: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Parse file
        cv_data = cv_parser.parse(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        if not cv_data['success']:
            raise HTTPException(status_code=400, detail=cv_data['error'])
        
        logger.info(f"CV file parsed successfully: {len(cv_data['sections'])} sections")
        return cv_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing CV file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse-jd")
async def parse_jd(request: JDParseRequest):
    """
    Parse job description
    
    Args:
        request: JDParseRequest with JD text
        
    Returns:
        Parsed JD data with skills, experience, education
    """
    try:
        logger.info("Parsing job description")
        
        jd_data = jd_parser.parse(request.text)
        
        if not jd_data['success']:
            raise HTTPException(status_code=400, detail=jd_data['error'])
        
        logger.info(f"JD parsed: {len(jd_data['required_skills'])} required, {len(jd_data['preferred_skills'])} preferred")
        return jd_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing JD: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match")
async def match_cv_jd(request: MatchRequest):
    """
    Complete CV-JD matching with scoring and explainability
    
    Args:
        request: MatchRequest with CV and JD text
        
    Returns:
        Match score, breakdown, and explanations
    """
    try:
        logger.info("Processing CV-JD match request")
        
        # Parse CV
        cv_data = {
            'text': request.cv_text,
            'sections': cv_parser._segment_sections(request.cv_text)
        }
        
        # Parse JD
        jd_data = jd_parser.parse(request.jd_text)
        
        if not jd_data['success']:
            raise HTTPException(status_code=400, detail=f"JD parsing failed: {jd_data['error']}")
        
        # Compute match score
        match_result = scoring_engine.compute_match_score(cv_data, jd_data)
        
        # Add explainability if requested
        if request.include_explainability:
            explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
            match_result['explainability'] = explanation
        
        logger.info(f"Match computed: {match_result['overall_percentage']}")
        
        return {
            'success': True,
            'match_result': match_result,
            'cv_sections_found': list(cv_data['sections'].keys()),
            'jd_requirements': {
                'required_skills': len(jd_data['required_skills']),
                'preferred_skills': len(jd_data['preferred_skills']),
                'experience': jd_data.get('experience_years')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Form

@app.post("/api/match/file")
async def match_cv_jd_file(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """
    CV-JD matching with CV file upload
    
    Args:
        cv_file: Uploaded CV file
        jd_text: Job description text
        
    Returns:
        Match result with explainability
    """
    try:
        logger.info(f"Processing match with CV file: {cv_file.filename}")
        
        # Parse CV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(cv_file.filename)[1]) as tmp:
            content = await cv_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        cv_data = cv_parser.parse(tmp_path)
        os.unlink(tmp_path)
        
        if not cv_data['success']:
            raise HTTPException(status_code=400, detail=f"CV parsing failed: {cv_data['error']}")
        
        # Parse JD
        jd_data = jd_parser.parse(jd_text)
        
        if not jd_data['success']:
            raise HTTPException(status_code=400, detail=f"JD parsing failed: {jd_data['error']}")
        
        # Compute match
        match_result = scoring_engine.compute_match_score(cv_data, jd_data)
        explanation = explainability_engine.explain_match(cv_data, jd_data, match_result)
        match_result['explainability'] = explanation
        
        logger.info(f"Match computed: {match_result['overall_percentage']}")
        
        return {
            'success': True,
            'match_result': match_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": "Visit /docs for API documentation"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc)
        }
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    logger.info("=" * 70)
    logger.info("CareerLens AI API Starting...")
    logger.info("=" * 70)
    logger.info("✅ CV Parser ready")
    logger.info("✅ JD Parser ready")
    logger.info("✅ Embedding Engine ready")
    logger.info("✅ Scoring Engine ready")
    logger.info("✅ Explainability Engine ready")
    logger.info("=" * 70)
    logger.info("🚀 API is ready to accept requests!")
    logger.info("📖 Documentation: http://localhost:8000/docs")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    logger.info("CareerLens AI API shutting down...")