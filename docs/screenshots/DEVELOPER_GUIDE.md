# 🛠️ CareerLens AI - Developer Guide

**Technical documentation for developers, contributors, and maintainers**

---

## 📑 Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Backend Components](#backend-components)
4. [Frontend Components](#frontend-components)
5. [AI/ML Models](#aiml-models)
6. [API Documentation](#api-documentation)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Development Setup

### Prerequisites

**Required:**

- Python 3.10 or higher
- pip (Python package manager)
- Git
- 8GB+ RAM (for ML models)

**Optional:**

- VS Code (recommended IDE)
- Postman (for API testing)
- Docker (for containerization)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/careerlens-ai.git
cd careerlens-ai
```

#### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install all packages
pip install -r requirements.txt

# Verify installation
pip list
```

#### 4. Download ML Models

Models are downloaded automatically on first run, but you can pre-download:

```python
# Run this in Python interpreter
from sentence_transformers import SentenceTransformer

# Download embeddings model (~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model downloaded successfully!")
```

#### 5. Verify Setup

```bash
# Run tests
python -m pytest tests/

# Start application
streamlit run streamlit_app/Home.py
```

Open `http://localhost:8501` - you should see the home page.

---

## 🏗️ Project Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│           Frontend (Streamlit)              │
│  ┌────────┬──────────┬────────┬──────────┐ │
│  │  Home  │ Matcher  │  Gen   │ Interview│ │
│  └────────┴──────────┴────────┴──────────┘ │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Session Manager (State)             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│            Backend Modules                  │
│  ┌──────────┬──────────┬──────────────────┐ │
│  │ Parsers  │ Embeddings│  Scoring       │ │
│  ├──────────┼──────────┼──────────────────┤ │
│  │Generation│ Validation│  Guidance      │ │
│  └──────────┴──────────┴──────────────────┘ │
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          AI/ML Models                       │
│  • SentenceTransformers (Embeddings)       │
│  • Ollama LLM (Text Generation)            │
│  • spaCy (NLP)                              │
└─────────────────────────────────────────────┘
```

### Directory Structure

```
careerlens-ai/
│
├── src/                      # Backend source code
│   ├── parsers/              # Document parsing
│   │   ├── cv_parser.py      # CV extraction
│   │   └── jd_parser.py      # JD extraction
│   │
│   ├── embeddings/           # Semantic embeddings
│   │   └── embedding_engine.py
│   │
│   ├── scoring/              # Match scoring
│   │   ├── scoring_engine.py # Core scoring logic
│   │   ├── explainability.py # Evidence generation
│   │   └── counterfactual.py # What-if analysis
│   │
│   ├── generation/           # CV generation
│   │   └── cv_generator.py   # DOCX creation
│   │
│   ├── validation/           # Validators
│   │   ├── cv_analyzer.py    # CV quality check
│   │   └── academic_validator.py # Eligibility
│   │
│   ├── guidance/             # Career guidance
│   │   ├── learning_pathways.py
│   │   └── interview_guidance.py
│   │
│   └── api/                  # REST API (optional)
│       └── app.py            # FastAPI endpoints
│
├── streamlit_app/            # Frontend
│   ├── Home.py               # Landing page
│   │
│   ├── pages/                # App pages
│   │   ├── 1_📊_CV_Matcher.py
│   │   ├── 2_📝_CV_Generator.py
│   │   ├── 3_🎓_Interview_Prep.py
│   │   └── 4_📈_Analytics.py
│   │
│   └── utils/                # Utilities
│       ├── session_manager.py # State management
│       └── api_client.py      # Backend client
│
├── tests/                    # Test files
│   ├── test_parsers.py
│   ├── test_scoring.py
│   ├── test_generation.py
│   └── sample_data/
│
├── docs/                     # Documentation
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   └── screenshots/
│
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project overview
```

---

## 🔧 Backend Components

### 1. CV Parser (`src/parsers/cv_parser.py`)

**Purpose:** Extract structured data from CV files

**Key Functions:**

```python
class CVParser:
    def parse_pdf(self, pdf_path: str) -> Dict:
        """Parse PDF CV file"""

    def parse_docx(self, docx_path: str) -> Dict:
        """Parse DOCX CV file"""

    def parse_text(self, text: str) -> Dict:
        """Parse plain text CV"""

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using NLP"""

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience"""
```

**Output Structure:**

```python
{
    "personal_info": {
        "name": "John Doe",
        "email": "john@email.com",
        "phone": "+1-555-0123"
    },
    "experience": [
        {
            "title": "Senior Engineer",
            "company": "TechCorp",
            "duration": "2020-Present",
            "description": "..."
        }
    ],
    "education": [...],
    "skills": ["Python", "FastAPI", ...],
    "raw_text": "..."
}
```

### 2. JD Parser (`src/parsers/jd_parser.py`)

**Purpose:** Extract requirements from job descriptions

**Key Functions:**

```python
class JDParser:
    def parse(self, jd_text: str) -> Dict:
        """Parse job description"""

    def extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills"""

    def extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred skills"""

    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience required"""
```

**Output Structure:**

```python
{
    "required_skills": ["Python", "Docker", ...],
    "preferred_skills": ["GraphQL", "MongoDB", ...],
    "experience_required": 5,
    "education_required": "Bachelor's",
    "raw_text": "..."
}
```

### 3. Embedding Engine (`src/embeddings/embedding_engine.py`)

**Purpose:** Generate semantic embeddings

**Key Functions:**

```python
class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text: str) -> np.ndarray:
        """Generate 384-dim embedding"""

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encoding for efficiency"""

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate similarity score"""
```

**Technical Details:**

- Model: `all-MiniLM-L6-v2`
- Dimensions: 384
- Max sequence length: 256 tokens
- Processing time: ~50ms per sentence

### 4. Scoring Engine (`src/scoring/scoring_engine.py`)

**Purpose:** Calculate CV-JD match scores

**Key Functions:**

```python
class MatchScorer:
    def calculate_match(self, cv_data: Dict, jd_data: Dict) -> Dict:
        """Calculate overall match score"""

    def score_skills(self, cv_skills: List, jd_skills: List) -> float:
        """Score skill match using embeddings"""

    def score_experience(self, cv_exp: int, jd_exp: int) -> float:
        """Score experience match"""
```

**Scoring Algorithm:**

```python
Overall Score = (
    0.60 * Required Skills Score +
    0.25 * Preferred Skills Score +
    0.15 * Experience Score
)
```

**Skill Matching:**

```python
# Semantic matching using cosine similarity
for cv_skill in cv_skills:
    for jd_skill in jd_skills:
        similarity = cosine_sim(
            embed(cv_skill),
            embed(jd_skill)
        )
        if similarity > 0.7:  # Threshold
            matched.append((cv_skill, jd_skill, similarity))
```

### 5. Explainability Engine (`src/scoring/explainability.py`)

**Purpose:** Generate evidence and recommendations

**Key Functions:**

```python
class ExplainabilityEngine:
    def generate_evidence(self, cv_data: Dict, jd_data: Dict,
                         match_result: Dict) -> List[Dict]:
        """Find supporting evidence in CV"""

    def generate_recommendations(self, cv_data: Dict, jd_data: Dict,
                                match_result: Dict) -> List[str]:
        """Generate actionable recommendations"""
```

### 6. CV Generator (`src/generation/cv_generator.py`)

**Purpose:** Create ATS-optimized CVs

**Key Functions:**

```python
class CVGenerator:
    def generate_cv(self, cv_data: Dict, output_path: str = None) -> str:
        """Generate DOCX CV"""

    def create_header(self, doc: Document, personal_info: Dict):
        """Add header section"""

    def create_experience(self, doc: Document, experience: List[Dict]):
        """Add experience section"""

    def create_skills(self, doc: Document, skills: List[str]):
        """Add skills section"""
```

**Template Features:**

- ATS-friendly layout
- Professional fonts (Calibri, Arial)
- Proper heading hierarchy
- Bullet point formatting
- White space optimization

---

## 🎨 Frontend Components

### Streamlit Pages

#### 1. Home.py

**Purpose:** Landing page with overview

**Key Features:**

- Animated gradient text
- Feature cards with hover effects
- Quick start buttons
- Platform statistics

**Session State:**

```python
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = None
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
```

#### 2. CV_Matcher.py

**Purpose:** CV-JD matching interface

**Flow:**

```python
1. File upload / text paste
2. Parse CV and JD
3. Calculate embeddings
4. Compute match score
5. Display results
6. Export JSON
```

**State Management:**

```python
if st.button("Calculate Match"):
    with st.spinner("Processing..."):
        cv_data = parser.parse(cv_file)
        jd_data = parser.parse(jd_text)
        match = scorer.calculate_match(cv_data, jd_data)
        st.session_state.last_match = match
```

#### 3. CV_Generator.py

**Purpose:** CV creation interface

**Three Modes:**

1. Manual Entry (tabbed forms)
2. Auto-Generate from JD
3. Document Upload (OCR)

**Generation:**

```python
if st.button("Generate CV"):
    cv_data = collect_form_data()
    cv_path = generator.generate_cv(cv_data)
    st.download_button("Download DOCX", cv_path)
```

#### 4. Interview_Prep.py

**Purpose:** Interview practice

**Two Features:**

1. Question Generation
2. Answer Evaluation

**Integration:**

```python
from src.guidance.interview_guidance import InterviewGuidanceSystem

system = InterviewGuidanceSystem()
questions = system.get_recommended_questions(skills, num=10)
evaluation = system.evaluate_answer(question, answer)
```

#### 5. Analytics.py

**Purpose:** Progress tracking

**Charts:**

- Match Score Trends (Plotly line chart)
- Skill Gaps (Plotly bar chart)
- CV Generation Stats (Plotly pie charts)
- Platform Usage (Plotly grouped bars)

**Demo Data:**

```python
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = {
        'total_matches': 0,
        'avg_score': 87.0,
        'cvs_generated': 0,
        'interviews_practiced': 0,
        'match_history': [],
        'skill_gaps': {}
    }
```

### Session Manager (`utils/session_manager.py`)

**Purpose:** Cross-page state management

**Key Functions:**

```python
class SessionManager:
    @staticmethod
    def init():
        """Initialize default session state"""

    @staticmethod
    def save_match_result(result: Dict):
        """Save match result and update analytics"""

    @staticmethod
    def save_generated_cv(cv_data: Dict, cv_path: str):
        """Track CV generation"""

    @staticmethod
    def get_analytics():
        """Get current analytics data"""
```

---

## 🤖 AI/ML Models

### 1. Sentence Embeddings

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Specifications:**

- Architecture: MiniLM (distilled BERT)
- Dimensions: 384
- Max tokens: 256
- Size: ~80MB
- Speed: ~50ms per sentence (CPU)

**Usage:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Python FastAPI developer")
# Returns: np.array of shape (384,)
```

**Why This Model:**

- ✅ Balance of speed and accuracy
- ✅ Lightweight (good for local deployment)
- ✅ Strong semantic understanding
- ✅ Well-suited for short texts (skills, job titles)

### 2. Ollama LLM (Optional)

**Model:** `gemma2:2b`

**Usage:**

```python
import ollama

response = ollama.chat(
    model='gemma2:2b',
    messages=[{
        'role': 'user',
        'content': 'Generate interview questions for Python developer'
    }]
)
```

**Note:** Used for text generation (interview questions, CV bullets). Optional dependency - can use templates if Ollama not available.

### 3. spaCy (NLP)

**Model:** `en_core_web_sm`

**Usage:**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Senior Software Engineer at Google")

# Named entity recognition
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Used For:**

- Named entity recognition (companies, roles)
- Part-of-speech tagging
- Skill extraction

---

## 🔌 API Documentation

### Backend API (Optional)

**Base URL:** `http://localhost:8000`

#### Endpoints

**1. Health Check**

```http
GET /health

Response:
{
  "status": "healthy",
  "services": {
    "cv_parser": "ready",
    "embeddings": "ready",
    "scorer": "ready"
  }
}
```

**2. Match CV-JD**

```http
POST /api/match

Request Body:
{
  "cv_text": "string",
  "jd_text": "string"
}

Response:
{
  "overall_score": 75.5,
  "breakdown": {...},
  "matched_skills": [...],
  "missing_skills": [...],
  "evidence": [...],
  "recommendations": [...]
}
```

**3. Generate CV**

```http
POST /api/cv/generate

Request Body:
{
  "personal_info": {...},
  "experience": [...],
  "education": [...],
  "skills": [...]
}

Response:
{
  "status": "success",
  "cv_path": "/path/to/cv.docx"
}
```

**4. Interview Questions**

```http
POST /api/interview/questions

Request Body:
{
  "skills": ["python", "fastapi"],
  "num_questions": 10
}

Response:
{
  "total_questions": 10,
  "by_category": {...},
  "preparation_tips": [...]
}
```

**5. Evaluate Answer**

```http
POST /api/interview/evaluate

Request Body:
{
  "question": "string",
  "answer": "string"
}

Response:
{
  "overall_score": 85,
  "rating": "Good",
  "breakdown": {...},
  "feedback": [...],
  "suggestions": [...]
}
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_parsers.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Test Structure

```
tests/
├── test_parsers.py          # CV/JD parsing tests
├── test_embeddings.py       # Embedding generation tests
├── test_scoring.py          # Match scoring tests
├── test_generation.py       # CV generation tests
├── test_integration.py      # End-to-end tests
└── sample_data/
    ├── sample_cv.pdf
    ├── sample_jd.txt
    └── expected_outputs/
```

### Example Test

```python
# tests/test_parsers.py
import pytest
from src.parsers.cv_parser import CVParser

def test_cv_parser_extracts_email():
    parser = CVParser()
    cv_text = "John Doe\njohn.doe@email.com"

    result = parser.parse_text(cv_text)

    assert result['personal_info']['email'] == 'john.doe@email.com'
    assert result['personal_info']['name'] == 'John Doe'

def test_skill_extraction():
    parser = CVParser()
    text = "Experienced with Python, Docker, and Kubernetes"

    skills = parser.extract_skills(text)

    assert 'Python' in skills
    assert 'Docker' in skills
    assert 'Kubernetes' in skills
```

### Manual Testing

See `tests/manual_test_guide.md` for comprehensive manual testing procedures.

---

## 🤝 Contributing

### Contribution Workflow

1. **Fork the repository**
2. **Create feature branch**

```bash
   git checkout -b feature/your-feature-name
```

3. **Make changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test thoroughly**

```bash
   pytest tests/
```

5. **Commit with clear messages**

```bash
   git commit -m "Add: Feature description"
```

6. **Push to your fork**

```bash
   git push origin feature/your-feature-name
```

7. **Create Pull Request**

### Code Style

**Python:**

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Maximum line length: 100 characters

**Example:**

```python
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0 and 1

    Example:
        >>> calculate_similarity("Python developer", "Python engineer")
        0.92
    """
    emb1 = self.encode(text1)
    emb2 = self.encode(text2)
    return cosine_similarity(emb1, emb2)
```

### Commit Message Format

```
Type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

**Types:**

- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Improve existing feature
- `Docs:` Documentation only
- `Test:` Add/update tests
- `Refactor:` Code restructuring

---

## 🚀 Deployment

### Local Deployment

**Standard:**

```bash
streamlit run streamlit_app/Home.py
```

**With custom port:**

```bash
streamlit run streamlit_app/Home.py --server.port 8080
```

**With backend API:**

```bash
# Terminal 1: Start backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
streamlit run streamlit_app/Home.py
```

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**

```bash
docker build -t careerlens-ai .
docker run -p 8501:8501 careerlens-ai
```

### Cloud Deployment

**Streamlit Cloud:**

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy

**Heroku:**

```bash
# Create Procfile
echo "web: streamlit run streamlit_app/Home.py --server.port=$PORT" > Procfile

# Deploy
heroku create careerlens-ai
git push heroku main
```

---

## 🔧 Troubleshooting

### Common Development Issues

**Issue: Model download fails**

```bash
# Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Issue: Import errors**

```bash
# Ensure project root in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/careerlens-ai"
```

**Issue: Streamlit port already in use**

```bash
# Kill process on port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9
```

**Issue: Memory errors with models**

```bash
# Reduce batch size or use smaller model
# Check available RAM
free -h  # Linux
wmic OS get FreePhysicalMemory  # Windows
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = scorer.calculate_match(cv_data, jd_data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## 📚 Additional Resources

### Documentation

- [Streamlit Docs](https://docs.streamlit.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

### Learning Resources

- [Semantic Similarity Tutorial](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Streamlit Components](https://streamlit.io/components)

### Community

- GitHub Discussions
- Stack Overflow tag: `careerlens-ai`

---

## 📞 Developer Support

**Questions or issues?**

- 📧 zamanratul419@gmail.com
- 💬 GitHub Issues
- 📖 This guide

---

**Happy coding! 🚀**

_Last Updated: March 2026_
_Version: 1.0.0_
