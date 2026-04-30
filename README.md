# CareerLens AI

**Explainable CV–Job Matching & Career Guidance System**

> Academic project — CSE 299 Junior Design, North South University, Spring 2026  
> Group 4 · Section 22 · Instructor: Silvia Ahmed

---

## What it does

CareerLens AI is a locally-running web application that helps job seekers understand how well their CV matches a job description — and what to do about it. It combines semantic NLP, a local language model, and a multi-criteria scoring engine to give candidates the kind of feedback that ATS systems silently withhold.

The system does five things:

1. **Semantic CV–JD matching** — scores a CV against a job description using sentence embeddings, with a per-skill breakdown and confidence-weighted evidence citations.
2. **Counterfactual skill impact simulation** — answers "if I learned Docker, how much would my score improve?" by re-scoring the CV with each missing skill added.
3. **CV generation and improvement** — extracts structured data from uploaded documents (PDF, DOCX, images via OCR), then generates an ATS-optimised CV in DOCX and PDF formats.
4. **Personalized learning pathways** — produces 7-, 14-, or 30-day skill development plans based on the identified gaps, including daily tasks, resource links, and mini-projects.
5. **Interview preparation** — generates role-specific questions and evaluates STAR-method answers with an AI-powered rubric.

All processing runs locally. No CV data is sent to external servers.

---

## Why it was built

A survey of 70 final-year students and fresh graduates at Dhaka universities found:

- 72% did not know how well their CV matched a given job description
- 65.7% had been rejected without receiving any explanation
- 62.9% could not identify which skill to learn next
- 67.1% said a time-based learning plan would be the most helpful tool they could have

Existing tools (Jobscan, Resumeworded, LinkedIn Skills) are either paywalled, send data to external servers, or produce a score without explaining it. None offer counterfactual reasoning or time-bound learning plans. This project is an attempt to close that gap with open-source tooling and no recurring cost.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  Home · CV Matcher · CV Generator · Interview Prep · Analytics│
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Session Manager   │
              │   (cross-page state)│
              └──────────┬──────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌───────▼───────┐   ┌───────▼───────┐
│ Parsers │      │  Scoring &    │   │  Generation & │
│ CV · JD │      │  Explainability│   │  Guidance     │
└────┬────┘      └───────┬───────┘   └───────┬───────┘
     │                   │                   │
     └─────────┬─────────┘                   │
               │                             │
    ┌──────────▼──────────┐      ┌───────────▼───────────┐
    │  Embedding Engine   │      │  Ollama (Gemma)        │
    │  all-MiniLM-L6-v2  │      │  Local LLM inference   │
    │  384-dim vectors    │      │  No external API       │
    └─────────────────────┘      └───────────────────────┘
```

### Scoring algorithm

Match score is a weighted sum across three criteria:

```
Overall = 0.60 × required_skills + 0.25 × preferred_skills + 0.15 × experience
```

Skill matching uses cosine similarity on sentence embeddings with a 0.70 threshold. This catches semantic equivalents ("REST APIs" ≈ "RESTful API design") that keyword matching misses.

### Counterfactual simulation

For each missing skill, the system clones the CV data, injects the skill, re-runs the full scoring pipeline, and reports the delta. The result is a ranked list of skills sorted by score impact — the closest thing to an ROI calculation for a learning decision.

---

## Tech stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | Streamlit | 1.31 |
| Backend API | FastAPI + Uvicorn | 0.109 |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) | 2.2.2 |
| Local LLM | Ollama — Gemma / Gemma2 | latest |
| ML utilities | scikit-learn, numpy, pandas | — |
| Document parsing | python-docx, pypdf, pdfplumber, mammoth | — |
| OCR | Tesseract via pytesseract | 0.3.10 |
| PDF generation | ReportLab (pure Python, no LibreOffice needed) | 4.x |
| Security | bleach, python-magic, threading.Lock | — |
| Testing | pytest, pytest-cov, psutil | — |
| Visualisation | Plotly | 5.18 |

The embedding model runs entirely on CPU. No GPU is required, though inference is faster with one.

---

## Project structure

```
careerlens-ai/
├── src/
│   ├── parsers/
│   │   ├── cv_parser.py          # PDF/DOCX/TXT extraction + LLM-assisted section parsing
│   │   └── jd_parser.py          # JD requirement extraction via regex + LLM hybrid
│   ├── embeddings/
│   │   └── embedding_engine.py   # SentenceTransformer wrapper with LRU cache
│   ├── scoring/
│   │   ├── scoring_engine.py     # Weighted multi-criteria scorer
│   │   ├── explainability.py     # Evidence citation + recommendation generator
│   │   └── counterfactual.py     # Skill impact simulator (re-score with injected skill)
│   ├── generation/
│   │   ├── cv_generator.py       # ATS-optimised DOCX builder (python-docx)
│   │   └── cv_optimizer.py       # LLM-powered skill reordering and summary rewriting
│   ├── guidance/
│   │   ├── learning_pathways.py  # 7/14/30-day roadmap generator
│   │   └── interview_guidance.py # Question generation + STAR-method evaluator
│   ├── validation/
│   │   ├── cv_analyzer.py        # LLM-based CV quality analysis
│   │   └── eligibility_validator.py # OCR-based academic credential checker
│   └── security/
│       ├── input_validator.py    # XSS/injection sanitisation (bleach)
│       ├── rate_limiter.py       # Token bucket rate limiter (threading-safe)
│       └── file_security.py      # MIME type + magic bytes validation
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── 1_📊_CV_Matcher.py
│       ├── 2_📝_CV_Generator.py
│       ├── 3_🎓_Interview_Prep.py
│       └── 4_📈_Analytics.py
├── tests/
│   ├── test_cv_parser.py
│   ├── test_scoring.py
│   ├── test_counterfactual.py
│   ├── test_interview_guidance.py
│   ├── test_learning_pathway.py
│   ├── integration/
│   ├── performance/
│   └── security/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for running the LLM locally)
- [Ollama](https://ollama.ai) installed and running

### Installation

```bash
git clone https://github.com/your-username/careerlens-ai.git
cd careerlens-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Pull the LLM

```bash
ollama pull gemma:latest
ollama serve                    # keep this running in a separate terminal
```

### Configure environment

```bash
cp .env.example .env
# Edit .env — set SECRET_KEY at minimum
```

### Run

```bash
streamlit run streamlit_app/Home.py
```

Open `http://localhost:8501`.

---

## Running tests

```bash
# All tests, skipping slow load/stress tests
pytest tests/ -v -m "not slow"

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Security tests only
pytest tests/security/ -v

# Integration tests
pytest tests/integration/ -v
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma:latest` | Model to use for generation |
| `SECRET_KEY` | — | Session signing key (required) |
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Requests allowed per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `3600` | Rate limit window in seconds |
| `MAX_UPLOAD_SIZE_MB` | `10` | File upload size cap |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Limitations

**LLM dependency.** The interview prep, CV optimization, and learning pathway features require Ollama running locally. If Ollama is unavailable, these features fall back to template-based outputs, which are functional but less tailored.

**OCR accuracy.** The academic eligibility validator uses Tesseract OCR. Scanned documents with low resolution, handwriting, or non-standard layouts may produce poor extraction results. The system flags low-confidence extractions rather than silently accepting them.

**CV parsing quality.** The CV parser uses a regex + LLM hybrid. Heavily formatted CVs (multi-column layouts, tables, text boxes) may not parse cleanly — plain text or simply formatted DOCX files produce the best results.

**Session persistence.** There is no database. All session data lives in Streamlit's session state and resets on browser refresh. Export your results before closing the tab.

**PDF generation.** The ReportLab-based PDF converter reproduces the semantic structure of the CV (headings, bullets, bold) but does not perfectly replicate the visual styling of the DOCX. For a pixel-accurate PDF, open the DOCX in Word or LibreOffice and export from there.

**Concurrency.** The system is tested and stable up to ~15 concurrent users on a 16 GB machine. Beyond that, LLM inference becomes the bottleneck. A production deployment would need a request queue or multiple Ollama instances.

---

## Performance benchmarks

Measured on a Windows 10 machine, Python 3.10, Ollama Gemma2:2b, CPU-only inference.

| Operation | Average | Notes |
|---|---|---|
| CV parsing (TXT/DOCX) | ~2.1 s | Includes LLM section extraction |
| JD parsing | ~1.3 s | Regex + LLM hybrid |
| Match scoring | ~1.7 s | Embedding + cosine similarity |
| CV generation (DOCX) | ~1.8 s | python-docx only |
| PDF generation | ~0.8 s | ReportLab, pure Python |
| LLM optimization (CV) | ~12–20 s | Ollama inference, CPU |
| Interview question generation | ~11–15 s | 10 questions |
| Learning pathway (7-day) | ~18–26 s | Full plan with resources |

Throughput: ~2 match requests/second. 10–15 concurrent users before degradation.

---

## Security

- All user input is sanitised through a whitelist validator before processing (bleach for HTML, regex for emails/URLs/skills).
- Uploaded files are validated by extension, MIME type, size limit, and magic bytes signature check before any parsing occurs.
- The rate limiter uses a token bucket algorithm implemented with `threading.Lock` — safe for concurrent Streamlit sessions.
- No CV or JD data leaves the machine. All embedding and LLM inference is local.
- Session tokens are signed with a configurable `SECRET_KEY`.

---

## Cost

The entire project runs on open-source software with no licensing fees or API costs.

| Item | Cost |
|---|---|
| All software (Python, Streamlit, FastAPI, Ollama, SentenceTransformers, etc.) | ৳0 |
| Cloud / API fees | ৳0 |
| Internet (3 months, shared) | ৳1,500 |
| Electricity estimate | ৳600 |
| **Total** | **৳2,100 (~$19 USD)** |

For reference: a Jobscan subscription costs ~৳12,000/year. Cloud-based OpenAI API usage for 10,000 CV matches would run ~৳50,000+.

---

## Team

| Name | ID | Primary contribution |
|---|---|---|
| Arafat Zaman Ratul | 2311539042 | Semantic matching engine, CV/JD parsers, embedding pipeline, scoring algorithm, FastAPI `/match` endpoint |
| Hasibul Islam Rony | 2312439042 | Counterfactual simulator, Ollama LLM integration, prompt engineering, CV bullet rewriting |
| Ashikur Rahman | 2311555642 | CV generation suite, ATS formatting, PDF export, academic eligibility validator (OCR), FastAPI `/generate-cv` and `/validate-eligibility` |
| Mahfuzur Rahman Sazid | 2312169042 | Streamlit frontend (5 pages), analytics dashboard, session management, interview guidance, learning pathways, documentation |

---

## References

1. Deshmukh & Ratul, "Applying BERT-Based NLP for Automated Resume Screening," *Annals Data Sci.*, vol. 12, no. 2, 2025.
2. Ali et al., "Job Matching and Skill Recommendation Using Transformers and the O\*NET Database," *Expert Syst. Appl.*, vol. 258, 2025.
3. Joshi et al., "Resume2Vec: Transforming ATS with Intelligent Resume Embeddings," *Electronics*, vol. 14, no. 4, 2025.
4. Verma et al., "Counterfactual Explanations and Algorithmic Recourses for ML: A Review," *ACM Comput. Surveys*, vol. 57, no. 3, 2025.
5. Molnar, *Interpretable Machine Learning*, 2nd ed. — [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
6. Smith et al., "Multi-Criteria Resume Ranking with Weighted BERT Embeddings," *IEEE Trans. Services Computing*, vol. 18, no. 2, 2025.

---

## License

MIT — see [LICENSE](LICENSE).
