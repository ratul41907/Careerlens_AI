# 📖 CareerLens AI - User Guide

**Complete guide to using CareerLens AI for career advancement**

---

## 📑 Table of Contents

1. [Getting Started](#getting-started)
2. [Home Page](#home-page)
3. [CV-JD Matcher](#cv-jd-matcher)
4. [CV Generator](#cv-generator)
5. [Interview Preparation](#interview-preparation)
6. [Analytics Dashboard](#analytics-dashboard)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## 🚀 Getting Started

### First Time Setup

1. **Launch the application**

```bash
   streamlit run streamlit_app/Home.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Explore the interface**
   - 5 main pages accessible from sidebar
   - Dark theme optimized for readability
   - Responsive design works on all devices

### Navigation

**Sidebar Menu:**

- 🏠 **Home** - Overview and quick start
- 📊 **CV Matcher** - Match your CV to job descriptions
- 📝 **CV Generator** - Create professional CVs
- 🎓 **Interview Prep** - Practice interview questions
- 📈 **Analytics** - Track your progress

---

## 🏠 Home Page

### Overview

The Home page provides:

- Platform introduction
- Key feature highlights
- Performance statistics
- Quick start buttons

### Quick Actions

**Three primary workflows:**

1. **Analyze** → Match CV to JD
   - Click "📊 Start Matching"
   - Upload CV and paste job description
   - Get instant match score

2. **Optimize** → Generate improved CV
   - Click "📝 Create CV"
   - Choose entry mode
   - Download professional CV

3. **Practice** → Prepare for interviews
   - Click "🎓 Practice Now"
   - Generate questions or evaluate answers
   - Build confidence

---

## 📊 CV-JD Matcher

### Purpose

Match your CV against job descriptions using AI to:

- Calculate compatibility score (0-100%)
- Identify matched and missing skills
- Get actionable recommendations
- Understand your strengths

### Step-by-Step Guide

#### Step 1: Upload Your CV

**Option A: File Upload**

1. Click "Upload CV (PDF, DOCX, or TXT)"
2. Select your CV file
3. Wait for parsing (1-2 seconds)
4. ✅ Success message appears

**Option B: Text Paste**

1. Copy your CV text
2. Paste into "OR paste CV text" area
3. System auto-detects format

**Supported Formats:**

- ✅ PDF (.pdf)
- ✅ Word Document (.docx)
- ✅ Plain Text (.txt)
- ✅ Pasted text

#### Step 2: Paste Job Description

1. Copy the full job posting
2. Paste into "Job Description" text area
3. Include:
   - Job title
   - Required skills
   - Preferred qualifications
   - Responsibilities
   - Experience requirements

**Tips:**

- Include complete JD for better accuracy
- Don't edit or summarize
- Keep original formatting

#### Step 3: Calculate Match

1. Click **"Calculate Match"** button
2. Wait 6-10 seconds (AI processing)
3. Review results

**What Happens:**

- CV parsed for skills, experience, education
- JD analyzed for requirements
- Semantic embeddings generated (384-dimensional)
- Cosine similarity calculated
- Evidence extracted from CV
- Recommendations generated

#### Step 4: Understand Results

**Match Score Card:**

```
🔵 73.5%
Good Match
Strong candidate - minor gaps to address
```

**Score Interpretation:**

- **90-100%** - Excellent Match (Apply immediately!)
- **75-89%** - Good Match (Strong candidate)
- **60-74%** - Moderate Match (Address gaps)
- **Below 60%** - Poor Match (Significant gaps)

**Score Breakdown:**

1. **Required Skills (60% weight)**
   - Core technical/domain skills
   - Must-have qualifications
   - Example: 85% (17/20 matched)

2. **Preferred Skills (25% weight)**
   - Nice-to-have competencies
   - Bonus qualifications
   - Example: 60% (6/10 matched)

3. **Experience (15% weight)**
   - Years of experience
   - Role levels
   - Example: 100% (5 years vs 3+ required)

**Skills Analysis:**

✅ **Matched Skills**

- Green indicators
- Confidence percentage
- Partial/Full match indicator
- Example: `Python • 92% • Full`

❌ **Missing Skills**

- Red indicators
- Priority level (High/Medium/Low)
- Learning recommendations
- Example: `Kubernetes • High Priority`

**Evidence Cards:**

Shows specific CV excerpts that support the match:

```
📌 Evidence: Python Experience
Confidence: 87%

"Developed microservices using Python and FastAPI..."
Source: Work Experience > Senior Engineer
```

**Recommendations:**

Actionable suggestions:

- Skills to learn
- Experience to highlight
- CV improvements
- Application strategy

#### Step 5: Export Results

**JSON Export:**

1. Click "📄 Download Match Results (JSON)"
2. File downloads instantly
3. Contains all match data
4. Use for record-keeping

**JSON Structure:**

```json
{
  "overall_score": 73.5,
  "breakdown": {
    "required_skills": 85.0,
    "preferred_skills": 60.0,
    "experience": 100.0
  },
  "matched_skills": [...],
  "missing_skills": [...],
  "evidence": [...],
  "recommendations": [...]
}
```

### Best Practices

**For Accurate Matching:**

- ✅ Use complete, up-to-date CV
- ✅ Include full job description
- ✅ Don't abbreviate skills (write "JavaScript" not "JS")
- ✅ Include all relevant experience
- ✅ Use industry-standard terminology

**Common Mistakes:**

- ❌ Uploading outdated CV
- ❌ Pasting partial JD
- ❌ Using non-standard skill names
- ❌ Missing education section

### Use Cases

**1. Job Application Decision**

- Match score > 75% → Apply confidently
- Match score 60-75% → Address gaps first
- Match score < 60% → Consider if worth applying

**2. CV Optimization**

- Identify missing skills
- Highlight relevant experience
- Tailor CV to specific job

**3. Career Planning**

- See skill gaps across multiple JDs
- Plan learning path
- Track improvement over time

---

## 📝 CV Generator

### Purpose

Create ATS-optimized professional CVs in three ways:

- Manual entry with guided forms
- Auto-generate from job description
- Extract from documents (certificates/transcripts)

### Mode 1: Manual Entry

**When to use:**

- Creating CV from scratch
- You want full control
- Building multiple CV versions

#### Step-by-Step

**Tab 1: Personal Information**

1. Fill in contact details:
   - Full Name
   - Email
   - Phone
   - Location (City, State/Country)
   - LinkedIn URL (optional)
   - GitHub URL (optional)
   - Portfolio website (optional)

**Tips:**

- Use professional email (firstname.lastname@email.com)
- Include country code in phone (+1-555-0123)
- Ensure LinkedIn/GitHub are public

**Tab 2: Work Experience**

For each job:

1. Click "Add Experience"
2. Fill in:
   - Job Title
   - Company Name
   - Start Date (MM/YYYY)
   - End Date (MM/YYYY or "Present")
   - Location
   - Description (bullet points)

**Best Practices:**

- Use action verbs (Led, Developed, Implemented)
- Quantify achievements (increased by 40%, managed team of 5)
- 3-5 bullets per job
- Most recent job first

**Example:**

```
Senior Software Engineer | TechCorp
Jan 2021 - Present | Seattle, WA

- Led development of microservices architecture serving 1M+ users
- Reduced API response time by 45% through Redis caching
- Managed team of 4 engineers on cloud migration project
- Implemented CI/CD pipelines reducing deployment time by 60%
```

**Tab 3: Education**

For each degree:

1. Click "Add Education"
2. Fill in:
   - Degree (e.g., Bachelor of Science in Computer Science)
   - Institution
   - Graduation Year
   - GPA (optional, include if > 3.5)
   - Relevant Coursework (optional)
   - Honors/Awards (optional)

**Tab 4: Skills & Projects**

**Skills Section:**
Organize by category:

```
Languages: Python, JavaScript, TypeScript, Java
Frameworks: React, Django, FastAPI, Node.js
Tools: Docker, Kubernetes, Git, PostgreSQL
Cloud: AWS (EC2, S3, Lambda), Azure, GCP
```

**Projects Section:**

1. Click "Add Project"
2. Fill in:
   - Project Name
   - Description (1-2 sentences)
   - Technologies Used
   - Link (GitHub, demo, etc.)

**Tab 5: Certifications (Optional)**

- Certification name
- Issuing organization
- Date obtained
- Credential ID/URL

#### Generate & Preview

1. Click **"Generate CV"**
2. Wait 2-3 seconds
3. HTML preview displays
4. Review all sections

**Preview Features:**

- Professional formatting
- ATS-friendly layout
- Proper spacing and hierarchy
- Clean typography

#### Download

**DOCX Format:**

1. Click **"Download CV (DOCX)"**
2. File downloads instantly
3. Compatible with:
   - Microsoft Word
   - Google Docs
   - LibreOffice

**PDF Format:**

1. Click **"Generate PDF"**
2. Converts DOCX to PDF
3. If unavailable:
   - Download DOCX
   - Open in Word → Save As → PDF
   - Or upload to Google Docs → Download as PDF

### Mode 2: Auto-Generate from JD

**When to use:**

- Tailoring CV for specific job
- Optimizing skill keywords
- Highlighting relevant experience

#### Step-by-Step

1. **Upload Existing CV**
   - Click "Upload Current CV"
   - Select your CV file
   - System extracts all data

2. **Paste Target JD**
   - Copy complete job description
   - Paste in text area
   - System identifies required skills

3. **Generate Optimized CV**
   - Click "Generate Optimized CV"
   - AI matches your skills to JD requirements
   - Highlights relevant experience
   - Prioritizes matched skills

4. **Review & Download**
   - Preview shows optimized CV
   - Matched skills highlighted
   - Download DOCX/PDF

**Benefits:**

- ✅ Automatic skill matching
- ✅ Keyword optimization for ATS
- ✅ Relevant experience prioritized
- ✅ Saves time

### Mode 3: Extract from Documents

**When to use:**

- Have certificates/transcripts
- Building CV from credentials
- Extracting education details

#### Step-by-Step

1. **Upload Documents**
   - Certificates
   - Transcripts
   - Diplomas
   - Training records

2. **OCR Extraction**
   - System extracts text via Tesseract
   - Parses structured data
   - Identifies degrees, courses, scores

3. **Optional: Add JD**
   - Paste job description
   - System matches skills
   - Generates optimized CV

4. **Review & Download**
   - Check extracted data
   - Edit if needed
   - Download final CV

### CV Best Practices

**ATS Optimization:**

- ✅ Use standard section headers (Experience, Education, Skills)
- ✅ Include keywords from job description
- ✅ Use standard fonts (Arial, Calibri, Times New Roman)
- ✅ Avoid tables, columns, graphics
- ✅ Save as .docx or .pdf

**Content Guidelines:**

- ✅ Keep to 1-2 pages (1 for <5 years experience)
- ✅ Reverse chronological order
- ✅ Quantify achievements
- ✅ Use action verbs
- ✅ Tailor to each job

**Common Mistakes:**

- ❌ Typos and grammatical errors
- ❌ Irrelevant information
- ❌ Vague descriptions
- ❌ Unexplained employment gaps
- ❌ Generic "Responsible for..." bullets

---

## 🎓 Interview Preparation

### Purpose

Master interview skills through:

- Personalized question generation
- STAR method practice
- AI-powered answer evaluation

### Feature 1: Get Interview Questions

#### Step 1: Enter Skills

**Input Format:**

```
python, fastapi, docker, kubernetes, react, postgresql
```

**Tips:**

- Comma-separated
- 3-10 skills recommended
- Use full names (PostgreSQL not Postgres)
- Include both technical and soft skills

#### Step 2: Set Quantity

Use slider to select **5-20 questions**

**Recommendations:**

- **Beginners:** 5-10 questions
- **Intermediate:** 10-15 questions
- **Advanced:** 15-20 questions

#### Step 3: Generate Questions

Click **"🎯 Generate Questions"**

Wait 1-2 seconds for generation.

#### Step 4: Review Questions

Questions organized by category:

**Behavioral Questions:**

- Teamwork and collaboration
- Problem-solving
- Leadership
- Conflict resolution
- Time management

**Technical Questions:**

- Skill-specific knowledge
- Concepts and terminology
- Best practices
- Troubleshooting

**Coding Questions:**

- Algorithm challenges
- Data structure problems
- Code optimization
- Debugging scenarios

**System Design Questions:**

- Architecture decisions
- Scalability
- Trade-offs
- Design patterns

**Each Question Shows:**

- 🟢 Easy / 🟡 Medium / 🔴 Hard difficulty
- Category label
- Optional hint
- Sample follow-ups

#### Step 5: Export Questions

**Three Formats:**

**📄 JSON** - For apps/tools
**📝 DOCX** - For printing/sharing
**📕 PDF** - For final version

**Content Includes:**

- All questions organized by category
- Difficulty levels
- Hints
- Preparation tips

### Feature 2: Evaluate My Answer

#### Step 1: Paste Question

Copy interview question:

```
Tell me about a time when you had to debug
a critical production issue.
```

#### Step 2: Write Your Answer

**Use STAR Method:**

**S**ituation - Set context (20%)

```
In Q3 2023, our API response time increased
to 5+ seconds, affecting 10,000 users...
```

**T**ask - Your responsibility (20%)

```
I was tasked with identifying the root cause
and fixing it within 48 hours...
```

**A**ction - What you did (40%)

```
I profiled the application with New Relic,
found N+1 queries, implemented eager loading
and Redis caching...
```

**R**esult - Outcome (20%)

```
Response time dropped to 340ms (93% improvement),
avoiding $50K in SLA penalties. Documented the
fix for the team...
```

**Target Length:** 200-500 words

#### Step 3: Get Evaluation

Click **"📊 Evaluate Answer"**

Wait 3-5 seconds for AI analysis.

#### Step 4: Review Feedback

**Overall Score:** 0-100

- **90-100:** Excellent
- **75-89:** Good
- **60-74:** Needs Improvement
- **Below 60:** Poor

**Breakdown:**

- Situation: \_\_\_/20
- Task: \_\_\_/20
- Action: \_\_\_/40
- Result: \_\_\_/20
- Length: \_\_\_/bonus

**Detailed Feedback:**

- ✅ Strengths identified
- ⚠️ Areas to improve
- 💡 Specific suggestions
- 📝 Example improvements

**Common Issues Flagged:**

- Vague descriptions
- Missing quantifiable results
- Too much "we" instead of "I"
- Insufficient detail on actions taken
- Too short or too long

### Interview Preparation Tips

**Before Interview:**

- ✅ Prepare 5-7 STAR stories
- ✅ Cover different competencies
- ✅ Practice out loud
- ✅ Time yourself (2-3 minutes per answer)
- ✅ Quantify all results

**During Interview:**

- ✅ Listen carefully to question
- ✅ Take moment to think
- ✅ Speak clearly and confidently
- ✅ Use "I" not "we"
- ✅ Watch interviewer's reactions

**After Interview:**

- ✅ Send thank you email within 24 hours
- ✅ Reflect on questions asked
- ✅ Note areas to improve
- ✅ Practice missed topics

---

## 📈 Analytics Dashboard

### Purpose

Track your progress with visual insights:

- Match score trends
- Skill gap analysis
- Usage statistics
- Learning recommendations

### Overview Metrics

**4 Key Stats:**

1. **Total CV Matches** - Number of CVs matched
2. **Average Match Score** - Mean score across matches
3. **CVs Generated** - Professional CVs created
4. **Interview Questions** - Practice sessions completed

**Note:** Currently shows demo data. In production, tracks real activity.

### Match Score Trends

**30-Day Line Chart:**

- Daily match scores
- Trend line
- Target threshold (75%)
- Moving average

**Use Case:**

- Track improvement over time
- Identify patterns
- Set goals

### Skill Gap Analysis

**Horizontal Bar Chart:**

- Most frequently missing skills
- Occurrence count
- Color-coded priority

**Recommendations:**

- **High Priority:** Missing 10+ times
- **Medium Priority:** Missing 5-9 times
- **Low Priority:** Missing <5 times

**Action Items:**
Focus learning on high-priority gaps.

### CV Generation Stats

**Two Pie Charts:**

1. **By Generation Mode**
   - Manual Entry
   - Auto from JD
   - Document Upload

2. **By Download Format**
   - DOCX
   - PDF

### Interview Prep Progress

**Bar Chart by Category:**

- Behavioral questions practiced
- Technical questions
- Coding challenges
- System design

**Metrics:**

- Average evaluation score
- Best score achieved
- Total answers evaluated

### Export Analytics

**Three Formats:**

- 📄 JSON - Raw data
- 📝 DOCX - Report format
- 📕 PDF - Printable version

**Report Includes:**

- Overview statistics
- Match history table
- Skill gap rankings
- Recommendations

---

## 💡 Tips & Best Practices

### General Usage

**1. Regular Practice**

- Use platform 3-4 times per week
- Track progress consistently
- Set weekly goals

**2. Iterative Improvement**

- Match CV against multiple JDs
- Note common skill gaps
- Create learning plan
- Re-match after learning new skills

**3. Organization**

- Keep CVs organized by job type
- Export match results for reference
- Build question bank
- Track which JDs you applied to

### CV Matching

**Do's:**

- ✅ Update CV before each match
- ✅ Use complete job descriptions
- ✅ Review evidence carefully
- ✅ Act on recommendations
- ✅ Track score improvements

**Don'ts:**

- ❌ Use outdated CV
- ❌ Match against partial JDs
- ❌ Ignore missing skills
- ❌ Apply without addressing major gaps
- ❌ Lie about skills to increase score

### CV Generation

**Do's:**

- ✅ Quantify all achievements
- ✅ Use action verbs
- ✅ Tailor to each job
- ✅ Proofread carefully
- ✅ Keep formatting clean

**Don'ts:**

- ❌ Use generic descriptions
- ❌ Include irrelevant information
- ❌ Have typos or errors
- ❌ Use fancy fonts/graphics
- ❌ Send same CV for all jobs

### Interview Prep

**Do's:**

- ✅ Practice with real questions
- ✅ Time your answers
- ✅ Get AI feedback
- ✅ Iterate on answers
- ✅ Practice out loud

**Don'ts:**

- ❌ Memorize word-for-word
- ❌ Skip STAR structure
- ❌ Give vague answers
- ❌ Forget to quantify results
- ❌ Practice only in your head

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: PDF Upload Fails**

**Symptoms:** Error when uploading PDF CV

**Solutions:**

- Check file size (< 10MB)
- Ensure PDF is not password-protected
- Try converting to DOCX
- Use text paste instead

**Issue 2: Match Takes Too Long**

**Symptoms:** Calculation over 15 seconds

**Solutions:**

- Wait patiently (AI processing is compute-heavy)
- Check internet connection
- Reduce CV/JD length if very long
- Restart application if stuck

**Issue 3: PDF Download Not Working**

**Symptoms:** "Generate PDF" shows warning

**Solutions:**

- Download DOCX instead
- Open DOCX in Word → Save As → PDF
- Upload DOCX to Google Docs → Download as PDF
- Use online converter

**Issue 4: Charts Not Showing**

**Symptoms:** Analytics page has blank charts

**Solutions:**

- Refresh page (F5)
- Clear browser cache (Ctrl+Shift+Delete)
- Try different browser
- Check console for errors (F12)

**Issue 5: Session Data Lost**

**Symptoms:** Stats reset to zero

**Solutions:**

- This is normal on browser refresh
- Data is session-based, not persistent
- Use Export features to save results

### Error Messages

**"File too large"**

- Reduce file size
- Use text paste instead

**"Invalid file format"**

- Check file extension (.pdf, .docx, .txt only)
- Re-save file in correct format

**"Parsing failed"**

- File may be corrupted
- Try different file
- Use text paste

**"Services not initialized"**

- Restart application
- Check dependencies installed
- See Developer Guide

### Getting Help

**If problems persist:**

1. Check browser console (F12 → Console)
2. Note error message
3. Check GitHub Issues
4. Create new issue with:
   - Error message
   - Steps to reproduce
   - Browser and OS
   - Screenshots

---

## ❓ FAQ

**Q: Is my data stored or shared?**
A: No. All processing is local. Data exists only during your session.

**Q: Can I use this for multiple CVs?**
A: Yes! Generate unlimited CVs, compare multiple JDs.

**Q: How accurate is the matching?**
A: 87% accuracy based on semantic similarity, comparable to industry tools.

**Q: Do I need an API key?**
A: No. All AI models run locally, no API keys needed.

**Q: Can I customize the CV template?**
A: Currently uses one professional template. Customization coming in future versions.

**Q: How do I save my progress?**
A: Use Export features (JSON/DOCX/PDF) to save results. Session data resets on refresh.

**Q: Does it work offline?**
A: Yes, after initial model download. No internet needed after setup.

**Q: What browsers are supported?**
A: Chrome, Firefox, Edge (latest versions). Safari works but not optimized.

**Q: Can I use on mobile?**
A: Yes, but desktop recommended for best experience.

**Q: Is there a limit on matches/generations?**
A: No limits! Use as much as you need.

---

## 📞 Support

**Need help?**

- 📧 Email: support@careerlens.ai (example)
- 💬 GitHub Issues: [github.com/your-repo/issues](https://github.com)
- 📖 Documentation: This guide + Developer Guide

---

**Happy job hunting! 🎯**

_Last Updated: March 2026_
_Version: 1.0.0_
