"""
End-to-End Workflow Tests
Test complete user journeys through the system
"""
import pytest
import time
from pathlib import Path
import tempfile
import os


class TestCompleteWorkflow:
    """Test complete CV-JD matching workflow"""
    
    def test_complete_cv_jd_matching_pipeline(self, cv_parser, jd_parser, scoring_engine):
        """Test complete workflow from CV upload to match results"""
        
        print("\n" + "=" * 70)
        print("END-TO-END WORKFLOW TEST: CV-JD MATCHING")
        print("=" * 70)
        
        # Sample CV text
        cv_text = """
JANE SMITH
Senior Full-Stack Developer
jane.smith@email.com | +1-555-0123

PROFESSIONAL SUMMARY
Experienced Full-Stack Developer with 6+ years building scalable web applications.
Expertise in Python, JavaScript, React, and cloud technologies.

EXPERIENCE

Senior Full-Stack Developer | TechCorp Inc. | 2021 - Present
- Built microservices architecture serving 1M+ daily users using Python FastAPI
- Developed React-based dashboard reducing load time by 60%
- Implemented Docker containerization for 15+ services
- Led migration to AWS, reducing infrastructure costs by 40%
- Mentored team of 4 junior developers

Full-Stack Developer | StartupXYZ | 2019 - 2021
- Developed REST APIs handling 500K+ requests/day
- Built CI/CD pipeline using GitHub Actions
- Optimized PostgreSQL queries improving response time by 75%
- Worked with Redis for caching, reducing database load by 50%

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, SQL
Frontend: React, Vue.js, HTML5, CSS3
Backend: FastAPI, Django, Node.js, Express
Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, Jenkins
Databases: PostgreSQL, MongoDB, Redis

EDUCATION
Bachelor of Science in Computer Science | University of California | 2019
GPA: 3.9/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect (2022)
- Certified Kubernetes Administrator (2023)
"""
        
        # Sample JD text
        jd_text = """
Senior Full-Stack Engineer

TechVision Solutions is seeking a Senior Full-Stack Engineer to join our growing team.

REQUIRED QUALIFICATIONS:
- 5+ years of professional software development experience
- Strong proficiency in Python and JavaScript
- Experience with React or Vue.js frontend frameworks
- Backend development with FastAPI, Django, or Node.js
- Proficiency with PostgreSQL or MySQL databases
- Experience with Docker containerization
- AWS cloud services experience required
- Bachelor's degree in Computer Science or related field

PREFERRED QUALIFICATIONS:
- Kubernetes orchestration experience
- Redis caching implementation
- CI/CD pipeline setup (GitHub Actions, Jenkins)
- Microservices architecture experience
- Team leadership or mentoring experience

RESPONSIBILITIES:
- Design and develop scalable full-stack applications
- Lead technical architecture decisions
- Mentor junior developers
- Optimize application performance
- Collaborate with cross-functional teams

WHAT WE OFFER:
- Competitive salary ($140K - $180K)
- Remote-first culture
- Health, dental, vision insurance
- 401(k) matching
- Professional development budget
"""
        
        # STEP 1: Parse CV
        print("\n📄 STEP 1: Parsing CV")
        print("-" * 70)
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(cv_text)
            cv_file_path = f.name
        
        try:
            cv_data = cv_parser.parse(cv_file_path)
            cv_parse_time = time.time() - start_time
            
            assert cv_data is not None
            assert 'text' in cv_data
            assert 'sections' in cv_data
            
            print(f"✅ CV parsed successfully in {cv_parse_time:.2f}s")
            print(f"   Sections extracted: {len(cv_data['sections'])}")
            
            # STEP 2: Parse JD
            print("\n📋 STEP 2: Parsing Job Description")
            print("-" * 70)
            start_time = time.time()
            
            jd_data = jd_parser.parse(jd_text)
            jd_parse_time = time.time() - start_time
            
            assert jd_data is not None
            assert 'text' in jd_data
            assert 'sections' in jd_data
            
            print(f"✅ JD parsed successfully in {jd_parse_time:.2f}s")
            
            sections = jd_data['sections']
            if 'required_skills' in sections:
                print(f"   Required skills: {len(sections['required_skills'])}")
            if 'preferred_skills' in sections:
                print(f"   Preferred skills: {len(sections['preferred_skills'])}")
            
            # STEP 3: Compute Match Score
            print("\n🎯 STEP 3: Computing Match Score")
            print("-" * 70)
            start_time = time.time()
            
            match_result = scoring_engine.compute_match_score(cv_data, jd_data)
            scoring_time = time.time() - start_time
            
            assert match_result is not None
            assert 'overall_score' in match_result
            assert 'breakdown' in match_result
            
            overall_score = match_result['overall_score']
            overall_pct = match_result.get('overall_percentage', f"{overall_score*100:.1f}%")
            
            print(f"✅ Match computed successfully in {scoring_time:.2f}s")
            print(f"\n🎯 MATCH RESULTS:")
            print(f"   Overall Score: {overall_pct}")
            print(f"   Raw Score: {overall_score:.4f}")
            
            # Display breakdown
            breakdown = match_result['breakdown']
            print(f"\n📊 SCORE BREAKDOWN:")
            
            for component, data in breakdown.items():
                comp_name = component.replace('_', ' ').title()
                if 'score' in data:
                    score = data['score']
                    print(f"   {comp_name}: {score:.4f}")
                if 'percentage' in data:
                    print(f"      → {data['percentage']}")
            
            # STEP 4: Performance Summary
            total_time = cv_parse_time + jd_parse_time + scoring_time
            
            print(f"\n⏱️  PERFORMANCE SUMMARY:")
            print("-" * 70)
            print(f"   CV Parsing: {cv_parse_time:.2f}s")
            print(f"   JD Parsing: {jd_parse_time:.2f}s")
            print(f"   Scoring: {scoring_time:.2f}s")
            print(f"   Total Time: {total_time:.2f}s")
            
            # Assertions
            assert overall_score > 0.70  # Should be good match
            assert total_time < 30  # Should complete in <30s
            
            print("\n" + "=" * 70)
            print("✅ END-TO-END WORKFLOW TEST PASSED!")
            print("=" * 70)
            
        finally:
            # Cleanup
            if os.path.exists(cv_file_path):
                os.remove(cv_file_path)
    
    def test_learning_pathway_generation_workflow(self, learning_pathway_generator):
        """Test learning pathway generation workflow"""
        
        print("\n" + "=" * 70)
        print("WORKFLOW TEST: LEARNING PATHWAY GENERATION")
        print("=" * 70)
        
        # Missing skills scenario
        skill_gaps = ['Docker', 'Kubernetes', 'AWS']
        jd_data = {
            'text': 'Need Docker, Kubernetes, and AWS experience',
            'sections': {
                'required_skills': ['Docker', 'Kubernetes', 'AWS']
            }
        }
        
        print(f"\n📊 Skill Gaps: {', '.join(skill_gaps)}")
        
        # Generate 7-day pathway
        print("\n🗓️  Generating 7-day pathway...")
        start_time = time.time()
        
        pathway = learning_pathway_generator.generate_pathway(
            skill_gaps=skill_gaps,
            jd_data=jd_data,
            num_days=7
        )
        
        generation_time = time.time() - start_time
        
        assert pathway is not None
        assert pathway['success'] is True
        assert pathway['timeline_days'] == 7
        assert len(pathway['daily_plans']) == 7
        
        print(f"✅ Pathway generated in {generation_time:.2f}s")
        print(f"   Timeline: {pathway['timeline_days']} days")
        print(f"   Focus skills: {', '.join(pathway['focus_skills'])}")
        
        # Verify each day has proper structure
        for day_plan in pathway['daily_plans']:
            assert 'day' in day_plan
            assert 'focus' in day_plan
            assert 'tasks' in day_plan
            assert 'resources' in day_plan
        
        print("\n✅ LEARNING PATHWAY WORKFLOW PASSED!")
    
    def test_interview_preparation_workflow(self, interview_guidance):
        """Test interview preparation workflow"""
        
        print("\n" + "=" * 70)
        print("WORKFLOW TEST: INTERVIEW PREPARATION")
        print("=" * 70)
        
        candidate_skills = ['Python', 'FastAPI', 'Docker', 'PostgreSQL']
        
        print(f"\n👤 Candidate Skills: {', '.join(candidate_skills)}")
        
        # Generate questions
        print("\n❓ Generating interview questions...")
        start_time = time.time()
        
        questions = interview_guidance.generate_questions(
            skills=candidate_skills,
            num_questions=10,
            question_type='mixed'
        )
        
        generation_time = time.time() - start_time
        
        assert questions is not None
        assert questions['success'] is True
        assert len(questions['questions']) >= 5
        
        print(f"✅ Questions generated in {generation_time:.2f}s")
        print(f"   Total questions: {len(questions['questions'])}")
        
        # Show sample questions
        print(f"\n📝 Sample Questions:")
        for i, q in enumerate(questions['questions'][:3], 1):
            print(f"   {i}. {q['question'][:80]}...")
        
        # Evaluate a sample answer
        sample_question = questions['questions'][0]['question']
        sample_answer = """
        In my previous role, I led the migration of our monolithic application to microservices.
        The situation was that our application was becoming difficult to maintain and scale.
        My task was to design and implement a microservices architecture without disrupting operations.
        I created a phased migration plan, implemented feature flags for gradual rollout, and used
        Docker containers with Kubernetes orchestration. As a result, we reduced deployment time
        by 70% and improved system reliability to 99.9% uptime.
        """
        
        print(f"\n💬 Evaluating sample answer...")
        start_time = time.time()
        
        evaluation = interview_guidance.evaluate_answer(sample_question, sample_answer)
        
        eval_time = time.time() - start_time
        
        assert evaluation is not None
        assert 'score' in evaluation
        assert 0 <= evaluation['score'] <= 100
        
        print(f"✅ Answer evaluated in {eval_time:.2f}s")
        print(f"   Score: {evaluation['score']}/100")
        
        print("\n✅ INTERVIEW PREPARATION WORKFLOW PASSED!")


class TestUserScenarios:
    """Test realistic user scenarios"""
    
    def test_first_time_user_scenario(self, cv_parser, jd_parser, scoring_engine):
        """Test: First-time user uploads CV and pastes JD"""
        
        print("\n" + "=" * 70)
        print("USER SCENARIO: First-Time User")
        print("=" * 70)
        
        # Simulate user uploading a simple CV
        simple_cv = """
John Doe
Python Developer
john@email.com

Skills: Python, JavaScript, SQL
Experience: 3 years
"""
        
        simple_jd = """
Looking for Python developer
Required: Python, SQL
3+ years experience
"""
        
        # User action: Upload CV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(simple_cv)
            cv_path = f.name
        
        try:
            cv_data = cv_parser.parse(cv_path)
            assert cv_data is not None
            print("✅ CV uploaded and parsed")
            
            # User action: Paste JD
            jd_data = jd_parser.parse(simple_jd)
            assert jd_data is not None
            print("✅ JD pasted and parsed")
            
            # User action: Click "Analyze Match"
            result = scoring_engine.compute_match_score(cv_data, jd_data)
            assert result is not None
            assert result['overall_score'] > 0
            print(f"✅ Match computed: {result['overall_score']*100:.1f}%")
            
            print("\n✅ FIRST-TIME USER SCENARIO PASSED!")
            
        finally:
            os.remove(cv_path)
    
    def test_experienced_user_scenario(self, cv_analyzer, learning_pathway_generator):
        """Test: Experienced user analyzes CV and generates pathway"""
        
        print("\n" + "=" * 70)
        print("USER SCENARIO: Experienced User Optimization")
        print("=" * 70)
        
        cv_data = {
            'text': """Senior Developer
            Experience: Built APIs, worked on projects
            Skills: Python, Docker"""
        }
        
        # User action 1: Analyze CV quality
        print("\n📊 Analyzing CV quality...")
        analysis = cv_analyzer.analyze_cv(cv_data)
        
        assert analysis is not None
        assert 'score' in analysis
        assert 'improvements' in analysis
        
        print(f"✅ CV analyzed: {analysis['score']}/100")
        print(f"   Improvements suggested: {len(analysis['improvements'])}")
        
        # User action 2: Generate learning pathway
        print("\n🗓️  Generating learning pathway...")
        pathway = learning_pathway_generator.generate_pathway(
            skill_gaps=['Kubernetes', 'AWS'],
            jd_data={'text': 'Need Kubernetes and AWS'},
            num_days=14
        )
        
        assert pathway is not None
        assert pathway['success'] is True
        
        print(f"✅ Pathway generated: {pathway['timeline_days']} days")
        
        print("\n✅ EXPERIENCED USER SCENARIO PASSED!")