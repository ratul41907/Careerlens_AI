"""
Complete End-to-End Pipeline Test
Test the entire user journey from CV upload to results
"""
import pytest
import time
import tempfile
import os


class TestCompletePipeline:
    """Test complete user pipeline"""
    
    def test_full_user_journey(self, cv_parser, jd_parser, scoring_engine, 
                               cv_analyzer, interview_guidance, learning_pathway_generator):
        """Test complete user journey through all features"""
        
        print("\n" + "=" * 70)
        print("COMPLETE PIPELINE TEST: Full User Journey")
        print("=" * 70)
        
        # STEP 1: User uploads CV
        print("\n📤 STEP 1: User Uploads CV")
        print("-" * 70)
        
        cv_text = """
JANE SMITH
Full-Stack Developer
jane.smith@email.com | +1-555-9999

SUMMARY
Full-stack developer with 4 years experience in web applications.

EXPERIENCE
Full-Stack Developer | WebCorp | 2022-Present
- Built web applications using React and Node.js
- Developed REST APIs
- Worked with MongoDB database

Junior Developer | StartupXYZ | 2020-2022
- Learned Python and JavaScript
- Helped with frontend development

SKILLS
JavaScript, React, Node.js, MongoDB, HTML, CSS, Git

EDUCATION
BS Computer Science | State University | 2020
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(cv_text)
            cv_path = f.name
        
        try:
            cv_parse_start = time.time()
            cv_data = cv_parser.parse(cv_path)
            cv_parse_time = time.time() - cv_parse_start
            
            assert cv_data is not None
            print(f"✅ CV uploaded and parsed in {cv_parse_time:.2f}s")
            
            # STEP 2: User pastes JD
            print("\n📋 STEP 2: User Pastes Job Description")
            print("-" * 70)
            
            jd_text = """
Senior Full-Stack Developer

Required Skills:
- JavaScript, React, Node.js
- Python, FastAPI
- Docker, Kubernetes
- PostgreSQL
- 5+ years experience

Preferred:
- AWS
- TypeScript
- GraphQL
"""
            
            jd_parse_start = time.time()
            jd_data = jd_parser.parse(jd_text)
            jd_parse_time = time.time() - jd_parse_start
            
            assert jd_data is not None
            print(f"✅ JD pasted and parsed in {jd_parse_time:.2f}s")
            
            # STEP 3: User clicks "Analyze Match"
            print("\n🎯 STEP 3: Computing Match Score")
            print("-" * 70)
            
            match_start = time.time()
            match_result = scoring_engine.compute_match_score(cv_data, jd_data)
            match_time = time.time() - match_start
            
            assert match_result is not None
            overall_score = match_result['overall_score']
            print(f"✅ Match computed in {match_time:.2f}s")
            print(f"   Overall Score: {overall_score*100:.1f}%")
            
            # STEP 4: User views CV analysis
            print("\n📊 STEP 4: Analyzing CV Quality")
            print("-" * 70)
            
            analysis_start = time.time()
            cv_analysis = cv_analyzer.analyze_cv(cv_data, jd_data)
            analysis_time = time.time() - analysis_start
            
            assert cv_analysis is not None
            print(f"✅ CV analyzed in {analysis_time:.2f}s")
            print(f"   Quality Score: {cv_analysis['score']}/100")
            print(f"   Grade: {cv_analysis['grade']}")
            print(f"   Issues found: {cv_analysis['total_issues']}")
            
            # STEP 5: User generates learning pathway
            print("\n🗓️  STEP 5: Generating Learning Pathway")
            print("-" * 70)
            
            # Get missing skills from match result
            missing_skills = []
            if 'breakdown' in match_result:
                req_details = match_result['breakdown'].get('required_skills', {}).get('details', {})
                missing_skills = req_details.get('missing_skills', ['Python', 'Docker', 'PostgreSQL'])
            
            if not missing_skills:
                missing_skills = ['Python', 'Docker', 'PostgreSQL']
            
            pathway_start = time.time()
            pathway = learning_pathway_generator.generate_pathway(
                skill_gaps=missing_skills[:3],  # Top 3 skills
                jd_data=jd_data,
                num_days=7
            )
            pathway_time = time.time() - pathway_start
            
            assert pathway is not None
            assert pathway['success'] is True
            print(f"✅ Pathway generated in {pathway_time:.2f}s")
            print(f"   Timeline: {pathway['timeline_days']} days")
            print(f"   Focus skills: {', '.join(pathway['focus_skills'][:3])}")
            
            # STEP 6: User gets interview questions
            print("\n❓ STEP 6: Generating Interview Questions")
            print("-" * 70)
            
            candidate_skills = cv_data['sections'].get('skills', ['JavaScript', 'React'])
            
            questions_start = time.time()
            questions = interview_guidance.generate_questions(
                skills=candidate_skills[:5],  # Top 5 skills
                num_questions=5,
                question_type='mixed'
            )
            questions_time = time.time() - questions_start
            
            assert questions is not None
            assert questions['success'] is True
            print(f"✅ Questions generated in {questions_time:.2f}s")
            print(f"   Total questions: {len(questions['questions'])}")
            
            # WORKFLOW SUMMARY
            total_time = (cv_parse_time + jd_parse_time + match_time + 
                         analysis_time + pathway_time + questions_time)
            
            print("\n" + "=" * 70)
            print("WORKFLOW SUMMARY")
            print("=" * 70)
            print(f"\n⏱️  TIME BREAKDOWN:")
            print(f"   CV Upload & Parse:     {cv_parse_time:.2f}s")
            print(f"   JD Parse:              {jd_parse_time:.2f}s")
            print(f"   Match Computation:     {match_time:.2f}s")
            print(f"   CV Analysis:           {analysis_time:.2f}s")
            print(f"   Learning Pathway:      {pathway_time:.2f}s")
            print(f"   Interview Questions:   {questions_time:.2f}s")
            print(f"   {'─' * 40}")
            print(f"   TOTAL:                 {total_time:.2f}s")
            
            print(f"\n✅ COMPLETE USER JOURNEY SUCCESSFUL!")
            print(f"   Total workflow time: {total_time:.2f}s")
            
            # Assertions for acceptable performance
            assert total_time < 120  # Complete workflow under 2 minutes
            assert cv_parse_time < 5
            assert jd_parse_time < 3
            assert match_time < 5
            
            print("\n" + "=" * 70)
            
        finally:
            if os.path.exists(cv_path):
                os.remove(cv_path)