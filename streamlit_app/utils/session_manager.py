"""
Session State Manager
Manages cross-page data persistence
"""
import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime

class SessionManager:
    """Manage session state across pages"""
    
    @staticmethod
    def init():
        """Initialize session state with defaults"""
        defaults = {
            # User profile
            'user_profile': None,
            
            # CV-JD Matcher data
            'last_match_result': None,
            'match_history': [],
            
            # CV Generator data
            'generated_cvs': [],
            'current_cv_data': None,
            
            # Interview Prep data
            'interview_questions': None,
            'practice_sessions': [],
            
            # Analytics data
            'analytics_data': {
                'total_matches': 0,
                'avg_score': 87.0,
                'cvs_generated': 0,
                'interviews_practiced': 0,
                'match_history': [],
                'skill_gaps': {}
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def save_match_result(result: Dict):
        """Save CV-JD match result"""
        st.session_state.last_match_result = result
        
        # Add to history
        st.session_state.match_history.append({
            'timestamp': datetime.now().isoformat(),
            'score': result['overall_score'],
            'breakdown': result['breakdown']
        })
        
        # Update analytics
        st.session_state.analytics_data['total_matches'] += 1
        
        # Update skill gaps
        for skill in result.get('missing_skills', []):
            if skill not in st.session_state.analytics_data['skill_gaps']:
                st.session_state.analytics_data['skill_gaps'][skill] = 0
            st.session_state.analytics_data['skill_gaps'][skill] += 1
    
    @staticmethod
    def save_generated_cv(cv_data: Dict, cv_path: str):
        """Save generated CV"""
        st.session_state.generated_cvs.append({
            'timestamp': datetime.now().isoformat(),
            'data': cv_data,
            'path': cv_path
        })
        
        # Update analytics
        st.session_state.analytics_data['cvs_generated'] += 1
    
    @staticmethod
    def save_interview_session(questions: Dict, evaluations: Optional[List[Dict]] = None):
        """Save interview practice session"""
        st.session_state.practice_sessions.append({
            'timestamp': datetime.now().isoformat(),
            'questions': questions,
            'evaluations': evaluations or []
        })
        
        # Update analytics
        total_questions = 0
        for category, cat_questions in questions.get('by_category', {}).items():
            total_questions += len(cat_questions)
        
        st.session_state.analytics_data['interviews_practiced'] += total_questions
    
    @staticmethod
    def get_analytics():
        """Get current analytics data"""
        # Calculate average score
        if st.session_state.match_history:
            scores = [m['score'] for m in st.session_state.match_history]
            st.session_state.analytics_data['avg_score'] = sum(scores) / len(scores)
        
        return st.session_state.analytics_data
    
    @staticmethod
    def clear_all():
        """Clear all session data"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        SessionManager.init()

# Initialize on import
SessionManager.init()