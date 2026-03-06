"""
API Client for Streamlit Frontend
Handles all backend API calls with error handling and caching
"""
import requests
import streamlit as st
from typing import Dict, List, Optional
import json

class APIClient:
    """Client for CareerLens AI backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def match_cv_jd(self, cv_text: str, jd_text: str) -> Dict:
        """Match CV against JD"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/match",
                json={"cv_text": cv_text, "jd_text": jd_text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def generate_cv(self, cv_data: Dict) -> Dict:
        """Generate CV"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/cv/generate",
                json=cv_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_interview_questions(self, skills: List[str], num_questions: int = 10) -> Dict:
        """Get interview questions"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/interview/questions",
                json={"skills": skills, "num_questions": num_questions},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """Evaluate interview answer"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/interview/evaluate",
                json={"question": question, "answer": answer},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def parse_cv(self, file_bytes: bytes, filename: str) -> Dict:
        """Parse CV file"""
        try:
            files = {'file': (filename, file_bytes)}
            response = self.session.post(
                f"{self.base_url}/api/parse/cv",
                files=files,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None

# Singleton instance
@st.cache_resource
def get_api_client():
    """Get cached API client instance"""
    return APIClient()