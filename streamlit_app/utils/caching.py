"""
Caching utilities for performance optimization
"""
import streamlit as st
from functools import lru_cache
import hashlib

class ModelCache:        # ← removed double underscores
    """Cache ML models and embeddings for performance"""
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_engine():
        from src.embeddings.embedding_engine import EmbeddingEngine
        return EmbeddingEngine()
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_scoring_engine(_embedding_engine):
        from src.scoring.scoring_engine import ScoringEngine
        return ScoringEngine(_embedding_engine)
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_explainability_engine(_embedding_engine):
        from src.scoring.explainability import ExplainabilityEngine
        return ExplainabilityEngine(_embedding_engine)
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_cv_generator():
        from src.generation.cv_generator import CVGenerator
        return CVGenerator()
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_cv_parser():
        from src.parsers.cv_parser import CVParser
        return CVParser()
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_jd_parser():
        from src.parsers.jd_parser import JDParser
        return JDParser()


class ComputationCache:  # ← removed double underscores
    """Cache expensive computations"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def cache_cv_parse(_parser, file_hash: str, file_path: str):
        return _parser.parse(file_path)
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def cache_jd_parse(_parser, jd_text: str):
        text_hash = hashlib.md5(jd_text.encode()).hexdigest()
        return _parser.parse(jd_text), text_hash
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def cache_match_score(_scorer, cv_hash: str, jd_hash: str, cv_data: dict, jd_data: dict):
        return _scorer.compute_match_score(cv_data, jd_data)