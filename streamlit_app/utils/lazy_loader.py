"""
Lazy loading utilities for better performance
"""
import streamlit as st
from typing import Callable, Any


class LazyLoader:
    """Load resources only when needed"""
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._loaded = False
        self._resource = None
    
    def get(self) -> Any:
        """Get resource, loading if necessary"""
        if not self._loaded:
            with st.spinner("Loading resource..."):
                self._resource = self.loader_func(*self.args, **self.kwargs)
                self._loaded = True
        return self._resource
    
    def is_loaded(self) -> bool:
        """Check if resource is loaded"""
        return self._loaded
    
    def unload(self):
        """Unload resource to free memory"""
        self._resource = None
        self._loaded = False


# Example usage
def lazy_load_model(model_name: str):
    """Lazy load ML model"""
    loader = LazyLoader(load_model_helper, model_name)
    return loader


def load_model_helper(model_name: str):
    """Helper to load model"""
    # Import here to avoid loading at module import time
    if model_name == 'embedding':
        from src.embeddings.embedding_engine import EmbeddingEngine
        return EmbeddingEngine()
    elif model_name == 'scoring':
        from src.scoring.scoring_engine import ScoringEngine
        embedding = load_model_helper('embedding')
        return ScoringEngine(embedding)
    return None