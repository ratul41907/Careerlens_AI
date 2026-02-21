"""
Embedding Engine - Generates semantic embeddings using SentenceTransformers
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Union
from loguru import logger
import os


class EmbeddingEngine:
    """Generate and cache embeddings for text using SentenceTransformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding engine
        
        Args:
            model_name: SentenceTransformer model to use
                       (all-MiniLM-L6-v2 is fast, 384-dim, good quality)
        """
        self.model_name = model_name
        self.model = None
        self.cache = {}  # Simple in-memory cache
        
        logger.info(f"Initializing EmbeddingEngine with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Load model (will download on first run)
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder="models")
            
            # Test the model
            test_embedding = self.model.encode("test", show_progress_bar=False)
            logger.info(f"✅ Model loaded successfully! Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               use_cache: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            use_cache: Whether to use cached embeddings
            show_progress: Show progress bar for batch encoding
            
        Returns:
            numpy array of embeddings (single vector or matrix)
        """
        # Convert single string to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Check cache
        if use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.cache:
                    cached_embeddings.append((i, self.cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                logger.debug(f"Generating embeddings for {len(uncached_texts)} new texts")
                new_embeddings = self.model.encode(
                    uncached_texts,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache[text] = embedding
                
                # Combine cached and new embeddings in correct order
                all_embeddings = [None] * len(texts)
                for idx, emb in cached_embeddings:
                    all_embeddings[idx] = emb
                for idx, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[idx] = emb
                
                embeddings = np.array(all_embeddings)
            else:
                # All cached
                embeddings = np.array([emb for _, emb in sorted(cached_embeddings)])
        
        else:
            # No caching
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        
        # Return single vector if single input
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Ensure 2D arrays for sklearn
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Convert from [-1, 1] to [0, 1] range
        # (though cosine similarity is usually [0, 1] for text)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def compute_similarity_matrix(self,
                                  embeddings1: np.ndarray,
                                  embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of embeddings
        
        Args:
            embeddings1: Matrix of embeddings (n x dim)
            embeddings2: Matrix of embeddings (m x dim)
            
        Returns:
            Similarity matrix (n x m)
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        # Normalize to [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2
        
        return similarity_matrix
    
    def find_most_similar(self,
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Dict]:
        """
        Find top-k most similar embeddings to query
        
        Args:
            query_embedding: Single query embedding
            candidate_embeddings: Matrix of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'index' and 'score'
        """
        # Compute similarities
        similarities = self.compute_similarity_matrix(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            {
                'index': int(idx),
                'score': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache = {}
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_items': len(self.cache),
            'model': self.model_name
        }


# Convenience functions
def create_engine(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingEngine:
    """Create and return embedding engine"""
    return EmbeddingEngine(model_name)


def compute_text_similarity(text1: str, text2: str, 
                           engine: EmbeddingEngine = None) -> float:
    """
    Quick function to compute similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        engine: Optional pre-initialized engine
        
    Returns:
        Similarity score (0 to 1)
    """
    if engine is None:
        engine = EmbeddingEngine()
    
    emb1 = engine.encode(text1)
    emb2 = engine.encode(text2)
    
    return engine.compute_similarity(emb1, emb2)