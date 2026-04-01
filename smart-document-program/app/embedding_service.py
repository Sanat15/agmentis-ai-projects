"""Embedding service for generating text embeddings."""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Uses all-mpnet-base-v2 by default:
    - 768-dimensional embeddings
    - Best balance of quality and speed
    - Trained on 1B+ sentence pairs
    """
    
    _instance: Optional['EmbeddingService'] = None
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default from settings)
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> 'EmbeddingService':
        """Get singleton instance of embedding service."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1 for normalized vectors)
        """
        # For normalized vectors, cosine similarity = dot product
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: Matrix of candidate vectors
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results


# Singleton accessor
def get_embedding_service() -> EmbeddingService:
    """Get the embedding service singleton."""
    return EmbeddingService.get_instance()


# Convenience functions
def encode_text(text: str) -> np.ndarray:
    """Encode single text to embedding."""
    return get_embedding_service().encode_text(text)


def encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode multiple texts to embeddings."""
    return get_embedding_service().encode_batch(texts, batch_size)
