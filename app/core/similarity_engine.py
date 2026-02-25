"""
Similarity Engine for comparing embeddings
Uses cosine similarity for vector comparison
"""

import numpy as np
from scipy.spatial.distance import cosine
from typing import Tuple, Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimilarityEngine:
    """
    Compute similarity between embeddings
    """

    def __init__(self, metric: str = "cosine"):
        """
        Initialize similarity engine
        
        Args:
            metric: Distance metric ('cosine' recommended)
        """
        self.metric = metric

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        try:
            # Normalize embeddings
            vec1 = embedding1.astype(np.float32)
            vec2 = embedding2.astype(np.float32)

            # Cosine similarity = 1 - cosine_distance
            distance = cosine(vec1, vec2)
            similarity = 1 - distance

            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def batch_similarity(
        self, embedding: np.ndarray, embeddings_list: list
    ) -> list:
        """
        Compute similarity between one embedding and multiple embeddings
        
        Args:
            embedding: Single embedding vector
            embeddings_list: List of embedding vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for emb in embeddings_list:
            sim = self.compute_similarity(embedding, emb)
            similarities.append(sim)
        return similarities

    def match_threshold(self, similarity: float, threshold: float) -> bool:
        """
        Check if similarity exceeds threshold
        
        Args:
            similarity: Computed similarity score
            threshold: Minimum similarity threshold
            
        Returns:
            True if similarity >= threshold
        """
        return similarity >= threshold
