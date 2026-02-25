"""
Comparison Agent
Finds similar past cases for reference
"""

from typing import Dict, List
from sqlalchemy.orm import Session
from app.core.embedding_engine import EmbeddingEngine
from app.core.faiss_store import FAISSStore
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ComparisonAgent:
    """
    Finds similar past cases using embeddings
    """

    def __init__(self, db: Session):
        """Initialize comparison agent"""
        self.db = db
        self.embedding_engine = EmbeddingEngine()
        self.faiss_store = FAISSStore()

    def find_similar(
        self,
        form_data: Dict,
        vision_data: Dict,
        speech_data: Dict,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find similar past cases
        
        Args:
            form_data: Patient form data
            vision_data: Vision analysis
            speech_data: Speech analysis
            top_k: Number of similar cases to return
            
        Returns:
            List of similar cases
        """
        try:
            # Create multimodal embedding
            embedding = self.embedding_engine.create_multimodal_embedding(
                form_data, vision_data, speech_data
            )

            if embedding is None:
                logger.warning("Failed to create embedding for comparison")
                return []

            # Search for similar cases
            similar = self.faiss_store.search_similar(embedding, k=top_k)

            return [
                {
                    "session_id": session_id,
                    "similarity": float(similarity),
                    "metadata": metadata,
                }
                for session_id, similarity, metadata in similar
            ]

        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
