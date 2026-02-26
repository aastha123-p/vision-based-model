"""
Embedding Engine Module
Generates multimodal embeddings from text, vision, and speech features
Uses SentenceTransformers for text embeddings
"""

import numpy as np
from typing import Dict, Optional, List, Union
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not installed. Install with: pip install sentence-transformers")


class EmbeddingEngine:
    """
    Generate multimodal embeddings from text and features
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding engine
        
        Args:
            model_name: SentenceTransformer model name
        """
        if not hasattr(self, "_initialized"):
            self.model_name = model_name
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.model = SentenceTransformer(model_name)
                    logger.info(f"Loaded embedding model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading embedding model: {e}")
                    self.model = None
            else:
                self.model = None
            self._initialized = True

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None
        """
        if not self.model or not text:
            return None

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings or None
        """
        if not self.model or not texts:
            return None

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch embedding: {e}")
            return None

    def create_multimodal_embedding(
        self,
        form_data: Dict,
        vision_features: Dict,
        speech_features: Dict,
    ) -> Optional[np.ndarray]:
        """
        Create multimodal embedding from form, vision, and speech features
        
        Args:
            form_data: form submission data
            vision_features: vision analysis features
            speech_features: speech analysis features
            
        Returns:
            Combined embedding vector
        """
        try:
            # Combine all text information
            text_parts = []

            # Form data
            if form_data:
                text_parts.append(form_data.get("chief_complaint", ""))
                symptoms = form_data.get("symptoms", [])
                if symptoms:
                    text_parts.append(" ".join(symptoms))

            # Vision summary
            if vision_features:
                emotion = vision_features.get("emotion", "")
                if emotion:
                    text_parts.append(f"Patient appears {emotion}")

            # Speech summary
            if speech_features:
                sentiment = speech_features.get("sentiment", "")
                transcript = speech_features.get("transcript", "")
                if sentiment:
                    text_parts.append(f"Patient sentiment is {sentiment}")
                if transcript:
                    text_parts.append(transcript[:200])

            combined_text = " ".join(filter(None, text_parts))
            if not combined_text:
                combined_text = "No data available"

            # Generate embedding
            embedding = self.embed_text(combined_text)
            return embedding

        except Exception as e:
            logger.error(f"Error creating multimodal embedding: {e}")
            return None

    def embed_vision_features(self, vision_features: Dict) -> Optional[np.ndarray]:
        """
        Create embedding from vision features
        
        Args:
            vision_features: Vision analysis dictionary
            
        Returns:
            Embedding vector
        """
        try:
            text_description = self._vision_features_to_text(vision_features)
            return self.embed_text(text_description)
        except Exception as e:
            logger.error(f"Error embedding vision features: {e}")
            return None

    def embed_session_summary(self, session_summary: str) -> Optional[np.ndarray]:
        """
        Create embedding from session summary
        
        Args:
            session_summary: Session summary text
            
        Returns:
            Embedding vector
        """
        try:
            return self.embed_text(session_summary)
        except Exception as e:
            logger.error(f"Error embedding session summary: {e}")
            return None

    @staticmethod
    def _vision_features_to_text(vision_features: Dict) -> str:
        """Convert vision features to text"""
        parts = []
        if vision_features.get("emotion"):
            parts.append(f"Emotion: {vision_features['emotion']}")
        if vision_features.get("eye_strain_score"):
            parts.append(f"Eye strain: {vision_features['eye_strain_score']}")
        if vision_features.get("blink_rate"):
            parts.append(f"Blink rate: {vision_features['blink_rate']}")
        return " ".join(parts) if parts else "Monitor vision only"

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return config.EMBEDDING_DIM
