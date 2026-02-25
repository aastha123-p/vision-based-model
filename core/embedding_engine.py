"""
Embedding Generation Module (Phase 3 – Task 1)

This module generates vector embeddings for text inputs using
Sentence Transformers (pretrained BERT-based models).

Inputs:
    - form_text: Main form/submission text
    - emotion_summary: Summary of emotional analysis
    - sentiment_summary: Summary of sentiment analysis

Output:
    - Structured dictionary containing embeddings for each input
"""

from typing import Dict, Optional, List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingEngine:
    """
    Generates vector embeddings for text inputs using Sentence Transformers.
    
    Uses pretrained models like all-MiniLM-L6-v2 for efficient
    and high-quality sentence embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Embedding Engine with a pretrained model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Default: all-MiniLM-L6-v2 (fast, 384-dim embeddings)
                       Other options: all-mpnet-base-v2 (higher quality, 768-dim)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SentenceTransformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            print(f"Embedding model '{model_name}' loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
        # Get embedding dimension for reference
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text input.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of embeddings (shape: [embedding_dim])
            Returns None if input is empty or None
        """
        # Handle empty or None inputs
        if text is None or (isinstance(text, str) and text.strip() == ""):
            return None
        
        # Ensure text is string
        text = str(text).strip()
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            return None
    
    def generate_embeddings(
        self,
        form_text: Optional[str] = None,
        emotion_summary: Optional[str] = None,
        sentiment_summary: Optional[str] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Generate embeddings for all three input texts.
        
        Args:
            form_text: Main form/submission text
            emotion_summary: Summary of emotional analysis
            sentiment_summary: Summary of sentiment analysis
            
        Returns:
            Dictionary containing embeddings for each input:
            {
                'form_text_embedding': np.ndarray or None,
                'emotion_summary_embedding': np.ndarray or None,
                'sentiment_summary_embedding': np.ndarray or None,
                'model_name': str,
                'embedding_dim': int,
                'device': str
            }
        """
        embeddings = {
            'form_text_embedding': self.generate_embedding(form_text),
            'emotion_summary_embedding': self.generate_embedding(emotion_summary),
            'sentiment_summary_embedding': self.generate_embedding(sentiment_summary),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device
        }
        
        return embeddings
    
    def get_embedding_info(self) -> Dict:
        """
        Get information about the loaded embedding model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length
        }
    
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
            Cosine similarity score (between -1 and 1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Compute cosine similarity using dot product
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


def generate_text_embeddings(
    form_text: str,
    emotion_summary: str,
    sentiment_summary: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Optional[np.ndarray]]:
    """
    Convenience function to generate embeddings for all inputs.
    
    Args:
        form_text: Main form/submission text
        emotion_summary: Summary of emotional analysis
        sentiment_summary: Summary of sentiment analysis
        model_name: Name of the SentenceTransformer model
        
    Returns:
        Dictionary containing embeddings and metadata
    """
    engine = EmbeddingEngine(model_name=model_name)
    return engine.generate_embeddings(
        form_text=form_text,
        emotion_summary=emotion_summary,
        sentiment_summary=sentiment_summary
    )


# Main test block
if __name__ == "__main__":
    # Sample inputs for testing
    sample_form_text = """
    I am writing to express my deep concern about the recent changes 
    to the workplace policy. The new requirements have significantly 
    impacted our team's productivity and morale.
    """
    
    sample_emotion_summary = """
    Primary emotion detected: Concern (75% intensity)
    Secondary emotions: Frustration (45%), Anxiety (30%)
    Overall emotional state: Negative with moderate stress indicators
    """
    
    sample_sentiment_summary = """
    Overall sentiment: Negative (0.75 confidence)
    Key themes: Policy concerns, productivity impact, morale issues
    Tone: Formal and professional with underlying frustration
    """
    
    print("=" * 60)
    print("Embedding Generation Test")
    print("=" * 60)
    
    # Initialize the embedding engine
    print("\n1. Initializing EmbeddingEngine...")
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    # Get model info
    model_info = engine.get_embedding_info()
    print(f"   Model: {model_info['model_name']}")
    print(f"   Embedding Dimension: {model_info['embedding_dimension']}")
    print(f"   Device: {model_info['device']}")
    
    # Generate embeddings
    print("\n2. Generating embeddings for sample inputs...")
    embeddings = engine.generate_embeddings(
        form_text=sample_form_text,
        emotion_summary=sample_emotion_summary,
        sentiment_summary=sample_sentiment_summary
    )
    
    # Display results
    print("\n3. Embedding Results:")
    print(f"   form_text_embedding: {embeddings['form_text_embedding'].shape if embeddings['form_text_embedding'] is not None else 'None'}")
    print(f"   emotion_summary_embedding: {embeddings['emotion_summary_embedding'].shape if embeddings['emotion_summary_embedding'] is not None else 'None'}")
    print(f"   sentiment_summary_embedding: {embeddings['sentiment_summary_embedding'].shape if embeddings['sentiment_summary_embedding'] is not None else 'None'}")
    
    # Test similarity between embeddings
    print("\n4. Computing Similarities:")
    if embeddings['form_text_embedding'] is not None and embeddings['emotion_summary_embedding'] is not None:
        sim_form_emotion = engine.compute_similarity(
            embeddings['form_text_embedding'],
            embeddings['emotion_summary_embedding']
        )
        print(f"   Form Text <-> Emotion Summary: {sim_form_emotion:.4f}")
    
    if embeddings['form_text_embedding'] is not None and embeddings['sentiment_summary_embedding'] is not None:
        sim_form_sentiment = engine.compute_similarity(
            embeddings['form_text_embedding'],
            embeddings['sentiment_summary_embedding']
        )
        print(f"   Form Text <-> Sentiment Summary: {sim_form_sentiment:.4f}")
    
    if embeddings['emotion_summary_embedding'] is not None and embeddings['sentiment_summary_embedding'] is not None:
        sim_emotion_sentiment = engine.compute_similarity(
            embeddings['emotion_summary_embedding'],
            embeddings['sentiment_summary_embedding']
        )
        print(f"   Emotion Summary <-> Sentiment Summary: {sim_emotion_sentiment:.4f}")
    
    # Test empty/None input handling
    print("\n5. Testing Empty/None Input Handling:")
    empty_embeddings = engine.generate_embeddings(
        form_text="",
        emotion_summary=None,
        sentiment_summary="Valid sentiment text"
    )
    print(f"   Empty form_text embedding: {empty_embeddings['form_text_embedding']}")
    print(f"   None emotion_summary embedding: {empty_embeddings['emotion_summary_embedding']}")
    print(f"   Valid sentiment_summary embedding: {empty_embeddings['sentiment_summary_embedding'].shape if empty_embeddings['sentiment_summary_embedding'] is not None else 'None'}")
    
    # Test convenience function
    print("\n6. Testing Convenience Function:")
    quick_embeddings = generate_text_embeddings(
        form_text="Quick test",
        emotion_summary="Emotion test",
        sentiment_summary="Sentiment test"
    )
    print(f"   Quick embeddings generated: {quick_embeddings['form_text_embedding'] is not None}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
