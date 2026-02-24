"""
Similarity Engine for Face Recognition Authentication

This module handles loading stored face embeddings and computing
similarity between live face embeddings and stored embeddings.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class SimilarityEngine:
    """
    Engine for comparing face embeddings and determining authentication matches.
    
    Uses cosine similarity or Euclidean distance to compare embeddings,
    with configurable threshold for authentication decisions.
    """
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        threshold: float = 0.5,
        metric: str = "cosine"
    ):
        """
        Initialize the similarity engine.
        
        Args:
            embeddings_dir: Directory containing stored embeddings
            threshold: Similarity threshold for authentication (0-1 for cosine, lower is better for euclidean)
            metric: Distance metric to use ('cosine' or 'euclidean')
        """
        self.embeddings_dir = embeddings_dir
        self.threshold = threshold
        self.metric = metric.lower()
        self.stored_embeddings: Dict[str, np.ndarray] = {}
        
        # Ensure embeddings directory exists
        os.makedirs(embeddings_dir, exist_ok=True)
    
    def load_embeddings(self) -> None:
        """
        Load all stored embeddings from the embeddings directory.
        
        Expected file format: JSON files with .json extension
        Each file should contain:
        {
            "user_id": "user_123",
            "embedding": [0.1, 0.2, ...],
            "created_at": "2024-01-01T00:00:00"
        }
        """
        self.stored_embeddings.clear()
        
        if not os.path.exists(self.embeddings_dir):
            return
        
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.embeddings_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        user_id = data.get('user_id')
                        embedding = data.get('embedding')
                        
                        if user_id and embedding:
                            self.stored_embeddings[user_id] = np.array(embedding)
                            print(f"Loaded embedding for user: {user_id}")
                except Exception as e:
                    print(f"Error loading embedding from {filename}: {str(e)}")
    
    def compute_similarity(
        self,
        live_embedding: np.ndarray,
        stored_embedding: np.ndarray
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            live_embedding: The embedding extracted from live capture
            stored_embedding: The stored embedding for comparison
            
        Returns:
            Similarity score (1 for cosine similarity, distance for euclidean)
        """
        # Reshape for sklearn functions
        live = live_embedding.reshape(1, -1)
        stored = stored_embedding.reshape(1, -1)
        
        if self.metric == "cosine":
            # Cosine similarity: higher is better (1 = identical)
            similarity = cosine_similarity(live, stored)[0][0]
            return float(similarity)
        else:
            # Euclidean distance: lower is better
            distance = euclidean_distances(live, stored)[0][0]
            return float(distance)
    
    def compare_with_stored(
        self,
        live_embedding: np.ndarray
    ) -> Tuple[bool, Optional[str], float]:
        """
        Compare live embedding with all stored embeddings.
        
        Args:
            live_embedding: The embedding extracted from live capture
            
        Returns:
            Tuple of (is_authenticated, user_id, confidence)
            - is_authenticated: True if match found above threshold
            - user_id: ID of matching user (None if no match)
            - confidence: Similarity score (cosine) or 1/distance (euclidean)
        """
        # Load embeddings if not already loaded
        if not self.stored_embeddings:
            self.load_embeddings()
        
        if not self.stored_embeddings:
            print("No stored embeddings found")
            return False, None, 0.0
        
        best_match = None
        best_score = -float('inf') if self.metric == "cosine" else float('inf')
        
        for user_id, stored_emb in self.stored_embeddings.items():
            try:
                score = self.compute_similarity(live_embedding, stored_emb)
                
                # For cosine: higher is better, for euclidean: lower is better
                if self.metric == "cosine":
                    if score > best_score:
                        best_score = score
                        best_match = user_id
                else:
                    if score < best_score:
                        best_score = score
                        best_match = user_id
                        
            except Exception as e:
                print(f"Error comparing with {user_id}: {str(e)}")
                continue
        
        # Determine authentication result based on metric
        if self.metric == "cosine":
            is_authenticated = best_score >= self.threshold
            confidence = best_score
        else:
            # For euclidean, convert distance to confidence-like score
            is_authenticated = best_score <= self.threshold
            confidence = 1.0 / (1.0 + best_score)  # Convert to 0-1 scale
        
        return is_authenticated, best_match, confidence
    
    def save_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save a new embedding for a user.
        
        Args:
            user_id: Unique identifier for the user
            embedding: Face embedding array
            metadata: Optional additional metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import datetime
            
            data = {
                "user_id": user_id,
                "embedding": embedding.tolist(),
                "created_at": datetime.datetime.now().isoformat()
            }
            
            if metadata:
                data.update(metadata)
            
            filename = f"{user_id}.json"
            filepath = os.path.join(self.embeddings_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update in-memory storage
            self.stored_embeddings[user_id] = embedding
            
            print(f"Embedding saved for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"Error saving embedding: {str(e)}")
            return False
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get a specific user's embedding.
        
        Args:
            user_id: The user ID to retrieve
            
        Returns:
            Embedding array if found, None otherwise
        """
        if user_id in self.stored_embeddings:
            return self.stored_embeddings[user_id]
        
        # Try loading from file
        filepath = os.path.join(self.embeddings_dir, f"{user_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return np.array(data.get('embedding', []))
            except Exception as e:
                print(f"Error loading embedding for {user_id}: {str(e)}")
        
        return None


def create_similarity_engine(
    embeddings_dir: str = "data/embeddings",
    threshold: float = 0.5,
    metric: str = "cosine"
) -> SimilarityEngine:
    """
    Factory function to create a similarity engine instance.
    
    Args:
        embeddings_dir: Directory containing stored embeddings
        threshold: Similarity threshold for authentication
        metric: Distance metric ('cosine' or 'euclidean')
        
    Returns:
        Configured SimilarityEngine instance
    """
    return SimilarityEngine(
        embeddings_dir=embeddings_dir,
        threshold=threshold,
        metric=metric
    )
