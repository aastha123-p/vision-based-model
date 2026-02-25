"""
FAISS Vector Store Module
Efficient similarity search using FAISS index
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")


class FAISSStore:
    """
    FAISS-based vector store for similarity search
    """

    def __init__(self, embedding_dim: int = 384, index_path: Optional[str] = None):
        """
        Initialize FAISS store
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load index
        """
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            self.index = None
            return

        self.embedding_dim = embedding_dim
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.index = None
        self.metadata = []  # Store metadata for each embedding

        # Create index directory if needed
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or load FAISS index"""
        try:
            if self.index_path and os.path.exists(self.index_path):
                self.load_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                # Create new flat index (L2 distance)
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            self.index = None

    def add_embedding(
        self,
        embedding: np.ndarray,
        metadata: Dict,
        session_id: int,
    ) -> bool:
        """
        Add embedding to index
        
        Args:
            embedding: Embedding vector
            metadata: Session metadata
            session_id: Session ID
            
        Returns:
            True if successful
        """
        if self.index is None:
            return False

        try:
            # Ensure embedding is float32
            embedding = embedding.astype(np.float32).reshape(1, -1)

            # Add to index
            self.index.add(embedding)

            # Store metadata
            metadata["session_id"] = session_id
            metadata["added_at"] = str(np.datetime64('now'))
            self.metadata.append(metadata)

            logger.info(f"Added embedding for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            return False

    def search_similar(
        self, embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar embeddings
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of (session_id, distance, metadata) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)

            # Search
            distances, indices = self.index.search(embedding, min(k, self.index.ntotal))

            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    distance = float(distances[0][i])
                    # Convert L2 distance to similarity (0-1)
                    similarity = 1 / (1 + distance)
                    metadata = self.metadata[idx]
                    results.append((metadata.get("session_id"), similarity, metadata))

            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def save_index(self, path: Optional[str] = None) -> bool:
        """
        Save index to disk
        
        Args:
            path: Save path
            
        Returns:
            True if successful
        """
        if self.index is None:
            return False

        try:
            path = path or self.index_path
            os.makedirs(os.path.dirname(path), exist_ok=True)

            faiss.write_index(self.index, path)

            # Save metadata
            metadata_path = path.replace(".bin", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f)

            logger.info(f"Saved FAISS index to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """
        Load index from disk
        
        Args:
            path: Load path
            
        Returns:
            True if successful
        """
        try:
            self.index = faiss.read_index(path)

            # Load metadata
            metadata_path = path.replace(".bin", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)

            logger.info(f"Loaded FAISS index from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def get_index_info(self) -> Dict:
        """Get index information"""
        if self.index is None:
            return {"status": "uninitialized"}

        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": "Flat (L2)",
            "metadata_count": len(self.metadata),
        }

    def clear_index(self) -> bool:
        """Clear index"""
        try:
            self._initialize_index()
            self.metadata = []
            logger.info("Cleared FAISS index")
            return True
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
