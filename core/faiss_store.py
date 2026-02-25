"""
FAISS Vector Store for Similarity Search (Phase 3 – Task 2)

This module provides FAISS-based vector storage for storing embeddings efficient
and performing similarity search.

Features:
- Store embeddings with case metadata
- Retrieve top similar cases
- Return similarity percentage
- Persist index to disk

Dependencies:
- FAISS (faiss-cpu or faiss-gpu)
- numpy
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import faiss
import json


class FAISSStore:
    """
    FAISS-based vector store for similarity search.
    
    Provides efficient storage and retrieval of embeddings using
    Facebook AI Similarity Search (FAISS).
    
    Attributes:
        embedding_dim: Dimension of the embedding vectors
        index: FAISS index
        case_ids: List of case IDs corresponding to embeddings
        metadata: List of metadata dictionaries for each case
    """
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "Flat"):
        """
        Initialize the FAISS store.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.case_ids: List[str] = []
        self.metadata: List[Dict] = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the index type."""
        if self.index_type == "Flat":
            # Flat index - exact search, good for small datasets
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVF":
            # IVF index - approximate search, faster for large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "HNSW":
            # HNSW index - fast approximate search with good accuracy
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            # Default to Flat index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        print(f"FAISS index initialized: {self.index_type}, dimension: {self.embedding_dim}")
    
    def add_embedding(
        self, 
        embedding: np.ndarray, 
        case_id: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add an embedding to the FAISS index.
        
        Args:
            embedding: Numpy array of shape (embedding_dim,) or (1, embedding_dim)
            case_id: Unique identifier for the case
            metadata: Optional metadata dictionary for the case
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure embedding is 2D array (required by FAISS)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Verify embedding dimension matches
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {embedding.shape[1]} does not match "
                    f"expected dimension {self.embedding_dim}"
                )
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Store case_id and metadata
            self.case_ids.append(case_id)
            self.metadata.append(metadata or {})
            
            return True
            
        except Exception as e:
            print(f"Error adding embedding: {e}")
            return False
    
    def add_embeddings(
        self, 
        embeddings: List[np.ndarray], 
        case_ids: List[str], 
        metadata_list: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add multiple embeddings to the FAISS index.
        
        Args:
            embeddings: List of numpy arrays
            case_ids: List of unique identifiers for each case
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(embeddings) != len(case_ids):
                raise ValueError("Number of embeddings must match number of case_ids")
            
            # Convert to 2D array if needed
            processed_embeddings = []
            for emb in embeddings:
                if emb.ndim == 1:
                    processed_embeddings.append(emb.reshape(1, -1))
                else:
                    processed_embeddings.append(emb)
            
            # Stack all embeddings
            embedding_matrix = np.vstack(processed_embeddings).astype('float32')
            
            # Verify dimensions
            if embedding_matrix.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {embedding_matrix.shape[1]} does not match "
                    f"expected dimension {self.embedding_dim}"
                )
            
            # Add to FAISS index
            self.index.add(embedding_matrix)
            
            # Store case_ids and metadata
            self.case_ids.extend(case_ids)
            
            # Handle metadata list
            if metadata_list:
                self.metadata.extend(metadata_list)
            else:
                self.metadata.extend([{}] * len(case_ids))
            
            return True
            
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            return False
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar cases in the FAISS index.
        
        Args:
            query_embedding: Numpy array of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of similar cases to return
            
        Returns:
            List of dictionaries containing case_id, similarity_score, 
            similarity_percentage, and metadata
        """
        try:
            # Ensure embedding is 2D array
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Adjust top_k if it's larger than the number of stored cases
            actual_top_k = min(top_k, self.index.ntotal)
            
            if actual_top_k == 0:
                print("No embeddings stored in the index")
                return []
            
            # Search the index
            distances, indices = self.index.search(query_embedding, actual_top_k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.case_ids):
                    similarity_percentage = self.get_similarity_percentage(distance)
                    results.append({
                        'rank': i + 1,
                        'case_id': self.case_ids[idx],
                        'distance': float(distance),
                        'similarity_score': float(1 / (1 + distance)),  # Convert distance to score
                        'similarity_percentage': similarity_percentage,
                        'metadata': self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching index: {e}")
            return []
    
    def get_similarity_percentage(self, distance: float) -> float:
        """
        Convert FAISS L2 distance to similarity percentage.
        
        Uses exponential decay to convert distance to 0-100% similarity.
        
        Args:
            distance: FAISS L2 distance
            
        Returns:
            Similarity percentage (0-100)
        """
        # Using exponential decay: similarity = 100 * e^(-distance)
        # This maps:
        #   distance = 0 -> similarity = 100%
        #   distance = 1 -> similarity = 36.8%
        #   distance = 2 -> similarity = 13.5%
        #   distance = 3 -> similarity = 5%
        
        similarity = 100 * np.exp(-distance)
        return max(0.0, min(100.0, float(similarity)))
    
    def save_index(self, path: str) -> bool:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            path: Directory path to save the index files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save case_ids
            case_ids_path = os.path.join(path, "case_ids.pkl")
            with open(case_ids_path, 'wb') as f:
                pickle.dump(self.case_ids, f)
            
            # Save metadata
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save configuration
            config = {
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Index saved to {path}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            path: Directory path containing the index files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            self.index = faiss.read_index(index_path)
            
            # Load case_ids
            case_ids_path = os.path.join(path, "case_ids.pkl")
            with open(case_ids_path, 'rb') as f:
                self.case_ids = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Update embedding_dim from loaded index
            self.embedding_dim = self.index.d
            
            print(f"Index loaded from {path}")
            print(f"   Total cases: {len(self.case_ids)}")
            print(f"   Embedding dimension: {self.embedding_dim}")
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the FAISS store.
        
        Returns:
            Dictionary containing store statistics
        """
        return {
            'total_cases': len(self.case_ids),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'index_is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'ntotal': self.index.ntotal if self.index else 0
        }
    
    def clear(self):
        """Clear all data from the store."""
        self._initialize_index()
        self.case_ids = []
        self.metadata = []
        print("FAISS store cleared")


def create_faiss_store(
    embedding_dim: int = 384,
    index_type: str = "Flat",
    load_path: Optional[str] = None
) -> FAISSStore:
    """
    Factory function to create a FAISS store.
    
    Args:
        embedding_dim: Dimension of the embedding vectors
        index_type: Type of FAISS index
        load_path: Optional path to load existing index
        
    Returns:
        FAISSStore instance
    """
    store = FAISSStore(embedding_dim=embedding_dim, index_type=index_type)
    
    if load_path and os.path.exists(load_path):
        store.load_index(load_path)
    
    return store


# Main test block
if __name__ == "__main__":
    print("=" * 60)
    print("FAISS Store Test")
    print("=" * 60)
    
    # Create a FAISS store
    print("\n1. Creating FAISS store...")
    embedding_dim = 384  # Match SentenceTransformer dimension
    store = FAISSStore(embedding_dim=embedding_dim, index_type="Flat")
    
    # Generate some test embeddings
    print("\n2. Generating test embeddings...")
    np.random.seed(42)
    test_embeddings = [
        np.random.randn(embedding_dim).astype('float32'),
        np.random.randn(embedding_dim).astype('float32'),
        np.random.randn(embedding_dim).astype('float32'),
        np.random.randn(embedding_dim).astype('float32'),
        np.random.randn(embedding_dim).astype('float32'),
    ]
    
    test_case_ids = [f"case_{i:03d}" for i in range(5)]
    test_metadata = [
        {"diagnosis": "Condition A", "patient_age": 30},
        {"diagnosis": "Condition B", "patient_age": 45},
        {"diagnosis": "Condition A", "patient_age": 25},
        {"diagnosis": "Condition C", "patient_age": 50},
        {"diagnosis": "Condition B", "patient_age": 35},
    ]
    
    # Add embeddings to store
    print("\n3. Adding embeddings to store...")
    success = store.add_embeddings(test_embeddings, test_case_ids, test_metadata)
    print(f"   Embeddings added: {success}")
    
    # Get stats
    stats = store.get_stats()
    print(f"\n4. Store Statistics:")
    print(f"   Total cases: {stats['total_cases']}")
    print(f"   Embedding dimension: {stats['embedding_dim']}")
    print(f"   Index type: {stats['index_type']}")
    
    # Search for similar cases
    print("\n5. Searching for similar cases...")
    query_embedding = test_embeddings[0]  # Query with first embedding
    results = store.search(query_embedding, top_k=3)
    
    print("\n   Search Results:")
    for result in results:
        print(f"   Rank {result['rank']}: {result['case_id']}")
        print(f"      Distance: {result['distance']:.4f}")
        print(f"      Similarity: {result['similarity_percentage']:.2f}%")
        print(f"      Metadata: {result['metadata']}")
    
    # Test similarity percentage conversion
    print("\n6. Similarity Percentage Conversion:")
    for dist in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        pct = store.get_similarity_percentage(dist)
        print(f"   Distance {dist:.1f} -> Similarity {pct:.2f}%")
    
    # Save index
    print("\n7. Saving index to disk...")
    save_path = "data/faiss_index/test"
    store.save_index(save_path)
    
    # Load index
    print("\n8. Loading index from disk...")
    new_store = create_faiss_store(load_path=save_path)
    print(f"   Loaded {new_store.get_stats()['total_cases']} cases")
    
    # Search again with loaded index
    print("\n9. Searching with loaded index...")
    results = new_store.search(query_embedding, top_k=3)
    print(f"   Found {len(results)} similar cases")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
