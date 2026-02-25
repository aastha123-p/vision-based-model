"""
Learning Agent
Stores embeddings for continuous learning
"""

from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.core.embedding_engine import EmbeddingEngine
from app.core.faiss_store import FAISSStore
from app.database.models import Session as SessionModel
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LearningAgent:
    """
    Stores session embeddings for future reference
    """

    def __init__(self, db: Session):
        """
        Initialize learning agent
        
        Args:
            db: Database session
        """
        self.db = db
        self.embedding_engine = EmbeddingEngine()
        self.faiss_store = FAISSStore()

    def store_session(
        self, patient_id: int, condition: str, features: Dict
    ) -> bool:
        """
        Store session for learning
        
        Args:
            patient_id: Patient ID
            condition: Predicted condition
            features: Session features
            
        Returns:
            True if successful
        """
        try:
            # Create embedding
            embedding = self.embedding_engine.create_multimodal_embedding(
                features.get("form", {}),
                features.get("vision", {}),
                features.get("speech", {}),
            )

            if embedding is None:
                logger.warning("Failed to create embedding for learning")
                return False

            # Store in FAISS
            metadata = {
                "patient_id": patient_id,
                "condition": condition,
                "features": {
                    "symptom_count": len(features.get("form", {}).get("symptoms", [])),
                    "emotion": features.get("vision", {}).get("emotion"),
                    "sentiment": features.get("speech", {}).get("sentiment"),
                },
            }

            success = self.faiss_store.add_embedding(
                embedding, metadata, session_id=patient_id
            )

            if success:
                # Save FAISS index
                self.faiss_store.save_index()
                logger.info(f"Stored session for patient {patient_id}")

            return success

        except Exception as e:
            logger.error(f"Error storing session: {e}")
            return False

    def get_similar_sessions(
        self, features: Dict, top_k: int = 5
    ) -> list:
        """
        Get similar past sessions
        
        Args:
            features: Current session features
            top_k: Number of results
            
        Returns:
            List of similar sessions
        """
        try:
            # Create embedding for query
            embedding = self.embedding_engine.create_multimodal_embedding(
                features.get("form", {}),
                features.get("vision", {}),
                features.get("speech", {}),
            )

            if embedding is None:
                return []

            # Search FAISS
            results = self.faiss_store.search_similar(embedding, k=top_k)

            return results

        except Exception as e:
            logger.error(f"Error retrieving similar sessions: {e}")
            return []
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            index_path: Path to store/load FAISS index
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
            auto_load: Whether to load existing index if available
        """
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.embedding_dim = self.embedding_engine.embedding_dim
        self.index_path = index_path
        self.index_type = index_type
        
        # Initialize FAISS store
        self.faiss_store = None
        
        # Try to load existing index
        if auto_load and os.path.exists(index_path):
            try:
                self.faiss_store = create_faiss_store(
                    embedding_dim=self.embedding_dim,
                    index_type=index_type,
                    load_path=index_path
                )
                print(f"Loaded existing FAISS index from {index_path}")
            except Exception as e:
                print(f"Could not load existing index: {e}")
                self.faiss_store = FAISSStore(
                    embedding_dim=self.embedding_dim,
                    index_type=index_type
                )
        else:
            self.faiss_store = FAISSStore(
                embedding_dim=self.embedding_dim,
                index_type=index_type
            )
        
        print(f"Learning Agent initialized with model: {embedding_model}")
    
    def store_case(
        self,
        form_text: str,
        emotion_summary: str,
        sentiment_summary: str,
        diagnosis: str,
        treatment: Optional[str] = None,
        patient_info: Optional[Dict] = None,
        save_index: bool = True
    ) -> bool:
        """
        Store a case with its embeddings for future similarity search.
        
        Args:
            form_text: Main form/submission text
            emotion_summary: Summary of emotional analysis
            sentiment_summary: Summary of sentiment analysis
            diagnosis: Diagnosed condition
            treatment: Treatment/medication provided (optional)
            patient_info: Additional patient information (optional)
            save_index: Whether to save index to disk after adding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings for all inputs
            embeddings = self.embedding_engine.generate_embeddings(
                form_text=form_text,
                emotion_summary=emotion_summary,
                sentiment_summary=sentiment_summary
            )
            
            # Combine all embeddings into a single vector (average)
            valid_embeddings = []
            if embeddings['form_text_embedding'] is not None:
                valid_embeddings.append(embeddings['form_text_embedding'])
            if embeddings['emotion_summary_embedding'] is not None:
                valid_embeddings.append(embeddings['emotion_summary_embedding'])
            if embeddings['sentiment_summary_embedding'] is not None:
                valid_embeddings.append(embeddings['sentiment_summary_embedding'])
            
            if not valid_embeddings:
                print("No valid embeddings generated")
                return False
            
            # Average the embeddings to create a combined case embedding
            combined_embedding = sum(valid_embeddings) / len(valid_embeddings)
            
            # Generate unique case ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_id = f"case_{timestamp}"
            
            # Prepare metadata
            metadata = {
                'diagnosis': diagnosis,
                'treatment': treatment,
                'patient_info': patient_info or {},
                'form_text_preview': form_text[:200] if form_text else "",
                'emotion_summary': emotion_summary,
                'sentiment_summary': sentiment_summary,
                'created_at': datetime.now().isoformat()
            }
            
            # Add to FAISS store
            success = self.faiss_store.add_embedding(
                embedding=combined_embedding,
                case_id=case_id,
                metadata=metadata
            )
            
            if success and save_index:
                # Save index to disk
                self.faiss_store.save_index(self.index_path)
                print(f"Case {case_id} stored and index saved")
            
            return success
            
        except Exception as e:
            print(f"Error storing case: {e}")
            return False
    
    def find_similar_cases(
        self,
        query_embedding: Any,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases using a pre-computed embedding.
        
        Args:
            query_embedding: Numpy array of query embedding
            top_k: Number of similar cases to return
            
        Returns:
            List of similar case dictionaries
        """
        try:
            results = self.faiss_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            return results
        except Exception as e:
            print(f"Error finding similar cases: {e}")
            return []
    
    def get_top_similar_cases(
        self,
        form_text: str,
        emotion_summary: str,
        sentiment_summary: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top similar cases based on form data, emotion, and sentiment.
        
        This is a convenience method that generates embeddings from the
        input texts and searches for similar cases.
        
        Args:
            form_text: Main form/submission text
            emotion_summary: Summary of emotional analysis
            sentiment_summary: Summary of sentiment analysis
            top_k: Number of similar cases to return
            
        Returns:
            List of similar case dictionaries with:
            - case_id: Unique identifier
            - similarity_percentage: Similarity score as percentage
            - diagnosis: Diagnosis from the similar case
            - treatment: Treatment from the similar case
            - metadata: Full metadata
        """
        try:
            # Generate embeddings for query
            embeddings = self.embedding_engine.generate_embeddings(
                form_text=form_text,
                emotion_summary=emotion_summary,
                sentiment_summary=sentiment_summary
            )
            
            # Combine all embeddings into a single vector
            valid_embeddings = []
            if embeddings['form_text_embedding'] is not None:
                valid_embeddings.append(embeddings['form_text_embedding'])
            if embeddings['emotion_summary_embedding'] is not None:
                valid_embeddings.append(embeddings['emotion_summary_embedding'])
            if embeddings['sentiment_summary_embedding'] is not None:
                valid_embeddings.append(embeddings['sentiment_summary_embedding'])
            
            if not valid_embeddings:
                print("No valid embeddings generated from query")
                return []
            
            # Average the embeddings
            combined_embedding = sum(valid_embeddings) / len(valid_embeddings)
            
            # Search for similar cases
            results = self.faiss_store.search(
                query_embedding=combined_embedding,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'case_id': result['case_id'],
                    'similarity_percentage': round(result['similarity_percentage'], 2),
                    'similarity_score': round(result['similarity_score'], 4),
                    'distance': round(result['distance'], 4),
                    'diagnosis': result['metadata'].get('diagnosis', 'N/A'),
                    'treatment': result['metadata'].get('treatment', 'N/A'),
                    'emotion_summary': result['metadata'].get('emotion_summary', ''),
                    'sentiment_summary': result['metadata'].get('sentiment_summary', ''),
                    'created_at': result['metadata'].get('created_at', 'N/A')
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error getting similar cases: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the learning agent.
        
        Returns:
            Dictionary with agent statistics
        """
        faiss_stats = self.faiss_store.get_stats()
        embedding_info = self.embedding_engine.get_embedding_info()
        
        return {
            'total_cases': faiss_stats['total_cases'],
            'embedding_model': embedding_info['model_name'],
            'embedding_dimension': embedding_info['embedding_dimension'],
            'index_type': self.index_type,
            'index_path': self.index_path
        }
    
    def save_index(self) -> bool:
        """
        Save the FAISS index to disk.
        
        Returns:
            True if successful, False otherwise
        """
        return self.faiss_store.save_index(self.index_path)
    
    def reload_index(self) -> bool:
        """
        Reload the FAISS index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.index_path):
                self.faiss_store.load_index(self.index_path)
                return True
            return False
        except Exception as e:
            print(f"Error reloading index: {e}")
            return False
    
    def clear_index(self):
        """Clear all cases from the index."""
        self.faiss_store.clear()
        print("Index cleared")


def create_learning_agent(
    embedding_model: str = "all-MiniLM-L6-v2",
    index_path: str = "data/faiss_index",
    index_type: str = "Flat"
) -> LearningAgent:
    """
    Factory function to create a LearningAgent.
    
    Args:
        embedding_model: Name of the SentenceTransformer model
        index_path: Path to store/load FAISS index
        index_type: Type of FAISS index
        
    Returns:
        LearningAgent instance
    """
    return LearningAgent(
        embedding_model=embedding_model,
        index_path=index_path,
        index_type=index_type,
        auto_load=True
    )


# Main test block
if __name__ == "__main__":
    print("=" * 60)
    print("Learning Agent Test")
    print("=" * 60)
    
    # Create learning agent
    print("\n1. Creating Learning Agent...")
    agent = LearningAgent(
        embedding_model="all-MiniLM-L6-v2",
        index_path="data/faiss_index/test",
        index_type="Flat",
        auto_load=False  # Start fresh for testing
    )
    
    # Get agent stats
    stats = agent.get_stats()
    print(f"   Agent initialized:")
    print(f"   - Embedding model: {stats['embedding_model']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    
    # Store sample cases
    print("\n2. Storing sample cases...")
    
    sample_cases = [
        {
            'form_text': "Patient presents with chronic fatigue, difficulty sleeping, and persistent sadness. Symptoms have been ongoing for 3 months.",
            'emotion_summary': "Primary emotion: Sadness (80%), Secondary: Fatigue (60%)",
            'sentiment_summary': "Negative sentiment with melancholic tone",
            'diagnosis': "Depressive Disorder",
            'treatment': "St. John's Wort 300mg twice daily + Therapy"
        },
        {
            'form_text': "Patient reports anxiety attacks, racing thoughts, restlessness, and difficulty concentrating. Symptoms exacerbated by stress.",
            'emotion_summary': "Primary emotion: Anxiety (85%), Secondary: Restlessness (70%)",
            'sentiment_summary': "Negative sentiment with worried tone",
            'diagnosis': "Generalized Anxiety Disorder",
            'treatment': "Lavender oil 80mg daily + Breathing exercises"
        },
        {
            'form_text': "Patient complains of joint pain, stiffness, and reduced mobility. Symptoms worse in morning and improve with movement.",
            'emotion_summary': "Primary emotion: Frustration (65%), Secondary: Discomfort (55%)",
            'sentiment_summary': "Negative sentiment with discomfort indicators",
            'diagnosis': "Arthritis",
            'treatment': "Rhus Tox 30C + Warm compress therapy"
        },
        {
            'form_text': "Patient experiences frequent headaches, sensitivity to light, and nausea. Migraine episodes occur 2-3 times per month.",
            'emotion_summary': "Primary emotion: Distress (75%), Secondary: Nausea (50%)",
            'sentiment_summary': "Negative sentiment with pain indicators",
            'diagnosis': "Migraine",
            'treatment': "Belladonna 30C + Feverfew supplement"
        },
        {
            'form_text': "Patient reports digestive issues, bloating, and irregular bowel movements. Symptoms correlate with certain food intake.",
            'emotion_summary': "Primary emotion: Discomfort (70%), Secondary: Frustration (45%)",
            'sentiment_summary': "Negative sentiment with digestive concern",
            'diagnosis': "Irritable Bowel Syndrome",
            'treatment': "Nux Vomica 30C + Dietary modification"
        }
    ]
    
    for i, case in enumerate(sample_cases):
        success = agent.store_case(
            form_text=case['form_text'],
            emotion_summary=case['emotion_summary'],
            sentiment_summary=case['sentiment_summary'],
            diagnosis=case['diagnosis'],
            treatment=case['treatment'],
            patient_info={'case_number': i + 1},
            save_index=False  # Don't save yet
        )
        print(f"   Case {i+1} stored: {success}")
    
    # Save index
    print("\n3. Saving index to disk...")
    agent.save_index()
    
    # Get updated stats
    stats = agent.get_stats()
    print(f"   Total cases stored: {stats['total_cases']}")
    
    # Test similarity search with a new query
    print("\n4. Testing similarity search...")
    query = {
        'form_text': "Patient has been feeling very tired lately, having trouble sleeping at night, and often feels down and hopeless.",
        'emotion_summary': "Primary emotion: Sadness (75%), Secondary: Fatigue (65%)",
        'sentiment_summary': "Negative sentiment, showing signs of depression"
    }
    
    similar_cases = agent.get_top_similar_cases(
        form_text=query['form_text'],
        emotion_summary=query['emotion_summary'],
        sentiment_summary=query['sentiment_summary'],
        top_k=3
    )
    
    print("\n   Similar Cases Found:")
    print("   " + "-" * 50)
    for case in similar_cases:
        print(f"   Rank {case['rank']}: {case['case_id']}")
        print(f"   Similarity: {case['similarity_percentage']:.2f}%")
        print(f"   Diagnosis: {case['diagnosis']}")
        print(f"   Treatment: {case['treatment']}")
        print()
    
    # Test with a different query (anxiety)
    print("\n5. Testing with anxiety-related query...")
    query2 = {
        'form_text': "Patient experiences intense worry, nervousness, and panic about everyday situations. Can't seem to relax.",
        'emotion_summary': "Primary emotion: Anxiety (90%), Secondary: Worry (75%)",
        'sentiment_summary': "Negative sentiment, high stress indicators"
    }
    
    similar_cases2 = agent.get_top_similar_cases(
        form_text=query2['form_text'],
        emotion_summary=query2['emotion_summary'],
        sentiment_summary=query2['sentiment_summary'],
        top_k=2
    )
    
    print("\n   Similar Cases Found:")
    print("   " + "-" * 50)
    for case in similar_cases2:
        print(f"   Rank {case['rank']}: {case['case_id']}")
        print(f"   Similarity: {case['similarity_percentage']:.2f}%")
        print(f"   Diagnosis: {case['diagnosis']}")
        print(f"   Treatment: {case['treatment']}")
        print()
    
    # Reload and verify
    print("\n6. Testing index reload...")
    new_agent = create_learning_agent(
        embedding_model="all-MiniLM-L6-v2",
        index_path="data/faiss_index/test"
    )
    new_stats = new_agent.get_stats()
    print(f"   Loaded {new_stats['total_cases']} cases from disk")
    
    # Search again with reloaded agent
    similar_cases_reloaded = new_agent.get_top_similar_cases(
        form_text=query['form_text'],
        emotion_summary=query['emotion_summary'],
        sentiment_summary=query['sentiment_summary'],
        top_k=2
    )
    
    print("\n   Search with reloaded index:")
    for case in similar_cases_reloaded:
        print(f"   {case['case_id']}: {case['similarity_percentage']:.2f}% - {case['diagnosis']}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
