"""
Face Embedding Storage and Retrieval Module
Handles persistent storage and efficient lookup of face embeddings
"""

import json
import numpy as np
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from app.database.models import FaceEmbedding, Patient
from app.utils.logger import setup_logger
from app.core.similarity_engine import SimilarityEngine

logger = setup_logger(__name__)


class FaceEmbeddingStore:
    """
    Manages face embedding storage and retrieval
    """

    def __init__(self):
        self.similarity_engine = SimilarityEngine()

    def save_face_embedding(
        self, db: Session, patient_id: int, embedding: np.ndarray
    ) -> bool:
        """
        Save face embedding for a patient
        
        Args:
            db: Database session
            patient_id: Patient ID
            embedding: Face embedding array
            
        Returns:
            True if successful
        """
        try:
            # Convert numpy array to JSON-serializable format
            embedding_json = json.dumps(embedding.tolist())

            # Create face embedding record
            face_emb = FaceEmbedding(patient_id=patient_id, embedding=embedding_json)
            db.add(face_emb)

            # Also update patient's primary face embedding
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if patient:
                patient.face_embedding = embedding_json
                
            db.commit()
            logger.info(f"Saved face embedding for patient {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving face embedding: {e}")
            db.rollback()
            return False

    def get_patient_embeddings(
        self, db: Session, patient_id: int
    ) -> List[np.ndarray]:
        """
        Get all embeddings for a patient
        
        Args:
            db: Database session
            patient_id: Patient ID
            
        Returns:
            List of embedding arrays
        """
        try:
            embeddings_data = (
                db.query(FaceEmbedding)
                .filter(FaceEmbedding.patient_id == patient_id)
                .all()
            )

            embeddings = []
            for emb_data in embeddings_data:
                embedding_array = np.array(json.loads(emb_data.embedding))
                embeddings.append(embedding_array)

            return embeddings
        except Exception as e:
            logger.error(f"Error retrieving embeddings for patient {patient_id}: {e}")
            return []

    def get_primary_embedding(self, db: Session, patient_id: int) -> Optional[np.ndarray]:
        """
        Get primary/latest face embedding for a patient
        
        Args:
            db: Database session
            patient_id: Patient ID
            
        Returns:
            Embedding array or None
        """
        try:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if patient and patient.face_embedding:
                return np.array(json.loads(patient.face_embedding))
            return None
        except Exception as e:
            logger.error(f"Error retrieving primary embedding for patient {patient_id}: {e}")
            return None

    def compare_embeddings(
        self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.60
    ) -> Tuple[bool, float]:
        """
        Compare two embeddings and check if they match
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold (default 0.60)
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = self.similarity_engine.compute_similarity(embedding1, embedding2)
        is_match = self.similarity_engine.match_threshold(similarity, threshold)
        return is_match, similarity

    def find_match(
        self, db: Session, new_embedding: np.ndarray, threshold: float = 0.60
    ) -> Optional[Tuple[int, float]]:
        """
        Find matching patient for a face embedding
        
        Args:
            db: Database session
            new_embedding: Face embedding to match
            threshold: Similarity threshold
            
        Returns:
            Tuple of (patient_id, similarity_score) or None
        """
        try:
            all_patients = db.query(Patient).filter(Patient.face_embedding.isnot(None)).all()

            best_match = None
            best_similarity = 0

            for patient in all_patients:
                stored_embedding = np.array(json.loads(patient.face_embedding))
                is_match, similarity = self.compare_embeddings(
                    new_embedding, stored_embedding, threshold
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    if is_match:
                        best_match = (patient.id, similarity)

            return best_match if best_similarity >= threshold else None
        except Exception as e:
            logger.error(f"Error finding match: {e}")
            return None

    def delete_embedding(self, db: Session, embedding_id: int) -> bool:
        """
        Delete a face embedding
        
        Args:
            db: Database session
            embedding_id: Embedding ID
            
        Returns:
            True if successful
        """
        try:
            db.query(FaceEmbedding).filter(FaceEmbedding.id == embedding_id).delete()
            db.commit()
            logger.info(f"Deleted embedding {embedding_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            db.rollback()
            return False
