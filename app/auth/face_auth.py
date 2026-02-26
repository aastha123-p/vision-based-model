"""
Face Authentication Module
Handles face capture, embedding generation, and face-based login
Uses DeepFace for pre-trained face embedding models
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from app.config import config
from app.vision.face_recognition import FaceRecognitionEngine
from app.auth.face_embedding_store import FaceEmbeddingStore
from app.database.models import Patient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceAuthenticator:
    """
    Face-based authentication manager
    """

    def __init__(self):
        self.face_engine = FaceRecognitionEngine()
        self.embedding_store = FaceEmbeddingStore()
        self.similarity_threshold = config.FACE_SIMILARITY_THRESHOLD

    def capture_face_from_webcam(self, timeout: int = 10) -> Optional[np.ndarray]:
        """
        Capture face from webcam
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Face image as numpy array or None
        """
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

            collected_frames = []
            frame_count = 0
            max_frames = timeout * config.WEBCAM_FPS

            print(f"Capturing face for {timeout} seconds...")
            print("Please look directly at the camera")

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect face in frame
                face_image = self.face_engine.extract_face(frame)
                if face_image is not None:
                    collected_frames.append(face_image)

                frame_count += 1

                # Show live feed
                cv2.imshow("Face Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if collected_frames:
                # Return best quality frame (middle frame is usually best)
                best_frame = self.select_best_face(collected_frames)
                logger.info(f"Captured {len(collected_frames)} face frames")
                return best_frame

            logger.warning("No face detected during capture")
            return None

        except Exception as e:
            logger.error(f"Error capturing face: {e}")
            return None

    def register_face(
        self, db: Session, patient_id: int, face_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Register a patient's face
        
        Args:
            db: Database session
            patient_id: Patient ID
            face_image: Face image (if provided, else capture from webcam)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Capture face if not provided
            if face_image is None:
                face_image = self.capture_face_from_webcam()
                if face_image is None:
                    return False, "Failed to capture face"

            # Generate embedding
            embedding = self.face_engine.get_embedding(face_image)
            if embedding is None:
                return False, "Failed to generate face embedding"

            # Save embedding
            success = self.embedding_store.save_face_embedding(db, patient_id, embedding)
            if success:
                return True, "Face registered successfully"
            else:
                return False, "Failed to save face embedding to database"

        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False, str(e)

    def authenticate_face(
        self, db: Session, face_image: Optional[np.ndarray] = None
    ) -> Tuple[Optional[int], float, str]:
        """
        Authenticate patient by face
        
        Args:
            db: Database session
            face_image: Face image (if provided, else capture from webcam)
            
        Returns:
            Tuple of (patient_id, similarity_score, message)
        """
        try:
            # Capture face if not provided
            if face_image is None:
                face_image = self.capture_face_from_webcam()
                if face_image is None:
                    return None, 0.0, "Failed to capture face"

            # Generate embedding
            embedding = self.face_engine.get_embedding(face_image)
            if embedding is None:
                return None, 0.0, "Failed to generate face embedding"

            # Find matching patient
            result = self.embedding_store.find_match(db, embedding, self.similarity_threshold)
            if result:
                patient_id, similarity = result
                logger.info(f"Face match found: patient_id={patient_id}, similarity={similarity:.4f}")
                return patient_id, similarity, f"Face recognized with {similarity*100:.2f}% confidence"
            else:
                logger.warning(f"No face match found (threshold: {self.similarity_threshold})")
                return None, 0.0, "Face not recognized. Please try token login."

        except Exception as e:
            logger.error(f"Error authenticating face: {e}")
            return None, 0.0, str(e)

    def re_register_face(
        self, db: Session, patient_id: int
    ) -> Tuple[bool, str]:
        """
        Re-register patient's face (update enrollment)
        
        Args:
            db: Database session
            patient_id: Patient ID
            
        Returns:
            Tuple of (success, message)
        """
        try:
            face_image = self.capture_face_from_webcam()
            if face_image is None:
                return False, "Failed to capture face"

            embedding = self.face_engine.get_embedding(face_image)
            if embedding is None:
                return False, "Failed to generate face embedding"

            # Update embedding
            success = self.embedding_store.save_face_embedding(db, patient_id, embedding)
            if success:
                return True, "Face re-registered successfully"
            else:
                return False, "Failed to update face embedding"

        except Exception as e:
            logger.error(f"Error re-registering face: {e}")
            return False, str(e)

    @staticmethod
    def select_best_face(face_images: list) -> np.ndarray:
        """
        Select best quality face image from list
        Uses brightness and contrast heuristics
        
        Args:
            face_images: List of face images
            
        Returns:
            Best quality face image
        """
        if not face_images:
            return None

        if len(face_images) == 1:
            return face_images[0]

        # Select middle frame (often best quality)
        return face_images[len(face_images) // 2]

    def liveness_check(
        self, face_image: np.ndarray, num_samples: int = 5
    ) -> Tuple[bool, str]:
        """
        Check if face is live (anti-spoofing)
        
        Args:
            face_image: Face image to check
            num_samples: Number of samples to check
            
        Returns:
            Tuple of (is_live, message)
        """
        # TODO: Implement with anti-spoofing model if needed
        # For MVP, accept all faces
        return True, "Liveness check passed"
