"""
Face Recognition Authentication Module

This module handles live face capture from webcam and extracts
face embeddings using DeepFace for authentication purposes.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth.core.similarity_engine import SimilarityEngine, create_similarity_engine


class FaceRecognitionAuth:
    """
    Face recognition authentication system using webcam capture
    and DeepFace embedding extraction.
    """
    
    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
        embeddings_dir: str = "data/embeddings",
        threshold: float = 0.5,
        model_name: str = "Facenet",
        detection_threshold: float = 0.95
    ):
        """
        Initialize the face recognition authentication system.
        
        Args:
            similarity_engine: Optional pre-configured similarity engine
            embeddings_dir: Directory containing stored embeddings
            threshold: Similarity threshold for authentication
            model_name: DeepFace model for embedding extraction
                       Options: 'Facenet', 'ArcFace', 'VGG-Face', etc.
            detection_threshold: Minimum face detection confidence
        """
        self.similarity_engine = similarity_engine or create_similarity_engine(
            embeddings_dir=embeddings_dir,
            threshold=threshold,
            metric="cosine"
        )
        self.model_name = model_name
        self.detection_threshold = detection_threshold
        self.embeddings_dir = embeddings_dir
        
        # Ensure embeddings directory exists
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize webcam
        self.cap = None
        
        # Face detection cascade (Haar Cascade as fallback)
        self.face_cascade = None
        
    def _load_face_detector(self):
        """Load Haar Cascade face detector as fallback."""
        if self.face_cascade is None:
            # Try to load OpenCV's default face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def _preprocess_face(self, face_region: np.ndarray) -> np.ndarray:
        """
        Preprocess face region for DeepFace.
        
        Args:
            face_region: Cropped face image
            
        Returns:
            Preprocessed face ready for embedding extraction
        """
        # Resize to standard size expected by DeepFace
        target_size = (160, 160)  # Standard for Facenet
        processed = cv2.resize(face_region, target_size)
        
        # Convert BGR to RGB if needed
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        return processed
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace.
        
        Args:
            face_image: Image array containing face (BGR format from OpenCV)
            
        Returns:
            Face embedding as numpy array, or None if extraction fails
        """
        try:
            # Import DeepFace
            from deepface import DeepFace
            
            # Preprocess the face image
            processed_face = self._preprocess_face(face_image)
            
            # Extract embedding using DeepFace
            # We use the represent function to get embeddings
            embedding = DeepFace.represent(
                img_path=processed_face,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            
            return None
            
        except ImportError:
            print("DeepFace not installed. Please install: pip install deepface")
            return None
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect face in the given frame.
        
        Args:
            frame: Video frame from webcam
            
        Returns:
            Tuple of (cropped_face, confidence) or None if no face detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try DeepFace's built-in detection first
        try:
            from deepface import DeepFace
            
            # DeepFace can detect faces
            faces = DeepFace.detectFace(
                frame,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            if faces is not None and len(faces) > 0:
                # Return the first detected face with confidence
                return faces[0], 0.95  # DeepFace doesn't always return confidence
            
        except Exception as e:
            print(f"DeepFace detection error: {str(e)}")
        
        # Fallback to Haar Cascade
        self._load_face_detector()
        
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add padding
                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_crop = frame[y1:y2, x1:x2]
                return face_crop, 0.8
        
        return None
    
    def capture_and_authenticate(
        self,
        timeout_seconds: int = 10,
        show_preview: bool = True
    ) -> Tuple[bool, Optional[str], float, Optional[np.ndarray]]:
        """
        Capture face from webcam and authenticate against stored embeddings.
        
        Args:
            timeout_seconds: Maximum time to wait for face capture
            show_preview: Whether to show webcam preview window
            
        Returns:
            Tuple of (success, user_id, confidence, captured_embedding)
        """
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False, None, 0.0, None
        
        print("Starting face capture... Press 'q' to quit or wait for timeout.")
        
        import time
        start_time = time.time()
        captured_embedding = None
        captured_face = None
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print("Timeout reached without successful capture")
                break
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Try to detect and extract face
            result = self.detect_face(frame)
            
            if result is not None:
                face_region, confidence = result
                captured_face = face_region
                
                # Extract embedding
                embedding = self.extract_embedding(face_region)
                
                if embedding is not None:
                    captured_embedding = embedding
                    
                    # Compare with stored embeddings
                    is_authenticated, user_id, match_confidence = (
                        self.similarity_engine.compare_with_stored(embedding)
                    )
                    
                    # Draw face rectangle and status
                    if show_preview:
                        # Convert to RGB for display
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Draw status on frame
                        status_color = (0, 255, 0) if is_authenticated else (0, 0, 255)
                        status_text = f"Authenticated: {user_id}" if is_authenticated else "Not Authenticated"
                        
                        # Put text on frame
                        cv2.putText(
                            frame,
                            status_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            status_color,
                            2
                        )
                        
                        cv2.imshow('Face Recognition Auth', frame)
                    
                    # Release resources
                    self.cap.release()
                    if show_preview:
                        cv2.destroyAllWindows()
                    
                    return is_authenticated, user_id, match_confidence, captured_embedding
            
            # Show preview even without face detected
            if show_preview:
                cv2.putText(
                    frame,
                    f"Looking for face... ({int(timeout_seconds - elapsed)}s)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                cv2.imshow('Face Recognition Auth', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup on timeout or error
        self.cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        return False, None, 0.0, captured_embedding
    
    def register_face(
        self,
        user_id: str,
        face_image: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register a new user's face embedding.
        
        Args:
            user_id: Unique identifier for the user
            face_image: Image containing the user's face
            metadata: Optional additional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        # Extract embedding
        embedding = self.extract_embedding(face_image)
        
        if embedding is None:
            print("Failed to extract embedding for registration")
            return False
        
        # Save to similarity engine
        return self.similarity_engine.save_embedding(
            user_id=user_id,
            embedding=embedding,
            metadata=metadata
        )
    
    def authenticate_from_image(
        self,
        image_path: str
    ) -> Tuple[bool, Optional[str], float]:
        """
        Authenticate from a static image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_authenticated, user_id, confidence)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not read image from {image_path}")
                return False, None, 0.0
            
            # Detect face
            result = self.detect_face(image)
            
            if result is None:
                print("No face detected in image")
                return False, None, 0.0
            
            face_region, _ = result
            
            # Extract embedding
            embedding = self.extract_embedding(face_region)
            
            if embedding is None:
                print("Failed to extract embedding")
                return False, None, 0.0
            
            # Compare with stored embeddings
            return self.similarity_engine.compare_with_stored(embedding)
            
        except Exception as e:
            print(f"Error during authentication: {str(e)}")
            return False, None, 0.0
    
    def release(self):
        """Release webcam and other resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def create_face_recognition_auth(
    embeddings_dir: str = "data/embeddings",
    threshold: float = 0.5,
    model_name: str = "Facenet"
) -> FaceRecognitionAuth:
    """
    Factory function to create a FaceRecognitionAuth instance.
    
    Args:
        embeddings_dir: Directory containing stored embeddings
        threshold: Similarity threshold for authentication
        model_name: DeepFace model for embedding extraction
        
    Returns:
        Configured FaceRecognitionAuth instance
    """
    return FaceRecognitionAuth(
        embeddings_dir=embeddings_dir,
        threshold=threshold,
        model_name=model_name
    )


# Example usage and testing
if __name__ == "__main__":
    print("Face Recognition Authentication Module")
    print("=" * 50)
    
    # Create authentication instance
    auth = create_face_recognition_auth(
        embeddings_dir="data/embeddings",
        threshold=0.5,
        model_name="Facenet"
    )
    
    # Example: Load and check stored embeddings
    auth.similarity_engine.load_embeddings()
    print(f"Loaded {len(auth.similarity_engine.stored_embeddings)} embeddings")
    
    # Uncomment to test live authentication:
    # success, user_id, confidence, embedding = auth.capture_and_authenticate(
    #     timeout_seconds=10,
    #     show_preview=True
    # )
    # print(f"Auth result: {success}, User: {user_id}, Confidence: {confidence:.4f}")
    
    print("\nModule ready for use.")
    print("To authenticate, call: auth.capture_and_authenticate()")
