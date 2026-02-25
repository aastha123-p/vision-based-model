"""
Face Recognition Module
Uses DeepFace for face detection and embedding generation
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not installed. Install with: pip install deepface")

import mediapipe as mp


class FaceRecognitionEngine:
    """
    Face detection and embedding generation using DeepFace
    """

    def __init__(self, model_name: str = "Facenet"):
        """
        Initialize face recognition engine
        
        Args:
            model_name: DeepFace model ('Facenet', 'ArcFace', 'VGGFace', 'Dlib', 'OpenFace')
        """
        self.model_name = model_name
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        Returns bounding boxes as (x, y, width, height)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)

            boxes = []
            if results.detections:
                h, w, c = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    boxes.append((x, y, width, height))

            return boxes
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract largest face from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Face image or None
        """
        try:
            boxes = self.detect_faces(image)
            if not boxes:
                return None

            # Get largest face
            largest_box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, w, h = largest_box

            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            h_img, w_img = image.shape[:2]
            
            face = image[y:min(y + h, h_img), x:min(x + w, w_img)]
            return face if face.size > 0 else None

        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using DeepFace
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Embedding vector or None
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return None

        try:
            # Ensure image is proper shape
            if face_image is None or face_image.size == 0:
                return None

            # Get embedding
            embedding_objs = DeepFace.represent(
                face_image,
                model_name=self.model_name,
                enforce_detection=False,
                silent=True,
            )

            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"])
                return embedding
            return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def draw_boxes(
        self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            boxes: List of boxes (x, y, width, height)
            
        Returns:
            Image with drawn boxes
        """
        result = image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result

    def verify_faces(self, img1_path: str, img2_path: str) -> Tuple[bool, float]:
        """
        Verify if two images contain the same person
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Tuple of (match, distance)
        """
        if not DEEPFACE_AVAILABLE:
            return False, 1.0

        try:
            result = DeepFace.verify(
                img1_path, img2_path, model_name=self.model_name, silent=True
            )
            return result["verified"], result["distance"]
        except Exception as e:
            logger.error(f"Error verifying faces: {e}")
            return False, 1.0
