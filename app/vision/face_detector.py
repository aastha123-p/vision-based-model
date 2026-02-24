"""
Face Detection and Landmark Extraction using MediaPipe with Haar Cascade Fallback
Detects face presence, extracts facial landmarks for eye tracking and lip analysis
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, Dict, List, Tuple


class FaceDetector:
    """Detects faces and extracts facial landmarks using MediaPipe with fallback"""
    
    def __init__(self, use_mediapipe=True):
        """Initialize face detection with MediaPipe and Haar Cascade fallback"""
        self.use_mediapipe = use_mediapipe
        self.face_detector = None
        self.face_landmarker = None
        
        # Load Haar Cascade as fallback (always available)
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if use_mediapipe:
            self._try_load_mediapipe()
        
        # Key landmark indices for analysis
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146]
        self.FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136]
        
        self.mediapipe_available = self.face_detector is not None
        
    def _try_load_mediapipe(self):
        """Try to load MediaPipe models"""
        try:
            base_options = python.BaseOptions(model_asset_path=None)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.face_detector = vision.FaceDetector.create_from_options(options)
            print("[INFO] MediaPipe FaceDetector loaded successfully")
        except Exception as e:
            print(f"[WARNING] MediaPipe FaceDetector failed: {e}")
            print("[WARNING] Falling back to Haar Cascade detector")
            self.face_detector = None
        
        try:
            base_options = python.BaseOptions(model_asset_path=None)
            landmarker_options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
            print("[INFO] MediaPipe FaceLandmarker loaded successfully")
        except Exception as e:
            print(f"[WARNING] MediaPipe FaceLandmarker failed: {e}")
            print("[WARNING] Falling back to synthetic landmarks")
            self.face_landmarker = None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face in frame using MediaPipe or Haar Cascade
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary with face detection results or None if no face detected
        """
        h, w, _ = frame.shape
        
        # Try MediaPipe first
        if self.face_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.face_detector.detect(media_image)
                
                if detection_result.detections:
                    detection = detection_result.detections[0]
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x * w)
                    y = int(bbox.origin_y * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    score = detection.categories[0].score if detection.categories else 0.5
                    
                    return {
                        'bbox': (x, y, width, height),
                        'confidence': score,
                        'keypoints': None,
                        'method': 'mediapipe'
                    }
            except Exception as e:
                print(f"[DEBUG] MediaPipe detection failed: {e}")
        
        # Fallback to Haar Cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(int(w*0.8), int(h*0.8))
            )
            
            if len(faces) > 0:
                (x, y, width, height) = faces[0]  # Get largest face
                return {
                    'bbox': (x, y, width, height),
                    'confidence': 0.7,  # Haar doesn't give confidence
                    'keypoints': None,
                    'method': 'haar_cascade'
                }
        except Exception as e:
            print(f"[DEBUG] Haar Cascade detection failed: {e}")
        
        return None
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract facial landmarks
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary with landmark coordinates or None if face not detected
        """
        if self.face_landmarker is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                landmarker_result = self.face_landmarker.detect(media_image)
                
                if landmarker_result.face_landmarks:
                    landmarks_list = landmarker_result.face_landmarks[0]
                    landmark_points = []
                    for landmark in landmarks_list:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        z = landmark.z
                        landmark_points.append((x, y, z))
                    
                    return {
                        'all_landmarks': landmark_points,
                        'left_eye': [landmark_points[i] for i in self.LEFT_EYE_INDICES],
                        'right_eye': [landmark_points[i] for i in self.RIGHT_EYE_INDICES],
                        'mouth': [landmark_points[i] for i in self.MOUTH_INDICES],
                        'face_oval': [landmark_points[i] for i in self.FACE_OVAL_INDICES],
                        'method': 'mediapipe'
                    }
            except Exception as e:
                print(f"[DEBUG] MediaPipe landmarks failed: {e}")
        
        # Fallback: Generate synthetic landmarks from face detection
        # Try Haar Cascade directly if face_detector failed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w_face, h_face = faces[0]  # Get first face
            h, w, _ = frame.shape
            
            # Create synthetic landmarks - 468 points distributed around face
            landmark_points = []
            
            # Generate 468 points in a grid around detected face
            for i in range(468):
                row = i // 24
                col = i % 24
                px = int(x + (col * w_face / 24.0))
                py = int(y + (row * h_face / 12.0))
                px = np.clip(px, 0, w-1)
                py = np.clip(py, 0, h-1)
                landmark_points.append((int(px), int(py), 0))
            
            return {
                'all_landmarks': landmark_points,
                'left_eye': landmark_points[33:39],
                'right_eye': landmark_points[42:48],
                'mouth': landmark_points[48:68],
                'face_oval': landmark_points[0:17],
                'method': 'synthetic'
            }
        
        return None
    
    def get_eye_region(self, frame: np.ndarray, landmarks: Dict, eye_side: str = 'left') -> Optional[np.ndarray]:
        """
        Extract eye region from frame for detailed analysis
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks dictionary
            eye_side: 'left' or 'right'
            
        Returns:
            Cropped eye region image
        """
        if eye_side.lower() == 'left':
            eye_points = landmarks['left_eye']
        else:
            eye_points = landmarks['right_eye']
        
        if not eye_points:
            return None
            
        # Get bounding box of eye
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max]
    
    def get_mouth_region(self, frame: np.ndarray, landmarks: Dict) -> Optional[np.ndarray]:
        """
        Extract mouth region from frame
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks dictionary
            
        Returns:
            Cropped mouth region image
        """
        mouth_points = landmarks['mouth']
        
        if not mouth_points:
            return None
            
        x_coords = [p[0] for p in mouth_points]
        y_coords = [p[1] for p in mouth_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max]
    
    def get_iris_center(self, eye_landmarks: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        Calculate iris center from eye landmarks
        
        Args:
            eye_landmarks: List of eye landmark points
            
        Returns:
            Tuple of (x, y) coordinates of iris center
        """
        if not eye_landmarks or len(eye_landmarks) < 4:
            return None
            
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return (center_x, center_y)
    
    def calculate_lip_distance(self, mouth_landmarks: List[Tuple]) -> float:
        """
        Calculate vertical distance between lips (mouth openness)
        
        Args:
            mouth_landmarks: List of mouth landmark points
            
        Returns:
            Euclidean distance between upper and lower lips
        """
        if not mouth_landmarks or len(mouth_landmarks) < 12:
            return 0.0
            
        # Use landmarks for upper lip and lower lip
        upper_lip_y = mouth_landmarks[1][1]  # Approximate upper lip
        lower_lip_y = mouth_landmarks[11][1]  # Approximate lower lip
        
        distance = abs(upper_lip_y - lower_lip_y)
        return float(distance)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw facial landmarks on frame
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks dictionary
            
        Returns:
            Frame with drawn landmarks
        """
        output_frame = frame.copy()
        
        # Draw eye landmarks
        for eye_point in landmarks['left_eye']:
            cv2.circle(output_frame, (eye_point[0], eye_point[1]), 2, (0, 255, 0), -1)
        
        for eye_point in landmarks['right_eye']:
            cv2.circle(output_frame, (eye_point[0], eye_point[1]), 2, (0, 255, 0), -1)
        
        # Draw mouth landmarks
        for mouth_point in landmarks['mouth']:
            cv2.circle(output_frame, (mouth_point[0], mouth_point[1]), 2, (255, 0, 0), -1)
        
        return output_frame
    
    def release(self):
        """Release MediaPipe resources"""
        pass  # MediaPipe 0.10+ doesn't require explicit release
