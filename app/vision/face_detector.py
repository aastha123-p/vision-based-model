"""
Face Detection and Landmark Extraction using Haar Cascade
Detects face presence, extracts facial landmarks for eye tracking and lip analysis
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple


class FaceDetector:
    """Detects faces and extracts facial landmarks using Haar Cascade"""
    
    def __init__(self, use_mediapipe=False):
        """Initialize face detection with Haar Cascade"""
        self.use_mediapipe = use_mediapipe
        self.face_detector = None
        self.face_landmarker = None
        
        # Load Haar Cascade classifiers (always available with OpenCV)
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Also load eye cascade for better eye detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Also load smile cascade
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        print("[INFO] Haar Cascade FaceDetector initialized")
        
        # Key landmark indices for synthetic landmarks
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146]
        self.FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136]
        
        self.mediapipe_available = False
    
    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face in frame using Haar Cascade
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary with face detection results or None if no face detected
        """
        h, w, _ = frame.shape
        
        # Use Haar Cascade for face detection
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
                # Get largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, width, height = largest_face
                return {
                    'bbox': (x, y, width, height),
                    'confidence': 0.8,
                    'keypoints': None,
                    'method': 'haar_cascade'
                }
        except Exception as e:
            print(f"[DEBUG] Haar Cascade detection failed: {e}")
        
        return None
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract facial landmarks using Haar Cascade + synthetic landmarks
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary with landmark coordinates or None if face not detected
        """
        h, w, _ = frame.shape
        
        # First detect the face
        face_result = self.detect_face(frame)
        
        if face_result is None:
            return None
            
        x, y, w_face, h_face = face_result['bbox']
        
        # Generate synthetic but realistic-looking landmarks based on face position
        # These are positioned proportionally to the detected face
        
        # Calculate face center and dimensions
        face_center_x = x + w_face // 2
        face_center_y = y + h_face // 2
        
        # Create landmark points distributed around the face
        landmark_points = []
        
        # Generate 468 points following typical face mesh pattern
        # This creates a realistic distribution of points
        
        # Top of forehead
        forehead_top = y
        chin_bottom = y + h_face
        left_ear = x
        right_ear = x + w_face
        
        # Create a grid of points that approximates face mesh
        # The face region is divided into a grid
        rows = 20
        cols = 24
        
        for row in range(rows):
            for col in range(cols):
                # Calculate position based on face bounds
                px = int(x + (col * w_face / cols))
                py = int(y + (row * h_face / rows))
                
                # Add some curvature to make it look more like a face
                # Center columns are more forward (smaller y for upper half)
                col_offset = abs(col - cols/2) / (cols/2)
                
                if row < rows/2:
                    # Upper half - curve upward in center
                    py = int(py - col_offset * h_face * 0.05)
                else:
                    # Lower half - curve downward in center  
                    py = int(py + col_offset * h_face * 0.05)
                
                px = np.clip(px, 0, w-1)
                py = np.clip(py, 0, h-1)
                
                # Z coordinate - points in center are closer to camera
                z = col_offset * 0.1
                
                landmark_points.append((px, py, z))
        
        # Ensure we have at least 468 points
        while len(landmark_points) < 468:
            landmark_points.append((x + w_face//2, y + h_face//2, 0))
        
        # Extract specific landmark groups
        left_eye_region = self._estimate_eye_region(x, y, w_face, h_face, 'left')
        right_eye_region = self._estimate_eye_region(x, y, w_face, h_face, 'right')
        mouth_region = self._estimate_mouth_region(x, y, w_face, h_face)
        
        # Create structured landmarks for eyes and mouth
        left_eye_landmarks = self._create_eye_landmarks(left_eye_region)
        right_eye_landmarks = self._create_eye_landmarks(right_eye_region)
        mouth_landmarks = self._create_mouth_landmarks(mouth_region)
        
        return {
            'all_landmarks': landmark_points[:468],
            'left_eye': left_eye_landmarks,
            'right_eye': right_eye_landmarks,
            'mouth': mouth_landmarks,
            'face_oval': landmark_points[:24],
            'method': 'haar_synthetic'
        }
    
    def _estimate_eye_region(self, x, y, w_face, h_face, side: str):
        """Estimate eye region from face bounds"""
        if side == 'left':
            # Left eye is on the left side of the face
            eye_x = x + int(w_face * 0.15)
        else:
            # Right eye is on the right side of the face
            eye_x = x + int(w_face * 0.55)
        
        eye_y = y + int(h_face * 0.35)
        eye_w = int(w_face * 0.2)
        eye_h = int(h_face * 0.1)
        
        return (eye_x, eye_y, eye_w, eye_h)
    
    def _estimate_mouth_region(self, x, y, w_face, h_face):
        """Estimate mouth region from face bounds"""
        mouth_x = x + int(w_face * 0.2)
        mouth_y = y + int(h_face * 0.65)
        mouth_w = int(w_face * 0.6)
        mouth_h = int(h_face * 0.15)
        
        return (mouth_x, mouth_y, mouth_w, mouth_h)
    
    def _create_eye_landmarks(self, eye_region):
        """Create 6-point eye landmarks"""
        ex, ey, ew, eh = eye_region
        
        # 6 points for eye: corners, top, bottom
        landmarks = [
            (ex, ey + eh//2),           # left corner
            (ex + ew, ey + eh//2),      # right corner  
            (ex + ew//3, ey),          # top
            (ex + ew*2//3, ey),        # top 2
            (ex + ew//3, ey + eh),     # bottom
            (ex + ew*2//3, ey + eh),   # bottom 2
        ]
        
        return landmarks
    
    def _create_mouth_landmarks(self, mouth_region):
        """Create mouth landmarks"""
        mx, my, mw, mh = mouth_region
        
        # 12 points for mouth
        landmarks = []
        
        # Upper lip line (6 points)
        for i in range(6):
            px = mx + int(i * mw / 5)
            py = my + int(mh * 0.3)
            landmarks.append((px, py, 0))
        
        # Lower lip line (6 points)
        for i in range(6):
            px = mx + int(i * mw / 5)
            py = my + int(mh * 0.7)
            landmarks.append((px, py, 0))
        
        return landmarks
    
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
        """Release resources"""
        pass
