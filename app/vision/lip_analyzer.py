"""
Lip Tension and Mouth Analysis
Analyzes lip tension, mouth shape, and speech indicators
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class LipAnalyzer:
    """Analyzes lip tension, mouth openness, and mouth shape"""
    
    def __init__(self):
        """Initialize Lip Analyzer"""
        # Mouth landmark indices for different aspects
        self.UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146]
        self.LOWER_LIP_INDICES = [17, 84, 181, 91, 146, 84, 17, 314, 405, 321, 375, 291, 409]
        
        # Tension detection thresholds
        self.high_tension_threshold = 0.15  # Tension coefficient
        self.low_tension_threshold = 0.05
        
        # History for trend analysis
        self.tension_history = []
        self.mouth_openness_history = []
        self.max_history_size = 30
    
    def calculate_mouth_openness(self, mouth_landmarks: List[Tuple]) -> float:
        """
        Calculate mouth openness ratio (MOR)
        
        Args:
            mouth_landmarks: List of mouth landmark points
            
        Returns:
            Mouth openness ratio (0-1, where 0 is closed, 1 is wide open)
        """
        if not mouth_landmarks or len(mouth_landmarks) < 12:
            return 0.0
        
        # Get vertical distances for mouth openness
        # Typically index 0 is top, index 6 or 9 is bottom
        
        try:
            # Top lip point (could be index 0 or similar)
            top_y = mouth_landmarks[0][1]
            # Bottom lip point
            bottom_y = mouth_landmarks[9][1] if len(mouth_landmarks) > 9 else mouth_landmarks[-1][1]
            
            # Get horizontal span
            x_coords = [p[0] for p in mouth_landmarks]
            horizontal_span = max(x_coords) - min(x_coords)
            
            vertical_span = abs(bottom_y - top_y)
            
            # Mouth openness ratio (vertical / horizontal)
            if horizontal_span > 0:
                mouth_openness = vertical_span / horizontal_span
            else:
                mouth_openness = 0.0
            
            return float(np.clip(mouth_openness, 0, 1))
        
        except Exception as e:
            print(f"Error calculating mouth openness: {e}")
            return 0.0
    
    def calculate_lip_tension(self, mouth_landmarks: List[Tuple]) -> Dict:
        """
        Calculate lip tension based on mouth shape and perimeter
        
        Args:
            mouth_landmarks: List of mouth landmark points
            
        Returns:
            Dictionary with tension analysis
        """
        if not mouth_landmarks or len(mouth_landmarks) < 12:
            return {
                'tension_score': 0.0,
                'tension_level': 'unknown',
                'is_valid': False
            }
        
        try:
            # Calculate mouth perimeter
            perimeter = 0.0
            for i in range(len(mouth_landmarks)):
                p1 = mouth_landmarks[i]
                p2 = mouth_landmarks[(i + 1) % len(mouth_landmarks)]
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                perimeter += distance
            
            # Get mouth vertical span (height)
            y_coords = [p[1] for p in mouth_landmarks]
            mouth_height = max(y_coords) - min(y_coords)
            
            # Get mouth horizontal span (width)
            x_coords = [p[0] for p in mouth_landmarks]
            mouth_width = max(x_coords) - min(x_coords)
            
            # Calculate mouth area (approximate ellipse)
            mouth_area = (mouth_height / 2) * (mouth_width / 2) * 3.14159
            
            # Tension coefficient: how tightly drawn the lips are
            # Lower area with high perimeter = high tension
            if mouth_area > 0:
                tension_coefficient = perimeter / mouth_area
            else:
                tension_coefficient = 0.0
            
            # Normalize tension score to 0-100 range
            # High tension coefficient means more tension
            tension_score = min(100, max(0, (tension_coefficient - 0.5) * 20))
            
            # Determine tension level
            if tension_score < 20:
                tension_level = 'relaxed'
            elif tension_score < 40:
                tension_level = 'neutral'
            elif tension_score < 65:
                tension_level = 'tense'
            else:
                tension_level = 'very_tense'
            
            result = {
                'tension_score': float(tension_score),
                'tension_level': tension_level,
                'tension_coefficient': float(tension_coefficient),
                'perimeter': float(perimeter),
                'area': float(mouth_area),
                'width': float(mouth_width),
                'height': float(mouth_height),
                'is_valid': True
            }
            
            # Store in history
            self._add_to_history(tension_score)
            
            return result
        
        except Exception as e:
            print(f"Error calculating lip tension: {e}")
            return {
                'tension_score': 0.0,
                'tension_level': 'unknown',
                'is_valid': False,
                'error': str(e)
            }
    
    def _add_to_history(self, tension_score: float):
        """Add tension score to history for trend analysis"""
        self.tension_history.append(tension_score)
        if len(self.tension_history) > self.max_history_size:
            self.tension_history.pop(0)
    
    def detect_speaking(self, mouth_landmarks: List[Tuple], 
                       mouth_openness: float,
                       previous_openness: Optional[float] = None) -> Dict:
        """
        Detect if person is speaking based on mouth movement
        
        Args:
            mouth_landmarks: Current mouth landmarks
            mouth_openness: Current mouth openness ratio
            previous_openness: Previous frame's openness for speech detection
            
        Returns:
            Dictionary with speech detection results
        """
        # Simple heuristic: speaking detected if mouth openness > threshold
        # and there's significant movement
        
        speech_threshold = 0.15
        is_speaking = mouth_openness > speech_threshold
        
        # For better accuracy, check mouth movement
        mouth_movement = 0.0
        if previous_openness is not None:
            mouth_movement = abs(mouth_openness - previous_openness)
        
        # Check for sufficient mouth movement for speech
        speech_movement_threshold = 0.05
        is_speaking = is_speaking and (mouth_movement > speech_movement_threshold or mouth_openness > 0.2)
        
        # Calculate speech confidence
        if mouth_movement > 0:
            speech_confidence = min(1.0, mouth_movement / 0.1)
        else:
            speech_confidence = min(1.0, mouth_openness / 0.3) if is_speaking else 0.0
        
        return {
            'is_speaking': is_speaking,
            'speech_confidence': float(speech_confidence),
            'mouth_movement': float(mouth_movement),
            'mouth_openness': float(mouth_openness)
        }
    
    def analyze_lip_color_intensity(self, mouth_region: np.ndarray) -> Dict:
        """
        Analyze lip color and intensity from mouth region image
        
        Args:
            mouth_region: Cropped mouth region image
            
        Returns:
            Dictionary with color analysis results
        """
        if mouth_region is None or mouth_region.size == 0:
            return {
                'red_intensity': 0.0,
                'lip_color_valid': False
            }
        
        try:
            # Convert BGR to HSV for better color analysis
            hsv_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # Extract red channel for lips
            b, g, r = cv2.split(mouth_region)
            
            # Red intensity (red channel - blue channel)
            red_intensity = np.mean(r.astype(float) - b.astype(float))
            red_intensity = float(np.clip(red_intensity, 0, 255))
            
            # Saturation (indicator of lip color richness)
            saturation = np.mean(hsv_region[:, :, 1])
            
            return {
                'red_intensity': red_intensity,
                'saturation': float(saturation),
                'lip_color_valid': True,
                'is_visible': red_intensity > 10  # Lips have some red
            }
        
        except Exception as e:
            print(f"Error analyzing lip color: {e}")
            return {
                'red_intensity': 0.0,
                'lip_color_valid': False,
                'error': str(e)
            }
    
    def get_tension_trend(self) -> Dict:
        """
        Analyze tension trend over recent frames
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.tension_history:
            return {
                'trend': 'no_data',
                'average_tension': 0.0,
                'max_tension': 0.0,
                'min_tension': 0.0
            }
        
        avg_tension = np.mean(self.tension_history)
        max_tension = np.max(self.tension_history)
        min_tension = np.min(self.tension_history)
        
        # Determine trend direction
        if len(self.tension_history) >= 2:
            recent_trend = self.tension_history[-1] - self.tension_history[0]
            if recent_trend > 5:
                trend = 'increasing'
            elif recent_trend < -5:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'average_tension': float(avg_tension),
            'max_tension': float(max_tension),
            'min_tension': float(min_tension),
            'history_length': len(self.tension_history)
        }
    
    def draw_mouth_metrics(self, frame: np.ndarray, mouth_landmarks: List[Tuple],
                          tension_score: float, tension_level: str,
                          mouth_openness: float, position: Tuple = (10, 100)) -> np.ndarray:
        """
        Draw mouth metrics on frame
        
        Args:
            frame: Input frame
            mouth_landmarks: Mouth landmark points for visualization
            tension_score: Tension score (0-100)
            tension_level: Tension level string
            mouth_openness: Mouth openness ratio
            position: (x, y) position to start drawing
            
        Returns:
            Frame with drawn metrics
        """
        output_frame = frame.copy()
        x, y = position
        
        # Draw mouth contour
        if mouth_landmarks and len(mouth_landmarks) > 3:
            mouth_pts = np.array(mouth_landmarks, dtype=np.int32)
            
            # Color based on tension level
            if tension_level == 'relaxed':
                color = (0, 255, 0)  # Green
            elif tension_level == 'neutral':
                color = (255, 255, 0)  # Cyan
            elif tension_level == 'tense':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            cv2.polylines(output_frame, [mouth_pts], True, color, 2)
        
        # Draw metrics text
        metrics = [
            f"Lip Tension: {tension_level.upper()} ({tension_score:.1f})",
            f"Mouth Open: {mouth_openness:.2f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(output_frame, metric, (x, y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
