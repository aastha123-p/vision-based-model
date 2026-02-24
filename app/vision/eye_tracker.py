"""
Eye Tracking and Analysis
Detects eye movement, blink rate, and eye strain indicators
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class EyeTracker:
    """Tracks eye movement, blinks, and calculates eye strain metrics"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize Eye Tracker
        
        Args:
            window_size: Number of frames to use for averaging metrics
        """
        self.window_size = window_size
        self.blink_history = deque(maxlen=window_size)
        self.eye_distance_history = deque(maxlen=window_size)
        self.gaze_history = deque(maxlen=window_size)
        
        # Blink detection parameters
        self.blink_threshold = 0.2  # Eye aspect ratio threshold for blink
        self.blink_frames = 0
        self.consecutive_blink_frames = 0
        self.blink_count = 0
        self.last_blink_time = time.time()
        
        # Eye strain parameters
        self.min_blink_rate = 12  # Minimum blinks per minute (normal: 15-20)
        self.max_blink_duration = 0.3  # Maximum blink duration in seconds
        self.strain_threshold = 0.6
        
        # Pupils position tracking
        self.left_pupil_positions = deque(maxlen=window_size)
        self.right_pupil_positions = deque(maxlen=window_size)
        
        # Gaze direction (0: center, -1: left, 1: right, -2: up, 2: down)
        self.gaze_direction = 0
        
    def calculate_eye_aspect_ratio(self, eye_landmarks: List[Tuple]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        Uses the formula: EAR = ||p2 - p6|| + ||p3 - p5|| / 2 * ||p1 - p4||
        
        Args:
            eye_landmarks: List of 6 eye landmark points in order [corner1, corner2, top1, bottom1, top2, bottom2]
            
        Returns:
            Eye aspect ratio value
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return 0.5
        
        # Calculate distances between vertical landmarks
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Use indices for standard EAR calculation
        vertical_dist1 = distance(eye_landmarks[1], eye_landmarks[5])
        vertical_dist2 = distance(eye_landmarks[2], eye_landmarks[4])
        horizontal_dist = distance(eye_landmarks[0], eye_landmarks[3])
        
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
        
        return float(ear)
    
    def detect_blink(self, left_eye_landmarks: List[Tuple], 
                    right_eye_landmarks: List[Tuple]) -> Dict:
        """
        Detect blink from eye landmarks
        
        Args:
            left_eye_landmarks: Left eye landmarks
            right_eye_landmarks: Right eye landmarks
            
        Returns:
            Dictionary with blink detection results
        """
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        is_blink = avg_ear < self.blink_threshold
        
        if is_blink:
            self.consecutive_blink_frames += 1
        else:
            if self.consecutive_blink_frames > 0:
                # Blink completed
                self.blink_count += 1
                self.last_blink_time = time.time()
            self.consecutive_blink_frames = 0
        
        self.blink_history.append(is_blink)
        
        return {
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'avg_ear': float(avg_ear),
            'is_blink': is_blink,
            'blink_count': self.blink_count
        }
    
    def calculate_blink_rate(self, time_window: float = 60.0) -> float:
        """
        Calculate blink rate (blinks per minute)
        
        Args:
            time_window: Time window in seconds for calculation
            
        Returns:
            Blinks per minute in the specified time window
        """
        if self.blink_count == 0:
            return 0.0
        
        # Simplified calculation: use stored blink count
        # For real-time: track time between blinks
        blink_rate = self.blink_count / (time_window / 60.0)
        
        return min(max(float(blink_rate), 0), 100)  # Clamp between 0-100
    
    def track_pupil_position(self, eye_landmarks: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        Track pupil/iris center position in eye
        
        Args:
            eye_landmarks: Eye landmark points
            
        Returns:
            Tuple of (x, y) pupil position or None
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return None
        
        # Calculate centroid of eye region
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        
        pupil_x = sum(x_coords) // len(x_coords)
        pupil_y = sum(y_coords) // len(y_coords)
        
        return (pupil_x, pupil_y)
    
    def detect_gaze_direction(self, left_pupil: Optional[Tuple],
                             right_pupil: Optional[Tuple],
                             left_eye_landmarks: List[Tuple],
                             right_eye_landmarks: List[Tuple]) -> Dict:
        """
        Detect gaze direction (left, right, up, down, center)
        
        Args:
            left_pupil: Left pupil position (x, y)
            right_pupil: Right pupil position (x, y)
            left_eye_landmarks: Left eye landmarks for bounding box
            right_eye_landmarks: Right eye landmarks for bounding box
            
        Returns:
            Dictionary with gaze detection results
        """
        directions = {
            'direction': 'center',
            'left_gaze': 0.5,  # 0 = far left, 1 = far right
            'right_gaze': 0.5,
            'vertical_pos': 0.5  # 0 = up, 1 = down
        }
        
        if not left_pupil or not right_pupil:
            return directions
        
        # Calculate gaze position relative to eye region
        try:
            # Left eye gaze
            if len(left_eye_landmarks) >= 6:
                left_x_coords = [p[0] for p in left_eye_landmarks]
                left_x_min, left_x_max = min(left_x_coords), max(left_x_coords)
                left_gaze = (left_pupil[0] - left_x_min) / (left_x_max - left_x_min) if (left_x_max - left_x_min) > 0 else 0.5
                directions['left_gaze'] = float(np.clip(left_gaze, 0, 1))
            
            # Right eye gaze
            if len(right_eye_landmarks) >= 6:
                right_x_coords = [p[0] for p in right_eye_landmarks]
                right_x_min, right_x_max = min(right_x_coords), max(right_x_coords)
                right_gaze = (right_pupil[0] - right_x_min) / (right_x_max - right_x_min) if (right_x_max - right_x_min) > 0 else 0.5
                directions['right_gaze'] = float(np.clip(right_gaze, 0, 1))
            
            # Determine direction
            avg_horizontal = (directions['left_gaze'] + directions['right_gaze']) / 2
            
            if avg_horizontal < 0.35:
                directions['direction'] = 'left'
            elif avg_horizontal > 0.65:
                directions['direction'] = 'right'
            elif avg_horizontal > 0.35 and avg_horizontal < 0.65:
                directions['direction'] = 'center'
            
        except Exception as e:
            print(f"Error in gaze detection: {e}")
        
        return directions
    
    def calculate_eye_strain_score(self, blink_rate: float, eye_aspect_ratios: List[float],
                                  gaze_deviation: float) -> Dict:
        """
        Calculate overall eye strain score (0-100) based on multiple factors
        
        Args:
            blink_rate: Current blink rate (blinks per minute)
            eye_aspect_ratios: List of recent eye aspect ratios
            gaze_deviation: Deviation of gaze from center (0-1)
            
        Returns:
            Dictionary with strain score and breakdown
        """
        strain_score = 0.0
        factors = {}
        
        # Factor 1: Low blink rate (ideal: 15-20 blinks/min)
        if blink_rate < self.min_blink_rate:
            blink_score = ((self.min_blink_rate - blink_rate) / self.min_blink_rate) * 40
            factors['blink_rate_score'] = float(min(40, blink_score))
        else:
            factors['blink_rate_score'] = 0.0
        
        # Factor 2: Sustained eye opening (high EAR for long periods)
        if eye_aspect_ratios:
            avg_ear = np.mean(eye_aspect_ratios)
            eye_opening_score = max(0, (avg_ear - 0.3) * 20) if avg_ear > 0.3 else 0
            factors['eye_opening_score'] = float(min(30, eye_opening_score))
        else:
            factors['eye_opening_score'] = 0.0
        
        # Factor 3: Excessive gaze fixation
        gaze_score = gaze_deviation * 30
        factors['gaze_fixation_score'] = float(min(30, gaze_score))
        
        # Calculate total strain
        strain_score = sum(factors.values())
        
        # Determine strain level
        if strain_score < 20:
            strain_level = 'low'
        elif strain_score < 50:
            strain_level = 'moderate'
        elif strain_score < 75:
            strain_level = 'high'
        else:
            strain_level = 'severe'
        
        return {
            'strain_score': float(min(100, strain_score)),
            'strain_level': strain_level,
            'factors': factors,
            'recommendations': self._get_strain_recommendations(strain_level)
        }
    
    def _get_strain_recommendations(self, strain_level: str) -> List[str]:
        """
        Get recommendations based on strain level
        
        Args:
            strain_level: 'low', 'moderate', 'high', or 'severe'
            
        Returns:
            List of recommendation strings
        """
        recommendations = {
            'low': [
                'Maintain current eye health practices',
                'Continue regular breaks'
            ],
            'moderate': [
                'Take 20-20-20 breaks (20 sec, 20 feet, every 20 min)',
                'Adjust screen brightness',
                'Ensure proper posture'
            ],
            'high': [
                'Take frequent breaks immediately',
                'Reduce screen time',
                'Use eye drops or blink more consciously',
                'Adjust lighting and screen position'
            ],
            'severe': [
                'Stop screen use immediately',
                'Close eyes and rest for 5 minutes',
                'Consult an eye care professional',
                'Consider professional help for prolonged symptoms'
            ]
        }
        
        return recommendations.get(strain_level, [])
    
    def draw_eye_metrics(self, frame: np.ndarray, left_pupil: Optional[Tuple],
                        right_pupil: Optional[Tuple], eye_aspect_ratio: float,
                        blink_rate: float, position: Tuple = (10, 70)) -> np.ndarray:
        """
        Draw eye metrics on frame
        
        Args:
            frame: Input frame
            left_pupil: Left pupil position
            right_pupil: Right pupil position
            eye_aspect_ratio: Current eye aspect ratio
            blink_rate: Current blink rate
            position: (x, y) position to start drawing
            
        Returns:
            Frame with drawn metrics
        """
        output_frame = frame.copy()
        x, y = position
        
        # Draw pupil positions
        if left_pupil:
            cv2.circle(output_frame, left_pupil, 3, (255, 0, 0), -1)
        if right_pupil:
            cv2.circle(output_frame, right_pupil, 3, (255, 0, 0), -1)
        
        # Draw metrics text
        metrics = [
            f"EAR: {eye_aspect_ratio:.3f}",
            f"Blink Rate: {blink_rate:.1f} bpm",
            f"Blinks: {self.blink_count}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(output_frame, metric, (x, y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output_frame
    
    def reset_blink_counter(self):
        """Reset blink counter for new session"""
        self.blink_count = 0
        self.consecutive_blink_frames = 0
    
    def get_statistics(self) -> Dict:
        """
        Get overall eye statistics
        
        Returns:
            Dictionary with aggregated statistics
        """
        return {
            'total_blinks': self.blink_count,
            'blink_history_length': len(self.blink_history),
            'is_blinking_now': len(self.blink_history) > 0 and self.blink_history[-1],
            'window_size': self.window_size
        }
