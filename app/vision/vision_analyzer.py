"""
Comprehensive Vision Analysis Module
Integrates face detection, emotion detection, eye tracking, and lip analysis
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import time

from app.vision.face_detector import FaceDetector
from app.vision.emotion_detector import EmotionDetector
from app.vision.eye_tracker import EyeTracker
from app.vision.lip_analyzer import LipAnalyzer


class VisionAnalyzer:
    """
    Main vision analysis class that coordinates all vision-based feature extraction
    Provides unified interface for face, emotion, eye, and lip analysis
    """
    
    def __init__(self):
        """Initialize all vision analysis components"""
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()
        self.eye_tracker = EyeTracker()
        self.lip_analyzer = LipAnalyzer()
        
        # Frame tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Session metrics storage
        self.session_metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'emotions_detected': [],
            'total_blinks': 0,
            'eye_strain_scores': [],
            'lip_tensions': []
        }
        
        # Previous frame data for trend detection
        self.previous_mouth_openness = None
        self.previous_landmarks = None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for all vision features
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        
        analysis_result = {
            'frame_id': self.frame_count,
            'timestamp': current_time,
            'fps': self.fps,
            'face_detected': False,
            'face_info': None,
            'emotion': None,
            'eyes': None,
            'lips': None,
            'overall_strain': None,
            'error': None
        }
        
        try:
            # Step 1: Detect face
            face_result = self.face_detector.detect_face(frame)
            
            if face_result is None:
                analysis_result['error'] = 'No face detected'
                self.session_metrics['frames_processed'] += 1
                return analysis_result
            
            analysis_result['face_detected'] = True
            self.session_metrics['faces_detected'] += 1
            
            # Step 2: Extract facial landmarks
            landmarks = self.face_detector.extract_landmarks(frame)
            
            if landmarks is None:
                analysis_result['error'] = 'Could not extract landmarks'
                self.session_metrics['frames_processed'] += 1
                return analysis_result
            
            # Step 3: Emotion Detection on face region
            face_x, face_y, face_w, face_h = face_result['bbox']
            face_region = frame[face_y:face_y+face_h, face_x:face_x+face_w]
            emotion_result = self.emotion_detector.detect_emotion(face_region)
            
            analysis_result['emotion'] = {
                'primary': emotion_result['primary_emotion'],
                'confidence': emotion_result['confidence'],
                'all_emotions': emotion_result['all_emotions'],
                'intensity': self.emotion_detector.get_emotion_intensity(
                    emotion_result['primary_emotion'],
                    emotion_result['confidence']
                )
            }
            
            if emotion_result['is_valid']:
                self.session_metrics['emotions_detected'].append(emotion_result['primary_emotion'])
            
            # Step 4: Eye Analysis
            left_eye_landmarks = landmarks['left_eye']
            right_eye_landmarks = landmarks['right_eye']
            
            # Detect blinks
            blink_result = self.eye_tracker.detect_blink(left_eye_landmarks, right_eye_landmarks)
            
            # Get pupil positions
            left_pupil = self.eye_tracker.track_pupil_position(left_eye_landmarks)
            right_pupil = self.eye_tracker.track_pupil_position(right_eye_landmarks)
            
            # Detect gaze direction
            gaze_result = self.eye_tracker.detect_gaze_direction(
                left_pupil, right_pupil,
                left_eye_landmarks, right_eye_landmarks
            )
            
            # Calculate blink rate
            blink_rate = self.eye_tracker.calculate_blink_rate()
            
            # Store eye aspect ratios for strain calculation
            ear_history = [blink_result['left_ear'], blink_result['right_ear']]
            
            # Calculate eye strain
            gaze_deviation = abs(float(gaze_result['left_gaze']) - 0.5) + abs(float(gaze_result['right_gaze']) - 0.5)
            strain_result = self.eye_tracker.calculate_eye_strain_score(
                blink_rate, ear_history, gaze_deviation
            )
            
            analysis_result['eyes'] = {
                'left_eye_aspect_ratio': blink_result['left_ear'],
                'right_eye_aspect_ratio': blink_result['right_ear'],
                'avg_eye_aspect_ratio': blink_result['avg_ear'],
                'is_blinking': blink_result['is_blink'],
                'blink_count': blink_result['blink_count'],
                'blink_rate': blink_rate,
                'gaze_direction': gaze_result['direction'],
                'gaze_details': {
                    'left_gaze_position': gaze_result['left_gaze'],
                    'right_gaze_position': gaze_result['right_gaze'],
                    'vertical_fixation': gaze_result['vertical_pos']
                },
                'strain_score': strain_result['strain_score'],
                'strain_level': strain_result['strain_level'],
                'strain_factors': strain_result['factors'],
                'strain_recommendations': strain_result['recommendations']
            }
            
            self.session_metrics['total_blinks'] = blink_result['blink_count']
            self.session_metrics['eye_strain_scores'].append(strain_result['strain_score'])
            
            # Step 5: Lip Analysis
            mouth_landmarks = landmarks['mouth']
            mouth_openness = self.lip_analyzer.calculate_mouth_openness(mouth_landmarks)
            
            lip_tension_result = self.lip_analyzer.calculate_lip_tension(mouth_landmarks)
            
            # Detect speaking
            speech_result = self.lip_analyzer.detect_speaking(
                mouth_landmarks,
                mouth_openness,
                self.previous_mouth_openness
            )
            
            # Get mouth region for color analysis
            mouth_region = self.face_detector.get_mouth_region(frame, landmarks)
            lip_color = self.lip_analyzer.analyze_lip_color_intensity(mouth_region)
            
            # Get lip tension trend
            tension_trend = self.lip_analyzer.get_tension_trend()
            
            analysis_result['lips'] = {
                'lip_tension_score': lip_tension_result['tension_score'],
                'lip_tension_level': lip_tension_result['tension_level'],
                'lip_tension_coefficient': lip_tension_result.get('tension_coefficient', 0.0),
                'mouth_openness': mouth_openness,
                'is_speaking': speech_result['is_speaking'],
                'speech_confidence': speech_result['speech_confidence'],
                'mouth_movement': speech_result['mouth_movement'],
                'lip_color_intensity': lip_color['red_intensity'],
                'lip_color_saturation': lip_color.get('saturation', 0.0),
                'tension_trend': tension_trend['trend'],
                'average_tension': tension_trend['average_tension']
            }
            
            self.session_metrics['lip_tensions'].append(lip_tension_result['tension_score'])
            
            # Step 6: Store face detection info
            analysis_result['face_info'] = {
                'bbox': face_result['bbox'],
                'confidence': face_result['confidence'],
                'face_size': (face_w, face_h)
            }
            
            # Step 7: Calculate overall strain metric
            overall_strain = (
                strain_result['strain_score'] * 0.4 +
                lip_tension_result['tension_score'] * 0.3 +
                (100 - min(100, blink_rate * 5)) * 0.3  # Adjust blink rate to 0-100
            ) / 3
            
            analysis_result['overall_strain'] = {
                'score': float(overall_strain),
                'level': self._get_overall_strain_level(overall_strain)
            }
            
            # Update session metrics
            self.session_metrics['frames_processed'] = self.frame_count
            
            # Store for next frame
            self.previous_mouth_openness = mouth_openness
            self.previous_landmarks = landmarks
            
        except Exception as e:
            analysis_result['error'] = f'Analysis error: {str(e)}'
            print(f"Error in frame analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return analysis_result
    
    def _get_overall_strain_level(self, score: float) -> str:
        """Convert strain score to level"""
        if score < 20:
            return 'minimal'
        elif score < 40:
            return 'low'
        elif score < 60:
            return 'moderate'
        elif score < 80:
            return 'high'
        else:
            return 'severe'
    
    def process_video(self, video_source: str, callback=None) -> Dict:
        """
        Process entire video file or webcam stream
        
        Args:
            video_source: Path to video file or 0 for webcam
            callback: Optional callback function for each frame analysis
            
        Returns:
            Dictionary with overall session statistics
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            return {'error': 'Could not open video source'}
        
        frame_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Analyze frame
                result = self.analyze_frame(frame)
                frame_results.append(result)
                
                # Call callback if provided
                if callback:
                    callback(frame, result)
                
                # Optional: limit for testing
                if self.frame_count >= 1000:
                    break
        
        finally:
            cap.release()
        
        return self._compile_session_statistics(frame_results)
    
    def _compile_session_statistics(self, frame_results: List[Dict]) -> Dict:
        """
        Compile overall statistics from all frames
        
        Args:
            frame_results: List of analysis results for each frame
            
        Returns:
            Dictionary with session summary
        """
        if not frame_results:
            return {'error': 'No frames processed'}
        
        # Filter valid frames
        valid_frames = [f for f in frame_results if f['face_detected']]
        
        if not valid_frames:
            return {'error': 'No faces detected in session'}
        
        # Compile statistics
        emotions = [f['emotion']['primary'] for f in valid_frames if f['emotion']]
        eye_strains = [f['eyes']['strain_score'] for f in valid_frames if f['eyes']]
        lip_tensions = [f['lips']['lip_tension_score'] for f in valid_frames if f['lips']]
        overall_strains = [f['overall_strain']['score'] for f in valid_frames if f['overall_strain']]
        
        session_stats = {
            'total_frames': len(frame_results),
            'frames_with_face': len(valid_frames),
            'detection_success_rate': len(valid_frames) / len(frame_results) if frame_results else 0,
            'duration_seconds': (frame_results[-1]['timestamp'] - frame_results[0]['timestamp']) if len(frame_results) > 1 else 0,
            'avg_fps': np.mean([f['fps'] for f in frame_results if f.get('fps', 0) > 0]) if frame_results else 0,
            'emotion_distribution': self._get_emotion_distribution(emotions),
            'dominant_emotion': max(set(emotions), key=emotions.count) if emotions else None,
            'eye_strain': {
                'average': np.mean(eye_strains) if eye_strains else 0,
                'max': np.max(eye_strains) if eye_strains else 0,
                'min': np.min(eye_strains) if eye_strains else 0
            },
            'lip_tension': {
                'average': np.mean(lip_tensions) if lip_tensions else 0,
                'max': np.max(lip_tensions) if lip_tensions else 0,
                'min': np.min(lip_tensions) if lip_tensions else 0
            },
            'overall_strain': {
                'average': np.mean(overall_strains) if overall_strains else 0,
                'max': np.max(overall_strains) if overall_strains else 0,
                'min': np.min(overall_strains) if overall_strains else 0,
                'level': self._get_overall_strain_level(np.mean(overall_strains)) if overall_strains else 'unknown'
            },
            'total_blinks': self.session_metrics['total_blinks'],
            'session_metrics': self.session_metrics
        }
        
        return session_stats
    
    def _get_emotion_distribution(self, emotions: List[str]) -> Dict:
        """Get emotion frequency distribution"""
        if not emotions:
            return {}
        
        distribution = {}
        for emotion in emotions:
            distribution[emotion] = distribution.get(emotion, 0) + 1
        
        # Convert to percentages
        total = len(emotions)
        return {k: (v / total) * 100 for k, v in distribution.items()}
    
    def draw_analysis_results(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """
        Draw all analysis results on frame
        
        Args:
            frame: Input frame
            analysis: Analysis results from analyze_frame
            
        Returns:
            Frame with all metrics drawn
        """
        output_frame = frame.copy()
        
        if not analysis['face_detected']:
            cv2.putText(output_frame, 'No face detected', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return output_frame
        
        # Draw frame info
        cv2.putText(output_frame, f"FPS: {analysis['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(output_frame, f"Frame: {analysis['frame_id']}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Draw face detection box
        if analysis['face_info']:
            x, y, w, h = analysis['face_info']['bbox']
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw emotion
        if analysis['emotion']:
            emotion_color = self.emotion_detector.get_emotion_color(analysis['emotion']['primary'])
            emotion_color_bgr = (emotion_color[2], emotion_color[1], emotion_color[0])
            emotion_text = f"{analysis['emotion']['primary'].upper()} ({analysis['emotion']['confidence']:.2f})"
            cv2.putText(output_frame, emotion_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color_bgr, 2)
        
        # Draw eye metrics
        if analysis['eyes']:
            eye_text = f"Blink Rate: {analysis['eyes']['blink_rate']:.1f} bpm | Strain: {analysis['eyes']['strain_level'].upper()}"
            cv2.putText(output_frame, eye_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Draw lip metrics
        if analysis['lips']:
            lip_text = f"Lip Tension: {analysis['lips']['lip_tension_level'].upper()} | Speaking: {'Yes' if analysis['lips']['is_speaking'] else 'No'}"
            cv2.putText(output_frame, lip_text, (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # Draw overall strain
        if analysis['overall_strain']:
            strain_color = (0, 255, 0) if analysis['overall_strain']['level'] == 'minimal' else \
                          (0, 255, 255) if analysis['overall_strain']['level'] == 'low' else \
                          (0, 165, 255) if analysis['overall_strain']['level'] == 'moderate' else \
                          (0, 0, 255)
            strain_text = f"Overall Strain: {analysis['overall_strain']['level'].upper()} ({analysis['overall_strain']['score']:.1f})"
            cv2.putText(output_frame, strain_text, (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, strain_color, 2)
        
        return output_frame
    
    def get_session_summary(self) -> Dict:
        """Get current session summary"""
        return {
            'frames_processed': self.session_metrics['frames_processed'],
            'faces_detected': self.session_metrics['faces_detected'],
            'total_blinks': self.session_metrics['total_blinks'],
            'duration_seconds': time.time() - self.start_time,
            'session_metrics': self.session_metrics
        }
    
    def reset(self):
        """Reset analyzer for new session"""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.eye_tracker.reset_blink_counter()
        self.session_metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'emotions_detected': [],
            'total_blinks': 0,
            'eye_strain_scores': [],
            'lip_tensions': []
        }
        self.previous_mouth_openness = None
        self.previous_landmarks = None
    
    def release(self):
        """Release all resources"""
        self.face_detector.release()
