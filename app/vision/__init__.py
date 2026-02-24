"""
Vision Module - Multimodal Feature Extraction
Provides comprehensive vision-based analysis using:
- MediaPipe for face detection and landmarks
- Hugging Face transformers for emotion detection
- OpenCV for video processing
- Custom algorithms for eye tracking and lip analysis
"""

from app.vision.face_detector import FaceDetector
from app.vision.emotion_detector import EmotionDetector
from app.vision.eye_tracker import EyeTracker
from app.vision.lip_analyzer import LipAnalyzer
from app.vision.vision_analyzer import VisionAnalyzer

__all__ = [
    'FaceDetector',
    'EmotionDetector',
    'EyeTracker',
    'LipAnalyzer',
    'VisionAnalyzer'
]
