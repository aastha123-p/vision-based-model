"""
Emotion Detection using Hugging Face Pretrained Models
Classifies facial emotions: happy, sad, angry, surprised, neutral, disgusted, fearful
"""

import cv2
import numpy as np
from typing import Dict, Optional, List
from transformers import pipeline, AutoProcessor, AutoModelForImageClassification
import torch


class EmotionDetector:
    """Detects facial emotions using Hugging Face Vision Transformer models"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """
        Initialize Emotion Detection model
        
        Args:
            model_name: Hugging Face model identifier for emotion classification
                       Options: microsoft/resnet-50, google/vit-base-patch16-224, etc.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use the Hugging Face image classification pipeline for emotions
        # This is a more reliable emotion detection model
        try:
            self.emotion_pipeline = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",  # Specialized emotion detection model
                device=0 if self.device == "cuda" else -1
            )
            self.model_type = "vit-face-expression"
        except Exception as e:
            print(f"Loading specialized model failed: {e}")
            # Fallback to general image classification
            self.emotion_pipeline = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224-in21k",
                device=0 if self.device == "cuda" else -1
            )
            self.model_type = "vit-base"
        
        # Emotion labels for classification
        self.emotion_labels = [
            'angry',
            'disgusted',
            'fearful',
            'happy',
            'neutral',
            'sad',
            'surprised'
        ]
        
        # Emotion intensity calibration values
        self.emotion_calibration = {
            'anger': {'intensity_threshold': 0.6, 'rgb': (0, 0, 255)},      # Red
            'happiness': {'intensity_threshold': 0.5, 'rgb': (0, 255, 0)},   # Green
            'sadness': {'intensity_threshold': 0.5, 'rgb': (255, 0, 0)},     # Blue
            'surprise': {'intensity_threshold': 0.55, 'rgb': (0, 255, 255)}, # Yellow
            'fear': {'intensity_threshold': 0.6, 'rgb': (165, 42, 42)},      # Brown
            'disgust': {'intensity_threshold': 0.55, 'rgb': (128, 0, 128)},  # Purple
            'neutral': {'intensity_threshold': 0.4, 'rgb': (128, 128, 128)}  # Gray
        }
    
    def detect_emotion(self, face_image: np.ndarray, top_k: int = 3) -> Dict:
        """
        Detect emotion from face image
        
        Args:
            face_image: Face region image (BGR format)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with emotion predictions and confidence scores
        """
        if face_image is None or face_image.size == 0:
            return {
                'primary_emotion': 'unknown',
                'confidence': 0.0,
                'all_emotions': {},
                'is_valid': False
            }
        
        try:
            # Convert BGR to RGB for model input
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            
            # Convert numpy array to PIL Image for the pipeline
            from PIL import Image
            pil_image = Image.fromarray(rgb_image)
            
            # Perform emotion classification
            results = self.emotion_pipeline(pil_image, top_k=top_k)
            
            if not results:
                return {
                    'primary_emotion': 'unknown',
                    'confidence': 0.0,
                    'all_emotions': {},
                    'is_valid': False
                }
            
            # Parse results
            emotion_dict = {}
            primary_emotion = None
            primary_confidence = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                emotion_dict[label] = score
                
                if score > primary_confidence:
                    primary_confidence = score
                    primary_emotion = label
            
            return {
                'primary_emotion': primary_emotion if primary_emotion else 'neutral',
                'confidence': float(primary_confidence),
                'all_emotions': emotion_dict,
                'is_valid': True
            }
            
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return {
                'primary_emotion': 'unknown',
                'confidence': 0.0,
                'all_emotions': {},
                'is_valid': False,
                'error': str(e)
            }
    
    def get_emotion_intensity(self, emotion: str, confidence: float) -> float:
        """
        Calculate emotion intensity based on confidence score
        
        Args:
            emotion: Emotion label
            confidence: Confidence score (0-1)
            
        Returns:
            Scaled intensity value (0-100)
        """
        emotion_lower = emotion.lower()
        
        # Get calibration value or default
        if emotion_lower in self.emotion_calibration:
            threshold = self.emotion_calibration[emotion_lower]['intensity_threshold']
        else:
            threshold = 0.5
        
        # Scale confidence to intensity (0-100)
        if confidence >= threshold:
            intensity = ((confidence - threshold) / (1 - threshold)) * 100
        else:
            intensity = (confidence / threshold) * 50
        
        return min(100.0, max(0.0, intensity))
    
    def analyze_emotion_trend(self, emotion_history: List[Dict]) -> Dict:
        """
        Analyze emotion trends over time
        
        Args:
            emotion_history: List of emotion detection results over time
            
        Returns:
            Dictionary with emotion trend analysis
        """
        if not emotion_history:
            return {}
        
        emotion_counts = {}
        total_confidence = {}
        
        for record in emotion_history:
            emotion = record.get('primary_emotion', 'unknown')
            confidence = record.get('confidence', 0.0)
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                total_confidence[emotion] = 0.0
            
            emotion_counts[emotion] += 1
            total_confidence[emotion] += confidence
        
        # Calculate average confidence for each emotion
        emotion_averages = {}
        for emotion in emotion_counts:
            emotion_averages[emotion] = {
                'count': emotion_counts[emotion],
                'avg_confidence': total_confidence[emotion] / emotion_counts[emotion],
                'frequency': emotion_counts[emotion] / len(emotion_history)
            }
        
        # Find most common emotion
        most_common = max(emotion_averages.items(), key=lambda x: x[1]['count'])[0]
        
        return {
            'emotion_distribution': emotion_averages,
            'most_common_emotion': most_common,
            'total_frames': len(emotion_history),
            'confidence_trend': [r.get('confidence', 0.0) for r in emotion_history]
        }
    
    def get_emotion_color(self, emotion: str) -> tuple:
        """
        Get RGB color for emotion visualization
        
        Args:
            emotion: Emotion label
            
        Returns:
            RGB tuple for color visualization
        """
        emotion_lower = emotion.lower()
        
        if emotion_lower in self.emotion_calibration:
            return self.emotion_calibration[emotion_lower]['rgb']
        
        return (128, 128, 128)  # Default gray
    
    def draw_emotion_label(self, frame: np.ndarray, emotion: str, 
                          confidence: float, position: tuple = (10, 30)) -> np.ndarray:
        """
        Draw emotion label and confidence on frame
        
        Args:
            frame: Input frame
            emotion: Detected emotion
            confidence: Confidence score
            position: (x, y) position to draw text
            
        Returns:
            Frame with drawn emotion information
        """
        output_frame = frame.copy()
        
        text = f"Emotion: {emotion.upper()} ({confidence:.2f})"
        color = self.get_emotion_color(emotion)
        
        # Convert BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        
        cv2.putText(
            output_frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2
        )
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = position[0]
        bar_y = position[1] + 30
        
        # Background bar (gray)
        cv2.rectangle(output_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (200, 200, 200), -1)
        
        # Confidence bar (colored)
        filled_width = int(bar_width * confidence)
        cv2.rectangle(output_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                     color_bgr, -1)
        
        return output_frame
    
    def batch_detect_emotions(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Detect emotions from multiple face images
        
        Args:
            face_images: List of face region images
            
        Returns:
            List of emotion detection results
        """
        results = []
        for face_image in face_images:
            result = self.detect_emotion(face_image)
            results.append(result)
        
        return results
