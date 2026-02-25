"""
Vision Analysis FastAPI Routes
Integrates vision analysis into the FastAPI application
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import cv2
import numpy as np
import io
from PIL import Image

from app.vision.vision_analyzer import VisionAnalyzer

router = APIRouter(prefix="/api/vision", tags=["vision"])

# Global analyzer instance (in production, use dependency injection)
analyzer = None


def get_analyzer() -> VisionAnalyzer:
    """Get or create analyzer instance"""
    global analyzer
    if analyzer is None:
        analyzer = VisionAnalyzer()
    return analyzer


# Pydantic models for API responses
class EmotionResponse(BaseModel):
    primary: str
    confidence: float
    intensity: float
    all_emotions: Dict[str, float]


class EyeMetrics(BaseModel):
    blink_rate: float
    eye_aspect_ratio: float
    gaze_direction: str
    strain_score: float
    strain_level: str
    strain_recommendations: List[str]


class LipMetrics(BaseModel):
    tension_score: float
    tension_level: str
    mouth_openness: float
    is_speaking: bool
    speech_confidence: float


class FrameAnalysisResponse(BaseModel):
    frame_id: int
    face_detected: bool
    emotion: Optional[EmotionResponse] = None
    eyes: Optional[EyeMetrics] = None
    lips: Optional[LipMetrics] = None
    overall_strain_score: Optional[float] = None
    overall_strain_level: Optional[str] = None
    fps: float


class SessionSummary(BaseModel):
    total_frames: int
    frames_with_face: int
    detection_success_rate: float
    dominant_emotion: Optional[str]
    total_blinks: int
    avg_eye_strain: float
    avg_lip_tension: float
    overall_strain_level: str


# API Routes
@router.post("/analyze-image", response_model=FrameAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image for vision features
    
    Extracts:
    - Face detection
    - Emotion classification
    - Eye metrics (blink rate, strain)
    - Lip tension and speech detection
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Comprehensive vision analysis results
    """
    try:
        # Read uploaded image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze
        analyzer = get_analyzer()
        result = analyzer.analyze_frame(frame)
        
        # Format response
        response = FrameAnalysisResponse(
            frame_id=result['frame_id'],
            face_detected=result['face_detected'],
            emotion=EmotionResponse(
                primary=result['emotion']['primary'],
                confidence=result['emotion']['confidence'],
                intensity=result['emotion']['intensity'],
                all_emotions=result['emotion']['all_emotions']
            ) if result['emotion'] else None,
            eyes=EyeMetrics(
                blink_rate=result['eyes']['blink_rate'],
                eye_aspect_ratio=result['eyes']['avg_eye_aspect_ratio'],
                gaze_direction=result['eyes']['gaze_direction'],
                strain_score=result['eyes']['strain_score'],
                strain_level=result['eyes']['strain_level'],
                strain_recommendations=result['eyes']['strain_recommendations']
            ) if result['eyes'] else None,
            lips=LipMetrics(
                tension_score=result['lips']['lip_tension_score'],
                tension_level=result['lips']['lip_tension_level'],
                mouth_openness=result['lips']['mouth_openness'],
                is_speaking=result['lips']['is_speaking'],
                speech_confidence=result['lips']['speech_confidence']
            ) if result['lips'] else None,
            overall_strain_score=result['overall_strain']['score'] if result['overall_strain'] else None,
            overall_strain_level=result['overall_strain']['level'] if result['overall_strain'] else None,
            fps=result['fps']
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/analyze-base64")
async def analyze_base64(data: Dict):
    """
    Analyze image from base64 encoded string
    
    Args:
        data: JSON with 'image' field containing base64 encoded image
        
    Returns:
        Vision analysis results
    """
    try:
        import base64
        
        image_base64 = data.get('image')
        if not image_base64:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Analyze
        analyzer = get_analyzer()
        result = analyzer.analyze_frame(frame)
        
        return {
            "status": "success",
            "face_detected": result['face_detected'],
            "emotion": result['emotion'],
            "eyes": result['eyes'],
            "lips": result['lips'],
            "overall_strain": result['overall_strain']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/session-summary", response_model=SessionSummary)
async def get_session_summary():
    """
    Get summary of current analysis session
    
    Returns:
        Session statistics including emotion distribution, strain levels, blink count
    """
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_session_summary()
        
        response = SessionSummary(
            total_frames=summary['frames_processed'],
            frames_with_face=summary['faces_detected'],
            detection_success_rate=summary['faces_detected'] / summary['frames_processed'] 
                                 if summary['frames_processed'] > 0 else 0,
            dominant_emotion=summary.get('dominant_emotion'),
            total_blinks=summary['total_blinks'],
            avg_eye_strain=np.mean(summary['session_metrics']['eye_strain_scores']) 
                          if summary['session_metrics']['eye_strain_scores'] else 0,
            avg_lip_tension=np.mean(summary['session_metrics']['lip_tensions']) 
                           if summary['session_metrics']['lip_tensions'] else 0,
            overall_strain_level='minimal' if summary.get('eye_strain', {}).get('average', 0) < 20 else 'low'
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@router.post("/reset-session")
async def reset_session():
    """Reset analysis session for new recording"""
    try:
        analyzer = get_analyzer()
        analyzer.reset()
        
        return {
            "status": "success",
            "message": "Session reset successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting session: {str(e)}")


@router.get("/health")
async def health_check():
    """Check vision module health"""
    try:
        analyzer = get_analyzer()
        return {
            "status": "healthy",
            "module": "vision_analyzer",
            "models_loaded": [
                "mediapipe_face_detection",
                "mediapipe_face_mesh",
                "huggingface_emotion_transformer"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/config")
async def get_config():
    """Get vision module configuration"""
    return {
        "vision_features": {
            "face_detection": {
                "tool": "MediaPipe Face Detection",
                "provides": ["face_bbox", "facial_landmarks"]
            },
            "emotion_detection": {
                "tool": "Hugging Face Vision Transformer",
                "model": "trpakov/vit-face-expression",
                "emotions": ["happy", "sad", "angry", "surprised", "neutral", "disgusted", "fearful"]
            },
            "eye_analysis": {
                "tool": "MediaPipe Face Mesh + Custom Algorithms",
                "metrics": ["blink_rate", "eye_aspect_ratio", "gaze_direction", "eye_strain_score"]
            },
            "lip_analysis": {
                "tool": "MediaPipe Face Mesh + Custom Algorithms",
                "metrics": ["lip_tension", "mouth_openness", "speech_detection"]
            }
        },
        "performance": {
            "face_detection_fps": "30-100",
            "emotion_detection_fps": "5-15",
            "overall_fps": "5-15",
            "hardware": "CPU with GPU acceleration optional"
        }
    }


# Error handlers
@router.post("/analyze-image")
async def handle_file_error():
    """Handle file upload errors"""
    pass


if __name__ == "__main__":
    # For testing
    import uvicorn
    uvicorn.run("app.api.routes_vision:router", host="0.0.0.0", port=8000, reload=True)
