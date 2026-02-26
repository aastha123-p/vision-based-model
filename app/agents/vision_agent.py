"""
Vision Analysis Agent
Analyzes facial and eye-based features
"""

from typing import Dict
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class VisionAgent:
    """
    Analyzes vision-based patient observations
    """

    EMOTION_IMPLICATIONS = {
        "happy": "Good overall state, positive mood",
        "sad": "Low mood, possible depression",
        "angry": "Agitation, pain or frustration",
        "fear": "Anxiety or nervousness",
        "neutral": "Calm demeanor",
        "surprise": "Unexpected reaction",
        "disgust": "Discomfort or revulsion",
    }

    EYE_STRAIN_THRESHOLDS = {
        "normal": (0, 0.3),
        "mild": (0.3, 0.6),
        "moderate": (0.6, 0.8),
        "severe": (0.8, 1.0),
    }

    def analyze(self, vision_data: Dict) -> Dict:
        """
        Analyze vision features
        
        Args:
            vision_data: Vision analysis data
            
        Returns:
            Vision analysis result
        """
        try:
            emotion = vision_data.get("emotion", "neutral")
            emotion_confidence = vision_data.get("emotion_confidence", 0.5)
            blink_rate = vision_data.get("blink_rate", 0)
            eye_strain = vision_data.get("eye_strain_score", 0.3)
            lip_tension = vision_data.get("lip_tension", 0.5)

            # Assess emotional state
            emotional_assessment = self._assess_emotion(emotion, emotion_confidence)

            # Assess eye health
            eye_assessment = self._assess_eye_health(blink_rate, eye_strain)

            # Generate summary
            summary = self._generate_summary(
                emotion, eye_strain, blink_rate, emotional_assessment
            )

            return {
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "emotional_assessment": emotional_assessment,
                "blink_rate": blink_rate,
                "eye_strain_level": self._get_eye_strain_level(eye_strain),
                "eye_assessment": eye_assessment,
                "lip_tension": lip_tension,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Error analyzing vision: {e}")
            return {"error": str(e)}

    def _assess_emotion(self, emotion: str, confidence: float) -> str:
        """Assess emotional state"""
        assessment = self.EMOTION_IMPLICATIONS.get(emotion.lower(), "Neutral state")
        confidence_text = "Confident" if confidence > 0.7 else "Moderate" if confidence > 0.5 else "Uncertain"
        return f"{assessment} ({confidence_text} - {confidence*100:.1f}%)"

    @staticmethod
    def _assess_eye_health(blink_rate: float, eye_strain: float) -> str:
        """Assess eye health"""
        if blink_rate < 10:
            return "Low blink rate - possible concentration or discomfort"
        elif blink_rate > 30:
            return "High blink rate - possible irritation or nervousness"
        elif eye_strain > 0.7:
            return "Significant eye strain detected"
        else:
            return "Normal eye function"

    @staticmethod
    def _get_eye_strain_level(strain_score: float) -> str:
        """Get eye strain level"""
        if strain_score < 0.3:
            return "Normal"
        elif strain_score < 0.6:
            return "Mild"
        elif strain_score < 0.8:
            return "Moderate"
        else:
            return "Severe"

    @staticmethod
    def _generate_summary(
        emotion: str, eye_strain: float, blink_rate: float, emotional_assessment: str
    ) -> str:
        """Generate vision summary"""
        return f"Patient displays {emotion} emotion. Eye strain: {eye_strain*100:.1f}%. Blink rate: {blink_rate:.1f} blinks/min. {emotional_assessment}."
