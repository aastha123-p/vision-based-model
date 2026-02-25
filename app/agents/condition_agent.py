"""
Condition Prediction Agent
Uses LLM to predict medical condition
"""

from typing import Dict, List
from app.core.llm_engine import LLMEngine
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConditionAgent:
    """
    Predicts patient condition using LLM
    """

    def __init__(self, provider: str = "openai"):
        """
        Initialize condition agent
        
        Args:
            provider: LLM provider ('openai' or 'anthropic')
        """
        self.llm = LLMEngine(provider)

    def predict(
        self,
        symptom_analysis: Dict,
        vision_analysis: Dict,
        similar_cases: List[Dict],
    ) -> Dict:
        """
        Predict condition based on analysis
        
        Args:
            symptom_analysis: Symptom analysis
            vision_analysis: Vision analysis
            similar_cases: Similar past cases
            
        Returns:
            Condition prediction result
        """
        try:
            # Compile features dict
            features = {
                "form_data": {
                    "chief_complaint": symptom_analysis.get("chief_complaint"),
                    "symptoms": symptom_analysis.get("symptoms", []),
                    "severity": symptom_analysis.get("severity"),
                },
                "vision_analysis": {
                    "emotion": vision_analysis.get("emotion"),
                    "eye_strain_score": vision_analysis.get("eye_strain_level"),
                },
                "similar_cases": [
                    {
                        "condition": case.get("metadata", {}).get("condition", "Unknown"),
                        "similarity": case.get("similarity", 0),
                    }
                    for case in similar_cases[:3]
                ],
            }

            # Get LLM prediction
            result = self.llm.analyze_condition(features)

            return {
                "condition": result.get("condition", "Unknown"),
                "confidence": result.get("confidence", 0.5),
                "explanation": result.get("explanation", ""),
            }

        except Exception as e:
            logger.error(f"Error predicting condition: {e}")
            return {
                "condition": "Unable to diagnose",
                "confidence": 0.0,
                "explanation": str(e),
            }
