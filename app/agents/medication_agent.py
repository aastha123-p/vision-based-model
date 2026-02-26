"""
Medication Suggestion Agent
Suggests appropriate remedies
"""

from typing import Dict
from app.core.llm_engine import LLMEngine
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class MedicationAgent:
    """
    Suggests medications/remedies based on condition
    """

    def __init__(self, provider: str = "openai"):
        """Initialize medication agent"""
        self.llm = LLMEngine(provider)

    def suggest(self, condition: str, patient_history: Dict) -> Dict:
        """
        Suggest medication for condition
        
        Args:
            condition: Predicted condition
            patient_history: Patient medical history
            
        Returns:
            Medication suggestion
        """
        try:
            result = self.llm.suggest_medication(condition, patient_history)

            return {
                "medication": result.get("medication", "Consult physician"),
                "dosage": result.get("dosage", "As prescribed"),
                "frequency": result.get("frequency", "As needed"),
                "warnings": result.get("warnings", ""),
                "condition": condition,
            }

        except Exception as e:
            logger.error(f"Error suggesting medication: {e}")
            return {
                "medication": "Consult physician",
                "dosage": "As prescribed",
                "frequency": "As needed",
                "warnings": str(e),
                "condition": condition,
            }
