"""
Safety Checking Agent
Validates medication safety and contraindications
"""

from typing import Dict
from sqlalchemy.orm import Session
from app.core.safety_rules import SafetyChecker
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyAgent:
    """
    Performs safety checks on medications
    """

    def __init__(self, db: Session):
        """Initialize safety agent"""
        self.db = db
        self.safety_checker = SafetyChecker(db)

    def check(self, patient_id: int, medication: str) -> Dict:
        """
        Check medication safety for patient
        
        Args:
            patient_id: Patient ID
            medication: Medication to check
            
        Returns:
            Safety check result
        """
        try:
            result = self.safety_checker.check_patient_safety(
                patient_id, medication, self.db
            )

            return {
                "is_safe": result.get("is_safe", True),
                "medication": medication,
                "patient_id": patient_id,
                "warnings": result.get("warnings", []),
                "status": "SAFE" if result.get("is_safe") else "WARNING",
            }

        except Exception as e:
            logger.error(f"Error checking medication safety: {e}")
            return {
                "is_safe": False,
                "medication": medication,
                "patient_id": patient_id,
                "warnings": [f"Safety check error: {str(e)}"],
                "status": "ERROR",
            }
