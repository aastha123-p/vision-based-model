"""
Symptom Analysis Agent
Analyzes patient symptoms from form submission
"""

from typing import Dict, List
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SymptomAgent:
    """
    Analyzes patient symptoms and medical history
    """

    SYMPTOM_SEVERITY = {
        "mild": 1,
        "moderate": 2,
        "severe": 3,
    }

    SYMPTOM_CATEGORIES = {
        "pain": ["pain", "ache", "hurt", "soreness"],
        "respiratory": ["cough", "throat", "asthma", "breath"],
        "digestive": ["nausea", "vomiting", "diarrhea", "acid"],
        "fever": ["fever", "temperature", "chills"],
        "allergy": ["allergy", "rash", "itch", "hive"],
    }

    def analyze(self, form_data: Dict) -> Dict:
        """
        Analyze symptoms from form
        
        Args:
            form_data: Pre-consultation form data
            
        Returns:
            Symptom analysis result
        """
        try:
            chief_complaint = form_data.get("chief_complaint", "")
            symptoms = form_data.get("symptoms", [])
            duration = form_data.get("symptom_duration", "")
            severity = form_data.get("severity", "moderate")

            # Categorize symptoms
            categories = self._categorize_symptoms(symptoms)

            # Assess severity
            severity_score = self.SYMPTOM_SEVERITY.get(severity.lower(), 2)

            # Generate summary
            summary = self._generate_summary(chief_complaint, symptoms, duration, severity)

            return {
                "chief_complaint": chief_complaint,
                "symptoms": symptoms,
                "symptom_count": len(symptoms),
                "categories": categories,
                "duration": duration,
                "severity": severity,
                "severity_score": severity_score,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            return {"error": str(e)}

    def _categorize_symptoms(self, symptoms: List[str]) -> Dict[str, List[str]]:
        """Categorize symptoms"""
        categorized = {}
        for category, keywords in self.SYMPTOM_CATEGORIES.items():
            categorized[category] = [
                s for s in symptoms if any(k in s.lower() for k in keywords)
            ]

        return {k: v for k, v in categorized.items() if v}

    @staticmethod
    def _generate_summary(
        chief_complaint: str, symptoms: List[str], duration: str, severity: str
    ) -> str:
        """Generate symptom summary"""
        symptom_list = ", ".join(symptoms) if symptoms else "Not specified"
        return f"Patient reports {chief_complaint} with symptoms: {symptom_list}. Duration: {duration}. Severity: {severity}."
