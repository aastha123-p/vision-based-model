"""
Supervisor Agent
Orchestrates the entire analysis workflow
"""

from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.utils.logger import setup_logger
from app.agents.symptom_agent import SymptomAgent
from app.agents.vision_agent import VisionAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.condition_agent import ConditionAgent
from app.agents.medication_agent import MedicationAgent
from app.agents.safety_agent import SafetyAgent
from app.agents.learning_agent import LearningAgent

logger = setup_logger(__name__)


class SupervisorAgent:
    """
    Master orchestrator managing all analysis agents
    """

    def __init__(self, db: Session):
        """
        Initialize supervisor with all agents
        
        Args:
            db: Database session
        """
        self.db = db
        self.symptom_agent = SymptomAgent()
        self.vision_agent = VisionAgent()
        self.comparison_agent = ComparisonAgent(db)
        self.condition_agent = ConditionAgent()
        self.medication_agent = MedicationAgent()
        self.safety_agent = SafetyAgent(db)
        self.learning_agent = LearningAgent(db)

    def analyze_patient(
        self,
        patient_id: int,
        form_data: Dict,
        vision_data: Dict,
        speech_data: Dict,
    ) -> Dict:
        """
        Run complete patient analysis pipeline
        
        Args:
            patient_id: Patient ID
            form_data: Pre-consultation form data
            vision_data: Vision analysis features
            speech_data: Speech analysis features
            
        Returns:
            Complete analysis result
        """
        logger.info(f"Starting analysis for patient {patient_id}")

        try:
            # Step 1: Symptom Analysis
            symptom_analysis = self.symptom_agent.analyze(form_data)
            logger.info(f"Symptom analysis complete: {symptom_analysis.get('summary')}")

            # Step 2: Vision Analysis
            vision_analysis = self.vision_agent.analyze(vision_data)
            logger.info(f"Vision analysis complete: {vision_analysis.get('summary')}")

            # Step 3: Find Similar Cases
            similar_cases = self.comparison_agent.find_similar(
                form_data, vision_data, speech_data
            )
            logger.info(f"Found {len(similar_cases)} similar cases")

            # Step 4: Condition Prediction
            condition_result = self.condition_agent.predict(
                symptom_analysis,
                vision_analysis,
                similar_cases,
            )
            logger.info(f"Condition predicted: {condition_result.get('condition')}")

            # Step 5: Medication Suggestion
            medication_result = self.medication_agent.suggest(
                condition=condition_result.get("condition"),
                patient_history=self._get_patient_history(patient_id),
            )
            logger.info(f"Medication suggested: {medication_result.get('medication')}")

            # Step 6: Safety Check
            safety_result = self.safety_agent.check(
                patient_id=patient_id,
                medication=medication_result.get("medication"),
            )
            logger.info(f"Safety check: {'PASSED' if safety_result.get('is_safe') else 'WARNINGS'}")

            # Step 7: Learning (Store for future reference)
            self.learning_agent.store_session(
                patient_id=patient_id,
                condition=condition_result.get("condition"),
                features={
                    "form": form_data,
                    "vision": vision_data,
                    "speech": speech_data,
                },
            )

            # Compile final report
            final_report = self._compile_report(
                patient_id,
                symptom_analysis,
                vision_analysis,
                similar_cases,
                condition_result,
                medication_result,
                safety_result,
            )

            logger.info(f"Analysis complete for patient {patient_id}")
            return final_report

        except Exception as e:
            logger.error(f"Error in patient analysis: {e}")
            return {"error": str(e), "status": "failed"}

    def _compile_report(
        self,
        patient_id: int,
        symptom_analysis: Dict,
        vision_analysis: Dict,
        similar_cases: list,
        condition_result: Dict,
        medication_result: Dict,
        safety_result: Dict,
    ) -> Dict:
        """Compile final analysis report"""
        return {
            "patient_id": patient_id,
            "status": "success",
            "analysis": {
                "symptoms": symptom_analysis,
                "vision": vision_analysis,
                "similar_cases": similar_cases,
                "condition": condition_result,
                "medication": medication_result,
                "safety": safety_result,
            },
            "final_recommendation": self._generate_recommendation(
                condition_result, medication_result, safety_result
            ),
        }

    @staticmethod
    def _generate_recommendation(condition: Dict, medication: Dict, safety: Dict) -> str:
        """Generate final clinical recommendation"""
        if not safety.get("is_safe"):
            return f"Condition: {condition.get('condition')} - CAUTION: See safety warnings. Consult physician before medication."

        return f"Condition: {condition.get('condition')} (Confidence: {condition.get('confidence', 0)*100:.1f}%). Suggested remedy: {medication.get('medication')}. {medication.get('dosage')}. Frequency: {medication.get('frequency')}."

    def _get_patient_history(self, patient_id: int) -> Dict:
        """Get patient medical history"""
        from app.database.models import Patient

        patient = self.db.query(Patient).filter(Patient.id == patient_id).first()
        if patient:
            return {
                "medical_history": patient.medical_history or "",
                "current_medications": patient.current_medications or [],
                "allergies": patient.allergies or [],
            }
        return {}
