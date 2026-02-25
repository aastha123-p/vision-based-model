"""
Safety Rules Module
Medication safety checks and contraindications
"""

from typing import Dict, List, Tuple
from sqlalchemy.orm import Session
from app.database.models import SafetyRule, Patient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyChecker:
    """
    Medical safety rule enforcement
    """

    # Hardcoded medication conflict database
    MEDICATION_CONFLICTS = {
        "Aspirin": {
            "conflicts": ["Warfarin", "Clopidogrel"],
            "warnings": "Increased bleeding risk",
        },
        "Ibuprofen": {
            "conflicts": ["Aspirin", "Naproxen"],
            "warnings": "Increased GI complications",
        },
        "Metformin": {
            "conflicts": ["Contrast dye"],
            "warnings": "Lactic acidosis risk",
        },
        "Lisinopril": {
            "conflicts": ["Potassium supplements"],
            "warnings": "Hyperkalemia risk",
        },
        "Amoxicillin": {
            "conflicts": ["Methotrexate"],
            "warnings": "Reduced methotrexate clearance",
        },
    }

    # Contraindications (condition restrictions)
    CONTRAINDICATIONS = {
        "Aspirin": ["Bleeding disorder", "Peptic ulcer"],
        "NSAIDs": ["Kidney disease", "Heart failure"],
        "ACE inhibitors": ["Pregnancy", "Angioedema"],
        "Metformin": ["Kidney disease (eGFR <30)", "Heart failure"],
        "Statins": ["Liver disease"],
    }

    # Condition severity mapping
    CONDITION_SEVERITY = {
        "Fever condition": 1,
        "Pain condition": 2,
        "Allergy condition": 1,
        "General wellness check": 0,
    }

    def __init__(self, db: Session = None):
        """
        Initialize safety checker
        
        Args:
            db: Database session
        """
        self.db = db

    def check_medication_conflicts(
        self, new_medication: str, current_medications: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check for medication conflicts
        
        Args:
            new_medication: New medication to check
            current_medications: List of current medications
            
        Returns:
            Tuple of (is_safe, list of conflicts)
        """
        conflicts = []

        new_med_conflicts = self.MEDICATION_CONFLICTS.get(
            new_medication, {}
        ).get("conflicts", [])

        for current_med in current_medications:
            if current_med in new_med_conflicts:
                conflicts.append(
                    f"{new_medication} conflicts with {current_med}: {self.MEDICATION_CONFLICTS.get(new_medication, {}).get('warnings', 'Check interactions')}"
                )

        is_safe = len(conflicts) == 0
        return is_safe, conflicts

    def check_contraindications(
        self, medication: str, patient_conditions: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check for contraindications
        
        Args:
            medication: Medication to check
            patient_conditions: Patient's conditions
            
        Returns:
            Tuple of (is_safe, list of contraindications)
        """
        contraindications = []
        med_contraindications = self.CONTRAINDICATIONS.get(medication, [])

        for condition in patient_conditions:
            for contra in med_contraindications:
                if contra.lower() in condition.lower():
                    contraindications.append(
                        f"{medication} is contraindicated in {condition}"
                    )

        is_safe = len(contraindications) == 0
        return is_safe, contraindications

    def check_patient_safety(
        self, patient_id: int, medication: str, db: Session = None
    ) -> Dict:
        """
        Comprehensive safety check for a patient
        
        Args:
            patient_id: Patient ID
            medication: Medication to check
            db: Database session
            
        Returns:
            Safety check results dictionary
        """
        db = db or self.db
        if not db:
            return {"is_safe": True, "warnings": [], "note": "Database not available"}

        try:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return {"is_safe": False, "warnings": ["Patient not found"]}

            current_meds = patient.current_medications or []
            allergies = patient.allergies or []
            medical_history = patient.medical_history or ""

            warnings = []

            # Check conflicts
            is_safe_conflicts, conflict_warnings = self.check_medication_conflicts(
                medication, current_meds
            )
            warnings.extend(conflict_warnings)

            # Check contraindications
            is_safe_contra, contra_warnings = self.check_contraindications(
                medication, medical_history.split(",")
            )
            warnings.extend(contra_warnings)

            # Check allergies
            if medication.lower() in [a.lower() for a in allergies]:
                warnings.append(f"Patient has documented allergy to {medication}")
                is_safe_conflicts = False

            return {
                "is_safe": is_safe_conflicts and is_safe_contra,
                "warnings": warnings,
                "medication": medication,
                "patient_id": patient_id,
            }

        except Exception as e:
            logger.error(f"Error checking patient safety: {e}")
            return {"is_safe": False, "warnings": [str(e)]}

    def get_safety_profile(self, medication: str) -> Dict:
        """
        Get safety profile for a medication
        
        Args:
            medication: Medication name
            
        Returns:
            Safety profile dictionary
        """
        return {
            "medication": medication,
            "conflicts": self.MEDICATION_CONFLICTS.get(medication, {}).get(
                "conflicts", []
            ),
            "contraindications": self.CONTRAINDICATIONS.get(medication, []),
            "common_side_effects": self._get_common_side_effects(medication),
            "safe_in_pregnancy": not any(
                "pregnancy" in c.lower()
                for c in self.CONTRAINDICATIONS.get(medication, [])
            ),
        }

    def validate_condition_severity(self, condition: str, medication: str) -> Dict:
        """
        Validate if medication is appropriate for condition severity
        
        Args:
            condition: Patient condition
            medication: Proposed medication
            
        Returns:
            Validation result dictionary
        """
        severity = self.CONDITION_SEVERITY.get(condition, 1)

        return {
            "condition": condition,
            "medication": medication,
            "severity": severity,
            "appropriate": True,
            "recommendation": f"{medication} is appropriate for {condition}",
        }

    @staticmethod
    def _get_common_side_effects(medication: str) -> List[str]:
        """Get common side effects for medication"""
        side_effects = {
            "Aspirin": ["Stomach upset", "GI bleeding"],
            "Ibuprofen": ["Stomach pain", "Drowsiness"],
            "Metformin": ["Nausea", "Diarrhea"],
            "Lisinopril": ["Dry cough", "Dizziness"],
            "Amoxicillin": ["Allergic reaction", "Diarrhea"],
        }
        return side_effects.get(medication, [])

    @staticmethod
    def format_safety_report(safety_check: Dict) -> str:
        """
        Format safety check for display
        
        Args:
            safety_check: Safety check result
            
        Returns:
            Formatted report string
        """
        status = "✓ SAFE" if safety_check.get("is_safe") else "✗ WARNING"
        warnings = "\n".join(safety_check.get("warnings", []))
        return f"{status}\nMedication: {safety_check.get('medication')}\nWarnings:\n{warnings or 'None'}"
