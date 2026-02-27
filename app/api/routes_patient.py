"""
Patient API Routes
Handles patient registration and form submission
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.database.models import Patient
from app.auth.token_auth import TokenAuthenticator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/patient", tags=["patient"])


# Pydantic models for request body
class RegisterRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class FormSubmissionRequest(BaseModel):
    patient_id: int
    chief_complaint: str
    symptoms: List[str] = Field(..., min_items=1, description="List of symptoms")
    symptom_duration: str
    severity: str
    medical_history: Optional[str] = None


@router.post("/register")
async def register_patient(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Register new patient"""
    try:
        patient, token = TokenAuthenticator.create_patient(
            db, 
            request.name, 
            request.email, 
            request.phone, 
            request.age, 
            request.gender
        )

        if not patient:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register patient",
            )

        return {
            "patient_id": patient.id,
            "token": token,
            "message": "Patient registered successfully",
        }

    except Exception as e:
        logger.error(f"Error registering patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/form")
async def submit_form(
    request: FormSubmissionRequest,
    db: Session = Depends(get_db),
):
    """Submit pre-consultation form"""
    try:
        patient = db.query(Patient).filter(Patient.id == request.patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
            )

        # Update patient with form data
        patient.medical_history = request.medical_history
        db.commit()

        return {
            "patient_id": request.patient_id,
            "chief_complaint": request.chief_complaint,
            "symptoms": request.symptoms,
            "symptom_duration": request.symptom_duration,
            "severity": request.severity,
            "message": "Form submitted successfully",
        }

    except Exception as e:
        logger.error(f"Error submitting form: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/profile/{patient_id}")
async def get_patient_profile(patient_id: int, db: Session = Depends(get_db)):
    """Get patient profile"""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
            )

        return {
            "id": patient.id,
            "name": patient.name,
            "email": patient.email,
            "phone": patient.phone,
            "age": patient.age,
            "gender": patient.gender,
            "medical_history": patient.medical_history,
            "created_at": patient.created_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/update")
async def update_patient(
    patient_id: int,
    medical_history: str = None,
    current_medications: list = None,
    allergies: list = None,
    db: Session = Depends(get_db),
):
    """Update patient information"""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
            )

        if medical_history:
            patient.medical_history = medical_history
        if current_medications:
            patient.current_medications = current_medications
        if allergies:
            patient.allergies = allergies

        db.commit()

        return {"patient_id": patient_id, "message": "Patient updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
