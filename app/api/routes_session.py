"""
Session API Routes
Handles consultation sessions and analysis
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.database.models import Session as SessionModel, Patient
from app.agents.supervisor_agent import SupervisorAgent
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("/start")
async def start_session(
    patient_id: int,
    chief_complaint: str,
    symptoms: list,
    symptom_duration: str,
    severity: str,
    db: Session = Depends(get_db),
):
    """Start analysis session"""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
            )

        # Create session record
        session = SessionModel(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            symptom_duration=symptom_duration,
            severity=severity,
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info(f"Started session {session.id} for patient {patient_id}")

        return {
            "session_id": session.id,
            "patient_id": patient_id,
            "status": "started",
            "message": "Session started. Ready for vision and speech analysis.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/analyze")
async def analyze_session(
    session_id: int,
    vision_data: dict,
    speech_data: dict,
    db: Session = Depends(get_db),
):
    """Analyze session with vision and speech data"""
    try:
        session_rec = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        # Run analysis
        supervisor = SupervisorAgent(db)

        form_data = {
            "chief_complaint": session_rec.chief_complaint,
            "symptoms": session_rec.symptoms,
            "symptom_duration": session_rec.symptom_duration,
            "severity": session_rec.severity,
        }

        analysis_result = supervisor.analyze_patient(
            session_rec.patient_id,
            form_data,
            vision_data,
            speech_data,
        )

        # Update session with results
        if analysis_result.get("status") == "success":
            result = analysis_result.get("analysis", {})

            session_rec.emotion = result.get("vision", {}).get("emotion")
            session_rec.emotion_confidence = result.get("vision", {}).get(
                "emotion_confidence"
            )
            session_rec.blink_rate = result.get("vision", {}).get("blink_rate")
            session_rec.eye_strain_score = result.get("vision", {}).get(
                "eye_strain_level"
            )
            session_rec.vision_features = vision_data

            session_rec.transcript = result.get("speech", {}).get("transcript", "")
            session_rec.sentiment = result.get("speech", {}).get("sentiment")
            session_rec.sentiment_score = result.get("speech", {}).get(
                "sentiment_score"
            )
            session_rec.speech_features = speech_data

            session_rec.predicted_condition = result.get("condition", {}).get(
                "condition"
            )
            session_rec.condition_confidence = result.get("condition", {}).get(
                "confidence"
            )
            session_rec.condition_explanation = result.get("condition", {}).get(
                "explanation"
            )

            session_rec.suggested_medication = result.get("medication", {}).get(
                "medication"
            )
            session_rec.medication_dosage = result.get("medication", {}).get("dosage")
            session_rec.medication_warnings = result.get("medication", {}).get(
                "warnings"
            )

            session_rec.safety_check_passed = 1 if result.get("safety", {}).get("is_safe") else 0
            session_rec.safety_warnings = result.get("safety", {}).get("warnings")

            session_rec.similar_cases = result.get("similar_cases", [])
            session_rec.similar_cases_count = len(result.get("similar_cases", []))

            db.commit()

        return {
            "session_id": session_id,
            "status": analysis_result.get("status"),
            "analysis": analysis_result.get("analysis"),
            "recommendation": analysis_result.get("final_recommendation"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing session: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/{session_id}")
async def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get session details"""
    try:
        session_rec = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return {
            "id": session_rec.id,
            "patient_id": session_rec.patient_id,
            "chief_complaint": session_rec.chief_complaint,
            "predicted_condition": session_rec.predicted_condition,
            "confidence": session_rec.condition_confidence,
            "emotion": session_rec.emotion,
            "sentiment": session_rec.sentiment,
            "medication": session_rec.suggested_medication,
            "created_at": session_rec.created_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
