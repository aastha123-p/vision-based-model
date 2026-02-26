"""
Report API Routes
Handles report generation and sending
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.database.models import Session as SessionModel
from app.reports.pdf_generator import PDFGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("/generate/{session_id}")
async def generate_report(session_id: int, db: Session = Depends(get_db)):
    """Generate PDF report for session"""
    try:
        session_rec = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        pdf_gen = PDFGenerator()
        pdf_path = pdf_gen.generate_session_report(session_rec)

        return {
            "session_id": session_id,
            "report_path": pdf_path,
            "message": "Report generated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/email/{session_id}")
async def email_report(
    session_id: int, recipient_email: str, db: Session = Depends(get_db)
):
    """Email report to patient"""
    try:
        session_rec = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        from app.reports.email_service import EmailService

        email_service = EmailService()
        success = email_service.send_session_report(session_rec, recipient_email)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send email",
            )

        return {
            "session_id": session_id,
            "recipient": recipient_email,
            "message": "Report sent successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/list/{patient_id}")
async def list_patient_reports(patient_id: int, db: Session = Depends(get_db)):
    """List all reports for patient"""
    try:
        sessions = db.query(SessionModel).filter(
            SessionModel.patient_id == patient_id
        ).all()

        return {
            "patient_id": patient_id,
            "total_sessions": len(sessions),
            "sessions": [
                {
                    "session_id": s.id,
                    "condition": s.predicted_condition,
                    "confidence": s.condition_confidence,
                    "created_at": s.created_at,
                }
                for s in sessions
            ],
        }

    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
