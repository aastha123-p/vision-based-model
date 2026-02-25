"""
Authentication API Routes
Handles face login and token-based authentication
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
import cv2
import numpy as np
from io import BytesIO
import os

from app.database.db import get_db
from app.auth.token_auth import TokenAuthenticator
from app.auth.face_auth import FaceAuthenticator
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class TokenRequest(BaseModel):
    token: str


class FaceLoginRequest(BaseModel):
    image_data: str = None  # Base64 encoded image


class FaceRegistrationRequest(BaseModel):
    patient_id: int
    name: str


@router.post("/register")
async def register_new_patient(
    name: str,
    email: str = None,
    phone: str = None,
    age: int = None,
    gender: str = None,
    db: Session = Depends(get_db),
):
    """Register new patient with token"""
    try:
        patient, token = TokenAuthenticator.create_patient(
            db, name, email, phone, age, gender
        )

        if not patient:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register patient",
            )

        return {
            "patient_id": patient.id,
            "token": token,
            "message": "Patient registered successfully. Use this token for login.",
        }

    except Exception as e:
        logger.error(f"Error registering patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/face-register")
async def register_face(patient_id: int, db: Session = Depends(get_db)):
    """Register patient's face for authentication"""
    try:
        face_auth = FaceAuthenticator()
        success, message = face_auth.register_face(db, patient_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=message
            )

        return {
            "patient_id": patient_id,
            "message": message,
            "status": "registered",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/face-login")
async def face_login(db: Session = Depends(get_db)):
    """Authenticate patient by face capture"""
    try:
        face_auth = FaceAuthenticator()
        patient_id, similarity, message = face_auth.authenticate_face(db)

        if patient_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=message
            )

        return {
            "patient_id": patient_id,
            "similarity_score": similarity,
            "message": message,
            "status": "authenticated",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/token-login")
async def token_login(request: TokenRequest, db: Session = Depends(get_db)):
    """Authenticate patient by token (fallback)"""
    try:
        patient_id, message = TokenAuthenticator.authenticate_token(db, request.token)

        if patient_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        return {
            "patient_id": patient_id,
            "message": message,
            "status": "authenticated",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in token login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/logout")
async def logout(patient_id: int, db: Session = Depends(get_db)):
    """Logout (revoke current token)"""
    try:
        success = TokenAuthenticator.revoke_token(db, patient_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to logout",
            )

        return {
            "patient_id": patient_id,
            "message": "Logged out successfully",
            "status": "logged_out",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/face-upload")
async def upload_face_image(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload face image for registration from webcam capture"""
    try:
        # Read image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file",
            )

        # Register face
        face_auth = FaceAuthenticator()
        success, message = face_auth.register_face(db, patient_id, face_image)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=message
            )

        return {
            "patient_id": patient_id,
            "message": message,
            "status": "registered",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading face image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/status/{patient_id}")
async def get_auth_status(patient_id: int, db: Session = Depends(get_db)):
    """Get patient authentication status"""
    try:
        from app.database.models import Patient

        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
            )

        has_face_embedding = patient.face_embedding is not None

        return {
            "patient_id": patient_id,
            "has_token": True,
            "has_face_embedding": has_face_embedding,
            "can_use_face_login": has_face_embedding,
            "last_login": patient.last_login,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )