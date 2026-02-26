"""
Token-Based Authentication Module
Handles token validation and patient authentication
"""

import secrets
import json
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from app.database.models import Patient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TokenAuthenticator:
    """
    Token-based authentication manager
    """

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate a secure token
        
        Args:
            length: Token length
            
        Returns:
            Secure random token
        """
        return secrets.token_urlsafe(length)

    @staticmethod
    def authenticate_token(db: Session, token: str) -> Tuple[Optional[int], str]:
        """
        Authenticate patient by token
        
        Args:
            db: Database session
            token: Patient token
            
        Returns:
            Tuple of (patient_id, message)
        """
        try:
            patient = db.query(Patient).filter(Patient.token == token).first()
            
            if patient:
                patient.last_login = datetime.utcnow()
                db.commit()
                logger.info(f"Token authenticated patient: {patient.id}")
                return patient.id, f"Welcome back, {patient.name}"
            else:
                logger.warning(f"Invalid token attempt")
                return None, "Invalid token"

        except Exception as e:
            logger.error(f"Error authenticating token: {e}")
            return None, str(e)

    @staticmethod
    def validate_token(db: Session, token: str) -> bool:
        """
        Check if token is valid
        
        Args:
            db: Database session
            token: Patient token
            
        Returns:
            True if valid
        """
        try:
            patient = db.query(Patient).filter(Patient.token == token).first()
            return patient is not None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False

    @staticmethod
    def get_patient_by_token(db: Session, token: str) -> Optional[Patient]:
        """
        Get patient object by token
        
        Args:
            db: Database session
            token: Patient token
            
        Returns:
            Patient object or None
        """
        try:
            return db.query(Patient).filter(Patient.token == token).first()
        except Exception as e:
            logger.error(f"Error getting patient by token: {e}")
            return None

    @staticmethod
    def create_patient(
        db: Session,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> Tuple[Optional[Patient], str]:
        """
        Create new patient with token
        
        Args:
            db: Database session
            name: Patient name
            email: Patient email
            phone: Patient phone
            age: Patient age
            gender: Patient gender
            
        Returns:
            Tuple of (Patient object, token)
        """
        try:
            token = TokenAuthenticator.generate_token()
            
            patient = Patient(
                name=name,
                email=email,
                phone=phone,
                age=age,
                gender=gender,
                token=token,
            )
            
            db.add(patient)
            db.commit()
            db.refresh(patient)
            
            logger.info(f"Created new patient: {patient.id}")
            return patient, token

        except Exception as e:
            logger.error(f"Error creating patient: {e}")
            db.rollback()
            return None, None

    @staticmethod
    def revoke_token(db: Session, patient_id: int) -> bool:
        """
        Revoke patient's token (logout)
        
        Args:
            db: Database session
            patient_id: Patient ID
            
        Returns:
            True if successful
        """
        try:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if patient:
                patient.token = TokenAuthenticator.generate_token()
                db.commit()
                logger.info(f"Revoked token for patient: {patient_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            db.rollback()
            return False