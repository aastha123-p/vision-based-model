from sqlalchemy.orm import Session
from app.database.crud import PatientCRUD

class TokenAuthenticator:
    """
    Handles token-based authentication (Fallback Login)
    """

    def __init__(self, db: Session):
        self.db = db

    def authenticate(self, token: str):
        patient = PatientCRUD.get_patient_by_token(self.db, token)

        if not patient:
            return None

        return {
            "id": patient.id,
            "name": patient.name,
            "token": patient.token
        }