from sqlalchemy.orm import Session
from app.database.models import Patient

class PatientCRUD:

    @staticmethod
    def get_patient_by_token(db: Session, token: str):
        return db.query(Patient).filter(Patient.token == token).first()

    @staticmethod
    def create_patient(db: Session, name: str, token: str, face_embedding: str = None):
        patient = Patient(
            name=name,
            token=token,
            face_embedding=face_embedding
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient