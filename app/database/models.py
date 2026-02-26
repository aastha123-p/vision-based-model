"""
SQLAlchemy ORM models for Vision Agentic AI MVP
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Patient(Base):
    """Patient information table"""

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    phone = Column(String(20), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    
    # Authentication
    token = Column(String(255), unique=True, index=True, nullable=False)
    face_embedding = Column(Text, nullable=True)  # Stored as JSON string
    
    # Medical History
    medical_history = Column(Text, nullable=True)
    current_medications = Column(JSON, nullable=True)
    allergies = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.name}, email={self.email})>"


class Session(Base):
    """Session/Consultation table"""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=False, index=True)
    
    # Pre-consultation form data
    chief_complaint = Column(String(255), nullable=True)
    symptoms = Column(JSON, nullable=True)
    symptom_duration = Column(String(100), nullable=True)
    severity = Column(String(50), nullable=True)
    
    # Vision Features
    emotion = Column(String(50), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    blink_rate = Column(Float, nullable=True)
    eye_strain_score = Column(Float, nullable=True)
    lip_tension = Column(Float, nullable=True)
    vision_features = Column(JSON, nullable=True)
    
    # Speech Features
    transcript = Column(Text, nullable=True)
    sentiment = Column(String(50), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    speech_features = Column(JSON, nullable=True)
    
    # Analysis Results
    similar_cases = Column(JSON, nullable=True)  # Top 5 similar cases
    similar_cases_count = Column(Integer, default=0)
    
    # Prediction Results
    predicted_condition = Column(String(255), nullable=True)
    condition_confidence = Column(Float, nullable=True)
    condition_explanation = Column(Text, nullable=True)
    
    # Recommendations
    suggested_medication = Column(String(255), nullable=True)
    medication_dosage = Column(String(255), nullable=True)
    medication_warnings = Column(JSON, nullable=True)
    general_advice = Column(Text, nullable=True)
    
    # Safety Checks
    safety_check_passed = Column(Integer, default=1)
    safety_warnings = Column(JSON, nullable=True)
    
    # Session metadata
    duration_seconds = Column(Integer, nullable=True)
    embedding = Column(Text, nullable=True)  # Multimodal embedding
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Session(id={self.id}, patient_id={self.patient_id}, condition={self.predicted_condition})>"


class FaceEmbedding(Base):
    """Store face embeddings for face authentication"""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=False, index=True)
    embedding = Column(Text, nullable=False)  # Stored as JSON string
    captured_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<FaceEmbedding(id={self.id}, patient_id={self.patient_id})>"


class SafetyRule(Base):
    """Hardcoded medical safety rules"""

    __tablename__ = "safety_rules"

    id = Column(Integer, primary_key=True, index=True)
    medication_name = Column(String(255), unique=True, nullable=False)
    conflict_medications = Column(JSON, nullable=True)  # List of conflicting drugs
    contraindications = Column(JSON, nullable=True)    # List of conditions
    warnings = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<SafetyRule(id={self.id}, medication={self.medication_name})>"