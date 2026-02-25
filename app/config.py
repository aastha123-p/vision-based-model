"""
Configuration module for Vision Agentic AI MVP
Centralized settings for all components
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vision_ai.db")
    SQLALCHEMY_ECHO = False

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Face Recognition & Auth
    FACE_EMBEDDING_DIM = 128  # DeepFace embedding dimension
    FACE_SIMILARITY_THRESHOLD = 0.60  # Similarity threshold for face match
    FACE_MODEL_NAME = "Facenet"  # DeepFace model: Facenet, ArcFace, etc.

    # Paths
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.bin")
    EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./data/embeddings/")
    REPORTS_PATH = os.getenv("REPORTS_PATH", "./data/reports/")
    UPLOADS_PATH = os.getenv("UPLOADS_PATH", "./data/uploads/")

    # Vision Settings
    WEBCAM_WIDTH = 640
    WEBCAM_HEIGHT = 480
    WEBCAM_FPS = 30
    CAPTURE_MODE = "video"  # 'image' or 'video'

    # Speech Settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 30  # seconds
    WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large

    # Emotion Detection
    EMOTION_MODEL = "michellejieli/emotion_text_classifier"  # HuggingFace
    EMOTION_THRESHOLD = 0.5

    # Sentiment Analysis
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # HuggingFace

    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformers
    EMBEDDING_DIM = 384

    # LLM Settings
    LLM_MODEL = "gpt-4"  # or "claude-3-opus"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1000

    # Safety Rules
    ENABLE_SAFETY_CHECKS = True
    MEDICATION_CONFLICT_CHECK = True

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "./logs/app.log"

    # CORS
    CORS_ORIGINS = ["*"]  # Should be restricted in production

    # Timeouts
    WEBRTC_TIMEOUT = 60  # seconds
    ANALYSIS_TIMEOUT = 300  # seconds

    # Debug Mode
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    CORS_ORIGINS = [os.getenv("FRONTEND_URL", "http://localhost:3000")]


class TestingConfig(Config):
    """Testing configuration"""

    DATABASE_URL = "sqlite:///./test.db"
    TESTING = True


# Select config based on environment
ENV = os.getenv("ENV", "development")

if ENV == "production":
    config = ProductionConfig()
elif ENV == "testing":
    config = TestingConfig()
else:
    config = DevelopmentConfig()
