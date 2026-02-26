from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database.db import init_db
from app.api.routes_auth import router as auth_router
from app.api.routes_patient import router as patient_router
from app.api.routes_session import router as session_router
from app.api.routes_report import router as report_router
from app.api.routes_vision import router as vision_router
from app.api.routes_speech import router as speech_router
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="Vision Agentic AI MVP", description="Multimodal AI medical consultation system")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# Include routers
app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(session_router)
app.include_router(report_router)
app.include_router(vision_router)
app.include_router(speech_router)


@app.get("/")
async def root():
    return {
        "message": "Vision Agentic AI MVP",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/auth",
            "patient": "/api/patient",
            "session": "/api/session",
            "report": "/api/report",
            "vision": "/api/vision",
            "speech": "/api/speech",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
