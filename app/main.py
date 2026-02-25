from fastapi import FastAPI
from app.database.db import init_db
from app.api.routes_auth import router as auth_router
from app.api.routes_speech import router as speech_router

app = FastAPI(title="Vision Agentic AI MVP")

init_db()

app.include_router(auth_router)
app.include_router(speech_router)