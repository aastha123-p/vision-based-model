from fastapi import FastAPI
from app.database.db import init_db
from app.api.routes_auth import router as auth_router

app = FastAPI(title="Vision Agentic AI MVP")

init_db()

app.include_router(auth_router)