from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import SessionLocal
from app.auth.token_auth import TokenAuthenticator

router = APIRouter()

class TokenRequest(BaseModel):
    token: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/auth/token-login")
def token_login(request: TokenRequest, db: Session = Depends(get_db)):
    authenticator = TokenAuthenticator(db)
    user = authenticator.authenticate(request.token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "message": "Login successful",
        "user": user
    }