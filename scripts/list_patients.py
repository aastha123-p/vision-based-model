from app.database.db import init_db, SessionLocal
from app.database.models import Patient


def list_patients():
    init_db()
    db = SessionLocal()
    try:
        patients = db.query(Patient).all()
        if not patients:
            print("No patients found.")
            return
        for p in patients:
            print(f"ID: {p.id}  Name: {p.name}  Token: {p.token}  Created: {p.created_at}")
    finally:
        db.close()


if __name__ == "__main__":
    list_patients()
