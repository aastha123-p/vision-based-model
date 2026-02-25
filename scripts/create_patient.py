from app.database.db import init_db, SessionLocal
from app.database.crud import PatientCRUD
import uuid


def main(name: str = "test_patient"):
    init_db()
    db = SessionLocal()
    try:
        token = str(uuid.uuid4())
        patient = PatientCRUD.create_patient(db, name=name, token=token)
        print("Created patient:", patient.name)
        print("Token:", patient.token)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="test_patient", help="Patient name")
    args = p.parse_args()
    main(args.name)
