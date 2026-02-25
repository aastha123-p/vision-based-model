"""
Vision Agentic AI MVP - Quick Start Guide

Get the system running in 5 minutes!
"""

import os
import sys

print("""
╔══════════════════════════════════════════════════════════╗
║  Vision Agentic AI MVP - Setup Wizard                   ║
╚══════════════════════════════════════════════════════════╝

This script will help you set up the application in minutes.
""")

# Step 1: Check Python version
print("\n[1/5] Checking Python version...")
if sys.version_info < (3, 10):
    print("❌ Python 3.10+ required")
    sys.exit(1)
print(f"✓ Python {sys.version.split()[0]} detected")

# Step 2: Check/Create directories
print("\n[2/5] Setting up directories...")
dirs = [
    "data/faiss_index",
    "data/embeddings",
    "data/reports",
    "data/uploads",
    "logs"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")

# Step 3: Check environment
print("\n[3/5] Checking environment...")
if not os.path.exists(".env"):
    print("⚠ .env file not found")
    print("  → Copy .env.example to .env and fill in your API keys")
else:
    print("✓ .env file found")

# Step 4: Test imports
print("\n[4/5] Testing imports...")
try:
    from app.database.db import init_db
    print("✓ Database module")
    from app.config import config
    print("✓ Config module")
    from app.utils.logger import setup_logger
    print("✓ Logger module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("  → Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 5: Initialize database
print("\n[5/5] Initializing database...")
try:
    init_db()
    print("✓ Database tables created")
except Exception as e:
    print(f"⚠ Database init: {e}")

print("""
╔══════════════════════════════════════════════════════════╗
║  ✓ Setup Complete!                                      ║
╚══════════════════════════════════════════════════════════╝

Next steps:

1. Install dependencies (if not done):
   $ pip install -r requirements.txt

2. Configure environment:
   $ cp .env.example .env
   $ nano .env  # Add your API keys

3. Start the server:
   $ python -m uvicorn app.main:app --reload

4. Access the API:
   → http://localhost:8000/docs (Swagger UI)
   → http://localhost:8000/redoc (API docs)

5. Try a test request:
   $ curl http://localhost:8000/health

6. (Optional) Start Streamlit frontend:
   $ streamlit run frontend/streamlit_app.py

Useful endpoints to test:
- POST /api/auth/register              - Register new patient
- POST /api/auth/face-login            - Face authentication
- POST /api/session/start              - Start analysis session
- POST /api/session/analyze            - Run full analysis
- GET  /api/report/generate/{id}       - Generate report

See IMPLEMENTATION_GUIDE.md for detailed documentation.

Happy coding! 🚀
""")
