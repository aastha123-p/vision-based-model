# Vision Agentic AI MVP - Complete Implementation Guide

A fully functional multimodal AI-powered medical consultation system with biometric authentication, vision analysis, speech processing, and agentic AI reasoning.

## 🎯 Project Overview

This MVP implements a comprehensive healthcare AI system that combines:
- **Biometric Authentication** (Face + Token-based)
- **Multimodal Feature Extraction** (Vision + Speech + Form Data)
- **AI-Powered Analysis** (Embeddings + Similarity Search + LLM Reasoning)
- **Agentic Workflow** (7+ specialized agents for different analysis tasks)
- **Safety Checks** (Medication conflict detection & contraindication checks)
- **Report Generation** (PDF reports + Email delivery)

## 📁 Project Structure

```
vision_agentic_ai_mvp/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Centralized configuration
│   ├── api/                    # HTTP API routes
│   │   ├── routes_auth.py      # Authentication endpoints
│   │   ├── routes_patient.py   # Patient management
│   │   ├── routes_session.py   # Consultation sessions
│   │   ├── routes_report.py    # Report generation
│   │   ├── routes_vision.py    # Vision analysis
│   │   └── routes_speech.py    # Speech analysis
│   ├── auth/                   # Authentication layer
│   │   ├── face_auth.py        # Face login orchestration
│   │   ├── face_embedding_store.py  # Face embedding storage
│   │   └── token_auth.py       # Token-based authentication
│   ├── vision/                 # Vision processing
│   │   ├── webcam_capture.py   # Webcam frame capture
│   │   ├── face_recognition.py # DeepFace integration
│   │   ├── emotion_detector.py # Emotion detection
│   │   ├── eye_tracker.py      # Eye movement tracking
│   │   └── lip_analyzer.py     # Lip tension analysis
│   ├── speech/                 # Speech processing
│   │   ├── speech_to_text.py   # Whisper integration
│   │   ├── sentiment_analyzer.py # Sentiment analysis
│   │   └── processor.py        # Audio processing
│   ├── agents/                 # AI agents
│   │   ├── supervisor_agent.py # Master orchestrator
│   │   ├── symptom_agent.py    # Symptom analysis
│   │   ├── vision_agent.py     # Vision interpretation
│   │   ├── comparison_agent.py # Similar case retrieval
│   │   ├── condition_agent.py  # Condition prediction (LLM)
│   │   ├── medication_agent.py # Remedy suggestion (LLM)
│   │   ├── safety_agent.py     # Safety checks
│   │   └── learning_agent.py   # Embedding storage & retrieval
│   ├── core/                   # Core intelligence
│   │   ├── embedding_engine.py # Text embeddings (SentenceTransformers)
│   │   ├── faiss_store.py      # Vector similarity search
│   │   ├── llm_engine.py       # LLM API wrapper (OpenAI/Claude)
│   │   ├── safety_rules.py     # Medication safety rules
│   │   └── similarity_engine.py # Cosine similarity scoring
│   ├── database/               # Data persistence
│   │   ├── db.py               # Connection management
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── crud.py             # Database operations
│   ├── reports/                # Report generation
│   │   ├── pdf_generator.py    # PDF creation (ReportLab)
│   │   └── email_service.py    # Email delivery (SMTP)
│   └── utils/
│       ├── logger.py           # Logging setup
│       └── helpers.py          # Utility functions
├── frontend/
│   └── streamlit_app.py        # Streamlit UI
├── data/
│   ├── faiss_index/            # Vector search index
│   ├── embeddings/             # Stored embeddings
│   ├── reports/                # Generated PDF reports
│   └── uploads/                # Patient uploaded files
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🚀 Installation & Setup

### 1. Prerequisites

- Python 3.10+
- pip or conda
- Git
- Webcam (for face authentication)
- Microphone (for speech analysis)

### 2. Clone & Navigate

```bash
git clone <repository-url>
cd vision-agentic_ai_mvp
```

### 3. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n vision-ai python=3.10
conda activate vision-ai
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** On first install, some packages may need compilation. If deepface or FAISS fail:

```bash
# For DeepFace (if it fails)
pip install --no-cache-dir deepface

# For FAISS
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
```

### 5. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (for LLM)
- `SENDER_EMAIL` and `SENDER_PASSWORD` (for email reports)
- Other optional configs in `.env.example`

### 6. Create Data Directories

```bash
mkdir -p data/faiss_index
mkdir -p data/embeddings
mkdir -p data/reports
mkdir -p data/uploads
mkdir -p logs
```

## 🏃 Running the Application

### Start Backend Server

```bash
# Development mode
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Start Streamlit Frontend (Optional)

```bash
streamlit run frontend/streamlit_app.py
```

## 📊 Complete Workflow

### Phase 1: Authentication

**Option 1 - Face Login (Primary)**
1. Patient captures face via webcam
2. System generates embedding using DeepFace (Facenet)
3. Compares with stored face embedding
4. Similarity threshold check (default 0.60)
5. If match → Login success
6. If no match → Show token fallback option

**Option 2 - Token Login (Fallback)**
1. Patient provides token
2. System validates against database
3. Returns patient data

**Registration:**
```bash
POST /api/auth/register
{
  "name": "John Doe",
  "email": "john@example.com",
  "age": 35,
  "gender": "M"
}
→ Returns: patient_id, token
```

### Phase 2: Pre-Consultation Form

```bash
POST /api/patient/form
{
  "patient_id": 1,
  "chief_complaint": "Migraine",
  "symptoms": ["headache", "nausea", "light sensitivity"],
  "symptom_duration": "3 days",
  "severity": "moderate"
}
```

### Phase 3: Vision & Speech Analysis

**Start Session:**
```bash
POST /api/session/start
{
  "patient_id": 1,
  "chief_complaint": "Migraine",
  "symptoms": ["headache", "nausea"],
  "symptom_duration": "3 days",
  "severity": "moderate"
}
→ Returns: session_id
```

**Run Full Analysis:**
```bash
POST /api/session/analyze
{
  "session_id": 1,
  "vision_data": {
    "emotion": "sad",
    "emotion_confidence": 0.85,
    "blink_rate": 12.5,
    "eye_strain_score": 0.45
  },
  "speech_data": {
    "transcript": "I have severe headache...",
    "sentiment": "NEGATIVE",
    "sentiment_score": 0.92
  }
}
```

### Phase 4: AI Analysis Pipeline

The supervisor agent orchestrates:
1. **Symptom Agent** - Analyzes form data and symptoms
2. **Vision Agent** - Interprets emotion, eye strain, lip tension
3. **Comparison Agent** - Finds 5 similar past cases using embeddings
4. **Condition Agent** - Uses LLM to predict condition + confidence
5. **Medication Agent** - Suggests appropriate remedy
6. **Safety Agent** - Checks conflicts and contraindications
7. **Learning Agent** - Stores embedding for future reference

### Phase 5: Report Generation

**Generate PDF:**
```bash
GET /api/report/generate/{session_id}
→ Returns: PDF file path
```

**Send via Email:**
```bash
POST /api/report/email/{session_id}
{
  "recipient_email": "patient@example.com"
}
```

## 🧠 AI Models & Technologies

### Vision Processing
- **DeepFace** - Face detection & embedding (Facenet/ArcFace)
- **MediaPipe** - Face landmarks, eye tracking, lip tracking
- **OpenCV** - Webcam capture & frame processing

### Speech Processing
- **Whisper** - Speech-to-text (ASR)
- **HuggingFace Transformers** - Sentiment analysis

### Embeddings & Search
- **SentenceTransformers** - Text embeddings (all-MiniLM-L6-v2, 384-dim)
- **FAISS** - Fast similarity search

### LLM & Reasoning
- **OpenAI GPT-4** or **Claude 3 Opus** - Condition prediction & medication suggestion

### Safety & Validation
- **Rule-based engine** - Medication conflicts & contraindications
- **Hardcoded knowledge base** - Medical safety rules

### Reporting
- **ReportLab** - PDF generation
- **SMTP** - Email delivery

## 🔑 API Endpoints

### Authentication
```
POST   /api/auth/register           - Register new patient
POST   /api/auth/face-register      - Register face for login
POST   /api/auth/face-login         - Authenticate via face capture
POST   /api/auth/token-login        - Authenticate via token
POST   /api/auth/logout             - Logout (revoke token)
GET    /api/auth/status/{patient_id} - Get auth status
```

### Patient Management
```
POST   /api/patient/register        - Register patient
POST   /api/patient/form            - Submit pre-consultation form
GET    /api/patient/profile/{id}    - Get patient profile
POST   /api/patient/update          - Update patient info
```

### Sessions & Analysis
```
POST   /api/session/start           - Start new session
POST   /api/session/analyze         - Run analysis with vision/speech data
GET    /api/session/{session_id}    - Get session details
```

### Reports
```
GET    /api/report/generate/{session_id}  - Generate PDF report
POST   /api/report/email/{session_id}     - Email report to patient
GET    /api/report/list/{patient_id}      - List patient's reports
```

## 📝 Example Usage

### 1. Register Patient
```python
import requests
import json

url = "http://localhost:8000/api/auth/register"
payload = {
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "age": 30,
    "gender": "F"
}
response = requests.post(url, json=payload)
patient_data = response.json()
print(f"Patient ID: {patient_data['patient_id']}")
print(f"Token: {patient_data['token']}")
```

### 2. Start Analysis Session
```python
url = "http://localhost:8000/api/session/start"
payload = {
    "patient_id": 1,
    "chief_complaint": "Persistent cough",
    "symptoms": ["cough", "fatigue"],
    "symptom_duration": "5 days",
    "severity": "mild"
}
response = requests.post(url, json=payload)
session_data = response.json()
session_id = session_data['session_id']
```

### 3. Analyze with Vision & Speech
```python
url = f"http://localhost:8000/api/session/analyze"
payload = {
    "session_id": session_id,
    "vision_data": {
        "emotion": "anxious",
        "emotion_confidence": 0.75,
        "blink_rate": 18.0,
        "eye_strain_score": 0.35
    },
    "speech_data": {
        "transcript": "I have been coughing for 5 days, feeling tired",
        "sentiment": "NEGATIVE",
        "sentiment_score": 0.65
    }
}
response = requests.post(url, json=payload)
analysis = response.json()
print(json.dumps(analysis, indent=2))
```

### 4. Generate & Send Report
```python
# Generate PDF
url = f"http://localhost:8000/api/report/generate/{session_id}"
response = requests.get(url)
print(f"Report: {response.json()['report_path']}")

# Send via email
url = f"http://localhost:8000/api/report/email/{session_id}"
payload = {"recipient_email": "alice@example.com"}
response = requests.post(url, json=payload)
print(response.json()['message'])
```

## 🔐 Security Considerations

- **API Keys:** Store in environment variables, never commit to repo
- **Database:** Use strong passwords in production
- **CORS:** Configure properly for production (currently open)
- **Face Embeddings:** Stored securely in database, never exposed
- **Email:** Use app-specific passwords for Gmail (not main password)

## 🧪 Testing

### Test Authentication
```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com"}'
```

### Test Health
```bash
curl http://localhost:8000/health
```

### View Swagger UI
Open `http://localhost:8000/docs` in browser

## 📊 Database Models

### Patient Table
- id, name, email, phone, age, gender
- token (unique), face_embedding
- medical_history, current_medications, allergies
- created_at, updated_at, last_login

### Session Table
- id, patient_id
- chief_complaint, symptoms, symptom_duration, severity
- emotion, sentiment, vision_features, speech_features
- predicted_condition, condition_confidence, medication suggestions
- similar_cases, embedding, safety_check_passed

### FaceEmbedding Table
- id, patient_id, embedding, captured_at

### SafetyRule Table
- medication_name, conflict_medications, contraindications, warnings

## 🎯 Next Steps for Production

1. **Authentication Improvements**
   - Add liveness detection for faces
   - Implement 2FA
   - Add session management

2. **Data Privacy**
   - Encrypt face embeddings at rest
   - Add PII masking in logs
   - Implement audit trails

3. **Scalability**
   - Use PostgreSQL instead of SQLite
   - Add Redis for caching
   - Implement async tasks with Celery
   - Use connection pooling

4. **Monitoring & Logging**
   - Add Sentry for error tracking
   - Implement health checks
   - Add performance metrics

5. **Frontend**
   - Complete Streamlit/React UI
   - Add real-time status updates
   - Implement webcam preview

6. **APIs**
   - Add webhook callbacks
   - Implement pagination
   - Add filtering & search
   - Add rate limiting

## 📚 Documentation

- **API Documentation:** http://localhost:8000/docs (Swagger)
- **ReDoc:** http://localhost:8000/redoc
- **Code Comments:** Extensive docstrings in all modules

## 🐛 Troubleshooting

### DeepFace Not Working
```bash
pip install --no-cache-dir deepface
# May need to install PyTorch first
pip install torch torchvision
```

### FAISS Import Error
```bash
pip install faiss-cpu
# Or for GPU: pip install faiss-gpu
```

### Whisper Issues
```bash
pip install openai-whisper
# Download model: whisper base
python -c "import whisper; whisper.load_model('base')"
```

### Database Errors
```bash
# Clean and reinitialize
rm vision_ai.db
python -c "from app.database.db import init_db; init_db()"
```

## 📄 License & Credits

- **DeepFace:** https://github.com/serengp/deepface
- **FAISS:** https://github.com/facebookresearch/faiss
- **Whisper:** https://github.com/openai/whisper
- **MediaPipe:** https://github.com/google/mediapipe

## 📞 Support

For issues or questions:
1. Check the `/logs` directory
2. Enable debug mode in `.env`
3. Review API documentation at `/docs`
4. Check error messages in terminal output

---

**Status:** ✅ Fully Functional MVP
**Last Updated:** February 2026
