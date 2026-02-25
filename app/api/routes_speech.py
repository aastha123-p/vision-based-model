"""API Routes for Speech and Sentiment Analysis"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import shutil
import logging
import uuid
import time

from app.speech.processor import process_audio_file, process_audio_batch
from app.database.db import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/speech", tags=["speech"])

# Temporary directory for uploaded files
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Convert to absolute path and ensure it exists
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory: {UPLOAD_DIR}")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def validate_audio_file(filename: str) -> bool:
    """Validate if file has allowed audio extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Transcribe audio file to text."""
    logger.info(f"[ROUTE] POST /transcribe called")
    
    file_path = None
    try:
        # Step 1: Validate
        logger.info(f"[1] filename={file.filename}")
        if not validate_audio_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # Step 2: Create path
        logger.info(f"[2] Creating unique path")
        file_ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_name))
        logger.info(f"[3] path={file_path}")
        
        # Step 3: Prepare directory
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        logger.info(f"[4] dir ready")
        
        # Step 4: Read file content
        contents = await file.read()
        logger.info(f"[5] read {len(contents)} bytes")
        await file.close()
        logger.info(f"[6] file closed")
        
        # Step 5: Write to disk
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"[7] written to {file_path}")
        
        # Step 6: Verify
        time.sleep(1)
        if not os.path.exists(file_path):
            raise Exception(f"File missing: {file_path}")
        sz = os.path.getsize(file_path)
        logger.info(f"[8] verified: {sz} bytes")
        
        # Step 7: Transcribe
        logger.info(f"[9] starting whisper...")
        from app.speech.processor import transcribe_audio as transcribe
        result = transcribe(file_path)
        logger.info(f"[10] result={result}")
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "filename": file.filename,
            "transcript": result["transcript"],
            "language": result["language"],
            "duration": result["duration"],
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-sentiment")
async def analyze_sentiment_endpoint(
    text: str,
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment of provided text.
    
    Query Parameters:
    - text: Text to analyze
    
    Response:
    - sentiment: "positive", "neutral", or "negative"
    - confidence: Confidence score (0-1)
    - raw_label: Raw model output
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        from app.speech.processor import analyze_sentiment
        result = analyze_sentiment(text)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "text": text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "raw_label": result["raw_label"],
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-audio")
async def process_audio_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Process audio file: transcribe and analyze sentiment.
    
    Request:
    - file: Audio file (mp3, wav, m4a, flac, ogg, webm)
    
    Response:
    - transcript: Transcribed text
    - sentiment: "positive", "neutral", or "negative"
    - confidence: Sentiment confidence (0-1)
    - language: Detected language
    - duration: Audio duration in seconds
    - success: Boolean
    """
    
    try:
        if not validate_audio_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Create path with absolute path
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))
        
        logger.info(f"[UPLOAD] Target: {file_path}")
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        logger.info(f"[UPLOAD] Dir ready")
        
        # Save uploaded file
        contents = await file.read()
        logger.info(f"[UPLOAD] Read {len(contents)} bytes from {file.filename}")
        
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        logger.info(f"[UPLOAD] Write complete")
        
        # Close the UploadFile
        await file.close()
        logger.info(f"[UPLOAD] UploadFile closed")
        
        time.sleep(0.5)  # Ensure file is flushed to disk
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"File not saved to: {file_path}")
        
        file_size = os.path.getsize(file_path)
        logger.info(f"[UPLOAD] Saved: {file_path} ({file_size} bytes)")
        
        # Process audio
        logger.info(f"[PROCESS] Starting audio processing...")
        result = process_audio_file(file_path)
        logger.info(f"[PROCESS] Result: {result}")
        
        # Keep file for 1 hour (for debugging)
        logger.info(f"[CLEANUP] File kept at: {file_path} (will be cleaned up later)")
        
        if not result["success"]:
            return {
                "filename": file.filename,
                "transcript": result.get("transcript"),
                "sentiment": result.get("sentiment"),
                "confidence": result.get("confidence"),
                "language": result.get("language"),
                "duration": result.get("duration"),
                "success": False,
                "errors": result.get("errors")
            }

        return {
            "filename": file.filename,
            "transcript": result["transcript"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "language": result["language"],
            "duration": result["duration"],
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for speech module"""
    return {
        "status": "healthy",
        "module": "speech",
        "features": ["transcription", "sentiment-analysis"],
        "upload_dir": UPLOAD_DIR,
        "upload_dir_exists": os.path.exists(UPLOAD_DIR)
    }


@router.get("/cleanup")
async def cleanup_old_files():
    """Clean up files older than 1 hour"""
    import glob
    current_time = time.time()
    hour_ago = current_time - 3600  # 1 hour
    deleted = []
    
    files = glob.glob(os.path.join(UPLOAD_DIR, "*"))
    for file_path in files:
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            if file_time < hour_ago:
                try:
                    os.remove(file_path)
                    deleted.append(file_path)
                    logger.info(f"[CLEANUP] Deleted: {file_path}")
                except Exception as e:
                    logger.error(f"[CLEANUP] Failed to delete {file_path}: {e}")
    
    return {"deleted_files": len(deleted), "files": deleted}


@router.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test upload endpoint for debugging"""
    try:
        logger.info(f"[TEST] Received file: {file.filename}")
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"test_{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        logger.info(f"[TEST] Upload dir: {UPLOAD_DIR}")
        logger.info(f"[TEST] Full path: {file_path}")
        
        # Ensure dir exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        logger.info(f"[TEST] Dir created/exists")
        
        # Read and save
        contents = await file.read()
        logger.info(f"[TEST] Read {len(contents)} bytes")
        
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"[TEST] File written")
        
        # Close UploadFile
        await file.close()
        logger.info(f"[TEST] UploadFile closed")
        
        time.sleep(0.3)
        
        # Verify
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        logger.info(f"[TEST] File exists: {exists}, size: {size}")
        
        # Try to transcribe
        logger.info(f"[TEST] Attempting transcription...")
        from app.speech.processor import transcribe_audio as transcribe
        result = transcribe(file_path)
        logger.info(f"[TEST] Transcribe result: success={result.get('success')}, error={result.get('error')}")
        
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
        
        return {"status": "ok", "file": file.filename, "transcribe_result": result}
    except Exception as e:
        logger.error(f"[TEST] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
