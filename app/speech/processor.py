"""Combined Speech and Sentiment Processing (single-file)

This file consolidates transcription and sentiment helper functions so the
`app/speech` package can be reduced to a single final file (`processor.py`).
It keeps the same public API: `process_audio_file` and `process_audio_batch`.
"""

from typing import Dict, Optional, List
import logging
import os

logger = logging.getLogger(__name__)

# --- Whisper transcription helpers (inlined) ---
_whisper_model = None

def get_whisper_model(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except Exception:
            raise RuntimeError("Whisper library is required for transcription")
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


def transcribe_audio(audio_path: str, model_name: str = "base") -> Dict:
    try:
        logger.info(f"[WHISPER-START] Path: {audio_path}")

        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            return {"transcript": None, "language": None, "duration": 0, "success": False, "error": error_msg}

        if not os.access(audio_path, os.R_OK):
            error_msg = f"No read permission: {audio_path}"
            logger.error(error_msg)
            return {"transcript": None, "language": None, "duration": 0, "success": False, "error": error_msg}

        model = get_whisper_model(model_name)
        result = model.transcribe(audio_path, language="en")

        return {
            "transcript": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "duration": result.get("duration", 0),
            "success": True,
            "error": None,
        }
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"transcript": None, "language": None, "duration": 0, "success": False, "error": error_msg}


# --- Sentiment helpers (inlined) ---
_sentiment_pipeline = None

def get_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError("transformers library is required for sentiment analysis")
        logger.info(f"Loading sentiment model: {model_name}")
        _sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=-1)
    return _sentiment_pipeline


def analyze_sentiment(text: str, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> Dict:
    try:
        if not text or not text.strip():
            return {"sentiment": None, "confidence": 0, "raw_label": None, "raw_score": None, "success": False, "error": "Empty text provided"}

        pipeline = get_sentiment_pipeline(model_name)
        result = pipeline(text)[0]
        label = result["label"].upper()
        sentiment_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        sentiment = sentiment_map.get(label, "neutral")
        return {"sentiment": sentiment, "confidence": round(result["score"], 4), "raw_label": label, "raw_score": result["score"], "success": True, "error": None}
    except Exception as e:
        error_msg = f"Sentiment analysis error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"sentiment": None, "confidence": 0, "raw_label": None, "raw_score": None, "success": False, "error": error_msg}


def process_audio_file(
    audio_path: str,
    whisper_model: str = "base",
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
) -> Dict:
    """
    Process audio file: transcribe and analyze sentiment.
    
    This is the main entry point for processing an audio file. It:
    1. Transcribes the audio using Whisper
    2. Analyzes the sentiment of the transcript
    
    Args:
        audio_path: Path to audio file
        whisper_model: Whisper model size
        sentiment_model: HuggingFace sentiment model
        
    Returns:
        Dictionary containing:
            - transcript: Transcribed text
            - sentiment: Sentiment classification (positive/neutral/negative)
            - confidence: Sentiment confidence score (0-1)
            - language: Detected language
            - duration: Audio duration in seconds
            - success: Boolean indicating overall success
            - errors: List of any errors encountered
            
    Example:
        >>> result = process_audio_file("patient_interview.mp3")
        >>> print(f"Transcript: {result['transcript']}")
        >>> print(f"Sentiment: {result['sentiment']} ({result['confidence']})")
    """
    errors: List[str] = []
    try:
        logger.info(f"Processing audio file: {audio_path}")

        # Step 1: Transcribe audio
        transcription_result = transcribe_audio(audio_path, whisper_model)
        if not transcription_result["success"]:
            errors.append(transcription_result.get("error"))
            return {"transcript": None, "sentiment": None, "confidence": None, "language": None, "duration": None, "success": False, "errors": errors}

        transcript = transcription_result["transcript"]

        # Step 2: Analyze sentiment
        sentiment_result = analyze_sentiment(transcript, sentiment_model)
        if not sentiment_result["success"]:
            errors.append(sentiment_result.get("error"))

        return {
            "transcript": transcript,
            "sentiment": sentiment_result.get("sentiment"),
            "confidence": sentiment_result.get("confidence"),
            "raw_label": sentiment_result.get("raw_label"),
            "language": transcription_result.get("language"),
            "duration": transcription_result.get("duration"),
            "success": transcription_result.get("success") and sentiment_result.get("success"),
            "errors": errors if errors else None
        }
    except Exception as e:
        error_msg = f"Audio processing error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
        return {"transcript": None, "sentiment": None, "confidence": None, "language": None, "duration": None, "success": False, "errors": errors}


def process_audio_batch(
    audio_paths: list,
    whisper_model: str = "base",
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
) -> Dict:
    """
    Process multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        whisper_model: Whisper model size
        sentiment_model: HuggingFace sentiment model
        
    Returns:
        Dictionary containing:
            - results: List of processing results
            - total: Total number of files processed
            - successful: Number of successful processes
            - failed: Number of failed processes
    """
    results = []
    successful = 0
    failed = 0
    for audio_path in audio_paths:
        result = process_audio_file(audio_path, whisper_model, sentiment_model)
        results.append({"file": audio_path, "result": result})
        if result.get("success"):
            successful += 1
        else:
            failed += 1
    return {"results": results, "total": len(audio_paths), "successful": successful, "failed": failed}
