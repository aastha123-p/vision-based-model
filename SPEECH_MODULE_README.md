# Speech & Sentiment Analysis Module - Phase 2 Task 2

## Overview

The Speech & Sentiment Analysis module provides comprehensive audio processing capabilities for your Vision-based AI application. It combines two powerful technologies:

1. **OpenAI Whisper** - State-of-the-art speech recognition
2. **HuggingFace Transformers** - Sentiment classification

## Features

### ✅ Audio Transcription
- **Whisper Integration**: Transcribes audio to text with high accuracy
- Multiple model sizes: tiny, base, small, medium, large
- Automatic language detection
- Batch processing support
- Segment-level timestamps

### ✅ Sentiment Analysis
- **HuggingFace Models**: Pre-trained sentiment classifiers
- Three classifications: Positive, Neutral, Negative
- Confidence scores (0-1)
- Batch processing for multiple texts
- Sentiment distribution analysis

### ✅ Combined Processing
- Single endpoint to transcribe and analyze sentiment
- Error handling and validation
- File upload support

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The following new packages were added:
- `openai-whisper` - Speech recognition
- `transformers` - Sentiment analysis models
- `torch` - Deep learning framework
- `torchaudio` - Audio processing
- `librosa` - Music and audio analysis
- `soundfile` - Audio file I/O
- `numpy` - Numerical computing

### 2. Project Structure
```
app/speech/
├── __init__.py           # Module exports
├── transcription.py      # Whisper audio transcription
├── sentiment.py          # HuggingFace sentiment analysis
├── processor.py          # Combined audio processing
├── models.py             # Pydantic response models
├── config.py             # Configuration constants
├── examples.py           # Usage examples
```

## API Endpoints

### 1. Health Check
```http
GET /api/speech/health
```
**Response:**
```json
{
  "status": "healthy",
  "module": "speech",
  "features": ["transcription", "sentiment-analysis"]
}
```

### 2. Transcribe Audio
```http
POST /api/speech/transcribe
Content-Type: multipart/form-data

file: <audio_file>
```

**Supported Formats:** mp3, wav, m4a, flac, ogg, webm

**Response:**
```json
{
  "filename": "audio.mp3",
  "transcript": "Hello, how are you today?",
  "language": "en",
  "duration": 5.2,
  "success": true
}
```

### 3. Analyze Sentiment
```http
POST /api/speech/analyze-sentiment?text=<text>
```

**Query Parameters:**
- `text`: Text to analyze (required)

**Response:**
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.9987,
  "raw_label": "POSITIVE",
  "success": true
}
```

### 4. Process Audio (Complete)
```http
POST /api/speech/process-audio
Content-Type: multipart/form-data

file: <audio_file>
```

**Response:**
```json
{
  "filename": "interview.mp3",
  "transcript": "The patient reported feeling much better.",
  "sentiment": "positive",
  "confidence": 0.8542,
  "language": "en",
  "duration": 12.5,
  "success": true
}
```

## Python Usage Examples

### Basic Transcription
```python
from app.speech.transcription import transcribe_audio

result = transcribe_audio("audio.mp3")
if result["success"]:
    print(result["transcript"])
else:
    print(f"Error: {result['error']}")
```

### Sentiment Analysis
```python
from app.speech.sentiment import analyze_sentiment

result = analyze_sentiment("I love this product!")
if result["success"]:
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
```

### Combined Processing
```python
from app.speech.processor import process_audio_file

result = process_audio_file("patient_interview.mp3")
if result["success"]:
    print(f"Transcript: {result['transcript']}")
    print(f"Sentiment: {result['sentiment']}")
```

### Batch Processing
```python
from app.speech.processor import process_audio_batch

files = ["audio1.mp3", "audio2.wav", "audio3.m4a"]
results = process_audio_batch(files)
print(f"Processed: {results['total']}")
print(f"Successful: {results['successful']}")
```

### Sentiment Distribution
```python
from app.speech.sentiment import analyze_sentiment_batch, get_sentiment_distribution

texts = ["I love it!", "It's okay", "Hate it", "Amazing!"]
results = analyze_sentiment_batch(texts)
distribution = get_sentiment_distribution(results["results"])
print(distribution)
```

## Model Selection

### Whisper Models
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39M | Very Fast | Low | Real-time, Low resources |
| base | 74M | Fast | Medium | **Default, Balanced** |
| small | 244M | Medium | Good | Good accuracy, reasonable speed |
| medium | 769M | Slow | High | High accuracy |
| large | 1550M | Very Slow | Highest | Maximum accuracy |

**Default:** `base` - Good balance of speed and accuracy

### Sentiment Models
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| distilbert | Small | Fast | Good | **Default, Lightweight** |
| bert | Medium | Medium | High | Better accuracy |
| roberta | Medium | Medium | Very High | Improved BERT |
| xlnet | Large | Slow | Highest | State-of-the-art |

**Default:** `distilbert-base-uncased-finetuned-sst-2-english` - Lightweight and accurate

## Configuration

Edit `app/speech/config.py` to customize:

```python
# Whisper model size (default: "base")
DEFAULT_WHISPER_MODEL = "base"

# Sentiment model (default: DistilBERT)
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Supported audio formats
SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"]

# Maximum file size (default: 25MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Confidence thresholds
SENTIMENT_THRESHOLDS = {
    "high_confidence": 0.80,
    "medium_confidence": 0.60,
}
```

## Performance Considerations

### 1. Model Caching
Models are cached in memory after first load for faster subsequent calls:
```python
# First call: ~60s (model download + initialization)
result = transcribe_audio("audio1.mp3")

# Subsequent calls: ~5-10s (model already loaded)
result = transcribe_audio("audio2.mp3")
```

### 2. GPU Support
Models use CPU by default. To enable GPU:
```python
# Edit sentiment.py
pipeline = pipeline("sentiment-analysis", device=0)  # 0 = GPU device
```

### 3. Batch Processing
Process multiple files efficiently:
```python
results = process_audio_batch(audio_files)  # Optimized for batch
```

## Error Handling

All functions return structured responses with error information:

```python
result = process_audio_file("audio.mp3")

if not result["success"]:
    print("Errors encountered:")
    for error in result["errors"]:
        print(f"  - {error}")
```

Common errors:
- File not found
- Invalid audio format
- Unsupported language
- Model download failure
- Processing timeout

## Logging

The module includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs will show model loading, processing, and errors
result = process_audio_file("audio.mp3")
```

## Integration with Database

Store transcription and sentiment results:

```python
from app.database.models import Patient
from app.database.db import SessionLocal

# Process audio
result = process_audio_file("patient_feedback.mp3")

# Store in database
db = SessionLocal()
patient = db.query(Patient).filter(Patient.id == 1).first()
# Add fields to Patient model as needed

# Example: Extend Patient model
# sentiment_analysis = Column(JSON)  
# recent_transcript = Column(String)
```

## Testing

Test the module with sample audio:

```bash
# Using curl
curl -X POST "http://localhost:8000/api/speech/process-audio" \
     -F "file=@sample_audio.mp3"

# Using Python requests
import requests
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/speech/process-audio",
        files={"file": f}
    )
    print(response.json())
```

## Troubleshooting

### 1. Model Download Issues
```
Error: Failed to download model
Solution: Check internet connection, disk space, HuggingFace API availability
```

### 2. Audio Format Not Supported
```
Error: librosa.util.exceptions.PytorchUnavailable
Solution: Update audio file format or install ffmpeg: pip install pydub
```

### 3. Out of Memory
```
Error: CUDA out of memory / out of memory
Solution: Use smaller model (tiny/base) or process files separately
```

### 4. Slow Performance
```
Solution 1: Use GPU (set device=0 in sentiment.py)
Solution 2: Use smaller model size
Solution 3: Use batch processing for multiple files
```

## Next Steps

1. **Database Integration**: Store transcripts and sentiments
   - Add fields to Patient model
   - Create API endpoints to retrieve history

2. **Real-time Processing**: Stream audio instead of uploading
   - Implement WebSocket support
   - Real-time transcription

3. **Advanced Analytics**: 
   - Emotion detection (anger, joy, etc.)
   - Speaker diarization
   - Language identification

4. **Performance Optimization**:
   - Implement speech-to-text streaming
   - Use faster-whisper for inference
   - GPU acceleration

5. **Integration with Vision Module**:
   - Combine face recognition with speech analysis
   - Multi-modal patient assessment

## References

- **Whisper**: https://github.com/openai/whisper
- **HuggingFace**: https://huggingface.co/models?pipeline_tag=sentiment-analysis
- **Transformers**: https://huggingface.co/docs/transformers/
- **Librosa**: https://librosa.org/

## Support

For issues or questions:
1. Check the `examples.py` file for usage patterns
2. Review error messages and logging output
3. Consult model documentation for specific limitations
