"""Speech and Sentiment Analysis Module"""

# Import all functions from processor.py which contains both transcription and sentiment analysis
from .processor import (
    transcribe_audio,
    analyze_sentiment,
    process_audio_file,
    process_audio_batch,
)

__all__ = [
    "transcribe_audio",
    "analyze_sentiment",
    "process_audio_file",
]
