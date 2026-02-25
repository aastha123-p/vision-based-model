"""
Speech-to-Text Module
Uses OpenAI Whisper for ASR
"""

import numpy as np
from typing import Optional, Tuple
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Install with: pip install openai-whisper")

import librosa


class SpeechToTextEngine:
    """
    Speech-to-text using OpenAI Whisper
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available")
            self.model = None
        else:
            try:
                self.model = whisper.load_model(model_size)
                logger.info(f"Loaded Whisper {model_size} model")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                self.model = None

    def transcribe_audio(self, audio_path: str) -> Tuple[Optional[str], dict]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (transcription, metadata)
        """
        if not self.model:
            return None, {}

        try:
            result = self.model.transcribe(audio_path, language="en")
            metadata = {
                "language": result.get("language"),
                "duration": result.get("duration"),
            }
            return result.get("text", ""), metadata
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None, {}

    def transcribe_audio_array(
        self, audio: np.ndarray, sr: int = 16000
    ) -> Tuple[Optional[str], dict]:
        """
        Transcribe audio from numpy array
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Tuple of (transcription, metadata)
        """
        if not self.model:
            return None, {}

        try:
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0

            result = self.model.transcribe(
                {"audio": audio}, language="en"
            )
            metadata = {
                "language": result.get("language"),
            }
            return result.get("text", ""), metadata
        except Exception as e:
            logger.error(f"Error transcribing audio array: {e}")
            return None, {}

    @staticmethod
    def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=sr)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None, None

    @staticmethod
    def save_audio(audio: np.ndarray, output_path: str, sr: int = 16000) -> bool:
        """
        Save audio to file
        
        Args:
            audio: Audio array
            output_path: Output file path
            sr: Sample rate
            
        Returns:
            True if successful
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, sr)
            logger.info(f"Saved audio to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False

    @staticmethod
    def extract_audio_features(
        audio: np.ndarray, sr: int = 16000
    ) -> dict:
        """
        Extract audio features (energy, spectral, temporal)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with features
        """
        try:
            features = {
                "duration": len(audio) / sr,
                "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
                "zcr": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
                "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))),
                "mfcc_mean": float(np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13))),
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
