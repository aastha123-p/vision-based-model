"""
Sentiment Analysis Module
Uses HuggingFace transformers for sentiment analysis
"""

from typing import Tuple, Optional, Dict
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. Install with: pip install transformers")


class SentimentAnalyzer:
    """
    Sentiment analysis using HuggingFace transformers
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            self.classifier = None
        else:
            try:
                self.classifier = pipeline(
                    "sentiment-classification",
                    model=model_name,
                    device=-1,  # CPU, use 0 for GPU
                )
                logger.info(f"Loaded sentiment model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {e}")
                self.classifier = None

    def analyze_sentiment(
        self, text: str, return_all_scores: bool = False
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            return_all_scores: Return all class scores
            
        Returns:
            Tuple of (sentiment_label, confidence)
        """
        if not self.classifier or not text:
            return None, None

        try:
            # Truncate text if too long
            text = text[:512]
            
            result = self.classifier(text)[0]
            sentiment = result["label"].upper()
            confidence = result["score"]
            
            return sentiment, confidence
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None, None

    def batch_analyze_sentiment(
        self, texts: list
    ) -> list:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of (sentiment, confidence) tuples
        """
        results = []
        for text in texts:
            sentiment, confidence = self.analyze_sentiment(text)
            results.append((sentiment, confidence))
        return results

    @staticmethod
    def sentiment_to_numeric(sentiment: str) -> float:
        """
        Convert sentiment label to numeric score (-1 to 1)
        
        Args:
            sentiment: Sentiment label
            
        Returns:
            Numeric score
        """
        sentiment_map = {
            "POSITIVE": 1.0,
            "NEUTRAL": 0.0,
            "NEGATIVE": -1.0,
        }
        return sentiment_map.get(sentiment, 0.0)

    def analyze_text_emotions(
        self, text: str
    ) -> Dict[str, float]:
        """
        Detect emotions from text using zero-shot classification
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion scores
        """
        if not TRANSFORMERS_AVAILABLE:
            return {}

        try:
            from transformers import pipeline
            
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
            
            emotions = ["happy", "sad", "angry", "fearful", "surprised", "disgusted"]
            result = classifier(text, emotions)
            
            emotion_scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                emotion_scores[label] = score
                
            return emotion_scores
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {}

    def get_sentiment_summary(self, text: str) -> Dict:
        """
        Get comprehensive sentiment analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        sentiment, confidence = self.analyze_sentiment(text)
        emotions = self.analyze_text_emotions(text)
        
        return {
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "sentiment_numeric": self.sentiment_to_numeric(sentiment),
            "emotions": emotions,
            "text_length": len(text),
            "text_preview": text[:100],
        }

    @staticmethod
    def extract_text_features(text: str) -> Dict:
        """
        Extract text-based features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text features
        """
        try:
            features = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "sentence_count": len(text.split(".")),
                "avg_word_length": sum(len(w) for w in text.split()) / max(len(text.split()), 1),
                "contains_question": "?" in text,
                "contains_exclamation": "!" in text,
                "capitalization_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}
