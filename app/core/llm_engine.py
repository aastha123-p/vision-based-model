"""
LLM Engine Module
Wrapper for OpenAI and Claude APIs
"""

from typing import Optional, Dict, List
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMEngine:
    """
    LLM wrapper for medical reasoning
    """

    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM engine
        
        Args:
            provider: 'openai' or 'anthropic'
        """
        self.provider = provider

        if provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = config.OPENAI_API_KEY
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def analyze_condition(self, features: Dict) -> Dict:
        """
        Analyze patient features and predict condition
        
        Args:
            features: Patient features dictionary
                - form_data: symptoms, duration
                - vision_analysis: emotion, eye_strain
                - speech_analysis: sentiment, transcript
                - similar_cases: top 5 similar past cases
                
        Returns:
            Dictionary with condition prediction
        """
        try:
            prompt = self._create_analysis_prompt(features)

            if self.provider == "openai" and OPENAI_AVAILABLE:
                response = self._openai_completion(prompt)
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                response = self._anthropic_completion(prompt)
            else:
                logger.error("No LLM provider available")
                return self._fallback_analysis(features)

            return self._parse_condition_response(response)

        except Exception as e:
            logger.error(f"Error analyzing condition: {e}")
            return self._fallback_analysis(features)

    def suggest_medication(self, condition: str, patient_history: Dict) -> Dict:
        """
        Suggest medication based on condition and history
        
        Args:
            condition: Predicted condition
            patient_history: Patient medical history
            
        Returns:
            Dictionary with medication suggestion
        """
        try:
            prompt = self._create_medication_prompt(condition, patient_history)

            if self.provider == "openai" and OPENAI_AVAILABLE:
                response = self._openai_completion(prompt)
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                response = self._anthropic_completion(prompt)
            else:
                return {"medication": "Consult physician", "dosage": "N/A"}

            return self._parse_medication_response(response)

        except Exception as e:
            logger.error(f"Error suggesting medication: {e}")
            return {"medication": "Consult physician", "dosage": "N/A"}

    def _create_analysis_prompt(self, features: Dict) -> str:
        """Create condition analysis prompt"""
        form = features.get("form_data", {})
        vision = features.get("vision_analysis", {})
        speech = features.get("speech_analysis", {})
        similar = features.get("similar_cases", [])

        prompt = f"""
You are a medical AI assistant analyzing patient symptoms for a homeopathic consultation system.

PATIENT SYMPTOMS:
- Chief Complaint: {form.get('chief_complaint', 'Not specified')}
- Symptoms: {', '.join(form.get('symptoms', []))}
- Duration: {form.get('symptom_duration', 'Not specified')}
- Severity: {form.get('severity', 'Not specified')}

PATIENT OBSERVATIONS:
- Emotional State: {vision.get('emotion', 'Neutral')}
- Eye Condition: {vision.get('eye_strain_score', 'Normal')}
- Speech Sentiment: {speech.get('sentiment', 'Neutral')}

SIMILAR PAST CASES:
{self._format_similar_cases(similar)}

Based on the above information, provide:
1. Most likely condition (2-3 words)
2. Confidence level (0-100%)
3. Brief explanation

Format response as:
CONDITION: [condition name]
CONFIDENCE: [percentage]
EXPLANATION: [brief explanation]
"""
        return prompt

    def _create_medication_prompt(self, condition: str, patient_history: Dict) -> str:
        """Create medication suggestion prompt"""
        prompt = f"""
You are a homeopathic remedy advisor. Based on the diagnosed condition, suggest appropriate remedy.

CONDITION: {condition}
PATIENT HISTORY: {patient_history.get('medical_history', 'None')}
CURRENT MEDICATIONS: {', '.join(patient_history.get('current_medications', []))}

Suggest ONE homeopathic remedy with:
1. Remedy name
2. Recommended dosage
3. Administration frequency
4. Important warnings (if any)

Format response as:
REMEDY: [remedy name]
DOSAGE: [dosage]
FREQUENCY: [frequency]
WARNINGS: [warnings or 'None']
"""
        return prompt

    def _openai_completion(self, prompt: str) -> str:
        """Get completion from OpenAI"""
        if not OPENAI_AVAILABLE:
            return ""

        try:
            response = openai.ChatCompletion.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def _anthropic_completion(self, prompt: str) -> str:
        """Get completion from Claude"""
        if not ANTHROPIC_AVAILABLE:
            return ""

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ""

    @staticmethod
    def _parse_condition_response(response: str) -> Dict:
        """Parse LLM response for condition"""
        lines = response.split("\n")
        result = {
            "condition": "Undiagnosed",
            "confidence": 0.0,
            "explanation": "",
        }

        for line in lines:
            if "CONDITION:" in line:
                result["condition"] = line.split("CONDITION:")[-1].strip()
            elif "CONFIDENCE:" in line:
                try:
                    result["confidence"] = (
                        float(line.split("CONFIDENCE:")[-1].strip().rstrip("%")) / 100
                    )
                except ValueError:
                    result["confidence"] = 0.5
            elif "EXPLANATION:" in line:
                result["explanation"] = line.split("EXPLANATION:")[-1].strip()

        return result

    @staticmethod
    def _parse_medication_response(response: str) -> Dict:
        """Parse LLM response for medication"""
        lines = response.split("\n")
        result = {
            "medication": "Consult physician",
            "dosage": "As prescribed",
            "frequency": "As needed",
            "warnings": "",
        }

        for line in lines:
            if "REMEDY:" in line:
                result["medication"] = line.split("REMEDY:")[-1].strip()
            elif "DOSAGE:" in line:
                result["dosage"] = line.split("DOSAGE:")[-1].strip()
            elif "FREQUENCY:" in line:
                result["frequency"] = line.split("FREQUENCY:")[-1].strip()
            elif "WARNINGS:" in line:
                result["warnings"] = line.split("WARNINGS:")[-1].strip()

        return result

    @staticmethod
    def _format_similar_cases(similar_cases: List) -> str:
        """Format similar cases for prompt"""
        if not similar_cases:
            return "No similar past cases found."

        formatted = ""
        for i, case in enumerate(similar_cases[:3], 1):
            formatted += f"{i}. {case.get('condition', 'Unknown')} - Similarity: {case.get('similarity', 0)*100:.1f}%\n"

        return formatted

    @staticmethod
    def _fallback_analysis(features: Dict) -> Dict:
        """Fallback when LLM unavailable"""
        symptoms = features.get("form_data", {}).get("symptoms", [])
        condition = "General wellness check"

        if symptoms:
            if any("pain" in s.lower() for s in symptoms):
                condition = "Pain condition"
            elif any("fever" in s.lower() for s in symptoms):
                condition = "Fever condition"
            elif any("allergy" in s.lower() for s in symptoms):
                condition = "Allergy condition"

        return {
            "condition": condition,
            "confidence": 0.5,
            "explanation": "Analysis requires API configuration",
        }
