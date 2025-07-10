import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import requests
import os

logger = logging.getLogger(__name__)

class BiasDetectorAgent:
    def __init__(self, config=None):
        self.config = config
        self.last_run = None
        self.last_error = None
        
        # Get API configuration dynamically
        if config:
            self.api_key = config.get_huggingface_key()
            models = config.get_huggingface_models()
            self.api_url = f"https://api-inference.huggingface.co/models/{models['bias']}"
        else:
            # Fallback to environment/secrets file
            self.api_key = self._load_api_key_from_secrets()
            self.api_url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
        
        if not self.api_key:
            logger.warning("No Hugging Face API key found. Bias detection will use fallback methods.")
    
    def _load_api_key_from_secrets(self) -> str:
        """Load API key from secrets file"""
        try:
            with open('secrets.env', 'r') as f:
                for line in f:
                    if line.strip().startswith('HF_API_KEY='):
                        return line.strip().split('=', 1)[1].strip()
        except Exception as e:
            logger.warning(f"Could not load Hugging Face API key from secrets: {e}")
        return None
    async def analyze_bias(self, text: str, source: str = None) -> Dict[str, Any]:
        """Bias analysis using Hugging Face Inference API with multiple bias detection approaches."""
        self.last_run = datetime.now().isoformat()
        
        if not text or not text.strip():
            return {
                'overall_bias_score': 0.0,
                'bias_breakdown': {},
                'method': 'empty_text'
            }
        
        cleaned_text = self._preprocess_text(text)
        
        if self.api_key:
            try:
                # Use multiple bias detection approaches
                results = await self._comprehensive_bias_analysis(cleaned_text, source)
                return results
            except Exception as e:
                logger.warning(f"Hugging Face API call failed: {e}")
                return self._fallback_bias_analysis(cleaned_text, source)
        else:
            return self._fallback_bias_analysis(cleaned_text, source)
    
    async def _comprehensive_bias_analysis(self, text: str, source: str = None) -> Dict[str, Any]:
        """Comprehensive bias analysis using multiple Hugging Face models"""
        results = {}
        
        # 1. Toxicity detection
        toxicity_result = self._huggingface_api_bias(text)
        results['toxicity'] = self._parse_toxicity_result(toxicity_result)
        
        # 2. Political bias indicators (using sentiment as proxy)
        political_bias = self._detect_political_bias_keywords(text)
        results['political_bias'] = political_bias
        
        # 3. Emotional manipulation detection
        emotional_bias = self._detect_emotional_manipulation(text)
        results['emotional_bias'] = emotional_bias
        
        # 4. Source credibility factor
        source_factor = self._get_source_credibility_factor(source)
        results['source_credibility'] = source_factor
        
        # Calculate overall bias score
        overall_score = self._calculate_overall_bias_score(results)
        
        return {
            'overall_bias_score': overall_score,
            'bias_breakdown': results,
            'method': 'comprehensive_huggingface_api',
            'analyzed_at': self.last_run
        }
    
    def _fallback_bias_analysis(self, text: str, source: str = None) -> Dict[str, Any]:
        """Fallback bias analysis using keyword-based detection"""
        bias_indicators = self._detect_bias_keywords(text)
        source_factor = self._get_source_credibility_factor(source)
        
        # Simple scoring based on keyword presence
        keyword_score = min(len(bias_indicators) * 0.2, 1.0)
        overall_score = (keyword_score + (1.0 - source_factor)) / 2
        
        return {
            'overall_bias_score': overall_score,
            'bias_breakdown': {
                'bias_keywords': bias_indicators,
                'source_credibility': source_factor
            },
            'method': 'keyword_fallback',
            'analyzed_at': self.last_run
        }

    def _huggingface_api_bias(self, text: str):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.api_url, headers=headers, json={"inputs": text})
        response.raise_for_status()
        return response.json()

    def _preprocess_text(self, text: str) -> str:
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$\$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = ' '.join(text.split())
        return text
    
    def _parse_toxicity_result(self, result) -> Dict[str, float]:
        """Parse toxicity detection result from Hugging Face API"""
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            scores = {}
            for label_data in result[0]:
                label = label_data.get('label', '').lower()
                score = label_data.get('score', 0.0)
                scores[label] = score
            return scores
        return {'toxic': 0.0}
    
    def _detect_political_bias_keywords(self, text: str) -> Dict[str, Any]:
        """Detect political bias indicators in text"""
        left_keywords = ['progressive', 'liberal', 'social justice', 'equality', 'diversity', 'climate change']
        right_keywords = ['conservative', 'traditional', 'law and order', 'national security', 'free market']
        center_keywords = ['moderate', 'bipartisan', 'compromise', 'balanced']
        
        text_lower = text.lower()
        
        left_count = sum(1 for word in left_keywords if word in text_lower)
        right_count = sum(1 for word in right_keywords if word in text_lower)
        center_count = sum(1 for word in center_keywords if word in text_lower)
        
        total_political = left_count + right_count + center_count
        
        if total_political == 0:
            return {'lean': 'neutral', 'strength': 0.0}
        
        if left_count > right_count and left_count > center_count:
            return {'lean': 'left', 'strength': left_count / max(total_political, 1)}
        elif right_count > left_count and right_count > center_count:
            return {'lean': 'right', 'strength': right_count / max(total_political, 1)}
        else:
            return {'lean': 'center', 'strength': center_count / max(total_political, 1)}
    
    def _detect_emotional_manipulation(self, text: str) -> Dict[str, Any]:
        """Detect emotional manipulation indicators"""
        emotion_keywords = {
            'fear': ['terrifying', 'shocking', 'alarming', 'devastating', 'crisis', 'emergency'],
            'anger': ['outrageous', 'disgusting', 'infuriating', 'scandal', 'betrayal'],
            'excitement': ['amazing', 'incredible', 'revolutionary', 'breakthrough', 'miracle'],
            'urgency': ['urgent', 'immediate', 'now', 'quickly', 'limited time']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in keywords if word in text_lower)
            emotion_scores[emotion] = count
        
        total_emotional = sum(emotion_scores.values())
        manipulation_score = min(total_emotional * 0.1, 1.0)
        
        return {
            'manipulation_score': manipulation_score,
            'emotion_breakdown': emotion_scores
        }
    
    def _get_source_credibility_factor(self, source: str) -> float:
        """Get source credibility factor (higher = more credible)"""
        if not source:
            return 0.5  # Unknown source
        
        source = source.lower()
        
        # High credibility sources
        high_credibility = ['reuters', 'ap', 'bbc', 'npr', 'pbs', 'bloomberg']
        if any(cred in source for cred in high_credibility):
            return 0.9
        
        # Medium credibility sources
        medium_credibility = ['cnn', 'fox', 'nbc', 'abc', 'cbs', 'washington post', 'wall street journal']
        if any(cred in source for cred in medium_credibility):
            return 0.7
        
        # Lower credibility indicators
        low_credibility = ['blog', 'opinion', 'editorial', 'social media']
        if any(low in source for low in low_credibility):
            return 0.3
        
        return 0.5  # Default
    
    def _detect_bias_keywords(self, text: str) -> List[str]:
        """Detect bias keywords for fallback analysis"""
        bias_keywords = [
            'allegedly', 'claims', 'supposedly', 'reportedly', 'sources say',
            'insiders reveal', 'shocking truth', 'exposed', 'cover-up',
            'conspiracy', 'mainstream media', 'fake news', 'propaganda'
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in bias_keywords if keyword in text_lower]
        return found_keywords
    
    def _calculate_overall_bias_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall bias score from multiple factors"""
        toxicity_score = results.get('toxicity', {}).get('toxic', 0.0)
        political_strength = results.get('political_bias', {}).get('strength', 0.0)
        emotional_score = results.get('emotional_bias', {}).get('manipulation_score', 0.0)
        source_credibility = results.get('source_credibility', 0.5)
        
        # Weighted combination
        bias_score = (
            toxicity_score * 0.3 +
            political_strength * 0.2 +
            emotional_score * 0.3 +
            (1.0 - source_credibility) * 0.2
        )
        
        return min(bias_score, 1.0)