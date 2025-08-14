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
            hf_base = (config.get_api_endpoints().get('huggingface_inference') if hasattr(config, 'get_api_endpoints') else 'https://api-inference.huggingface.co/models')
            # Use MNLI zero-shot for political leaning; toxicity optional
            self.zero_shot_url = f"{hf_base}/{models.get('political_bias', 'facebook/bart-large-mnli')}"
            self.toxicity_url = f"{hf_base}/{models.get('toxicity', 'unitary/toxic-bert')}"
        else:
            # Fallback to environment/secrets file
            self.api_key = self._load_api_key_from_secrets()
            self.zero_shot_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            self.toxicity_url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
        
        if not self.api_key:
            logger.warning("No Hugging Face API key found. Bias detection will return neutral placeholder (fallback removed).")
    
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
        """Bias analysis using zero-shot political leaning plus toxicity and emotion signals."""
        self.last_run = datetime.now().isoformat()
        
        if not text or not text.strip():
            return {'overall_bias_score': 0.0, 'bias_breakdown': {}, 'method': 'empty_text'}
        if not self.api_key:
            return {'overall_bias_score': 0.0, 'bias_breakdown': {}, 'method': 'no_api_key'}
        cleaned_text = self._preprocess_text(text)
        try:
            political = self._zero_shot_political(cleaned_text)
            toxicity = self._toxicity(cleaned_text)
            emotional = self._detect_emotional_manipulation(cleaned_text)  # still used auxiliary
            source_factor = self._get_source_credibility_factor(source)
            overall = self._compute_bias_score(political, toxicity, emotional, source_factor)
            return {
                'overall_bias_score': overall,
                'bias_breakdown': {
                    'political': political,
                    'toxicity': toxicity,
                    'emotional': emotional,
                    'source_credibility': source_factor
                },
                'method': 'zero_shot_political_bias',
                'analyzed_at': self.last_run
            }
        except Exception as e:
            logger.error(f"Bias detection failed irrecoverably: {e}")
            return {'overall_bias_score': 0.0, 'bias_breakdown': {}, 'method': 'error', 'error': str(e)}

    def _preprocess_text(self, text: str) -> str:
        import re
        text = re.sub(r'http[s]?://\S+', '', text)
        return ' '.join(text.split())

    def _zero_shot_political(self, text: str) -> Dict[str, Any]:
        """Classify political leaning via zero-shot labels."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        labels = ["left-leaning", "right-leaning", "centrist"]
        payload = {"inputs": text[:800], "parameters": {"candidate_labels": labels}}
        resp = requests.post(self.zero_shot_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'labels' in data and 'scores' in data:
            scores = dict(zip([l.lower() for l in data['labels']], data['scores']))
            # Normalize and expose lean + strength
            lean = max(scores, key=scores.get)
            strength = float(scores[lean])
            return {"lean": lean, "strength": strength, "scores": scores}
        # Some HF servers return list
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'labels' in data[0]:
            scores = dict(zip([l.lower() for l in data[0]['labels']], data[0]['scores']))
            lean = max(scores, key=scores.get)
            strength = float(scores[lean])
            return {"lean": lean, "strength": strength, "scores": scores}
        return {"lean": "neutral", "strength": 0.0, "scores": {}}

    def _toxicity(self, text: str) -> Dict[str, float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.toxicity_url, headers=headers, json={"inputs": text[:800]}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        scores = {}
        if isinstance(data, list) and data and isinstance(data[0], list):
            for item in data[0]:
                label = item.get('label', '').lower()
                score = item.get('score', 0.0)
                scores[label] = score
        return scores or {"toxic": 0.0}

    def _detect_emotional_manipulation(self, text: str) -> Dict[str, Any]:
        emotion_keywords = {
            'fear': ['terrifying', 'shocking', 'alarming', 'devastating', 'crisis', 'emergency'],
            'anger': ['outrageous', 'disgusting', 'infuriating', 'scandal', 'betrayal'],
            'excitement': ['amazing', 'incredible', 'revolutionary', 'breakthrough', 'miracle'],
            'urgency': ['urgent', 'immediate', 'now', 'quickly', 'limited time']
        }
        text_lower = text.lower()
        emotion_scores = {k: sum(1 for w in v if w in text_lower) for k, v in emotion_keywords.items()}
        total = sum(emotion_scores.values())
        manipulation = min(total * 0.1, 1.0)
        return {'manipulation_score': manipulation, 'emotion_breakdown': emotion_scores}

    def _get_source_credibility_factor(self, source: str) -> float:
        if not source:
            return 0.5
        source = source.lower()
        high = ['reuters', 'ap', 'bbc', 'npr', 'pbs', 'bloomberg']
        if any(s in source for s in high):
            return 0.9
        medium = ['cnn', 'fox', 'nbc', 'abc', 'cbs', 'washington post', 'wall street journal']
        if any(s in source for s in medium):
            return 0.7
        low = ['blog', 'opinion', 'editorial', 'social media']
        if any(s in source for s in low):
            return 0.3
        return 0.5

    def _compute_bias_score(self, political: Dict[str, Any], toxicity: Dict[str, float], emotional: Dict[str, Any], source_factor: float) -> float:
        pol_strength = float(political.get('strength', 0.0))
        # Treat stronger left/right than center as more biased; center lowers score
        lean = political.get('lean', 'neutral')
        if 'left' in lean or 'right' in lean:
            political_component = pol_strength
        elif 'centrist' in lean:
            political_component = max(0.0, 0.5 - pol_strength)  # strong centrist lowers bias
        else:
            political_component = 0.0
        toxicity_component = max(toxicity.get('toxic', 0.0), toxicity.get('toxicity', 0.0))
        emotional_component = float(emotional.get('manipulation_score', 0.0))
        source_component = (1.0 - source_factor)  # less credible source => more bias
        score = (
            political_component * 0.45 +
            toxicity_component * 0.2 +
            emotional_component * 0.2 +
            source_component * 0.15
        )
        return min(1.0, max(0.0, score))

    # Removed keyword fallback bias analysis.