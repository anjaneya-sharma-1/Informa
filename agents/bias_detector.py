import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import requests
import os

logger = logging.getLogger(__name__)

# Load Hugging Face API key from secrets file
HF_API_KEY = None
try:
    with open('secrets.env', 'r') as f:
        for line in f:
            if line.strip().startswith('HF_API_KEY='):
                HF_API_KEY = line.strip().split('=', 1)[1].strip()
except Exception as e:
    logger.warning(f"Could not load Hugging Face API key: {e}")

class BiasDetectorAgent:
    def __init__(self):
        self.last_run = None
        self.last_error = None
        self.api_url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
        self.api_key = HF_API_KEY
        if not self.api_key:
            logger.warning("No Hugging Face API key found. Please add HF_API_KEY to secrets.env.")

    async def analyze_bias(self, text: str, source: str = None) -> Dict[str, Any]:
        """Bias analysis using Hugging Face Inference API only."""
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
                api_result = self._huggingface_api_bias(cleaned_text)
                if isinstance(api_result, list) and len(api_result) > 0 and isinstance(api_result[0], list):
                    toxic_score = 0.0
                    for label in api_result[0]:
                        if label['label'].lower() == 'toxic':
                            toxic_score = label['score']
                    return {
                        'overall_bias_score': toxic_score,
                        'bias_breakdown': {'toxic': toxic_score},
                        'method': 'huggingface_api',
                        'raw': api_result
                    }
                else:
                    logger.warning(f"Unexpected API result: {api_result}")
                    return {
                        'overall_bias_score': 0.0,
                        'error': 'Unexpected API result',
                        'method': 'huggingface_api_error',
                        'raw': api_result
                    }
            except Exception as e:
                logger.warning(f"Hugging Face API call failed: {e}")
                return {
                    'overall_bias_score': 0.0,
                    'error': str(e),
                    'method': 'huggingface_api_error'
                }
        else:
            return {
                'overall_bias_score': 0.0,
                'error': 'No Hugging Face API key provided',
                'method': 'no_api_key'
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