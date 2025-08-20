import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import re
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

class SentimentAnalyzerAgent:
    def __init__(self):
        self.last_run = None
        self.last_error = None
        self.api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.api_key = HF_API_KEY
        if not self.api_key:
            logger.warning("No Hugging Face API key found. Please add HF_API_KEY to secrets.env.")
        # Label mapping for the model
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text using Hugging Face Inference API only."""
        try:
            self.last_run = datetime.now().isoformat()
            if not text or not text.strip():
                return {
                    'label': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'method': 'empty_text'
                }
            cleaned_text = self._preprocess_text(text)
            if self.api_key:
                try:
                    return self._huggingface_api_analysis(cleaned_text)
                except Exception as e:
                    logger.warning(f"Hugging Face API call failed: {e}")
                    return {
                        'label': 'neutral',
                        'score': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'method': 'huggingface_api_error'
                    }
            else:
                return {
                    'label': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'error': 'No Hugging Face API key provided',
                    'method': 'no_api_key'
                }
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    def _huggingface_api_analysis(self, text: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.api_url, headers=headers, json={"inputs": text})
        response.raise_for_status()
        results = response.json()
        if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            best_result = max(results[0], key=lambda x: x['score'])
            raw_label = best_result['label']
            mapped_label = self.label_mapping.get(raw_label, raw_label.lower())
            confidence = best_result['score']
            if mapped_label == 'positive':
                score = confidence
            elif mapped_label == 'negative':
                score = -confidence
            else:
                score = 0.0
            return {
                'label': mapped_label,
                'score': score,
                'confidence': confidence,
                'method': 'huggingface_api',
                'raw_results': results[0]
            }
        else:
            logger.warning(f"Unexpected API result: {results}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': 'Unexpected API result',
                'method': 'huggingface_api_error',
                'raw': results
            }

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$\$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        text = ' '.join(text.split())
        return text
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts efficiently"""
        try:
            results = []
            
            # Process in batches to avoid memory issues
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch
                batch_results = []
                for text in batch:
                    result = await self.analyze(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return [{'label': 'neutral', 'score': 0.0, 'error': str(e)} for _ in texts]
    
    def get_emotion_breakdown(self, text: str) -> Dict[str, float]:
        """Get detailed emotion breakdown (joy, anger, fear, etc.)"""
        try:
            emotions = {
                'joy': ['happy', 'excited', 'thrilled', 'delighted', 'cheerful', 'elated', 'jubilant'],
                'anger': ['angry', 'furious', 'outraged', 'irritated', 'annoyed', 'livid', 'enraged'],
                'fear': ['afraid', 'scared', 'worried', 'anxious', 'concerned', 'terrified', 'panicked'],
                'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'heartbroken', 'melancholy', 'grief'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
                'disgust': ['disgusted', 'revolted', 'appalled', 'sickened', 'repulsed', 'nauseated']
            }
            
            text_lower = text.lower()
            emotion_scores = {}
            
            for emotion, keywords in emotions.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                emotion_scores[emotion] = score / len(keywords) if keywords else 0
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion breakdown: {e}")
            return {}
    
