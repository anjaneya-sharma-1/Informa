import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import re
import requests
import os

logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    def __init__(self, config=None):
        self.config = config
        self.last_run = None
        self.last_error = None
        
        # Get API configuration dynamically
        if config:
            self.api_key = config.get_huggingface_key()
            models = config.get_huggingface_models()
            self.api_url = f"https://api-inference.huggingface.co/models/{models['sentiment']}"
        else:
            # Fallback to environment/secrets file
            self.api_key = self._load_api_key_from_secrets()
            self.api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
        
        if not self.api_key:
            logger.warning("No Hugging Face API key found. Sentiment analysis will use fallback methods.")
        
        # Label mapping for the model
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
    
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

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text using Hugging Face Inference API with fallback methods."""
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
                    return await self._comprehensive_sentiment_analysis(cleaned_text)
                except Exception as e:
                    logger.warning(f"Hugging Face API call failed: {e}")
                    return self._fallback_sentiment_analysis(cleaned_text)
            else:
                return self._fallback_sentiment_analysis(cleaned_text)
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'method': 'error_fallback'
            }
    
    async def _comprehensive_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Comprehensive sentiment analysis using Hugging Face API"""
        # Get primary sentiment from API
        api_result = self._huggingface_api_analysis(text)
        
        # Enhance with additional analysis
        keyword_sentiment = self._keyword_based_sentiment(text)
        intensity_score = self._calculate_intensity(text)
        
        # Combine results
        primary_label = api_result['label']
        primary_confidence = api_result['confidence']
        
        # Adjust confidence based on keyword analysis agreement
        if keyword_sentiment['label'] == primary_label:
            adjusted_confidence = min(primary_confidence + 0.1, 1.0)
        else:
            adjusted_confidence = max(primary_confidence - 0.1, 0.0)
        
        return {
            'label': primary_label,
            'score': api_result['score'],
            'confidence': adjusted_confidence,
            'intensity': intensity_score,
            'keyword_analysis': keyword_sentiment,
            'method': 'comprehensive_huggingface_api',
            'raw_api_result': api_result.get('raw_results', [])
        }
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keyword-based approach"""
        keyword_result = self._keyword_based_sentiment(text)
        intensity = self._calculate_intensity(text)
        
        return {
            'label': keyword_result['label'],
            'score': keyword_result['score'],
            'confidence': keyword_result['confidence'],
            'intensity': intensity,
            'method': 'keyword_fallback'
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

    def _keyword_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Keyword-based sentiment analysis as fallback"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive',
            'happy', 'joy', 'success', 'achievement', 'progress', 'improvement', 'beneficial'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry', 'hate',
            'failure', 'problem', 'crisis', 'disaster', 'dangerous', 'harmful', 'devastating'
        ]
        
        neutral_words = [
            'okay', 'average', 'normal', 'standard', 'typical', 'regular', 'moderate'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        neutral_count = sum(1 for word in words if word in neutral_words)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.1}
        
        if positive_count > negative_count and positive_count > neutral_count:
            confidence = positive_count / max(total_sentiment_words, 1)
            return {'label': 'positive', 'score': confidence, 'confidence': confidence}
        elif negative_count > positive_count and negative_count > neutral_count:
            confidence = negative_count / max(total_sentiment_words, 1)
            return {'label': 'negative', 'score': -confidence, 'confidence': confidence}
        else:
            confidence = max(neutral_count / max(total_sentiment_words, 1), 0.1)
            return {'label': 'neutral', 'score': 0.0, 'confidence': confidence}
    
    def _calculate_intensity(self, text: str) -> float:
        """Calculate sentiment intensity based on text features"""
        intensity_indicators = [
            '!', '!!', '!!!', '?', '??', '???',
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'entirely', 'utterly', 'really', 'truly'
        ]
        
        text_lower = text.lower()
        intensity_count = sum(text_lower.count(indicator) for indicator in intensity_indicators)
        
        # Normalize intensity score
        max_intensity = 10  # Arbitrary max
        intensity_score = min(intensity_count / max_intensity, 1.0)
        
        return intensity_score
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$\$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit text length for API (most models have token limits)
        if len(text) > 500:
            text = text[:500] + "..."
        
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

