import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

import streamlit as st

logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    def __init__(self):
        self.last_run = None
        self.last_error = None
        
        # Predefined sentiment lexicons for more accurate analysis
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'positive',
            'success', 'achievement', 'breakthrough', 'progress', 'improvement', 'benefit',
            'opportunity', 'hope', 'optimistic', 'confident', 'pleased', 'satisfied',
            'outstanding', 'remarkable', 'impressive', 'innovative', 'revolutionary'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'negative', 'failure', 'disaster',
            'crisis', 'problem', 'issue', 'concern', 'worry', 'fear', 'threat', 'risk',
            'decline', 'decrease', 'loss', 'damage', 'harm', 'danger', 'critical',
            'devastating', 'alarming', 'shocking', 'disturbing', 'troubling'
        }
        
        self.neutral_words = {
            'report', 'announce', 'state', 'according', 'data', 'information', 'study',
            'research', 'analysis', 'review', 'update', 'news', 'statement'
        }
        
        # Initialize Hugging Face sentiment analysis pipeline
        try:
            # Use a lightweight, fast model for sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                return_all_scores=True
            )
            
            # Label mapping for the model
            self.label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            logger.info("Sentiment analyzer initialized with Hugging Face model")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.sentiment_pipeline = None
            self.last_error = str(e)
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text"""
        try:
            self.last_run = datetime.now().isoformat()
            
            if not text or not text.strip():
                return {
                    'label': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'method': 'empty_text'
                }
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Multiple sentiment analysis approaches
            if self.sentiment_pipeline:
                try:
                    hf_result = await self._huggingface_analysis(cleaned_text)
                    
                    # Combine with lexicon-based analysis for better accuracy
                    lexicon_result = self._lexicon_based_analysis(cleaned_text)
                    
                    # Weighted combination
                    combined_result = self._combine_results(hf_result, lexicon_result)
                    
                    return combined_result
                    
                except Exception as e:
                    logger.warning(f"Hugging Face analysis failed, falling back to lexicon: {e}")
                    return self._lexicon_based_analysis(cleaned_text)
            else:
                # Fallback to lexicon-based analysis
                return self._lexicon_based_analysis(cleaned_text)
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _huggingface_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis using Hugging Face model"""
        try:
            # Truncate text if too long (model limit is usually 512 tokens)
            if len(text) > 400:  # Conservative limit
                text = text[:400] + "..."
            
            # Run inference
            results = self.sentiment_pipeline(text)
            
            # Process results
            if results and len(results) > 0:
                # Get the result with highest confidence
                best_result = max(results[0], key=lambda x: x['score'])
                
                # Map label
                raw_label = best_result['label']
                mapped_label = self.label_mapping.get(raw_label, raw_label.lower())
                
                # Convert score to our format (-1 to 1)
                confidence = best_result['score']
                if mapped_label == 'positive':
                    score = confidence
                elif mapped_label == 'negative':
                    score = -confidence
                else:  # neutral
                    score = 0.0
                
                return {
                    'label': mapped_label,
                    'score': score,
                    'confidence': confidence,
                    'method': 'huggingface',
                    'raw_results': results[0]
                }
            else:
                raise ValueError("No results from Hugging Face model")
                
        except Exception as e:
            logger.error(f"Hugging Face analysis error: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _lexicon_based_analysis(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis using predefined word lexicons"""
        try:
            words = text.split()
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            neutral_count = sum(1 for word in words if word in self.neutral_words)
            
            total_sentiment_words = positive_count + negative_count + neutral_count
            
            if total_sentiment_words == 0:
                return {
                    'label': 'neutral',
                    'score': 0.0,
                    'confidence': 0.5,
                    'method': 'lexicon'
                }
            
            # Calculate weighted score
            positive_weight = positive_count / len(words)
            negative_weight = negative_count / len(words)
            neutral_weight = neutral_count / len(words)
            
            score = positive_weight - negative_weight
            
            # Determine label
            if score > 0.02:
                label = 'positive'
                confidence = min(0.9, 0.5 + abs(score) * 2)
            elif score < -0.02:
                label = 'negative'
                confidence = min(0.9, 0.5 + abs(score) * 2)
            else:
                label = 'neutral'
                confidence = 0.6
            
            return {
                'label': label,
                'score': score,
                'confidence': confidence,
                'method': 'lexicon',
                'positive_words': positive_count,
                'negative_words': negative_count,
                'neutral_words': neutral_count
            }
            
        except Exception as e:
            logger.error(f"Lexicon analysis error: {e}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _combine_results(self, hf_result: Dict[str, Any], lexicon_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine Hugging Face and lexicon results for better accuracy"""
        try:
            # Weight the results (HF gets more weight)
            hf_weight = 0.7
            lexicon_weight = 0.3
            
            # Combine scores
            combined_score = (
                hf_result['score'] * hf_weight + 
                lexicon_result['score'] * lexicon_weight
            )
            
            # Determine final label based on combined score
            if combined_score > 0.1:
                final_label = 'positive'
            elif combined_score < -0.1:
                final_label = 'negative'
            else:
                final_label = 'neutral'
            
            # Calculate combined confidence
            combined_confidence = (
                hf_result['confidence'] * hf_weight + 
                lexicon_result['confidence'] * lexicon_weight
            )
            
            return {
                'label': final_label,
                'score': combined_score,
                'confidence': combined_confidence,
                'method': 'combined',
                'details': {
                    'huggingface': hf_result,
                    'lexicon': lexicon_result
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            # Return HF result as fallback
            return hf_result
    
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
    
    def health_check(self) -> bool:
        """Check if the sentiment analyzer is healthy"""
        try:
            # Test with simple text
            test_text = "This is a good test"
            
            if self.sentiment_pipeline:
                # Test Hugging Face pipeline
                result = self.sentiment_pipeline(test_text)
                return bool(result)
            else:
                # Test lexicon analysis
                result = self._lexicon_based_analysis(test_text)
                return result.get('label') is not None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear any cached data"""
        self.last_error = None
        logger.info("Sentiment analyzer cache cleared")
    
    def restart(self):
        """Restart the sentiment analyzer"""
        self.clear_cache()
        self.last_run = None
        logger.info("Sentiment analyzer restarted")
