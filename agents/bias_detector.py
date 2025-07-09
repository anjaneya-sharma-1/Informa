import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)

class BiasDetectorAgent:
    def __init__(self):
        self.last_run = None
        self.last_error = None
        
        # Try to initialize Hugging Face model for bias detection
        try:
            # Use a model trained for bias detection or classification
            # Note: This is a placeholder - you might want to use a specific bias detection model
            model_name = "unitary/toxic-bert"
            
            self.bias_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Bias detector initialized with Hugging Face model")
            
        except Exception as e:
            logger.warning(f"Could not initialize HF bias model, using rule-based approach: {e}")
            self.bias_pipeline = None
        
        # Rule-based bias indicators
        self.bias_indicators = {
            'political_left': {
                'keywords': ['progressive', 'liberal', 'social justice', 'equality', 'diversity', 
                           'systemic', 'marginalized', 'oppressed', 'activist', 'reform'],
                'phrases': ['fight for justice', 'systemic inequality', 'progressive values', 
                          'social change', 'human rights'],
                'patterns': [r'(fight|battle) (for|against)', r'systemic (racism|inequality|oppression)']
            },
            'political_right': {
                'keywords': ['conservative', 'traditional', 'patriotic', 'freedom', 'liberty',
                           'constitutional', 'family values', 'law and order', 'security', 'defense'],
                'phrases': ['traditional values', 'personal responsibility', 'free market',
                          'constitutional rights', 'law and order'],
                'patterns': [r'traditional (values|family)', r'personal (responsibility|freedom)']
            },
            'sensationalism': {
                'keywords': ['shocking', 'unbelievable', 'incredible', 'amazing', 'devastating',
                           'explosive', 'bombshell', 'stunning', 'outrageous', 'unprecedented'],
                'phrases': ['you won\'t believe', 'shocking truth', 'incredible discovery',
                          'breaking news', 'exclusive report'],
                'patterns': [r'[A-Z]{3,}', r'!!!+', r'\?{2,}', r'BREAKING:', r'EXCLUSIVE:']
            },
            'emotional_manipulation': {
                'keywords': ['outrageous', 'disgusting', 'heartbreaking', 'terrifying', 'infuriating',
                           'devastating', 'horrific', 'appalling', 'shocking', 'disturbing'],
                'phrases': ['makes you sick', 'will shock you', 'absolutely disgusting',
                          'heart-wrenching', 'blood-boiling'],
                'patterns': [r'(very|extremely|absolutely|completely)\s+(shocking|disgusting|outrageous)']
            },
            'confirmation_bias': {
                'keywords': ['obviously', 'clearly', 'undoubtedly', 'everyone knows', 'common sense',
                           'of course', 'naturally', 'certainly', 'definitely', 'surely'],
                'phrases': ['it\'s obvious that', 'everyone agrees', 'common knowledge',
                          'goes without saying', 'stands to reason'],
                'patterns': [r'(obviously|clearly|undoubtedly)', r'everyone (knows|agrees|thinks)']
            },
            'loaded_language': {
                'keywords': ['radical', 'extremist', 'fanatic', 'terrorist', 'hero', 'villain',
                           'monster', 'saint', 'evil', 'pure', 'corrupt', 'innocent'],
                'phrases': ['radical agenda', 'extremist views', 'dangerous ideology',
                          'corrupt system', 'pure evil'],
                'patterns': [r'(so-called|alleged)', r'(radical|extreme|dangerous)\s+(left|right|agenda)']
            },
            'false_balance': {
                'keywords': ['both sides', 'balanced view', 'fair and balanced', 'some say',
                           'critics argue', 'supporters claim', 'opponents believe'],
                'phrases': ['on the other hand', 'some say', 'critics argue',
                          'both sides of the story', 'fair and balanced'],
                'patterns': [r'some (people|experts|critics) (say|argue|believe)',
                           r'(supporters|opponents) (claim|argue|believe)']
            }
        }
        
        # Source reliability database
        self.source_reliability = {
            'high': ['reuters', 'ap news', 'bbc', 'npr', 'pbs', 'associated press'],
            'medium': ['cnn', 'fox news', 'msnbc', 'washington post', 'new york times', 
                      'wall street journal', 'usa today'],
            'low': ['infowars', 'breitbart', 'occupy democrats', 'natural news', 
                   'daily mail', 'buzzfeed news'],
            'social': ['reddit', 'twitter', 'facebook', 'hackernews']
        }
    
    async def analyze_bias(self, text: str, source: str = None) -> Dict[str, Any]:
        """Comprehensive bias analysis using multiple methods"""
        try:
            self.last_run = datetime.now().isoformat()
            
            if not text or not text.strip():
                return {
                    'overall_bias_score': 0.0,
                    'bias_breakdown': {},
                    'method': 'empty_text'
                }
            
            # Clean text
            cleaned_text = self._preprocess_text(text)
            
            # Rule-based analysis
            rule_based_result = await self._rule_based_analysis(cleaned_text, source)
            
            # Try Hugging Face analysis if available
            if self.bias_pipeline:
                try:
                    hf_result = await self._huggingface_bias_analysis(cleaned_text)
                    # Combine results
                    combined_result = self._combine_bias_results(rule_based_result, hf_result)
                    return combined_result
                except Exception as e:
                    logger.warning(f"HF bias analysis failed, using rule-based: {e}")
            
            return rule_based_result
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in bias analysis: {e}")
            return {
                'overall_bias_score': 0.0,
                'error': str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for bias analysis"""
        # Preserve case for pattern matching
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = ' '.join(text.split())
        return text
    
    async def _rule_based_analysis(self, text: str, source: str = None) -> Dict[str, Any]:
        """Rule-based bias detection"""
        try:
            bias_scores = {}
            detected_patterns = []
            
            # Analyze each bias type
            for bias_type, indicators in self.bias_indicators.items():
                score = self._calculate_bias_score(text, indicators)
                bias_scores[bias_type] = score
                
                if score > 0.3:  # Significant bias detected
                    patterns = self._find_bias_patterns(text, indicators)
                    if patterns:
                        detected_patterns.extend([
                            {'type': bias_type, 'pattern': pattern, 'score': score}
                            for pattern in patterns[:3]  # Limit patterns
                        ])
            
            # Source-based bias adjustment
            source_bias = self._analyze_source_bias(source)
            
            # Calculate overall bias score
            content_bias = max(bias_scores.values()) if bias_scores else 0.0
            overall_bias = min(1.0, (content_bias * 0.8) + (source_bias * 0.2))
            
            # Generate recommendations
            recommendations = self._generate_bias_recommendations(bias_scores, detected_patterns)
            
            return {
                'overall_bias_score': overall_bias,
                'bias_breakdown': bias_scores,
                'detected_patterns': detected_patterns,
                'source_bias': source_bias,
                'recommendations': recommendations,
                'method': 'rule_based',
                'confidence': self._calculate_confidence(bias_scores, detected_patterns)
            }
            
        except Exception as e:
            logger.error(f"Rule-based analysis error: {e}")
            return {'overall_bias_score': 0.0, 'error': str(e)}
    
    async def _huggingface_bias_analysis(self, text: str) -> Dict[str, Any]:
        """Bias analysis using Hugging Face model"""
        try:
            # Truncate text if too long
            if len(text) > 400:
                text = text[:400] + "..."
            
            # Run inference
            results = self.bias_pipeline(text)
            
            if results and len(results) > 0:
                # Process results - this depends on the specific model used
                # For toxic-bert, we get toxicity scores
                toxic_score = 0.0
                
                for result in results[0]:
                    if result['label'] == 'TOXIC':
                        toxic_score = result['score']
                        break
                
                return {
                    'hf_bias_score': toxic_score,
                    'method': 'huggingface',
                    'raw_results': results[0]
                }
            else:
                raise ValueError("No results from HF model")
                
        except Exception as e:
            logger.error(f"HF bias analysis error: {e}")
            raise
    
    def _combine_bias_results(self, rule_result: Dict[str, Any], hf_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and HF bias analysis results"""
        try:
            # Weight the results
            rule_weight = 0.7
            hf_weight = 0.3
            
            # Combine overall scores
            rule_score = rule_result.get('overall_bias_score', 0.0)
            hf_score = hf_result.get('hf_bias_score', 0.0)
            
            combined_score = (rule_score * rule_weight) + (hf_score * hf_weight)
            
            # Update the rule result with combined information
            rule_result['overall_bias_score'] = combined_score
            rule_result['method'] = 'combined'
            rule_result['hf_analysis'] = hf_result
            
            return rule_result
            
        except Exception as e:
            logger.error(f"Error combining bias results: {e}")
            return rule_result
    
    def _calculate_bias_score(self, text: str, indicators: Dict[str, List[str]]) -> float:
        """Calculate bias score for specific indicators"""
        text_lower = text.lower()
        total_score = 0.0
        max_possible_score = 0.0
        
        # Check keywords
        if 'keywords' in indicators:
            keyword_matches = sum(1 for keyword in indicators['keywords'] if keyword in text_lower)
            keyword_score = min(1.0, keyword_matches / max(len(indicators['keywords']), 1))
            total_score += keyword_score * 0.4
            max_possible_score += 0.4
        
        # Check phrases
        if 'phrases' in indicators:
            phrase_matches = sum(1 for phrase in indicators['phrases'] if phrase in text_lower)
            phrase_score = min(1.0, phrase_matches / max(len(indicators['phrases']), 1))
            total_score += phrase_score * 0.4
            max_possible_score += 0.4
        
        # Check patterns
        if 'patterns' in indicators:
            pattern_matches = 0
            for pattern in indicators['patterns']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                pattern_matches += matches
            
            pattern_score = min(1.0, pattern_matches / max(len(indicators['patterns']), 1))
            total_score += pattern_score * 0.2
            max_possible_score += 0.2
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
    
    def _find_bias_patterns(self, text: str, indicators: Dict[str, List[str]]) -> List[str]:
        """Find specific bias patterns in text"""
        found_patterns = []
        
        # Find keyword matches
        if 'keywords' in indicators:
            for keyword in indicators['keywords']:
                if keyword in text.lower():
                    # Find context
                    start_idx = text.lower().find(keyword)
                    if start_idx != -1:
                        context_start = max(0, start_idx - 20)
                        context_end = min(len(text), start_idx + len(keyword) + 20)
                        context = text[context_start:context_end].strip()
                        found_patterns.append(f"Keyword '{keyword}': ...{context}...")
        
        # Find phrase matches
        if 'phrases' in indicators:
            for phrase in indicators['phrases']:
                if phrase in text.lower():
                    start_idx = text.lower().find(phrase)
                    if start_idx != -1:
                        context_start = max(0, start_idx - 15)
                        context_end = min(len(text), start_idx + len(phrase) + 15)
                        context = text[context_start:context_end].strip()
                        found_patterns.append(f"Phrase '{phrase}': ...{context}...")
        
        return found_patterns[:5]  # Limit to top 5
    
    def _analyze_source_bias(self, source: str) -> float:
        """Analyze bias based on news source"""
        if not source:
            return 0.0
        
        source_lower = source.lower()
        
        # Check against known source reliability
        for reliability_level, sources in self.source_reliability.items():
            if any(known_source in source_lower for known_source in sources):
                bias_score = {
                    'high': 0.1,      # High reliability = low bias
                    'medium': 0.3,    # Medium reliability = moderate bias
                    'low': 0.8,       # Low reliability = high bias
                    'social': 0.5     # Social media = moderate bias
                }.get(reliability_level, 0.5)
                
                return bias_score
        
        # Unknown source - moderate bias assumption
        return 0.5
    
    def _generate_bias_recommendations(self, bias_breakdown: Dict[str, float], 
                                     detected_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for reducing bias"""
        recommendations = []
        
        # Check for high bias scores
        high_bias_types = [bias_type for bias_type, score in bias_breakdown.items() if score > 0.6]
        
        if 'sensationalism' in high_bias_types:
            recommendations.append("Consider using more neutral language and avoiding sensationalized terms.")
        
        if 'emotional_manipulation' in high_bias_types:
            recommendations.append("Focus on facts rather than emotional appeals to maintain objectivity.")
        
        if 'political_left' in high_bias_types or 'political_right' in high_bias_types:
            recommendations.append("Present multiple perspectives to provide balanced coverage.")
        
        if 'loaded_language' in high_bias_types:
            recommendations.append("Use neutral terminology instead of loaded or charged language.")
        
        if 'confirmation_bias' in high_bias_types:
            recommendations.append("Include evidence that challenges the main narrative for balance.")
        
        if 'false_balance' in high_bias_types:
            recommendations.append("Ensure different viewpoints are given appropriate weight based on evidence.")
        
        # General recommendations
        if max(bias_breakdown.values()) > 0.5:
            recommendations.append("Consider fact-checking claims and citing credible sources.")
            recommendations.append("Review the article for subjective language and replace with objective reporting.")
        
        return recommendations[:5]  # Limit to top 5
    
    def _calculate_confidence(self, bias_breakdown: Dict[str, float], 
                            detected_patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence in bias detection"""
        # Higher confidence when multiple bias types are detected
        significant_biases = sum(1 for score in bias_breakdown.values() if score > 0.3)
        pattern_count = len(detected_patterns)
        
        # Base confidence on number of detected patterns and bias types
        confidence = min(1.0, (significant_biases * 0.2) + (pattern_count * 0.1) + 0.5)
        
        return confidence
    
    def health_check(self) -> bool:
        """Check if the bias detector is healthy"""
        try:
            # Test with simple text
            test_text = "This is a neutral test statement"
            
            # Test rule-based analysis
            result = asyncio.run(self._rule_based_analysis(test_text))
            return result.get('overall_bias_score') is not None
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Bias detector health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear any cached data"""
        self.last_error = None
        logger.info("Bias detector cache cleared")
    
    def restart(self):
        """Restart the bias detector"""
        self.clear_cache()
        self.last_run = None
        logger.info("Bias detector restarted")
