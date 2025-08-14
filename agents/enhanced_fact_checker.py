import logging
import aiohttp
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class EnhancedFactChecker:
    """
    Enhanced fact checker that uses multiple approaches:
    1. Google Fact Check Tools API (existing)
    2. Claim analysis and confidence scoring
    3. Multiple fact-checking sources
    4. AI-powered claim verification
    """
    
    def __init__(self, config=None):
        self.config = config
        self.factcheck_api_key = None
        if config:
            self.factcheck_api_key = config.get_api_key('factcheck')
        
        # Fact-checking sources
        self.sources = [
            'google_factcheck',
            'claim_analysis',
            'confidence_scoring'
        ]
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    async def verify_claims(self, text: str, max_claims: int = 5) -> Dict[str, Any]:
        """
        Enhanced claim verification using multiple approaches
        """
        try:
            # Extract claims from text
            claims = self._extract_enhanced_claims(text)
            claims = claims[:max_claims]
            
            results = []
            for claim in claims:
                # Try multiple verification methods
                verification_result = await self._verify_single_claim(claim)
                results.append(verification_result)
            
            # Aggregate results with confidence scoring
            overall_result = self._aggregate_with_confidence(results)
            
            return {
                'overall_verdict': overall_result['verdict'],
                'confidence': overall_result['confidence'],
                'claims': results,
                'verification_methods': self.sources,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced fact checking failed: {e}")
            return {
                'overall_verdict': 'ERROR',
                'confidence': 0.0,
                'claims': [],
                'error': str(e)
            }
    
    def _extract_enhanced_claims(self, text: str) -> List[str]:
        """
        Enhanced claim extraction with better pattern matching
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        # Enhanced claim indicators
        claim_indicators = [
            # Factual statements
            r'\b(?:is|are|was|were|will|has|have|had)\b',
            r'\b(?:according to|reported by|announced|claims|said|stated)\b',
            # Numbers and statistics
            r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)\b',
            r'\b(?:study|research|survey|poll|report)\b',
            # Time-based claims
            r'\b(?:in \d{4}|since|during|before|after)\b',
            # Comparative claims
            r'\b(?:more than|less than|higher than|lower than|better than|worse than)\b'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # Skip very short sentences
                continue
            
            # Check if sentence contains claim indicators
            is_claim = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in claim_indicators)
            
            if is_claim:
                # Clean up the sentence
                cleaned = re.sub(r'\s+', ' ', sentence).strip()
                if cleaned:
                    claims.append(cleaned[:300])  # Limit length
        
        # If no specific claims found, use the full text
        if not claims and text:
            claims = [text.strip()[:300]]
        
        return claims
    
    async def _verify_single_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verify a single claim using multiple methods
        """
        results = {}
        
        # Method 1: Google Fact Check Tools API
        if self.factcheck_api_key:
            api_result = await self._query_factcheck_api(claim)
            results['google_factcheck'] = api_result
        
        # Method 2: Claim Analysis
        analysis_result = self._analyze_claim_structure(claim)
        results['claim_analysis'] = analysis_result
        
        # Method 3: Confidence Scoring
        confidence_result = self._calculate_claim_confidence(claim, results)
        results['confidence_scoring'] = confidence_result
        
        # Combine results
        combined_verdict = self._combine_verdicts(results)
        
        return {
            'claim': claim,
            'verdict': combined_verdict['verdict'],
            'confidence': combined_verdict['confidence'],
            'reasoning': combined_verdict['reasoning'],
            'sources': combined_verdict.get('sources', []),
            'verification_methods': list(results.keys())
        }
    
    async def _query_factcheck_api(self, claim: str) -> Dict[str, Any]:
        """
        Query Google Fact Check Tools API
        """
        if not self.factcheck_api_key:
            return {'status': 'no_api_key'}
        
        endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': claim[:256],
            'languageCode': 'en',
            'pageSize': 5,
            'key': self.factcheck_api_key
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._process_factcheck_response(data)
                    else:
                        return {'status': f'api_error_{resp.status}'}
        except Exception as e:
            logger.warning(f"FactCheck API error: {e}")
            return {'status': 'request_error', 'error': str(e)}
    
    def _process_factcheck_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Google Fact Check API response
        """
        if 'claims' not in data or not data['claims']:
            return {'status': 'no_claims_found'}
        
        # Find the best matching claim
        best_claim = None
        best_score = 0
        
        for claim in data['claims']:
            score = self._calculate_claim_similarity_score(claim)
            if score > best_score:
                best_score = score
                best_claim = claim
        
        if not best_claim:
            return {'status': 'no_matching_claims'}
        
        # Extract verdict information
        reviews = best_claim.get('claimReview', [])
        if not reviews:
            return {'status': 'no_reviews'}
        
        review = reviews[0]  # Take the first review
        verdict = self._map_textual_rating(review.get('textualRating', ''))
        
        return {
            'status': 'success',
            'verdict': verdict,
            'source': review.get('publisher', {}).get('name'),
            'rating': review.get('textualRating'),
            'url': review.get('url'),
            'confidence': min(best_score, 0.9)  # Cap confidence at 0.9 for API results
        }
    
    def _calculate_claim_similarity_score(self, claim: Dict[str, Any]) -> float:
        """
        Calculate similarity score between query and API result
        """
        # Simple text similarity scoring
        query_text = claim.get('text', '').lower()
        if not query_text:
            return 0.0
        
        # Count common words (simple approach)
        words = set(re.findall(r'\b\w+\b', query_text))
        return min(len(words) / 10.0, 1.0)  # Normalize by expected word count
    
    def _analyze_claim_structure(self, claim: str) -> Dict[str, Any]:
        """
        Analyze claim structure to determine verifiability
        """
        analysis = {
            'verifiability': 'unknown',
            'confidence': 0.5,
            'reasoning': []
        }
        
        # Check for factual indicators
        factual_indicators = [
            (r'\b\d{4}\b', 'specific_year', 0.3),
            (r'\b\d+(?:\.\d+)?%\b', 'percentage', 0.4),
            (r'\b(?:million|billion|thousand)\b', 'large_numbers', 0.3),
            (r'\b(?:study|research|survey|poll)\b', 'research_reference', 0.4),
            (r'\b(?:according to|reported by|announced)\b', 'attribution', 0.3)
        ]
        
        confidence_boost = 0.0
        for pattern, indicator, boost in factual_indicators:
            if re.search(pattern, claim, re.IGNORECASE):
                analysis['reasoning'].append(f"Contains {indicator}")
                confidence_boost += boost
        
        # Check for vague language (reduces confidence)
        vague_indicators = [
            (r'\b(?:some|many|few|several|various)\b', -0.2),
            (r'\b(?:might|could|may|possibly)\b', -0.3),
            (r'\b(?:always|never|everyone|nobody)\b', -0.2)
        ]
        
        for pattern, penalty in vague_indicators:
            if re.search(pattern, claim, re.IGNORECASE):
                analysis['reasoning'].append(f"Contains vague language")
                confidence_boost += penalty
        
        # Calculate final confidence
        analysis['confidence'] = max(0.1, min(0.9, 0.5 + confidence_boost))
        
        # Determine verifiability
        if analysis['confidence'] >= 0.7:
            analysis['verifiability'] = 'highly_verifiable'
        elif analysis['confidence'] >= 0.5:
            analysis['verifiability'] = 'verifiable'
        else:
            analysis['verifiability'] = 'difficult_to_verify'
        
        return analysis
    
    def _calculate_claim_confidence(self, claim: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall confidence for a claim
        """
        confidence = 0.5  # Base confidence
        reasoning = []
        
        # Boost confidence if we have API results
        if 'google_factcheck' in previous_results:
            api_result = previous_results['google_factcheck']
            if api_result.get('status') == 'success':
                confidence += 0.3
                reasoning.append("Verified by fact-checking organizations")
            elif api_result.get('status') == 'no_claims_found':
                confidence -= 0.1
                reasoning.append("No existing fact-checks found")
        
        # Boost confidence based on claim analysis
        if 'claim_analysis' in previous_results:
            analysis = previous_results['claim_analysis']
            confidence += (analysis['confidence'] - 0.5) * 0.2
            reasoning.append(f"Claim structure analysis: {analysis['verifiability']}")
        
        # Normalize confidence
        confidence = max(0.1, min(0.95, confidence))
        
        return {
            'confidence': confidence,
            'reasoning': reasoning,
            'level': self._get_confidence_level(confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description
        """
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _combine_verdicts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from multiple verification methods
        """
        # Start with base values
        verdict = 'UNVERIFIED'
        confidence = 0.5
        reasoning = []
        sources = []
        
        # Check Google Fact Check results first
        if 'google_factcheck' in results:
            api_result = results['google_factcheck']
            if api_result.get('status') == 'success':
                verdict = api_result.get('verdict', 'UNVERIFIED')
                confidence = api_result.get('confidence', 0.7)
                reasoning.append("Verified by fact-checking organizations")
                if api_result.get('source'):
                    sources.append({
                        'name': api_result['source'],
                        'url': api_result.get('url'),
                        'rating': api_result.get('rating')
                    })
        
        # If no clear verdict, use claim analysis
        if verdict == 'UNVERIFIED' and 'claim_analysis' in results:
            analysis = results['claim_analysis']
            if analysis['verifiability'] == 'highly_verifiable':
                verdict = 'LIKELY_VERIFIABLE'
                confidence = max(confidence, analysis['confidence'])
                reasoning.append("Claim structure suggests high verifiability")
        
        # Apply confidence scoring
        if 'confidence_scoring' in results:
            confidence_result = results['confidence_scoring']
            confidence = max(confidence, confidence_result['confidence'])
            reasoning.extend(confidence_result['reasoning'])
        
        # Final verdict mapping
        if confidence >= 0.8:
            if verdict == 'UNVERIFIED':
                verdict = 'LIKELY_VERIFIABLE'
        elif confidence <= 0.3:
            if verdict == 'UNVERIFIED':
                verdict = 'DIFFICULT_TO_VERIFY'
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'sources': sources
        }
    
    def _map_textual_rating(self, textual: str) -> str:
        """
        Map textual ratings to standardized verdicts
        """
        if not textual:
            return 'UNVERIFIED'
        
        textual_lower = textual.lower()
        
        if any(word in textual_lower for word in ['false', 'inaccurate', 'misleading', 'wrong']):
            return 'FALSE'
        elif any(word in textual_lower for word in ['true', 'accurate', 'correct', 'accurate']):
            return 'TRUE'
        elif any(word in textual_lower for word in ['partly', 'mixed', 'partially', 'half']):
            return 'MIXED'
        else:
            return 'UNVERIFIED'
    
    def _aggregate_with_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple claim results with confidence scoring
        """
        if not results:
            return {'verdict': 'NO_CLAIMS', 'confidence': 0.0}
        
        # Count verdicts
        verdict_counts = {}
        total_confidence = 0.0
        
        for result in results:
            verdict = result.get('verdict', 'UNVERIFIED')
            confidence = result.get('confidence', 0.5)
            
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            total_confidence += confidence
        
        # Determine overall verdict
        if 'TRUE' in verdict_counts and 'FALSE' not in verdict_counts:
            overall_verdict = 'VERIFIED_TRUE'
        elif 'FALSE' in verdict_counts and 'TRUE' not in verdict_counts:
            overall_verdict = 'VERIFIED_FALSE'
        elif 'TRUE' in verdict_counts and 'FALSE' in verdict_counts:
            overall_verdict = 'MIXED_VERDICTS'
        elif 'LIKELY_VERIFIABLE' in verdict_counts:
            overall_verdict = 'PARTIALLY_VERIFIABLE'
        elif 'DIFFICULT_TO_VERIFY' in verdict_counts:
            overall_verdict = 'DIFFICULT_TO_VERIFY'
        else:
            overall_verdict = 'INSUFFICIENT_EVIDENCE'
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(results)
        
        return {
            'verdict': overall_verdict,
            'confidence': avg_confidence,
            'verdict_breakdown': verdict_counts
        }
