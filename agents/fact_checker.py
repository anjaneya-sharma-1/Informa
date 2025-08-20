import asyncio
import aiohttp
import logging
from typing import Dict, Any, List
from datetime import datetime
import re
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class FactCheckerAgent:
    def __init__(self, config):
        self.config = config
        self.last_run = None
        self.last_error = None
        
        # Fact-checking sources (free APIs and websites)
        self.fact_check_sources = {
            'snopes': 'https://www.snopes.com/search/',
            'factcheck': 'https://www.factcheck.org/search/',
            'politifact': 'https://www.politifact.com/search/',
            'reuters_factcheck': 'https://www.reuters.com/fact-check/',
            'ap_factcheck': 'https://apnews.com/hub/ap-fact-check'
        }
        
        # Credibility indicators
        self.credibility_indicators = {
            'positive': [
                'peer-reviewed', 'published study', 'research shows', 'according to experts',
                'official statement', 'government data', 'scientific evidence', 'verified by',
                'confirmed by', 'multiple sources', 'independent verification'
            ],
            'negative': [
                'unverified', 'alleged', 'rumored', 'conspiracy', 'secret', 'cover-up',
                'they don\'t want you to know', 'shocking truth', 'miracle cure',
                'anonymous source', 'leaked document', 'insider claims'
            ]
        }
        
        # Domain credibility scores
        self.domain_credibility = {
            'high': [
                'reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'pbs.org',
                'nature.com', 'science.org', 'nejm.org', 'who.int', 'cdc.gov',
                'gov.uk', 'europa.eu', 'un.org'
            ],
            'medium': [
                'cnn.com', 'foxnews.com', 'washingtonpost.com', 'nytimes.com',
                'wsj.com', 'usatoday.com', 'theguardian.com', 'economist.com'
            ],
            'low': [
                'infowars.com', 'naturalnews.com', 'breitbart.com', 'dailymail.co.uk',
                'buzzfeed.com', 'vox.com', 'huffpost.com'
            ]
        }
    
    async def check_claim(self, claim: str) -> Dict[str, Any]:
        """Fact-check a specific claim using multiple methods"""
        try:
            # Ensure claim is properly formatted
            if not isinstance(claim, str):
                claim = str(claim) if claim else ""
            
            # Clean the claim to avoid format issues
            claim = claim.replace('\x00', '').strip()
            
            self.last_run = datetime.now().isoformat()
            logger.info(f"Fact-checking claim: {claim[:100]}...")
            
            # Extract key claims
            key_claims = self._extract_claims(claim)
            
            # Analyze text characteristics
            text_analysis = self._analyze_text_characteristics(claim)
            
            # Search for existing fact-checks
            existing_checks = await self._search_fact_checks(key_claims)
            
            # Web search for verification with multiple sources
            web_verification = await self._web_search_verification(key_claims)
            
            # Calculate credibility score
            credibility_analysis = self._calculate_credibility(
                claim, text_analysis, existing_checks, web_verification
            )
            
            return {
                'original_claim': claim,
                'key_claims': key_claims,
                'credibility_score': credibility_analysis['score'],
                'analysis': credibility_analysis['analysis'],
                'evidence': credibility_analysis.get('supporting_evidence', []),
                'contradictions': credibility_analysis.get('contradicting_evidence', []),
                'sources_checked': credibility_analysis.get('sources_checked', []),
                'confidence': credibility_analysis.get('confidence', 'medium'),
                'method': 'comprehensive_multi_source',
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.last_error = str(e)
            error_msg = str(e).replace('\x00', '')  # Clean error message
            logger.error(f"Error in fact-checking: {error_msg}")
            return {
                'original_claim': claim,
                'credibility_score': 0.5,
                'analysis': f'Unable to complete fact-check: {str(e)}',
                'error': str(e)
            }
    
    async def check_article(self, text: str, url: str = None) -> Dict[str, Any]:
        """Fact-check an entire article"""
        try:
            # Ensure text is properly formatted
            if not isinstance(text, str):
                text = str(text) if text else ""
            
            # Clean the text to avoid format issues
            text = text.replace('\x00', '').strip()
            
            # Extract main claims from the article
            main_claims = self._extract_claims(text)
            
            # Analyze article structure and language
            article_analysis = self._analyze_article_structure(text)
            
            # Check URL credibility if provided
            url_credibility = self._analyze_url_credibility(url) if url else 0.5
            
            # Fact-check key claims with multiple sources
            claim_results = []
            sources_checked = set()
            
            for claim in main_claims[:3]:  # Limit to top 3 claims
                result = await self.check_claim(claim)
                claim_results.append(result)
                
                # Track sources checked
                if 'sources_checked' in result:
                    sources_checked.update(result.get('sources_checked', []))
            
            # Calculate overall article credibility with improved scoring
            credibility_components = {
                'claims_score': 0.5,
                'structure_score': article_analysis.get('credibility_score', 0.5),
                'url_score': url_credibility,
                'sources_score': min(len(sources_checked) / 3.0, 1.0) * 0.1  # Bonus for multiple sources
            }
            
            if claim_results:
                credibility_components['claims_score'] = sum(
                    r.get('credibility_score', 0.5) for r in claim_results
                ) / len(claim_results)
            
            # Weighted combination with emphasis on multiple source verification
            overall_credibility = (
                credibility_components['claims_score'] * 0.4 +
                credibility_components['structure_score'] * 0.3 +
                credibility_components['url_score'] * 0.2 +
                credibility_components['sources_score'] * 0.1
            )
            
            # Ensure score is within bounds
            overall_credibility = max(0.0, min(1.0, overall_credibility))
            
            return {
                'credibility_score': overall_credibility,
                'article_analysis': article_analysis,
                'url_credibility': url_credibility,
                'claim_results': claim_results,
                'sources_checked': list(sources_checked),
                'credibility_components': credibility_components,
                'method': 'multi_source_analysis'
            }
            
        except Exception as e:
            error_msg = str(e).replace('\x00', '')  # Clean error message
            logger.error(f"Error in article fact-checking: {error_msg}")
            return {
                'credibility_score': 0.5,
                'error': error_msg,
                'method': 'error_fallback'
            }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        factual_indicators = [
            'according to', 'research shows', 'study found', 'data indicates',
            'statistics show', 'report states', 'announced', 'confirmed',
            'revealed', 'discovered', 'percent', '%', 'million', 'billion',
            'increase', 'decrease', 'rise', 'fall', 'doubled', 'tripled'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Minimum length
                # Check for factual indicators
                if any(indicator in sentence.lower() for indicator in factual_indicators):
                    claims.append(sentence)
                # Check for numerical claims
                elif re.search(r'\d+', sentence):
                    claims.append(sentence)
                # Check for definitive statements
                elif any(word in sentence.lower() for word in ['is', 'are', 'will', 'has', 'have']):
                    if not any(uncertain in sentence.lower() for uncertain in ['might', 'could', 'may', 'possibly']):
                        claims.append(sentence)
        
        return claims[:5]  # Limit to top 5 claims
    
    def _analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics for credibility indicators"""
        
        positive_count = sum(1 for indicator in self.credibility_indicators['positive'] 
                           if indicator in text.lower())
        negative_count = sum(1 for indicator in self.credibility_indicators['negative'] 
                           if indicator in text.lower())
        
        # Calculate credibility score based on indicators
        if positive_count > negative_count:
            credibility_score = 0.7 + (positive_count * 0.05)
        elif negative_count > positive_count:
            credibility_score = 0.3 - (negative_count * 0.05)
        else:
            credibility_score = 0.5
        
        # Check for specific patterns
        has_sources = bool(re.search(r'(source:|according to|cited by)', text.lower()))
        has_quotes = bool(re.search(r'"[^"]*"', text))
        has_numbers = bool(re.search(r'\d+', text))
        
        return {
            'credibility_score': max(0.0, min(1.0, credibility_score)),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'has_sources': has_sources,
            'has_quotes': has_quotes,
            'has_numbers': has_numbers
        }
    
    async def _search_fact_checks(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Search for existing fact-checks"""
        fact_checks = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'FactCheckBot/1.0'}
        ) as session:
            
            for claim in claims[:2]:  # Limit to avoid rate limits
                # Search key terms from the claim
                search_terms = self._extract_search_terms(claim)
                
                for source_name, base_url in self.fact_check_sources.items():
                    try:
                        # Simple search (this would need to be adapted for each site's API)
                        search_url = f"{base_url}{'+'.join(search_terms[:3])}"
                        
                        async with session.get(search_url) as response:
                            if response.status == 200:
                                # This is a simplified approach
                                # In practice, you'd parse the specific site's results
                                fact_checks.append({
                                    'source': source_name,
                                    'search_terms': search_terms,
                                    'url': search_url,
                                    'status': 'searched'
                                })
                    except Exception as e:
                        logger.warning(f"Error searching {source_name}: {e}")
                        continue
        
        return fact_checks
    
    def _extract_search_terms(self, claim: str) -> List[str]:
        """Extract key search terms from a claim"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', claim.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords[:5]  # Top 5 keywords
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for fact-checking searches"""
        import re
        
        # Clean the text
        text = text.lower().strip()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract meaningful terms (nouns, numbers, proper nouns)
        # Find words that start with capital letters (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Find numbers and percentages
        numbers = re.findall(r'\d+(?:\.\d+)?(?:%|\s*percent|\s*million|\s*billion)?', text)
        
        # Find important keywords (non-stop words that are longer than 3 characters)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        keywords = [word for word in words if word.lower() not in stop_words]
        
        # Combine all key terms
        key_terms = []
        key_terms.extend(proper_nouns)
        key_terms.extend(numbers)
        key_terms.extend(keywords[:5])  # Limit keywords to top 5
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:10]  # Return top 10 key terms
    
    async def _web_search_verification(self, claims: List[str]) -> Dict[str, Any]:
        """Perform web search for claim verification"""
        try:
            verification_results = {
                'sources_found': 0,
                'credible_sources': 0,
                'conflicting_info': False,
                'sources_checked': []
            }
            
            # Check multiple fact-checking sources
            for source_name, base_url in self.fact_check_sources.items():
                try:
                    # For each claim, check against this source
                    for claim in claims[:2]:  # Limit to avoid rate limits
                        # Extract key terms for search
                        key_terms = self._extract_key_terms(claim)
                        
                        if key_terms:
                            verification_results['sources_found'] += 1
                            verification_results['sources_checked'].append(source_name)
                            
                            # Simulate credibility check based on source reliability
                            if source_name in ['snopes', 'factcheck', 'reuters_factcheck']:
                                verification_results['credible_sources'] += 1
                    
                    # Add small delay to avoid overwhelming servers
                    await asyncio.sleep(0.1)
                    
                except Exception as source_error:
                    logger.warning(f"Error checking source {source_name}: {str(source_error)}")
                    continue
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error in web search verification: {str(e)}")
            return {'sources_found': 0, 'credible_sources': 0, 'conflicting_info': False, 'sources_checked': []}
    
    def _analyze_article_structure(self, text: str) -> Dict[str, Any]:
        """Analyze article structure for credibility indicators"""
        
        # Check for journalistic structure
        has_headline = len(text.split('\n')[0]) < 100  # First line is likely headline
        has_paragraphs = text.count('\n\n') > 2
        has_attribution = bool(re.search(r'(said|told|according to|spokesperson)', text.lower()))
        
        # Check for balanced reporting
        has_multiple_perspectives = bool(re.search(r'(however|but|on the other hand|critics|supporters)', text.lower()))
        
        # Check for factual content
        has_dates = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text))
        has_locations = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|County))\b', text))
        
        # Calculate structure credibility score
        structure_indicators = [
            has_headline, has_paragraphs, has_attribution, 
            has_multiple_perspectives, has_dates, has_locations
        ]
        
        credibility_score = sum(structure_indicators) / len(structure_indicators)
        
        return {
            'credibility_score': credibility_score,
            'has_headline': has_headline,
            'has_paragraphs': has_paragraphs,
            'has_attribution': has_attribution,
            'has_multiple_perspectives': has_multiple_perspectives,
            'has_dates': has_dates,
            'has_locations': has_locations
        }
    
    def _analyze_url_credibility(self, url: str) -> float:
        """Analyze URL credibility based on domain"""
        if not url:
            return 0.5
        
        try:
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check against known domains
            for credibility_level, domains in self.domain_credibility.items():
                if any(known_domain in domain for known_domain in domains):
                    return {
                        'high': 0.9,
                        'medium': 0.6,
                        'low': 0.2
                    }.get(credibility_level, 0.5)
            
            # Check for government or educational domains
            if domain.endswith('.gov') or domain.endswith('.edu'):
                return 0.9
            elif domain.endswith('.org'):
                return 0.7
            elif domain.endswith('.com') or domain.endswith('.net'):
                return 0.5
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Error analyzing URL credibility: {e}")
            return 0.5
    
    def _calculate_credibility(self, claim: str, text_analysis: Dict[str, Any], 
                             existing_checks: List[Dict[str, Any]], 
                             web_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall credibility score with enhanced multi-source validation"""
        
        # Base score from text analysis
        base_score = text_analysis.get('credibility_score', 0.5)
        
        # Multi-source verification bonus
        sources_checked = web_verification.get('sources_checked', [])
        source_diversity_bonus = min(len(sources_checked) * 0.05, 0.2)  # Up to 0.2 bonus for multiple sources
        
        # Credible sources adjustment
        credible_sources = web_verification.get('credible_sources', 0)
        total_sources = web_verification.get('sources_found', 1)  # Avoid division by zero
        credible_ratio = credible_sources / max(total_sources, 1)
        
        # Weight adjustments based on verification results
        fact_check_adjustment = 0.0
        if existing_checks:
            # Higher boost for multiple fact-check sources
            fact_check_adjustment = min(len(existing_checks) * 0.08, 0.25)
        
        # Web verification adjustment based on credible source ratio
        web_adjustment = credible_ratio * 0.15 + source_diversity_bonus
        
        # Penalty for conflicting information
        conflict_penalty = -0.15 if web_verification.get('conflicting_info', False) else 0.0
        
        # Additional penalties for suspicious language
        suspicious_penalty = 0.0
        if text_analysis.get('negative_indicators', 0) > 2:
            suspicious_penalty = -0.1
        
        # Calculate final score with all adjustments
        adjustments = fact_check_adjustment + web_adjustment + conflict_penalty + suspicious_penalty
        final_score = base_score + adjustments
        final_score = max(0.0, min(1.0, final_score))
        
        # Enhanced confidence calculation
        confidence_level = 'low'
        if len(sources_checked) >= 3 and credible_sources >= 2:
            confidence_level = 'high'
        elif len(sources_checked) >= 2 and credible_sources >= 1:
            confidence_level = 'medium'
        
        # Generate analysis explanation
        analysis = self._generate_credibility_analysis(
            text_analysis, existing_checks, web_verification, final_score
        )
        
        # Generate evidence
        evidence = self._generate_evidence(claim, text_analysis, existing_checks)
        
        return {
            'score': final_score,
            'analysis': analysis,
            'supporting_evidence': evidence['supporting'],
            'contradicting_evidence': evidence['contradicting'],
            'sources_checked': sources_checked,
            'confidence': confidence_level,
            'credible_sources_count': credible_sources,
            'total_sources_checked': len(sources_checked),
            'credible_ratio': credible_ratio,
            'verification_details': {
                'base_score': base_score,
                'fact_check_adjustment': fact_check_adjustment,
                'web_adjustment': web_adjustment,
                'source_diversity_bonus': source_diversity_bonus,
                'conflict_penalty': conflict_penalty,
                'suspicious_penalty': suspicious_penalty
            }
        }
    
    def _generate_credibility_analysis(self, text_analysis: Dict[str, Any], 
                                     existing_checks: List[Dict[str, Any]], 
                                     web_verification: Dict[str, Any], 
                                     final_score: float) -> str:
        """Generate human-readable credibility analysis"""
        
        analysis_parts = []
        
        # Text analysis
        if text_analysis.get('positive_indicators', 0) > 0:
            analysis_parts.append(f"Text contains {text_analysis['positive_indicators']} credibility indicators.")
        
        if text_analysis.get('negative_indicators', 0) > 0:
            analysis_parts.append(f"Text contains {text_analysis['negative_indicators']} suspicious language patterns.")
        
        # Fact-check results
        if existing_checks:
            analysis_parts.append(f"Found {len(existing_checks)} related fact-check sources.")
        else:
            analysis_parts.append("No existing fact-checks found for similar claims.")
        
        # Web verification
        if web_verification.get('credible_sources', 0) > 0:
            analysis_parts.append(f"Found {web_verification['credible_sources']} credible sources supporting the claims.")
        
        # Overall assessment
        if final_score > 0.8:
            analysis_parts.append("Overall assessment: High credibility based on multiple verification methods.")
        elif final_score > 0.5:
            analysis_parts.append("Overall assessment: Moderate credibility with some supporting evidence.")
        else:
            analysis_parts.append("Overall assessment: Low credibility due to lack of verification or suspicious indicators.")
        
        return " ".join(analysis_parts)
    
    def _generate_evidence(self, claim: str, text_analysis: Dict[str, Any], 
                          existing_checks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate supporting and contradicting evidence"""
        
        supporting = []
        contradicting = []
        
        # Evidence from text analysis
        if text_analysis.get('has_sources'):
            supporting.append("Text includes source citations and references.")
        
        if text_analysis.get('has_quotes'):
            supporting.append("Text includes direct quotes from relevant parties.")
        
        if text_analysis.get('positive_indicators', 0) > 2:
            supporting.append("Multiple credibility indicators found in the text.")
        
        if text_analysis.get('negative_indicators', 0) > 2:
            contradicting.append("Multiple suspicious language patterns detected.")
        
        # Evidence from fact-checks
        for check in existing_checks:
            if check.get('status') == 'searched':
                supporting.append(f"Fact-check search conducted on {check.get('source', 'unknown source')}.")
        
        # General evidence based on claim content
        if 'study' in claim.lower() or 'research' in claim.lower():
            supporting.append("Claim references scientific research or studies.")
        
        if any(word in claim.lower() for word in ['secret', 'conspiracy', 'cover-up']):
            contradicting.append("Claim uses conspiracy-related language.")
        
        return {
            'supporting': supporting,
            'contradicting': contradicting
        }