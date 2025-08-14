import logging
import aiohttp
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FactCheckVerdictAgent:
    """Lightweight fact-check verdict agent using Google Fact Check Tools API.
    Extracts simple factual claims and queries external API for verdicts.
    Requires FACTCHECK_API_KEY or GOOGLE_FACTCHECK_API_KEY in environment/secrets.

    Added diagnostics:
        - Debug mode (env FACTCHECK_DEBUG=1) returns raw API status and sample JSON.
        - Fallback: if individual claim queries return no data, tries the full original text once.
        - Relaxed minimum sentence length to capture shorter claims.
    """

    FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, config=None, simple_mode: bool = True):
        import os
        self.config = config
        self.debug = os.getenv('FACTCHECK_DEBUG', '0') == '1'
        # Simple mode: one direct API call using the full user input (recommended if prior extraction produced few matches)
        self.simple_mode = simple_mode or (os.getenv('FACTCHECK_SIMPLE', '1') == '1')
        api_key = None
        # Attempt config api_keys mapping if exists
        try:
            if config and hasattr(config, 'get_api_key'):
                api_key = config.get_api_key('factcheck') or None
        except Exception:
            if self.debug:
                logger.debug("Config get_api_key('factcheck') failed", exc_info=True)
        self.api_key = api_key or os.getenv('FACTCHECK_API_KEY') or os.getenv('GOOGLE_FACTCHECK_API_KEY')
        if not self.api_key:
            logger.warning("No FACTCHECK_API_KEY set; fact-check verdicts will be UNVERIFIED.")
        elif self.debug:
            logger.info(f"FactCheckVerdictAgent initialized with key prefix: {self.api_key[:6]}***")

    async def get_verdicts_for_text(self, text: str, max_claims: int = 3) -> Dict[str, Any]:
        """Return verdicts for input text.

        Modes:
          - simple_mode=True: single API call with the raw (truncated) text. Each returned claimReview becomes an entry.
          - simple_mode=False: extract sub-claims and query each separately with fallback full text.
        """
        api_events = [] if self.debug else None
        if self.simple_mode:
            query_text = (text or '').strip()
            if not query_text:
                return {'overall_verdict': 'UNVERIFIED', 'claims': [], 'error': 'empty'}
            api_json = await self._query_factcheck_api(query_text[:256])
            claims_out = []
            if 'claims' in api_json:
                for c in api_json['claims'][:max_claims]:
                    reviews = c.get('claimReview', []) or []
                    if not reviews:
                        continue
                    for r in reviews[:1]:  # take top review per claim
                        tr = (r.get('textualRating') or '').lower()
                        claims_out.append({
                            'claim': (c.get('text') or query_text)[:280],
                            'verdict': self._map_textual_rating(tr),
                            'source': r.get('publisher', {}).get('name'),
                            'rating': r.get('textualRating'),
                            'url': r.get('url'),
                            'textual_rating': r.get('textualRating'),
                            'mode': 'simple'
                        })
            if not claims_out:
                claims_out.append({'claim': query_text[:280], 'verdict': 'UNVERIFIED', 'mode': 'simple'})
            overall = self._aggregate_overall(claims_out)
            out = {'overall_verdict': overall, 'claims': claims_out, 'mode': 'simple'}
            if self.debug:
                api_events.append({'simple_mode': True, 'has_claims': 'claims' in api_json, 'keys': list(api_json.keys())[:5]})
                out['debug'] = {'api_events': api_events}
            return out
        # Advanced (previous) mode
        claims = self._extract_claims(text)[:max_claims]
        verdicts = []
        for claim in claims:
            api_result = await self._query_factcheck_api(claim)
            if self.debug:
                api_events.append({'claim_fragment': claim[:40], 'has_claims': 'claims' in api_result, 'raw_keys': list(api_result.keys())[:5]})
            verdict = self._select_best_verdict(api_result)
            verdicts.append({'claim': claim, **verdict})
        if verdicts and all(v.get('verdict') == 'UNVERIFIED' for v in verdicts) and text and text[:280] not in [v['claim'] for v in verdicts]:
            api_result = await self._query_factcheck_api(text[:280])
            if self.debug:
                api_events.append({'fallback_full_text': True, 'has_claims': 'claims' in api_result})
            fallback_verdict = self._select_best_verdict(api_result)
            verdicts.append({'claim': text[:280], **fallback_verdict, 'fallback': True})
        overall = self._aggregate_overall(verdicts)
        out = {'overall_verdict': overall, 'claims': verdicts, 'mode': 'extracted'}
        if self.debug:
            out['debug'] = {'api_events': api_events}
        return out

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]', text or '')
        out = []
        indicators = ['according to', 'reported', 'announced', 'claims', 'said', 'will', 'is', 'are', 'was', 'were', '%', 'million', 'billion', 'study', 'research']
        for s in sentences:
            s2 = s.strip()
            # Relaxed minimum length to catch shorter factual assertions
            if len(s2) < 12:
                continue
            lower = s2.lower()
            if any(k in lower for k in indicators) or re.search(r'\d', s2):
                out.append(s2[:280])
        if not out and text:
            out = [text.strip()[:280]]
        return out

    async def _query_factcheck_api(self, claim: str) -> Dict[str, Any]:
        if not self.api_key:
            return {}
        params = {
            'query': claim[:256],
            'languageCode': 'en',
            'pageSize': 5,
            'key': self.api_key
        }
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.FACTCHECK_ENDPOINT, params=params) as resp:
                    status = resp.status
                    if status != 200:
                        text = await resp.text()
                        logger.warning(f"FactCheck API {status}: {text[:180]}")
                        return {'error': f'status_{status}', 'raw': text[:300]}
                    try:
                        data = await resp.json()
                        if self.debug:
                            data['_debug_status'] = status
                        return data
                    except Exception as je:
                        logger.warning(f"FactCheck API JSON parse error: {je}")
                        return {'error': 'json_parse', 'raw': await resp.text()[:300]}
        except Exception as e:
            logger.warning(f"FactCheck API request error: {e}")
            return {'error': str(e)}

    def _select_best_verdict(self, api_json: Dict[str, Any]) -> Dict[str, Any]:
        if not api_json or 'claims' not in api_json:
            return {'verdict': 'UNVERIFIED', 'source': None, 'rating': None, 'url': None, 'textual_rating': None}
        best = None
        for claim in api_json.get('claims', []):
            reviews = claim.get('claimReview', []) or []
            for r in reviews:
                tr = (r.get('textualRating') or '').lower()
                score = 0
                if 'true' in tr or 'correct' in tr or 'accurate' in tr:
                    score = 3
                elif 'mostly true' in tr:
                    score = 2.5
                elif 'partly' in tr or 'mixed' in tr or 'partially' in tr:
                    score = 2
                elif 'false' in tr or 'inaccurate' in tr or 'misleading' in tr:
                    score = 1
                if not best or score > best['__score']:
                    best = {
                        '__score': score,
                        'verdict': self._map_textual_rating(tr),
                        'source': r.get('publisher', {}).get('name'),
                        'rating': r.get('textualRating'),
                        'url': r.get('url'),
                        'textual_rating': r.get('textualRating')
                    }
        if not best:
            return {'verdict': 'UNVERIFIED', 'source': None, 'rating': None, 'url': None, 'textual_rating': None}
        best.pop('__score', None)
        return best

    def _map_textual_rating(self, textual: str) -> str:
        if 'false' in textual or 'inaccurate' in textual or 'misleading' in textual:
            return 'FALSE'
        if 'true' in textual or 'accurate' in textual or 'correct' in textual:
            return 'TRUE'
        if 'partly' in textual or 'mixed' in textual or 'partially' in textual:
            return 'MIXED'
        return 'UNVERIFIED'

    def _aggregate_overall(self, verdicts: List[Dict[str, Any]]) -> str:
        if not verdicts:
            return 'UNVERIFIED'
        has_true = any(v.get('verdict') == 'TRUE' for v in verdicts)
        has_false = any(v.get('verdict') == 'FALSE' for v in verdicts)
        if has_true and not has_false:
            return 'PARTIALLY VERIFIED'
        if has_true and has_false:
            return 'MIXED'
        if has_false and not has_true:
            return 'LIKELY FALSE'
        return 'UNVERIFIED'
