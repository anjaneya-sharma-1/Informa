import logging
import aiohttp
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FactCheckVerdictAgent:
    """Lightweight fact-check verdict agent using Google Fact Check Tools API.
    Extracts simple factual claims and queries external API for verdicts.
    Requires FACTCHECK_API_KEY or GOOGLE_FACTCHECK_API_KEY in environment/secrets.
    """

    FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, config=None):
        self.config = config
        api_key = None
        # Attempt config api_keys mapping if exists
        try:
            if config and hasattr(config, 'get_api_key'):
                api_key = config.get_api_key('factcheck') or None
        except Exception:
            pass
        import os
        self.api_key = api_key or os.getenv('FACTCHECK_API_KEY') or os.getenv('GOOGLE_FACTCHECK_API_KEY')
        if not self.api_key:
            logger.warning("No FACTCHECK_API_KEY set; fact-check verdicts will be UNVERIFIED.")

    async def get_verdicts_for_text(self, text: str, max_claims: int = 3) -> Dict[str, Any]:
        claims = self._extract_claims(text)[:max_claims]
        verdicts = []
        for claim in claims:
            api_result = await self._query_factcheck_api(claim)
            verdict = self._select_best_verdict(api_result)
            verdicts.append({
                'claim': claim,
                **verdict
            })
        overall = self._aggregate_overall(verdicts)
        return {
            'overall_verdict': overall,
            'claims': verdicts
        }

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]', text or '')
        out = []
        indicators = ['according to', 'reported', 'announced', 'claims', 'said', 'will', 'is', 'are', 'was', 'were', '%', 'million', 'billion', 'study', 'research']
        for s in sentences:
            s2 = s.strip()
            if len(s2) < 30:
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
                    if resp.status != 200:
                        text = await resp.text()
                        logger.debug(f"FactCheck API non-200 {resp.status}: {text[:200]}")
                        return {}
                    return await resp.json()
        except Exception as e:
            logger.debug(f"FactCheck API error: {e}")
            return {}

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
