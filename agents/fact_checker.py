"""Lightweight fact checker wrapper retained for backward compatibility.
All legacy multi-factor credibility logic removed.
Provides simple verdict-oriented interface delegating to FactCheckVerdictAgent.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List
import re

from .fact_check_verdict import FactCheckVerdictAgent

logger = logging.getLogger(__name__)


class FactCheckerAgent:
    """Deprecated rich credibility checker replaced by lightweight verdict wrapper.

    Public methods:
      - check_claim(claim): returns overall_verdict + claims list
      - check_article(text, url): extracts claims then gets verdicts
    """

    def __init__(self, config=None):
        self.config = config
        self.verdict_agent = FactCheckVerdictAgent(config)
        self.last_run = None

    async def check_claim(self, claim: str) -> Dict[str, Any]:
        if not isinstance(claim, str):
            claim = str(claim) if claim else ""
        clean_claim = claim.replace("\x00", "").strip()
        self.last_run = datetime.now().isoformat()
        verdict_data = await self.verdict_agent.get_verdicts_for_text(clean_claim)
        return {
            'original_claim': clean_claim,
            'overall_verdict': verdict_data.get('overall_verdict'),
            'claims': verdict_data.get('claims', []),
            'method': 'external_fact_check_verdict',
            'checked_at': self.last_run
        }

    async def check_article(self, text: str, url: str = None) -> Dict[str, Any]:
        if not isinstance(text, str):
            text = str(text) if text else ""
        article_text = text.replace("\x00", "").strip()
        self.last_run = datetime.now().isoformat()
        verdict_data = await self.verdict_agent.get_verdicts_for_text(article_text)
        return {
            'overall_verdict': verdict_data.get('overall_verdict'),
            'claims': verdict_data.get('claims', []),
            'url': url,
            'method': 'external_fact_check_verdict',
            'checked_at': self.last_run
        }

    # Minimal helper retained for any residual imports expecting claim extraction
    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]', text or '')
        out = []
        for s in sentences:
            s2 = s.strip()
            if len(s2) < 30:
                continue
            if re.search(r'\d', s2):
                out.append(s2[:280])
        if not out and text:
            out = [text.strip()[:280]]
        return out[:5]