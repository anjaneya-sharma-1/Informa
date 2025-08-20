"""Minimal fact check pipeline: sentiment + bias + external verdict.
Avoids full news collection workflow; used by Fact Checking tab.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from .sentiment_analyzer import SentimentAnalyzerAgent
from .bias_detector import BiasDetectorAgent
from .fact_check_verdict import FactCheckVerdictAgent

logger = logging.getLogger(__name__)


class FactCheckPipeline:
    def __init__(self, config):
        self.config = config
        # Reuse lightweight agents
        self.sentiment = SentimentAnalyzerAgent(config)
        self.bias = BiasDetectorAgent(config)
        self.verdict = FactCheckVerdictAgent(config)

    async def run(self, claim: str) -> Dict[str, Any]:
        claim = (claim or "").strip()
        if not claim:
            return {
                'original_claim': claim,
                'error': 'empty_claim',
                'timestamp': datetime.utcnow().isoformat()
            }
        try:
            # Parallelize sentiment + bias + verdict
            import asyncio
            sentiment_task = asyncio.create_task(self.sentiment.analyze(claim))
            bias_task = asyncio.create_task(self.bias.analyze_bias(claim))
            verdict_task = asyncio.create_task(self.verdict.get_verdicts_for_text(claim))
            sentiment_res, bias_res, verdict_res = await asyncio.gather(sentiment_task, bias_task, verdict_task)
            return {
                'original_claim': claim,
                'sentiment': sentiment_res,
                'bias': bias_res,
                'fact_check': verdict_res,
                'overall_verdict': verdict_res.get('overall_verdict'),
                'timestamp': datetime.utcnow().isoformat(),
                'method': 'sentiment_bias_verdict_pipeline'
            }
        except Exception as e:
            logger.error(f"FactCheckPipeline error: {e}")
            return {
                'original_claim': claim,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'method': 'pipeline_error'
            }
