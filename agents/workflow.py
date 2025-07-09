import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .news_collector import NewsCollectorAgent
from .sentiment_analyzer import SentimentAnalyzerAgent
from .bias_detector import BiasDetectorAgent
from .fact_checker import FactCheckerAgent
from .rag_agent import RAGAgent
from utils.database import VectorDatabase
from utils.config import Config

logger = logging.getLogger(__name__)

class NewsAnalysisWorkflow:
    """Simplified workflow orchestrating all agents"""
    
    def __init__(self, config: Config, database: VectorDatabase):
        self.config = config
        self.database = database
        
        # Initialize agents
        self.news_collector = NewsCollectorAgent(config)
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.bias_detector = BiasDetectorAgent()
        self.fact_checker = FactCheckerAgent(config)
        self.rag_agent = RAGAgent(config, database)
        
        # Workflow state
        self.execution_logs: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_executions': 0,
            'successful_executions': 0,
            'avg_processing_time': 0.0,
            'articles_processed': 0
        }
        
        logger.info("NewsAnalysisWorkflow initialized successfully")
    
    async def run_complete_analysis(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete news analysis workflow sequentially"""
        start_time = datetime.now()
        execution_id = f"exec_{int(start_time.timestamp())}"
        
        try:
            logger.info(f"Starting workflow execution {execution_id}")
            
            # Step 1: Collect news
            logger.info("Step 1: Collecting news...")
            topics = user_input.get('topics', ['technology'])
            sources = user_input.get('sources', ['bbc'])
            max_articles = user_input.get('max_articles', 20)
            
            articles = await self.news_collector.collect_news(
                topics=topics,
                sources=sources,
                max_articles=max_articles,
                filter_type='latest'
            )
            
            if not articles:
                return {
                    'success': False,
                    'error': 'No articles collected',
                    'execution_id': execution_id
                }
            
            logger.info(f"Collected {len(articles)} articles")
            
            # Step 2: Analyze sentiment
            logger.info("Step 2: Analyzing sentiment...")
            sentiment_results = {}
            for article in articles:
                try:
                    title = str(article.get('title', '') or '')
                    content = str(article.get('content', '') or '')
                    text = f"{title} {content}"
                    
                    sentiment = await self.sentiment_analyzer.analyze(text)
                    sentiment_results[article['id']] = {
                        'sentiment': sentiment
                    }
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for article {article['id']}: {e}")
                    sentiment_results[article['id']] = {
                        'sentiment': {'label': 'neutral', 'score': 0.0}
                    }
            
            # Step 3: Detect bias
            logger.info("Step 3: Detecting bias...")
            bias_results = {}
            for article in articles:
                try:
                    title = str(article.get('title', '') or '')
                    content = str(article.get('content', '') or '')
                    text = f"{title} {content}"
                    source = str(article.get('source', '') or '')
                    
                    bias_analysis = await self.bias_detector.analyze_bias(text, source)
                    bias_results[article['id']] = {
                        'bias_score': bias_analysis.get('overall_bias_score', 0),
                        'bias_details': bias_analysis
                    }
                except Exception as e:
                    logger.error(f"Error detecting bias for article {article['id']}: {e}")
                    bias_results[article['id']] = {
                        'bias_score': 0.5,
                        'bias_details': {}
                    }
            
            # Step 4: Fact check (limited to first 3 articles)
            logger.info("Step 4: Fact checking...")
            fact_check_results = {}
            for article in articles[:3]:  # Limit to first 3 articles
                try:
                    title = article.get('title', '') or ''
                    content = article.get('content', '') or ''
                    text = f"{title} {content}"
                    
                    fact_check = await self.fact_checker.check_article(text, article.get('url'))
                    fact_check_results[article['id']] = {
                        'credibility_score': fact_check.get('credibility_score', 0.5),
                        'fact_check_details': fact_check
                    }
                except Exception as e:
                    logger.error(f"Error fact-checking article {article['id']}: {e}")
                    fact_check_results[article['id']] = {
                        'credibility_score': 0.5,
                        'fact_check_details': {}
                    }
            
            # Step 5: Store results
            logger.info("Step 5: Storing results...")
            stored_count = 0
            analysis_results = {}
            
            for article in articles:
                try:
                    # Combine all analysis results
                    article_analysis = {}
                    if article['id'] in sentiment_results:
                        article_analysis.update(sentiment_results[article['id']])
                    if article['id'] in bias_results:
                        article_analysis.update(bias_results[article['id']])
                    if article['id'] in fact_check_results:
                        article_analysis.update(fact_check_results[article['id']])
                    
                    analysis_results[article['id']] = article_analysis
                    
                    # Store in database
                    article_data = {
                        'id': article['id'],
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('published_at', ''),
                        'analysis': article_analysis
                    }
                    
                    self.database.add_article(article_data, article_analysis)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing article {article['id']}: {e}")
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update performance metrics
            self.performance_metrics['total_executions'] += 1
            self.performance_metrics['successful_executions'] += 1
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics['avg_processing_time'] * (self.performance_metrics['total_executions'] - 1) + duration) /
                self.performance_metrics['total_executions']
            )
            self.performance_metrics['articles_processed'] += len(articles)
            
            # Log execution
            execution_log = {
                'id': execution_id,
                'timestamp': start_time.isoformat(),
                'duration': duration,
                'status': 'success',
                'articles_processed': len(articles),
                'errors': [],
                'metrics': {
                    'articles_collected': len(articles),
                    'sentiment_analyzed': len(sentiment_results),
                    'bias_analyzed': len(bias_results),
                    'fact_checked': len(fact_check_results),
                    'stored': stored_count
                }
            }
            
            self.execution_logs.append(execution_log)
            
            # Return results
            return {
                'success': True,
                'articles': articles,
                'analysis_results': analysis_results,
                'workflow_summary': execution_log['metrics'],
                'execution_id': execution_id,
                'duration': duration,
                'errors': []
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            
            # Log failed execution
            execution_log = {
                'id': execution_id,
                'timestamp': start_time.isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'failed',
                'error': error_msg,
                'articles_processed': 0
            }
            
            self.execution_logs.append(execution_log)
            
            return {
                'success': False,
                'error': error_msg,
                'execution_id': execution_id
            }
    
    async def fact_check_claim(self, claim: str) -> Dict[str, Any]:
        """Fact-check a specific claim"""
        try:
            return await self.fact_checker.check_claim(claim)
        except Exception as e:
            logger.error(f"Error fact-checking claim: {e}")
            return None
    
    async def answer_question(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Answer a question using RAG"""
        try:
            return await self.rag_agent.answer_question(question, chat_history)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if all components are healthy"""
        try:
            # Check each agent
            agents_healthy = all([
                self.news_collector.health_check(),
                self.sentiment_analyzer.health_check(),
                self.bias_detector.health_check(),
                self.fact_checker.health_check(),
                self.rag_agent.health_check()
            ])
            
            # Check database
            database_healthy = self.database.health_check()
            
            return agents_healthy and database_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of individual components"""
        try:
            return {
                'News Collector': self.news_collector.health_check(),
                'Sentiment Analyzer': self.sentiment_analyzer.health_check(),
                'Bias Detector': self.bias_detector.health_check(),
                'Fact Checker': self.fact_checker.health_check(),
                'RAG Agent': self.rag_agent.health_check(),
                'Database': self.database.health_check()
            }
        except Exception as e:
            logger.error(f"Error getting component status: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        success_rate = (
            self.performance_metrics['successful_executions'] / 
            max(self.performance_metrics['total_executions'], 1)
        )
        
        return {
            'total_executions': self.performance_metrics['total_executions'],
            'success_rate': success_rate,
            'avg_processing_time': self.performance_metrics['avg_processing_time'],
            'articles_per_hour': self._calculate_articles_per_hour(),
            'memory_usage': self._get_memory_usage()
        }
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of each agent"""
        return {
            'News Collector': {
                'active': self.news_collector.health_check(),
                'last_run': getattr(self.news_collector, 'last_run', 'Never'),
                'error': getattr(self.news_collector, 'last_error', None)
            },
            'Sentiment Analyzer': {
                'active': self.sentiment_analyzer.health_check(),
                'last_run': getattr(self.sentiment_analyzer, 'last_run', 'Never'),
                'error': getattr(self.sentiment_analyzer, 'last_error', None)
            },
            'Bias Detector': {
                'active': self.bias_detector.health_check(),
                'last_run': getattr(self.bias_detector, 'last_run', 'Never'),
                'error': getattr(self.bias_detector, 'last_error', None)
            },
            'Fact Checker': {
                'active': self.fact_checker.health_check(),
                'last_run': getattr(self.fact_checker, 'last_run', 'Never'),
                'error': getattr(self.fact_checker, 'last_error', None)
            },
            'RAG Agent': {
                'active': self.rag_agent.health_check(),
                'last_run': getattr(self.rag_agent, 'last_run', 'Never'),
                'error': getattr(self.rag_agent, 'last_error', None)
            }
        }
    
    def get_execution_logs(self) -> List[Dict[str, Any]]:
        """Get workflow execution logs"""
        return self.execution_logs.copy()
    
    def restart(self):
        """Restart the workflow"""
        logger.info("Restarting workflow...")
        # Reset metrics and logs
        self.execution_logs.clear()
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'avg_processing_time': 0.0,
            'articles_processed': 0
        }
        
        # Restart agents
        for agent in [self.news_collector, self.sentiment_analyzer, self.bias_detector, 
                     self.fact_checker, self.rag_agent]:
            if hasattr(agent, 'restart'):
                agent.restart()
    
    def clear_cache(self):
        """Clear workflow cache"""
        logger.info("Clearing workflow cache...")
        # Clear agent caches
        for agent in [self.news_collector, self.sentiment_analyzer, self.bias_detector, 
                     self.fact_checker, self.rag_agent]:
            if hasattr(agent, 'clear_cache'):
                agent.clear_cache()
    
    def export_logs(self) -> str:
        """Export execution logs as JSON"""
        return json.dumps({
            'execution_logs': self.execution_logs,
            'performance_metrics': self.performance_metrics,
            'export_timestamp': datetime.now().isoformat()
        }, indent=2)
    
    def _calculate_articles_per_hour(self) -> int:
        """Calculate articles processed per hour"""
        if not self.execution_logs:
            return 0
        
        # Calculate based on recent executions
        recent_logs = self.execution_logs[-10:]  # Last 10 executions
        total_articles = sum(log.get('articles_processed', 0) for log in recent_logs)
        total_time_hours = sum(log.get('duration', 0) for log in recent_logs) / 3600
        
        if total_time_hours > 0:
            return int(total_articles / total_time_hours)
        return 0
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
