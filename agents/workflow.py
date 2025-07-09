import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from .news_collector import NewsCollectorAgent
from .sentiment_analyzer import SentimentAnalyzerAgent
from .bias_detector import BiasDetectorAgent
from .fact_checker import FactCheckerAgent
from .rag_agent import RAGAgent
from utils.database import VectorDatabase
from utils.config import Config

logger = logging.getLogger(__name__)

class NewsAnalysisWorkflow:
    """Main LangGraph workflow orchestrating all agents"""
    
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
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("NewsAnalysisWorkflow initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(Dict[str, Any])
        
        # Add nodes (agents)
        workflow.add_node("collect_news", self._collect_news_node)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment_node)
        workflow.add_node("detect_bias", self._detect_bias_node)
        workflow.add_node("fact_check", self._fact_check_node)
        workflow.add_node("store_results", self._store_results_node)
        
        # Define the workflow edges
        workflow.set_entry_point("collect_news")
        workflow.add_edge("collect_news", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "detect_bias")
        workflow.add_edge("detect_bias", "fact_check")
        workflow.add_edge("fact_check", "store_results")
        workflow.add_edge("store_results", END)
        
        return workflow.compile()
    
    async def _collect_news_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """News collection node"""
        try:
            logger.info("Starting news collection...")
            
            # Extract parameters from user input
            topics = state.get('user_input', {}).get('topics', ['technology'])
            sources = state.get('user_input', {}).get('sources', ['bbc'])
            max_articles = state.get('user_input', {}).get('max_articles', 20)
            filter_type = state.get('user_input', {}).get('filter_type', 'latest')
            
            # Collect news articles
            articles = await self.news_collector.collect_news(
                topics=topics,
                sources=sources,
                max_articles=max_articles,
                filter_type=filter_type
            )
            
            # Update state
            state['articles'] = articles
            state['metrics'] = state.get('metrics', {})
            state['metrics']['articles_collected'] = len(articles)
            state['errors'] = state.get('errors', [])
            
            logger.info(f"Collected {len(articles)} articles")
            
        except Exception as e:
            error_msg = f"Error in news collection: {str(e).replace(chr(0), '')}"
            state['errors'] = state.get('errors', []) + [error_msg]
            logger.error(error_msg)
        
        return state
    
    async def _analyze_sentiment_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sentiment analysis node"""
        try:
            logger.info("Starting sentiment analysis...")
            
            articles = state.get('articles', [])
            sentiment_results = {}
            
            for article in articles:
                # Combine title and content for analysis - ensure they are strings
                title = str(article.get('title', '') or '')
                content = str(article.get('content', '') or '')
                text = f"{title} {content}"
                
                # Analyze sentiment
                sentiment = await self.sentiment_analyzer.analyze(text)
                sentiment_results[article['id']] = {
                    'sentiment': sentiment
                }
            
            # Update analysis results
            analysis_results = state.get('analysis_results', {})
            for article_id, result in sentiment_results.items():
                if article_id not in analysis_results:
                    analysis_results[article_id] = {}
                analysis_results[article_id].update(result)
            
            state['analysis_results'] = analysis_results
            state['metrics'] = state.get('metrics', {})
            state['metrics']['sentiment_analyzed'] = len(sentiment_results)
            state['errors'] = state.get('errors', [])
            
            logger.info(f"Analyzed sentiment for {len(sentiment_results)} articles")
            
        except Exception as e:
            error_msg = f"Error in sentiment analysis: {str(e).replace(chr(0), '')}"
            state['errors'] = state.get('errors', []) + [error_msg]
            logger.error(error_msg)
        
        return state
    
    async def _detect_bias_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Bias detection node"""
        try:
            logger.info("Starting bias detection...")
            
            articles = state.get('articles', [])
            bias_results = {}
            
            for article in articles:
                # Combine title and content for analysis - ensure they are strings
                title = str(article.get('title', '') or '')
                content = str(article.get('content', '') or '')
                text = f"{title} {content}"
                source = str(article.get('source', '') or '')
                
                # Detect bias
                bias_analysis = await self.bias_detector.analyze_bias(text, source)
                bias_results[article['id']] = {
                    'bias_score': bias_analysis.get('overall_bias_score', 0),
                    'bias_details': bias_analysis
                }
            
            # Update analysis results
            analysis_results = state.get('analysis_results', {})
            for article_id, result in bias_results.items():
                if article_id not in analysis_results:
                    analysis_results[article_id] = {}
                analysis_results[article_id].update(result)
            
            state['analysis_results'] = analysis_results
            state['metrics'] = state.get('metrics', {})
            state['metrics']['bias_analyzed'] = len(bias_results)
            state['errors'] = state.get('errors', [])
            
            logger.info(f"Analyzed bias for {len(bias_results)} articles")
            
        except Exception as e:
            error_msg = f"Error in bias detection: {str(e).replace(chr(0), '')}"
            state['errors'] = state.get('errors', []) + [error_msg]
            logger.error(error_msg)
        
        return state
    
    async def _fact_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fact-checking node"""
        try:
            logger.info("Starting fact-checking...")
            
            articles = state.get('articles', [])
            fact_check_results = {}
            
            # Only fact-check a subset to avoid rate limits
            articles_to_check = articles[:5]  # Limit to first 5 articles
            
            for article in articles_to_check:
                # Combine title and content for fact-checking
                title = article.get('title', '') or ''
                content = article.get('content', '') or ''
                text = f"{title} {content}"
                
                # Fact-check the article
                fact_check = await self.fact_checker.check_article(text, article.get('url'))
                fact_check_results[article['id']] = {
                    'credibility_score': fact_check.get('credibility_score', 0.5),
                    'fact_check_details': fact_check
                }
            
            # Update analysis results
            analysis_results = state.get('analysis_results', {})
            for article_id, result in fact_check_results.items():
                if article_id not in analysis_results:
                    analysis_results[article_id] = {}
                analysis_results[article_id].update(result)
            
            state['analysis_results'] = analysis_results
            state['metrics'] = state.get('metrics', {})
            state['metrics']['fact_checked'] = len(fact_check_results)
            state['errors'] = state.get('errors', [])
            
            logger.info(f"Fact-checked {len(fact_check_results)} articles")
            
        except Exception as e:
            error_msg = f"Error in fact-checking: {str(e).replace(chr(0), '')}"
            state['errors'] = state.get('errors', []) + [error_msg]
            logger.error(error_msg)
        
        return state
    
    async def _store_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Store results in database node"""
        try:
            logger.info("Starting results storage...")
            
            articles = state.get('articles', [])
            analysis_results = state.get('analysis_results', {})
            
            # Store articles and analysis in vector database
            stored_count = 0
            for article in articles:
                try:
                    # Prepare article data with analysis
                    article_data = {
                        'id': article['id'],
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('published_at', ''),
                        'analysis': analysis_results.get(article['id'], {})
                    }
                    
                    # Store in database
                    self.database.add_article(article_data)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing article {article['id']}: {e}")
            
            state['metrics'] = state.get('metrics', {})
            state['metrics']['stored'] = stored_count
            state['errors'] = state.get('errors', [])
            
            logger.info(f"Stored {stored_count} articles in database")
            
        except Exception as e:
            error_msg = f"Error in results storage: {str(e).replace(chr(0), '')}"
            state['errors'] = state.get('errors', []) + [error_msg]
            logger.error(error_msg)
        
        return state
    
    async def run_complete_analysis(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete news analysis workflow"""
        start_time = datetime.now()
        execution_id = f"exec_{int(start_time.timestamp())}"
        
        try:
            logger.info(f"Starting workflow execution {execution_id}")
            
            # Initialize state as dictionary
            state = {
                'user_input': user_input,
                'articles': [],
                'analysis_results': {},
                'metrics': {},
                'errors': []
            }
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update performance metrics
            self.performance_metrics['total_executions'] += 1
            if not final_state.get('errors'):
                self.performance_metrics['successful_executions'] += 1
            
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics['avg_processing_time'] * (self.performance_metrics['total_executions'] - 1) + duration) /
                self.performance_metrics['total_executions']
            )
            
            self.performance_metrics['articles_processed'] += len(final_state.get('articles', []))
            
            # Log execution
            execution_log = {
                'id': execution_id,
                'timestamp': start_time.isoformat(),
                'duration': duration,
                'status': 'success' if not final_state.get('errors') else 'error',
                'articles_processed': len(final_state.get('articles', [])),
                'errors': final_state.get('errors', []),
                'metrics': final_state.get('metrics', {})
            }
            
            self.execution_logs.append(execution_log)
            
            # Return results
            return {
                'success': not final_state.get('errors'),
                'articles': final_state.get('articles', []),
                'analysis_results': final_state.get('analysis_results', {}),
                'workflow_summary': final_state.get('metrics', {}),
                'execution_id': execution_id,
                'duration': duration,
                'errors': final_state.get('errors', [])
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
