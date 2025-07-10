import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Local imports
from agents.news_collector import NewsCollectorAgent
from agents.sentiment_analyzer import SentimentAnalyzerAgent
from agents.bias_detector import BiasDetectorAgent
from agents.fact_checker import FactCheckerAgent
from utils.database import VectorDatabase
from utils.config import AppConfig

logger = logging.getLogger(__name__)

class WorkflowState:
    """State management for the news analysis workflow"""
    
    def __init__(self):
        # Configuration
        self.topics: List[str] = []
        self.sources: List[str] = []
        self.max_articles: int = 20
        
        # Data
        self.collected_articles: List[Dict[str, Any]] = []
        self.analyzed_articles: List[Dict[str, Any]] = []
        self.stored_articles: List[Dict[str, Any]] = []
        
        # LangGraph messaging
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {}
        
        # Status tracking
        self.errors: List[str] = []
        self.progress: int = 0
        self.current_step: str = ""
        self.start_time: str = datetime.now().isoformat()
        self.last_update_time: str = datetime.now().isoformat()

class NewsWorkflow:
    """Multi-agent news analysis workflow using LangGraph"""
    
    def __init__(self, config, database: VectorDatabase):
        self.config = config
        self.database = database
        
        # Initialize agents
        self.news_collector = NewsCollectorAgent(config)
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.bias_detector = BiasDetectorAgent()
        self.fact_checker = FactCheckerAgent(config)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper agent orchestration"""
        
        # Define the workflow graph with dynamic state management
        workflow = StateGraph(dict)
        
        # Add specialized agent nodes
        workflow.add_node("topic_selection", self._topic_selection_node)
        workflow.add_node("news_collection", self._news_collection_node)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
        workflow.add_node("bias_detection", self._bias_detection_node)
        workflow.add_node("content_synthesis", self._content_synthesis_node)
        workflow.add_node("vector_storage", self._vector_storage_node)
        workflow.add_node("completion", self._completion_node)
        
        # Define the workflow edges for proper orchestration
        workflow.add_edge("topic_selection", "news_collection")
        workflow.add_edge("news_collection", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "bias_detection")
        workflow.add_edge("bias_detection", "content_synthesis")
        workflow.add_edge("content_synthesis", "vector_storage")
        workflow.add_edge("vector_storage", "completion")
        workflow.add_edge("completion", END)
        
        # Set entry point
        workflow.set_entry_point("topic_selection")
        
        return workflow.compile()
    
    async def execute_workflow(
        self, 
        topics: List[str], 
        sources: List[str], 
        max_articles: int = 20,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Execute the complete news analysis workflow"""
        
        # Initialize state
        initial_state = {
            "topics": topics,
            "sources": sources,
            "max_articles": max_articles,
            "collected_articles": [],
            "analyzed_articles": [],
            "stored_articles": [],
            "errors": [],
            "progress": 0,
            "current_step": "",
            "progress_callback": progress_callback
        }
        
        try:
            logger.info(f"Starting workflow for topics: {topics}, sources: {sources}")
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"Workflow completed successfully. Processed {len(final_state.get('stored_articles', []))} articles")
            return final_state.get("stored_articles", [])
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _topic_selection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for topic selection and validation"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("🎯 Validating topics and sources...", 10)
            
            state["current_step"] = "topic_selection"
            state["progress"] = 10
            
            # Validate topics and sources
            valid_topics = []
            for topic in state["topics"]:
                if isinstance(topic, str) and topic.strip():
                    valid_topics.append(topic.strip().lower())
            
            valid_sources = []
            for source in state["sources"]:
                if isinstance(source, str) and source.strip():
                    valid_sources.append(source.strip().lower())
            
            if not valid_topics:
                valid_topics = ["technology"]  # Default topic
            
            if not valid_sources:
                valid_sources = ["bbc"]  # Default source
            
            state["topics"] = valid_topics
            state["sources"] = valid_sources
            
            logger.info(f"Topic selection complete: {valid_topics} from {valid_sources}")
            return state
            
        except Exception as e:
            error_msg = f"Topic selection error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _news_collection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for news collection from multiple sources"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("📰 Collecting news articles...", 30)
            
            state["current_step"] = "news_collection"
            state["progress"] = 30
            
            # Collect articles using news collector agent
            articles = await self.news_collector.collect_news(
                topics=state["topics"],
                sources=state["sources"],
                max_articles=state["max_articles"]
            )
            
            state["collected_articles"] = articles
            
            logger.info(f"Collected {len(articles)} articles")
            return state
            
        except Exception as e:
            error_msg = f"News collection error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for sentiment analysis using specialized agent"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("� Analyzing sentiment...", 50)
            
            state["current_step"] = "sentiment_analysis"
            state["progress"] = 50
            
            articles = state.get("collected_articles", [])
            analyzed_articles = []
            
            total_articles = len(articles)
            
            for i, article in enumerate(articles):
                try:
                    # Update progress for individual articles
                    if state.get("progress_callback") and total_articles > 0:
                        progress = 50 + (15 * i / total_articles)
                        state["progress_callback"](f"� Analyzing sentiment {i+1}/{total_articles}...", int(progress))
                    
                    # Perform sentiment analysis
                    title_content = f"{article.get('title', '')} {article.get('content', '')}"
                    sentiment_result = await self.sentiment_analyzer.analyze(title_content)
                    
                    # Add sentiment results to article
                    article_copy = article.copy()
                    article_copy.update({
                        'sentiment_label': sentiment_result.get('label', 'neutral'),
                        'sentiment_score': sentiment_result.get('score', 0.0),
                        'sentiment_confidence': sentiment_result.get('confidence', 0.0),
                        'sentiment_intensity': sentiment_result.get('intensity', 0.0),
                        'sentiment_method': sentiment_result.get('method', 'unknown')
                    })
                    
                    analyzed_articles.append(article_copy)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing sentiment for article {i}: {e}")
                    # Add article without sentiment analysis
                    analyzed_articles.append(article)
                    continue
            
            state["sentiment_analyzed_articles"] = analyzed_articles
            
            logger.info(f"Sentiment analysis complete for {len(analyzed_articles)} articles")
            return state
            
        except Exception as e:
            error_msg = f"Sentiment analysis error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _bias_detection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for bias detection using specialized agent"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("⚖️ Detecting bias...", 65)
            
            state["current_step"] = "bias_detection"
            state["progress"] = 65
            
            articles = state.get("sentiment_analyzed_articles", [])
            analyzed_articles = []
            
            total_articles = len(articles)
            
            for i, article in enumerate(articles):
                try:
                    # Update progress for individual articles
                    if state.get("progress_callback") and total_articles > 0:
                        progress = 65 + (15 * i / total_articles)
                        state["progress_callback"](f"⚖️ Detecting bias {i+1}/{total_articles}...", int(progress))
                    
                    # Perform bias detection
                    title_content = f"{article.get('title', '')} {article.get('content', '')}"
                    bias_result = await self.bias_detector.analyze_bias(title_content, article.get('source'))
                    
                    # Add bias results to article
                    article_copy = article.copy()
                    article_copy.update({
                        'bias_score': bias_result.get('overall_bias_score', 0.0),
                        'bias_breakdown': bias_result.get('bias_breakdown', {}),
                        'bias_method': bias_result.get('method', 'unknown')
                    })
                    
                    analyzed_articles.append(article_copy)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing bias for article {i}: {e}")
                    # Add article without bias analysis
                    analyzed_articles.append(article)
                    continue
            
            state["fully_analyzed_articles"] = analyzed_articles
            
            logger.info(f"Bias detection complete for {len(analyzed_articles)} articles")
            return state
            
        except Exception as e:
            error_msg = f"Bias detection error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _content_synthesis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for synthesizing and enhancing content analysis"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("🔗 Synthesizing analysis...", 80)
            
            state["current_step"] = "content_synthesis"
            state["progress"] = 80
            
            articles = state.get("fully_analyzed_articles", [])
            synthesized_articles = []
            
            for article in articles:
                try:
                    # Calculate overall article quality score
                    quality_score = self._calculate_article_quality(article)
                    
                    # Generate article summary if not present
                    if not article.get('summary'):
                        summary = self._generate_article_summary(article)
                        article['summary'] = summary
                    
                    # Add metadata
                    article['quality_score'] = quality_score
                    article['analysis_timestamp'] = datetime.now().isoformat()
                    
                    synthesized_articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Error synthesizing article: {e}")
                    synthesized_articles.append(article)
                    continue
            
            state["synthesized_articles"] = synthesized_articles
            
            logger.info(f"Content synthesis complete for {len(synthesized_articles)} articles")
            return state
            
        except Exception as e:
            error_msg = f"Content synthesis error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _vector_storage_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for storing articles in vector database"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("💾 Storing in vector database...", 90)
            
            state["current_step"] = "vector_storage"
            state["progress"] = 90
            
            stored_articles = []
            articles = state.get("synthesized_articles", [])
            
            for article in articles:
                try:
                    # Prepare comprehensive analysis data
                    analysis_data = {
                        'sentiment': {
                            'label': article.get('sentiment_label', 'neutral'),
                            'score': article.get('sentiment_score', 0.0),
                            'confidence': article.get('sentiment_confidence', 0.0),
                            'intensity': article.get('sentiment_intensity', 0.0)
                        },
                        'bias': {
                            'overall_score': article.get('bias_score', 0.0),
                            'breakdown': article.get('bias_breakdown', {})
                        },
                        'quality_score': article.get('quality_score', 0.5),
                        'credibility_score': article.get('credibility_score', 0.5)
                    }
                    
                    # Store in vector database
                    success = await self.database.store_article(article, analysis_data)
                    
                    if success:
                        stored_articles.append(article)
                    else:
                        logger.warning(f"Failed to store article: {article.get('title', 'Unknown')}")
                
                except Exception as e:
                    logger.warning(f"Error storing article: {e}")
                    continue
            
            state["stored_articles"] = stored_articles
            
            logger.info(f"Vector storage complete for {len(stored_articles)} articles")
            return state
            
        except Exception as e:
            error_msg = f"Vector storage error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    async def _completion_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Final node for workflow completion"""
        try:
            if state.get("progress_callback"):
                state["progress_callback"]("✅ Workflow completed successfully!", 100)
            
            state["current_step"] = "completion"
            state["progress"] = 100
            
            # Log final statistics
            total_collected = len(state.get("collected_articles", []))
            total_analyzed = len(state.get("analyzed_articles", []))
            total_stored = len(state.get("stored_articles", []))
            total_errors = len(state.get("errors", []))
            
            logger.info(f"Workflow completion summary:")
            logger.info(f"  - Collected: {total_collected} articles")
            logger.info(f"  - Analyzed: {total_analyzed} articles")
            logger.info(f"  - Stored: {total_stored} articles")
            logger.info(f"  - Errors: {total_errors}")
            
            return state
            
        except Exception as e:
            error_msg = f"Completion error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state
    
    def _calculate_article_quality(self, article: Dict[str, Any]) -> float:
        """Calculate overall article quality score"""
        try:
            quality_factors = {
                'has_content': 1.0 if article.get('content') else 0.0,
                'has_source': 1.0 if article.get('source') else 0.0,
                'has_url': 1.0 if article.get('url') else 0.0,
                'sentiment_confidence': article.get('sentiment_confidence', 0.0),
                'low_bias': 1.0 - article.get('bias_score', 0.0),
                'content_length': min(len(article.get('content', '')) / 500, 1.0)
            }
            
            # Weighted average
            weights = {
                'has_content': 0.3,
                'has_source': 0.2,
                'has_url': 0.1,
                'sentiment_confidence': 0.15,
                'low_bias': 0.15,
                'content_length': 0.1
            }
            
            quality_score = sum(
                quality_factors[factor] * weights[factor]
                for factor in quality_factors
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _generate_article_summary(self, article: Dict[str, Any]) -> str:
        """Generate article summary from content"""
        try:
            content = article.get('content', '')
            title = article.get('title', '')
            
            if not content:
                return title[:100] + "..." if len(title) > 100 else title
            
            # Simple extractive summarization
            sentences = content.split('.')
            if len(sentences) <= 2:
                return content[:200] + "..." if len(content) > 200 else content
            
            # Take first sentence and most informative sentences
            summary_sentences = [sentences[0]]
            
            # Add sentences with key information indicators
            key_indicators = ['according to', 'reported', 'announced', 'revealed', 'confirmed']
            for sentence in sentences[1:3]:  # Limit to avoid long summaries
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    summary_sentences.append(sentence)
            
            summary = '. '.join(summary_sentences)
            return summary[:300] + "..." if len(summary) > 300 else summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return article.get('title', 'Summary not available')[:100]

class ChatWorkflow:
    """Workflow for processing chat queries with RAG"""
    
    def __init__(self, config, database: VectorDatabase):
        self.config = config
        self.database = database
        self.workflow = self._build_chat_workflow()
    
    def _build_chat_workflow(self) -> StateGraph:
        """Build the chat workflow graph"""
        
        workflow = StateGraph(dict)
        
        # Add nodes for chat processing
        workflow.add_node("query_analysis", self._query_analysis_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("response_generation", self._response_generation_node)
        
        # Define edges
        workflow.add_edge("query_analysis", "retrieval")
        workflow.add_edge("retrieval", "response_generation")
        workflow.add_edge("response_generation", END)
        
        # Set entry point
        workflow.set_entry_point("query_analysis")
        
        return workflow.compile()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a chat query using the RAG workflow"""
        
        initial_state = {
            "query": query,
            "expanded_query": "",
            "retrieved_articles": [],
            "response": "",
            "confidence": 0.0
        }
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return final_state
            
        except Exception as e:
            logger.error(f"Chat workflow error: {e}")
            return {
                "query": query,
                "response": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0
            }
    
    async def _query_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and expand the user query"""
        try:
            query = state["query"]
            
            # Simple query expansion (can be enhanced with NLP models)
            expanded_query = self._expand_query(query)
            state["expanded_query"] = expanded_query
            
            return state
            
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return state
    
    async def _retrieval_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant articles from vector database"""
        try:
            query = state["expanded_query"] or state["query"]
            
            # Retrieve articles using semantic search
            articles = await self.database.semantic_search(
                query=query,
                embedding_or_model=None,  # Use ChromaDB's built-in embedding
                limit=5
            )
            
            state["retrieved_articles"] = articles
            
            return state
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state["retrieved_articles"] = []
            return state
    
    async def _response_generation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on retrieved articles"""
        try:
            query = state["query"]
            articles = state["retrieved_articles"]
            
            if not articles:
                response = "I couldn't find any relevant articles in the database. Try collecting some news first!"
                state["response"] = response
                state["confidence"] = 0.0
                return state
            
            # Generate response by summarizing relevant articles
            response = self._generate_summary_response(query, articles)
            state["response"] = response
            state["confidence"] = 0.8  # High confidence when we have articles
            
            return state
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            state["response"] = f"Error generating response: {str(e)}"
            state["confidence"] = 0.0
            return state
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        # Simple keyword expansion
        expansions = {
            "technology": ["tech", "digital", "AI", "software", "innovation"],
            "politics": ["political", "government", "election", "policy", "legislation"],
            "health": ["medical", "healthcare", "medicine", "wellness", "disease"],
            "business": ["economy", "market", "finance", "corporate", "industry"],
            "science": ["research", "study", "discovery", "scientific", "experiment"]
        }
        
        expanded = query.lower()
        for topic, terms in expansions.items():
            if topic in expanded:
                expanded += " " + " ".join(terms)
        
        return expanded
    
    def _generate_summary_response(self, query: str, articles: List[Dict[str, Any]]) -> str:
        """Generate a summary response from retrieved articles"""
        if not articles:
            return "No relevant articles found."
        
        # Extract key information from articles
        summaries = []
        for article in articles[:3]:  # Top 3 most relevant
            title = article.get('title', '')
            content = article.get('content', '')
            source = article.get('source', '')
            sentiment = article.get('sentiment_label', 'neutral')
            
            summary = f"**{title}** (Source: {source}, Sentiment: {sentiment})\n"
            summary += f"{content[:200]}...\n"
            summaries.append(summary)
        
        response = f"Based on the news articles in our database, here's what I found:\n\n"
        response += "\n".join(summaries)
        response += f"\n\nI found {len(articles)} relevant articles in total."
        
        return response