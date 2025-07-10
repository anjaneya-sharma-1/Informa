import asyncio
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agents.workflow import ChatWorkflow
from utils.database import VectorDatabase
from utils.config import AppConfig

logger = logging.getLogger(__name__)

class ChatAgent:
    """Intelligent chat agent for news queries with RAG capabilities"""
    
    def __init__(self, config, database: VectorDatabase):
        self.config = config
        self.database = database
        self.chat_workflow = ChatWorkflow(config, database)
        self.chat_history = []
        self.last_query_time = None
        
        # Initialize from config using proper methods
        self.api_key = config.get_huggingface_key() if config else ""
        
        # Get chat configuration
        chat_config = config.get_chat_config() if config else {}
        self.api_url = chat_config.get("api_url", "https://api-inference.huggingface.co/models/google/flan-t5-base")
        self.embedding_api_url = chat_config.get("embedding_api_url", "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2")
        self.max_sources = chat_config.get("max_sources", 5)
        self.context_window = chat_config.get("context_window", 4000)
        self.last_error = None
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query and return a response"""
        try:
            self.last_query_time = datetime.now()
            
            # Log the query
            logger.info(f"Processing query: {query[:100]}...")
            
            # Add query to chat history
            self.chat_history.append({
                "role": "user",
                "content": query,
                "timestamp": self.last_query_time.isoformat()
            })
            
            # Retrieve relevant articles
            relevant_articles = await self._retrieve_relevant_content(query)
            
            if not relevant_articles:
                # If no relevant content, try to fetch new articles via workflow
                logger.info("No relevant content found, triggering workflow")
                result = await self.chat_workflow.process_query(query)
                
                # Add response to chat history
                self.chat_history.append({
                    "role": "assistant",
                    "content": result.get("response", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Enhance response with metadata
                enhanced_result = {
                    "response": result.get("response", "I couldn't process your query."),
                    "confidence": result.get("confidence", 0.0),
                    "articles_found": len(result.get("retrieved_articles", [])),
                    "query_type": self._classify_query(query),
                    "suggestions": self._generate_suggestions(query, result),
                    "retrieved_articles": result.get("retrieved_articles", [])
                }
                
                return enhanced_result
                
            # Generate context from retrieved articles
            context = self._build_context(relevant_articles, query)
            
            # Generate answer using the context
            answer = await self._generate_answer(query, context, self.chat_history)
            
            # Prepare sources with credibility information
            sources = self._prepare_sources(relevant_articles)
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(query, relevant_articles, answer)
            
            # Add response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            result = {
                "response": answer,
                "confidence": 0.8,  # Default confidence
                "articles_found": len(relevant_articles),
                "query_type": self._classify_query(query),
                "suggestions": self._generate_suggestions(query, {"response": answer, "retrieved_articles": relevant_articles}),
                "retrieved_articles": sources
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "confidence": 0.0,
                "articles_found": 0,
                "error": str(e)
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in RAG processing: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "confidence": 0.0,
                "articles_found": 0,
                "error": str(e)
            }
    
    async def _retrieve_relevant_content(self, question: str) -> List[Dict[str, Any]]:
        try:
            # Use Hugging Face API for embedding
            question_embedding = self._get_embedding(question)
            if question_embedding is None:
                logger.error("Failed to get embedding from API.")
                return []
            articles = await self.database.semantic_search(question, question_embedding, limit=self.max_sources, use_api_embedding=True)
            return articles
        except Exception as e:
            logger.error(f"Error retrieving content: {e}")
            return []

    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings from Hugging Face API or fallback to simple embedding"""
        try:
            if not self.api_key:
                logger.error("No Hugging Face API key provided for embedding.")
                return self._create_simple_embedding(text)
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(self.embedding_api_url, headers=headers, json={"inputs": text})
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and isinstance(result[0], list):
                # If result is a list of lists (batch), take the first
                return result[0]
            elif isinstance(result, list):
                return result
            else:
                logger.error(f"Unexpected embedding API result: {result}")
                return self._create_simple_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding from API: {e}")
            return self._create_simple_embedding(text)
            
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple fallback embedding when API fails"""
        import numpy as np
        # Simple hash-based embedding
        text = text.lower()
        words = text.split()
        
        # Create a simple 384-dimensional vector (same as MiniLM)
        embedding = np.zeros(384)
        
        for i, word in enumerate(words[:100]):  # Limit to first 100 words
            # Simple hash-based approach
            word_hash = hash(word) % 384
            embedding[word_hash] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()

    def _build_context(self, articles: List[Dict[str, Any]], question: str) -> str:
        context_parts = []
        current_length = 0
        sorted_articles = sorted(
            articles, 
            key=lambda x: x.get('relevance_score', 0), 
            reverse=True
        )
        for i, article in enumerate(sorted_articles):
            article_text = f"""
Article {i+1}:
Title: {article.get('title', 'Unknown')}
Source: {article.get('source', 'Unknown')}
Published: {article.get('published_at', 'Unknown')}
Content: {article.get('content', 'No content available')}
Credibility Score: {article.get('credibility_score', 'Unknown')}
---
"""
            if current_length + len(article_text) > self.context_window:
                break
            context_parts.append(article_text)
            current_length += len(article_text)
        return "\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        history_context = ""
        if chat_history:
            recent_history = chat_history[-3:]
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                history_context += f"{role.title()}: {content}\n"
        if self.api_key:
            try:
                return self._generate_with_hf_api(question, context, history_context)
            except Exception as e:
                logger.warning(f"Hugging Face API generation failed, using template: {e}")
        return self._generate_with_template(question, context, history_context)

    def _generate_with_hf_api(self, question: str, context: str, history_context: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = f"""Based on the following news articles, answer the user's question accurately and comprehensively.\n\nPrevious conversation:\n{history_context}\n\nQuestion: {question}\n\nNews articles:\n{context[:1000]}  # Truncate context for model\n\nAnswer:"""
        response = requests.post(self.api_url, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 100, "temperature": 0.7}})
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            generated_text = result[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            return answer if answer else self._generate_with_template(question, context, history_context)
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            logger.warning(f"Unexpected API result: {result}")
            return self._generate_with_template(question, context, history_context)

    def _generate_with_template(self, question: str, context: str, history_context: str) -> str:
        return f"Based on the news articles, here is a summary answer to your question: {question}\n\n{context[:500]}"

    def _prepare_sources(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for article in articles:
            sources.append({
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'credibility_score': article.get('credibility_score', None)
            })
        return sources

    def _generate_reasoning(self, question: str, articles: List[Dict[str, Any]], answer: str) -> str:
        return f"The answer was generated using the most relevant news articles in the database for the question: '{question}'."
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query (news, fact-check, opinion, etc.)"""
        # Basic classification based on keywords
        query = query.lower()
        if any(word in query for word in ["fact", "check", "true", "false", "verify", "confirm"]):
            return "fact_check"
        elif any(word in query for word in ["latest", "recent", "today", "update", "news"]):
            return "news_update"
        elif any(word in query for word in ["bias", "sentiment", "opinion", "perspective"]):
            return "opinion_analysis"
        elif any(word in query for word in ["summarize", "summary", "brief", "overview"]):
            return "summarization"
        else:
            return "general_query"
    
    def _generate_suggestions(self, query: str, result: Dict[str, Any]) -> List[str]:
        """Generate follow-up question suggestions based on the query and result"""
        suggestions = []
        
        # Generate based on query type
        query_type = self._classify_query(query)
        
        # Get topics from retrieved articles
        topics = set()
        for article in result.get("retrieved_articles", []):
            if "title" in article:
                words = article["title"].lower().split()
                for word in words:
                    if len(word) > 4 and word not in ["about", "these", "those", "their", "there", "where"]:
                        topics.add(word.strip('.,?!'))
        
        # Generate suggestions based on query type and topics
        if query_type == "news_update":
            suggestions.append(f"What's the latest development on this topic?")
        elif query_type == "fact_check":
            suggestions.append("Is there any contradictory information in other sources?")
        elif query_type == "opinion_analysis":
            suggestions.append("What are the different perspectives on this issue?")
            
        # Add topic-based suggestions
        topic_list = list(topics)
        if topic_list and len(topic_list) >= 2:
            suggestions.append(f"How are {topic_list[0]} and {topic_list[1]} connected?")
        
        # Always add these general suggestions
        suggestions.append("Can you summarize this information?")
        suggestions.append("What's the political bias in the reporting?")
        
        # Return up to 3 suggestions
        return suggestions[:3]