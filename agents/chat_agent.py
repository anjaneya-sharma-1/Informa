import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agents.workflow import ChatWorkflow
from utils.database import VectorDatabase

logger = logging.getLogger(__name__)

class ChatAgent:
    """Intelligent chat agent for news queries with RAG capabilities"""
    
    def __init__(self, config, database: VectorDatabase):
        self.config = config
        self.database = database
        self.chat_workflow = ChatWorkflow(config, database)
        self.chat_history = []
        self.last_query_time = None
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query and return a response"""
        try:
            self.last_query_time = datetime.now()
            
            # Log the query
            logger.info(f"Processing query: {query[:100]}...")
            
            # Add query to chat history
            self.chat_history.append({
                "type": "user_query",
                "content": query,
                "timestamp": self.last_query_time.isoformat()
            })
            
            # Process query with chat workflow
            result = await self.chat_workflow.process_query(query)
            
            # Add response to chat history
            self.chat_history.append({
                "type": "assistant_response",
                "content": result.get("response", ""),
                "confidence": result.get("confidence", 0.0),
                "retrieved_articles": len(result.get("retrieved_articles", [])),
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
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "confidence": 0.0,
                "articles_found": 0,
                "error": str(e)
            }
                    'answer': "I don't have enough information in my knowledge base to answer that question. Please try collecting more news articles first.",
                    'sources': [],
                    'reasoning': "No relevant articles found in the database.",
                    'method': 'no_content'
                }
            
            # Generate context from retrieved articles
            context = self._build_context(relevant_articles, question)
            
            # Generate answer using the context
            answer = await self._generate_answer(question, context, chat_history)
            
            # Prepare sources with credibility information
            sources = self._prepare_sources(relevant_articles)
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(question, relevant_articles, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'reasoning': reasoning,
                'context_used': len(context),
                'articles_retrieved': len(relevant_articles),
                'method': 'rag'
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in RAG processing: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'reasoning': "Error occurred during processing.",
                'error': str(e)
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
        if not self.api_key:
            logger.error("No Hugging Face API key provided for embedding.")
            return None
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
            return None

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