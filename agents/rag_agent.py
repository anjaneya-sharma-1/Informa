import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from utils.database import VectorDatabase

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, config, database: VectorDatabase):
        self.config = config
        self.database = database
        self.last_run = None
        self.last_error = None
        
        # Initialize embedding model
        try:
            # Use a lightweight sentence transformer model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("RAG agent initialized with SentenceTransformer model")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
            self.last_error = str(e)
        
        # Initialize text generation (using a simple approach)
        try:
            from transformers import pipeline
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Text generation pipeline initialized")
        except Exception as e:
            logger.warning(f"Could not initialize text generation pipeline: {e}")
            self.text_generator = None
        
        self.context_window = 4000
        self.max_sources = 5
    
    async def answer_question(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Answer user question using RAG approach"""
        try:
            self.last_run = datetime.now().isoformat()
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant articles
            relevant_articles = await self._retrieve_relevant_content(question)
            
            if not relevant_articles:
                return {
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
        """Retrieve relevant articles from vector database"""
        try:
            # Get articles from database with similarity search
            if self.embedding_model:
                # Use semantic search
                articles = await self.database.semantic_search(question, self.embedding_model, limit=self.max_sources)
            else:
                # Fallback to keyword search
                articles = await self.database.keyword_search(question, limit=self.max_sources)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving content: {e}")
            return []
    
    def _build_context(self, articles: List[Dict[str, Any]], question: str) -> str:
        """Build context string from retrieved articles"""
        context_parts = []
        current_length = 0
        
        # Sort articles by relevance score if available
        sorted_articles = sorted(
            articles, 
            key=lambda x: x.get('relevance_score', 0), 
            reverse=True
        )
        
        for i, article in enumerate(sorted_articles):
            # Create article summary
            article_text = f"""
Article {i+1}:
Title: {article.get('title', 'Unknown')}
Source: {article.get('source', 'Unknown')}
Published: {article.get('published_at', 'Unknown')}
Content: {article.get('content', 'No content available')}
Credibility Score: {article.get('credibility_score', 'Unknown')}
---
"""
            
            # Check if adding this article would exceed context window
            if current_length + len(article_text) > self.context_window:
                break
            
            context_parts.append(article_text)
            current_length += len(article_text)
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate answer using context and question"""
        
        # Build conversation history context
        history_context = ""
        if chat_history:
            recent_history = chat_history[-3:]  # Last 3 exchanges
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                history_context += f"{role.title()}: {content}\n"
        
        # Try using the text generation model if available
        if self.text_generator:
            try:
                return await self._generate_with_model(question, context, history_context)
            except Exception as e:
                logger.warning(f"Model generation failed, using template: {e}")
        
        # Fallback to template-based generation
        return self._generate_with_template(question, context, history_context)
    
    async def _generate_with_model(self, question: str, context: str, history_context: str) -> str:
        """Generate answer using the language model"""
        try:
            # Create a prompt for the model
            prompt = f"""Based on the following news articles, answer the user's question accurately and comprehensively.

Previous conversation:
{history_context}

Question: {question}

News articles:
{context[:1000]}  # Truncate context for model

Answer:"""
            
            # Generate response
            response = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.text_generator.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else self._generate_with_template(question, context, history_context)
            
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            raise
    
    def _generate_with_template(self, question: str, context: str, history_context: str) -> str:
        """Generate answer using template-based approach"""
        
        # Extract key information from context
        articles_info = []
        if "Article 1:" in context:
            # Parse articles from context
            article_sections = context.split("Article ")[1:]  # Skip empty first element
            
            for section in article_sections[:3]:  # Limit to first 3 articles
                lines = section.strip().split('\n')
                title = ""
                source = ""
                content = ""
                
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Source:"):
                        source = line.replace("Source:", "").strip()
                    elif line.startswith("Content:"):
                        content = line.replace("Content:", "").strip()
                
                if title and source:
                    articles_info.append({
                        'title': title,
                        'source': source,
                        'content': content
                    })
        
        # Generate contextual answer based on question type
        question_lower = question.lower()
        
        if not articles_info:
            return "I don't have sufficient information in the current news articles to answer your question comprehensively."
        
        # Question type detection and response generation
        if any(word in question_lower for word in ['what', 'explain', 'describe']):
            # Explanatory questions
            answer = f"Based on the available news articles, here's what I can tell you:\n\n"
            
            for i, article in enumerate(articles_info):
                answer += f"According to {article['source']}, {article['content'][:200]}...\n\n"
            
            answer += "This information comes from multiple news sources in my database."
        
        elif any(word in question_lower for word in ['who', 'when', 'where']):
            # Factual questions
            answer = "Based on the news articles I have access to:\n\n"
            
            # Try to extract specific facts
            if articles_info:
                answer += f"From {articles_info[0]['source']}: {articles_info[0]['content'][:300]}...\n\n"
                if len(articles_info) > 1:
                    answer += f"Additional context from {articles_info[1]['source']}: {articles_info[1]['content'][:200]}..."
        
        elif any(word in question_lower for word in ['why', 'how']):
            # Analytical questions
            answer = "Based on my analysis of the available news articles:\n\n"
            
            for article in articles_info[:2]:
                answer += f"• {article['source']} reports: {article['content'][:250]}...\n\n"
            
            answer += "These sources provide insight into the underlying factors and mechanisms involved."
        
        elif any(word in question_lower for word in ['compare', 'difference', 'similar']):
            # Comparative questions
            answer = "Comparing information from different news sources:\n\n"
            
            for i, article in enumerate(articles_info[:2]):
                answer += f"**Source {i+1} ({article['source']}):** {article['content'][:200]}...\n\n"
            
            answer += "These different perspectives help provide a more complete picture of the situation."
        
        elif any(word in question_lower for word in ['trend', 'pattern', 'over time']):
            # Trend questions
            answer = "Based on the timeline of news articles in my database:\n\n"
            
            for article in articles_info:
                answer += f"• {article['source']}: {article['content'][:150]}...\n"
            
            answer += "\nThese articles suggest ongoing developments in this area."
        
        elif any(word in question_lower for word in ['sentiment', 'opinion', 'feel']):
            # Sentiment questions
            answer = "Regarding the sentiment and opinions expressed in the news:\n\n"
            
            for article in articles_info:
                answer += f"• {article['source']} perspective: {article['content'][:200]}...\n\n"
            
            answer += "Note that I analyze factual content; for detailed sentiment analysis, please check the Analysis Dashboard."
        
        else:
            # General questions
            answer = "Here's what I found in the news articles:\n\n"
            
            for article in articles_info[:2]:
                answer += f"**{article['source']}** reports: {article['content'][:250]}...\n\n"
        
        # Add credibility note if relevant
        if len(articles_info) > 1:
            answer += f"\n*This answer is based on {len(articles_info)} news sources from my database.*"
        
        return answer
    
    def _prepare_sources(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for display"""
        sources = []
        
        for article in articles:
            source_info = {
                'title': article.get('title', 'Unknown Title'),
                'source': article.get('source', 'Unknown Source'),
                'url': article.get('url', ''),
                'published_at': article.get('published_at', ''),
                'relevance_score': article.get('relevance_score', 0.0),
                'credibility_score': article.get('credibility_score', 0.0)
            }
            sources.append(source_info)
        
        return sources
    
    def _generate_reasoning(self, question: str, articles: List[Dict[str, Any]], answer: str) -> str:
        """Generate explanation of reasoning process"""
        
        reasoning_parts = []
        
        # Explain retrieval process
        reasoning_parts.append(f"I searched through {len(articles)} relevant news articles to answer your question.")
        
        # Explain source selection
        if articles:
            high_credibility = sum(1 for article in articles if article.get('credibility_score', 0) > 0.7)
            if high_credibility > 0:
                reasoning_parts.append(f"{high_credibility} of these sources have high credibility scores (>0.7).")
            
            # Mention source diversity
            sources = set(article.get('source', 'Unknown') for article in articles)
            if len(sources) > 1:
                reasoning_parts.append(f"Information comes from {len(sources)} different news sources for balanced perspective.")
        
        # Explain confidence level
        if len(articles) >= 3:
            reasoning_parts.append("High confidence in answer due to multiple corroborating sources.")
        elif len(articles) == 2:
            reasoning_parts.append("Moderate confidence - answer based on two sources.")
        else:
            reasoning_parts.append("Limited confidence - answer based on single source.")
        
        # Mention any limitations
        if not articles:
            reasoning_parts.append("No relevant articles found in database - answer may be incomplete.")
        
        return " ".join(reasoning_parts)
    
    def health_check(self) -> bool:
        """Check if the RAG agent is healthy"""
        try:
            # Test embedding model
            if self.embedding_model:
                test_embedding = self.embedding_model.encode("test sentence")
                if test_embedding is None or len(test_embedding) == 0:
                    return False
            
            # Test database connection
            if not self.database.health_check():
                return False
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"RAG agent health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear any cached data"""
        self.last_error = None
        logger.info("RAG agent cache cleared")
    
    def restart(self):
        """Restart the RAG agent"""
        self.clear_cache()
        self.last_run = None
        logger.info("RAG agent restarted")
