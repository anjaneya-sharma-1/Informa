# Patch sqlite3 for ChromaDB
import sys
try:
    import pysqlite3
    # Patch for chromadb compatibility when using streamlit
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # Use standard sqlite3 if pysqlite3 is not available
    pass

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime
import streamlit as st
import asyncio
import logging
from chromadb import PersistentClient
from chromadb.config import Settings


logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, db_path: str = "news_analysis.db", persist_directory: str = "./chroma_db"):
        self.db_path = db_path
        self.persist_directory = persist_directory
        self.init_database()
        self.init_chroma_db()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT,
                    source TEXT,
                    published_at TEXT,
                    topic TEXT,
                    source_type TEXT,
                    collected_at TEXT,
                    content_hash TEXT
                )
            ''')
            
            # Analysis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    article_id TEXT PRIMARY KEY,
                    sentiment_label TEXT,
                    sentiment_score REAL,
                    bias_score REAL,
                    credibility_score REAL,
                    analysis_data TEXT,
                    analyzed_at TEXT,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            ''')
            
            # Vector embeddings table (simplified - in production use proper vector DB)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    article_id TEXT PRIMARY KEY,
                    embedding_vector TEXT,
                    created_at TEXT,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            ''')
            
            # Search index for keywords
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT,
                    keyword TEXT,
                    frequency INTEGER,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def init_chroma_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="news_articles",
                metadata={"description": "News articles with analysis"}
            )
            
            logger.info(f"ChromaDB initialized with {self.collection.count()} articles")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    async def store_article(self, article: Dict[str, Any], analysis: Dict[str, Any] = None) -> bool:
        """Store article and its analysis in the database"""
        try:
            # Prepare document text - ensure strings are properly handled
            title = str(article.get('title', '') or '').replace('\x00', '')
            content = str(article.get('content', '') or '').replace('\x00', '')
            description = str(article.get('description', '') or '')
            if not description and content:
                description = content[:2000]
            document_text = f"{title} {content}"
            
            # Prepare metadata - ensure all values are properly stringified and cleaned
            metadata = {
                'title': str(article.get('title', '') or '').replace('\x00', '')[:1000],  # Limit length
                'source': str(article.get('source', '') or '').replace('\x00', '')[:200],
                'url': str(article.get('url', '') or '').replace('\x00', '')[:500],
                'published_at': str(article.get('published_at', '') or '').replace('\x00', ''),
                'topic': str(article.get('topic', '') or '').replace('\x00', '')[:100],
                'collected_at': str(article.get('collected_at', datetime.now().isoformat()) or datetime.now().isoformat())
            }
            
            # Add analysis results to metadata
            if analysis:
                sentiment = analysis.get('sentiment', {})
                metadata.update({
                    'sentiment_label': sentiment.get('label', 'neutral'),
                    'sentiment_score': sentiment.get('score', 0.0),
                    'bias_score': analysis.get('bias', {}).get('overall_score', analysis.get('bias_score', 0.0)),
                    'credibility_score': analysis.get('credibility_score', analysis.get('fact_check', {}).get('credibility_score', 0.5))
                })
            
            # Store in ChromaDB
            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[article['id']]
            )
            
            # Also persist minimal data to SQLite for keyword/embedding search used by chat
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert article (minimal fields)
                cursor.execute('''
                    INSERT OR REPLACE INTO articles 
                    (id, title, description, url, source, published_at, topic, source_type, collected_at, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['id'],
                    title[:1000],
                    description[:2000],
                    str(article.get('url', '') or '')[:500],
                    str(article.get('source', '') or '')[:200],
                    str(article.get('published_at', '') or ''),
                    str(article.get('topic', '') or '')[:100],
                    str(article.get('source_type', '') or '')[:100],
                    article.get('collected_at', datetime.now().isoformat()),
                    hashlib.md5((title + description).encode('utf-8', errors='ignore')).hexdigest()
                ))
                
                # Create keywords and fallback embedding
                self._create_search_index(article['id'], {
                    'title': title,
                    'description': description
                }, cursor)
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"SQLite persistence warning: {e}")
            
            logger.debug(f"Stored article: {title[:50]}...")
            return True
            
        except Exception as e:
            error_msg = str(e).replace('\x00', '')
            logger.error(f"Error storing article: {error_msg}")
            return False
    
    def add_article(self, article: Dict[str, Any], analysis: Dict[str, Any] = None):
        """Add article and its analysis to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert article
            cursor.execute('''
                INSERT OR REPLACE INTO articles 
                (id, title, description, url, source, published_at, topic, source_type, collected_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'],
                str(article.get('title', '') or '')[:1000],  # Limit length
                str(article.get('description', '') or '')[:2000],
                str(article.get('url', '') or '')[:500],
                str(article.get('source', '') or '')[:200],
                str(article.get('published_at', '') or ''),
                str(article.get('topic', '') or '')[:100],
                str(article.get('source_type', '') or '')[:100],
                article.get('collected_at', datetime.now().isoformat()),
                hashlib.md5((str(article.get('title', '') or '') + str(article.get('description', '') or '')).encode('utf-8', errors='ignore')).hexdigest()
            ))
            
            # Insert analysis if provided
            if analysis:
                sentiment = analysis.get('sentiment', {})
                fact_check = analysis.get('fact_check', {})
                
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_results
                    (article_id, sentiment_label, sentiment_score, bias_score, credibility_score, analysis_data, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['id'],
                    sentiment.get('label', 'neutral'),
                    sentiment.get('score', 0.0),
                    analysis.get('bias_score', 0.0),
                    fact_check.get('credibility_score', 0.0),
                    json.dumps(analysis),
                    datetime.now().isoformat()
                ))
            
            # Create simple embedding and search index
            self._create_search_index(article['id'], article, cursor)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.error(f"Error adding article to database: {str(e)}")
    
    def _create_search_index(self, article_id: str, article: Dict[str, Any], cursor):
        """Create search index for the article"""
        try:
            # Extract keywords from title and description - ensure strings are safe
            title = str(article.get('title', '') or '')
            description = str(article.get('description', '') or '')
            text = f"{title} {description}".lower().replace('\x00', '')
            
            # Simple keyword extraction (remove common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            
            words = text.split()
            keyword_freq = {}
            
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum()).lower()
                if len(word) > 2 and word not in stop_words:
                    keyword_freq[word] = keyword_freq.get(word, 0) + 1
            
            # Insert keywords into search index
            for keyword, freq in keyword_freq.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO search_index (article_id, keyword, frequency)
                    VALUES (?, ?, ?)
                ''', (article_id, keyword, freq))
            
            # Create simple embedding (in production, use proper embedding model)
            embedding = self._create_simple_embedding(text)
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (article_id, embedding_vector, created_at)
                VALUES (?, ?, ?)
            ''', (article_id, json.dumps(embedding.tolist()), datetime.now().isoformat()))
            
        except Exception as e:
            st.error(f"Error creating search index: {str(e)}")
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding vector (replace with proper embedding model)"""
        # Simple hash-based embedding for demonstration
        # In production, use models like sentence-transformers
        
        words = text.lower().split()
        
        # Create a 384-dimensional vector to align with MiniLM embeddings
        embedding = np.zeros(384)
        
        for i, word in enumerate(words[:100]):  # Limit to first 100 words
            # Simple hash-based approach
            word_hash = hash(word) % 384
            embedding[word_hash] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    async def semantic_search(self, query, embedding_or_model, limit: int = 5, use_api_embedding: bool = False) -> List[Dict[str, Any]]:
        """Perform semantic search using either a model or a precomputed embedding vector."""
        try:
            if use_api_embedding:
                # embedding_or_model is a precomputed embedding vector (list of floats)
                query_embedding = np.array(embedding_or_model)
                # Fetch all embeddings from DB
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT article_id, embedding_vector FROM embeddings')
                rows = cursor.fetchall()
                similarities = []
                for article_id, embedding_json in rows:
                    try:
                        embedding = np.array(json.loads(embedding_json))
                        # Ensure dimensionality match by padding/truncating
                        if embedding.shape[0] != query_embedding.shape[0]:
                            if embedding.shape[0] < query_embedding.shape[0]:
                                pad = np.zeros(query_embedding.shape[0] - embedding.shape[0])
                                embedding = np.concatenate([embedding, pad])
                            else:
                                embedding = embedding[:query_embedding.shape[0]]
                        # Cosine similarity
                        if np.linalg.norm(embedding) == 0 or np.linalg.norm(query_embedding) == 0:
                            sim = 0.0
                        else:
                            sim = float(np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding)))
                        similarities.append((article_id, sim))
                    except Exception:
                        continue
                # Sort by similarity
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
                # Fetch article data
                results = []
                for article_id, sim in similarities:
                    article_data = self._get_article_with_analysis(article_id, cursor)
                    if article_data:
                        article_data['relevance_score'] = sim
                        results.append(article_data)
                conn.close()
                return results
            else:
                # embedding_or_model is a model object, use ChromaDB as before
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=['documents', 'metadatas', 'distances']
                )
                articles = []
                if results['documents'] and len(results['documents']) > 0:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        relevance_score = max(0.0, 1.0 - distance)
                        article = {
                            'id': results['ids'][0][i],
                            'title': metadata.get('title', ''),
                            'content': doc,
                            'source': metadata.get('source', ''),
                            'url': metadata.get('url', ''),
                            'published_at': metadata.get('published_at', ''),
                            'topic': metadata.get('topic', ''),
                            'sentiment_label': metadata.get('sentiment_label', 'neutral'),
                            'sentiment_score': metadata.get('sentiment_score', 0.0),
                            'bias_score': metadata.get('bias_score', 0.0),
                            'credibility_score': metadata.get('credibility_score', 0.5),
                            'relevance_score': relevance_score
                        }
                        articles.append(article)
                logger.debug(f"Semantic search returned {len(articles)} articles")
                return articles
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def search_by_keywords(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search articles by keywords"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract keywords from query
            query_words = [word.lower().strip() for word in query.split() if len(word) > 2]
            
            if not query_words:
                return []
            
            # Search in search index
            article_scores = {}
            
            for word in query_words:
                cursor.execute('''
                    SELECT article_id, frequency FROM search_index 
                    WHERE keyword LIKE ?
                ''', (f'%{word}%',))
                
                results = cursor.fetchall()
                for article_id, freq in results:
                    article_scores[article_id] = article_scores.get(article_id, 0) + freq
            
            # Also search in titles and descriptions directly
            search_pattern = f"%{' '.join(query_words)}%"
            cursor.execute('''
                SELECT id FROM articles 
                WHERE title LIKE ? OR description LIKE ?
            ''', (search_pattern, search_pattern))
            
            direct_matches = cursor.fetchall()
            for (article_id,) in direct_matches:
                article_scores[article_id] = article_scores.get(article_id, 0) + 10  # Boost direct matches
            
            # Sort by score
            sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get article details
            results = []
            for article_id, score in sorted_articles[:limit]:
                article_data = self._get_article_with_analysis(article_id, cursor)
                if article_data:
                    article_data['relevance_score'] = min(1.0, score / 10.0)  # Normalize score
                    results.append(article_data)
            
            conn.close()
            return results
            
        except Exception as e:
            st.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _get_article_with_analysis(self, article_id: str, cursor) -> Optional[Dict[str, Any]]:
        """Get article with its analysis data"""
        try:
            # Get article data
            cursor.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
            article_row = cursor.fetchone()
            
            if not article_row:
                return None
            
            # Convert to dict
            article_columns = ['id', 'title', 'description', 'url', 'source', 'published_at', 'topic', 'source_type', 'collected_at', 'content_hash']
            article_data = dict(zip(article_columns, article_row))
            
            # Get analysis data
            cursor.execute('SELECT * FROM analysis_results WHERE article_id = ?', (article_id,))
            analysis_row = cursor.fetchone()
            
            if analysis_row:
                analysis_columns = ['article_id', 'sentiment_label', 'sentiment_score', 'bias_score', 'credibility_score', 'analysis_data', 'analyzed_at']
                analysis_data = dict(zip(analysis_columns, analysis_row))
                
                # Add analysis fields to article data
                article_data['sentiment_label'] = analysis_data['sentiment_label']
                article_data['sentiment_score'] = analysis_data['sentiment_score']
                article_data['bias_score'] = analysis_data['bias_score']
                article_data['credibility_score'] = analysis_data['credibility_score']
            
            return article_data
            
        except Exception as e:
            st.error(f"Error getting article data: {str(e)}")
            return None
    
    def get_count(self) -> int:
        """Get total number of articles in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM articles')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
    
    def get_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent articles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM articles 
                ORDER BY collected_at DESC 
                LIMIT ?
            ''', (limit,))
            
            article_ids = [row[0] for row in cursor.fetchall()]
            
            results = []
            for article_id in article_ids:
                article_data = self._get_article_with_analysis(article_id, cursor)
                if article_data:
                    results.append(article_data)
            
            conn.close()
            return results
            
        except Exception as e:
            st.error(f"Error getting recent articles: {str(e)}")
            return []
    
    def delete_article(self, article_id: str) -> bool:
        """Delete an article from the database"""
        try:
            self.collection.delete(ids=[article_id])
            logger.debug(f"Deleted article: {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting article: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all articles from the database"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("news_articles")
            self.collection = self.client.get_or_create_collection(
                name="news_articles",
                metadata={"description": "News articles with analysis"}
            )
            
            logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
    
     
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Get all metadata
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return {
                    'total_articles': 0,
                    'sources': {},
                    'topics': {},
                    'sentiment_distribution': {},
                    'avg_credibility': 0.0
                }
            
            # Analyze metadata
            sources = {}
            topics = {}
            sentiments = {}
            credibility_scores = []
            
            for metadata in results['metadatas']:
                # Count sources
                source = metadata.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
                
                # Count topics
                topic = metadata.get('topic', 'Unknown')
                topics[topic] = topics.get(topic, 0) + 1
                
                # Count sentiments
                sentiment = metadata.get('sentiment_label', 'neutral')
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
                
                # Collect credibility scores
                credibility = metadata.get('credibility_score', 0.5)
                if isinstance(credibility, (int, float)):
                    credibility_scores.append(credibility)
            
            avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.0
            
            return {
                'total_articles': len(results['metadatas']),
                'sources': sources,
                'topics': topics,
                'sentiment_distribution': sentiments,
                'avg_credibility': avg_credibility
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
