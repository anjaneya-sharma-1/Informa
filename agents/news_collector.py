import asyncio
import aiohttp
import feedparser
import requests
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Any
import logging
import json
import re
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class NewsCollectorAgent:
    def __init__(self, config):
        self.config = config
        self.session = None
        self.last_run = None
        self.last_error = None
        
        # RSS feed URLs for different sources
        self.rss_feeds = {
            'bbc': {
                'technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
                'politics': 'http://feeds.bbci.co.uk/news/politics/rss.xml',
                'health': 'http://feeds.bbci.co.uk/news/health/rss.xml',
                'business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
                'science': 'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
                'world': 'http://feeds.bbci.co.uk/news/world/rss.xml'
            },
            'reuters': {
                'technology': 'https://www.reuters.com/technology/rss',
                'politics': 'https://www.reuters.com/politics/rss',
                'health': 'https://www.reuters.com/healthcare-pharmaceuticals/rss',
                'business': 'https://www.reuters.com/business/rss',
                'science': 'https://www.reuters.com/science/rss',
                'world': 'https://www.reuters.com/world/rss'
            },
            'google_news': {
                'technology': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4ZERBU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
                'politics': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4ZERBU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
                'health': 'https://news.google.com/rss/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNR3QwTlRFU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
                'business': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4ZERBU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
                'science': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4ZERBU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
                'world': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4ZERBU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en'
            }
        }
        
        # Reddit API endpoints (no auth required for public posts)
        self.reddit_endpoints = {
            'technology': 'https://www.reddit.com/r/technology/hot.json?limit=25',
            'politics': 'https://www.reddit.com/r/politics/hot.json?limit=25',
            'health': 'https://www.reddit.com/r/health/hot.json?limit=25',
            'business': 'https://www.reddit.com/r/business/hot.json?limit=25',
            'science': 'https://www.reddit.com/r/science/hot.json?limit=25',
            'world': 'https://www.reddit.com/r/worldnews/hot.json?limit=25'
        }
        
        # HackerNews API
        self.hackernews_api = 'https://hacker-news.firebaseio.com/v0'
    
    async def collect_news(self, topics: List[str], sources: List[str], 
                          max_articles: int = 20, filter_type: str = 'latest') -> List[Dict[str, Any]]:
        """Collect news from multiple real sources"""
        try:
            self.last_run = datetime.now().isoformat()
            logger.info(f"Starting news collection for topics: {topics}, sources: {sources}")
            
            all_articles = []
            
            # Create aiohttp session
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'NewsAnalysisBot/1.0'}
            ) as session:
                
                # Collect from each source
                for source in sources:
                    try:
                        if source == 'bbc':
                            articles = await self._collect_from_rss(session, 'bbc', topics)
                        elif source == 'reuters':
                            articles = await self._collect_from_rss(session, 'reuters', topics)
                        elif source == 'google_news':
                            articles = await self._collect_from_rss(session, 'google_news', topics)
                        elif source == 'reddit':
                            articles = await self._collect_from_reddit(session, topics)
                        elif source == 'hackernews':
                            articles = await self._collect_from_hackernews(session)
                        else:
                            logger.warning(f"Unknown source: {source}")
                            continue
                        
                        all_articles.extend(articles)
                        logger.info(f"Collected {len(articles)} articles from {source}")
                        
                    except Exception as e:
                        logger.error(f"Error collecting from {source}: {e}")
                        continue
            
            # Remove duplicates
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Sort articles
            sorted_articles = self._sort_articles(unique_articles, filter_type)
            
            # Limit results
            final_articles = sorted_articles[:max_articles]
            
            logger.info(f"Final collection: {len(final_articles)} unique articles")
            return final_articles
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in news collection: {e}")
            return []
    
    async def _collect_from_rss(self, session: aiohttp.ClientSession, 
                               source: str, topics: List[str]) -> List[Dict[str, Any]]:
        """Collect articles from RSS feeds"""
        articles = []
        
        for topic in topics:
            if topic in self.rss_feeds.get(source, {}):
                feed_url = self.rss_feeds[source][topic]
                
                try:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Parse RSS feed
                            feed = feedparser.parse(content)
                            
                            for entry in feed.entries[:10]:  # Limit per topic
                                article = self._parse_rss_entry(entry, source, topic)
                                if article:
                                    articles.append(article)
                        else:
                            logger.warning(f"Failed to fetch RSS feed {feed_url}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                    continue
        
        return articles
    
    def _parse_rss_entry(self, entry, source: str, topic: str) -> Dict[str, Any]:
        """Parse RSS entry into article format"""
        try:
            # Generate unique ID
            content_hash = hashlib.md5(
                (entry.get('title', '') + entry.get('link', '')).encode()
            ).hexdigest()
            
            # Extract publication date
            published_at = ''
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6]).isoformat()
            elif hasattr(entry, 'published'):
                published_at = entry.published
            
            # Clean description
            description = entry.get('description', '')
            if description:
                # Remove HTML tags
                description = re.sub(r'<[^>]+>', '', description)
                description = description.strip()
            
            return {
                'id': content_hash,
                'title': entry.get('title', '').strip(),
                'content': description,
                'url': entry.get('link', ''),
                'source': source,
                'published_at': published_at,
                'topic': topic,
                'source_type': 'rss',
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    async def _collect_from_reddit(self, session: aiohttp.ClientSession, 
                                  topics: List[str]) -> List[Dict[str, Any]]:
        """Collect articles from Reddit"""
        articles = []
        
        for topic in topics:
            if topic in self.reddit_endpoints:
                endpoint = self.reddit_endpoints[topic]
                
                try:
                    async with session.get(endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for post in data.get('data', {}).get('children', []):
                                article = self._parse_reddit_post(post['data'], topic)
                                if article:
                                    articles.append(article)
                        else:
                            logger.warning(f"Failed to fetch Reddit data: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching Reddit data for {topic}: {e}")
                    continue
        
        return articles
    
    def _parse_reddit_post(self, post_data: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Parse Reddit post into article format"""
        try:
            # Skip self posts without URLs
            if post_data.get('is_self', False) and not post_data.get('url_overridden_by_dest'):
                return None
            
            # Generate unique ID
            content_hash = hashlib.md5(
                (post_data.get('title', '') + post_data.get('id', '')).encode()
            ).hexdigest()
            
            # Extract timestamp
            created_utc = post_data.get('created_utc', 0)
            published_at = datetime.fromtimestamp(created_utc).isoformat() if created_utc else ''
            
            return {
                'id': content_hash,
                'title': post_data.get('title', '').strip(),
                'content': post_data.get('selftext', '')[:500] or post_data.get('title', ''),  # Use title if no content
                'url': post_data.get('url', ''),
                'source': 'reddit',
                'published_at': published_at,
                'topic': topic,
                'source_type': 'reddit',
                'collected_at': datetime.now().isoformat(),
                'score': post_data.get('score', 0),
                'num_comments': post_data.get('num_comments', 0)
            }
            
        except Exception as e:
            logger.error(f"Error parsing Reddit post: {e}")
            return None
    
    async def _collect_from_hackernews(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Collect articles from Hacker News"""
        articles = []
        
        try:
            # Get top stories
            async with session.get(f"{self.hackernews_api}/topstories.json") as response:
                if response.status == 200:
                    story_ids = await response.json()
                    
                    # Get details for top 20 stories
                    for story_id in story_ids[:20]:
                        try:
                            async with session.get(f"{self.hackernews_api}/item/{story_id}.json") as story_response:
                                if story_response.status == 200:
                                    story_data = await story_response.json()
                                    article = self._parse_hackernews_story(story_data)
                                    if article:
                                        articles.append(article)
                        except Exception as e:
                            logger.error(f"Error fetching HN story {story_id}: {e}")
                            continue
                else:
                    logger.warning(f"Failed to fetch HN top stories: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error collecting from Hacker News: {e}")
        
        return articles
    
    def _parse_hackernews_story(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Hacker News story into article format"""
        try:
            # Skip stories without URLs (Ask HN, etc.)
            if not story_data.get('url'):
                return None
            
            # Generate unique ID
            content_hash = hashlib.md5(
                (story_data.get('title', '') + str(story_data.get('id', ''))).encode()
            ).hexdigest()
            
            # Extract timestamp
            timestamp = story_data.get('time', 0)
            published_at = datetime.fromtimestamp(timestamp).isoformat() if timestamp else ''
            
            return {
                'id': content_hash,
                'title': story_data.get('title', '').strip(),
                'content': story_data.get('title', ''),  # HN doesn't have descriptions
                'url': story_data.get('url', ''),
                'source': 'hackernews',
                'published_at': published_at,
                'topic': 'technology',  # HN is primarily tech
                'source_type': 'hackernews',
                'collected_at': datetime.now().isoformat(),
                'score': story_data.get('score', 0),
                'num_comments': story_data.get('descendants', 0)
            }
            
        except Exception as e:
            logger.error(f"Error parsing HN story: {e}")
            return None
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', title)
            normalized_title = ' '.join(normalized_title.split())
            
            # Check for duplicates
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check
                if self._calculate_similarity(normalized_title, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _sort_articles(self, articles: List[Dict[str, Any]], filter_type: str) -> List[Dict[str, Any]]:
        """Sort articles based on filter type"""
        if filter_type == 'latest':
            return sorted(articles, key=lambda x: x.get('published_at', ''), reverse=True)
        elif filter_type == 'popular':
            # Sort by score if available, otherwise by comments
            return sorted(articles, key=lambda x: (
                x.get('score', 0) + x.get('num_comments', 0)
            ), reverse=True)
        elif filter_type == 'trending':
            # Combine recency and popularity
            now = datetime.now()
            
            def trending_score(article):
                # Parse publication date
                pub_date_str = article.get('published_at', '')
                try:
                    if pub_date_str:
                        pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        hours_old = (now - pub_date.replace(tzinfo=None)).total_seconds() / 3600
                        recency_score = max(0, 24 - hours_old) / 24  # Decay over 24 hours
                    else:
                        recency_score = 0
                except:
                    recency_score = 0
                
                popularity_score = (article.get('score', 0) + article.get('num_comments', 0)) / 100
                
                return recency_score * 0.6 + popularity_score * 0.4
            
            return sorted(articles, key=trending_score, reverse=True)
        else:
            return articles
    
    def health_check(self) -> bool:
        """Check if the news collector is healthy"""
        try:
            # Simple health check - try to parse a sample RSS feed
            sample_feed = "<?xml version='1.0'?><rss><channel><item><title>Test</title></item></channel></rss>"
            feedparser.parse(sample_feed)
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def clear_cache(self):
        """Clear any cached data"""
        # Reset error state
        self.last_error = None
        logger.info("News collector cache cleared")
    
    def restart(self):
        """Restart the news collector"""
        self.clear_cache()
        self.last_run = None
        logger.info("News collector restarted")
