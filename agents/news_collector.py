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
        
        # Get configuration dynamically
        self.newsapi_key = config.get_newsapi_key() if config else None
        
        # Get news sources from config
        self.news_settings = config.get_news_setting if config else {}
        self.timeout = self.news_settings('timeout_seconds') if config else 30
        self.max_articles_per_source = self.news_settings('max_articles_per_source') if config else 50
        
        # Dynamic RSS feed configuration - no hardcoding
        self.rss_feeds = self._build_rss_feeds_config()
        
        # Dynamic Reddit and other API endpoints
        self.reddit_endpoints = self._build_reddit_config()
        self.hackernews_api = self._build_hackernews_config()
    
    def _build_rss_feeds_config(self) -> Dict[str, Dict[str, str]]:
        """Build RSS feed configuration dynamically"""
        base_feeds = {
            'bbc': {
                'base_url': 'http://feeds.bbci.co.uk/news',
                'topics': {
                    'technology': '/technology/rss.xml',
                    'politics': '/politics/rss.xml', 
                    'health': '/health/rss.xml',
                    'business': '/business/rss.xml',
                    'science': '/science_and_environment/rss.xml',
                    'world': '/world/rss.xml',
                    'entertainment': '/entertainment_and_arts/rss.xml'
                }
            },
            'reuters': {
                'base_url': 'https://www.reuters.com',
                'topics': {
                    'technology': '/technology/rss',
                    'politics': '/politics/rss',
                    'health': '/healthcare-pharmaceuticals/rss', 
                    'business': '/business/rss',
                    'science': '/science/rss',
                    'world': '/world/rss',
                    'entertainment': '/lifestyle/rss'
                }
            }
        }
        
        # Build full URLs
        rss_feeds = {}
        for source, config in base_feeds.items():
            rss_feeds[source] = {}
            for topic, path in config['topics'].items():
                rss_feeds[source][topic] = config['base_url'] + path
        
        return rss_feeds
    
    def _build_reddit_config(self) -> Dict[str, str]:
        """Build Reddit API configuration dynamically"""
        base_url = 'https://www.reddit.com/r'
        subreddits = {
            'technology': 'technology',
            'politics': 'politics', 
            'health': 'health',
            'business': 'business',
            'science': 'science',
            'world': 'worldnews',
            'entertainment': 'entertainment'
        }
        
        endpoints = {}
        for topic, subreddit in subreddits.items():
            endpoints[topic] = f"{base_url}/{subreddit}/hot.json?limit=25"
        
        return endpoints
    
    def _build_hackernews_config(self) -> str:
        """Build HackerNews API configuration"""
        return 'https://hacker-news.firebaseio.com/v0'
    
    async def collect_news(self, topics: List[str], sources: List[str], 
                          max_articles: int = 20, filter_type: str = 'latest') -> List[Dict[str, Any]]:
        """Collect news from multiple sources dynamically"""
        try:
            self.last_run = datetime.now().isoformat()
            logger.info(f"Starting dynamic news collection for topics: {topics}, sources: {sources}")
            
            all_articles = []
            
            # Create aiohttp session with dynamic timeout
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': 'Informa-NewsBot/1.0 (Multi-Agent News Analysis)',
                'Accept': 'application/json, text/xml, */*'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                # Collect from each source using dynamic dispatch
                collection_tasks = []
                
                for source in sources:
                    task = self._collect_from_source(session, source, topics, max_articles)
                    collection_tasks.append(task)
                
                # Run collections concurrently
                results = await asyncio.gather(*collection_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    source = sources[i]
                    if isinstance(result, Exception):
                        logger.error(f"Error collecting from {source}: {result}")
                        continue
                    
                    if isinstance(result, list):
                        all_articles.extend(result)
                        logger.info(f"Collected {len(result)} articles from {source}")
                    else:
                        logger.warning(f"Unexpected result type from {source}: {type(result)}")
            
            # Filter and process articles
            processed_articles = self._process_collected_articles(all_articles, filter_type, max_articles)
            
            logger.info(f"News collection completed. Total articles: {len(processed_articles)}")
            return processed_articles
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in news collection: {e}")
            return []
    
    async def _collect_from_source(self, session: aiohttp.ClientSession, source: str, 
                                 topics: List[str], max_articles: int) -> List[Dict[str, Any]]:
        """Dynamically collect from a specific source"""
        try:
            # Dynamic source dispatch
            if source in ['bbc', 'reuters']:
                return await self._collect_from_rss(session, source, topics)
            elif source == 'reddit':
                return await self._collect_from_reddit(session, topics)
            elif source == 'hackernews':
                return await self._collect_from_hackernews(session)
            elif source == 'newsapi' and self.newsapi_key:
                return await self._collect_from_newsapi(session, topics, max_articles)
            else:
                # Try to treat as RSS feed
                return await self._collect_from_generic_rss(session, source, topics)
                
        except Exception as e:
            logger.error(f"Error collecting from {source}: {e}")
            return []
    
    def _process_collected_articles(self, articles: List[Dict[str, Any]], 
                                  filter_type: str, max_articles: int) -> List[Dict[str, Any]]:
        """Process and filter collected articles"""
        try:
            # Remove duplicates based on title similarity
            unique_articles = self._remove_duplicates(articles)
            
            # Apply filtering
            filtered_articles = self._apply_filters(unique_articles, filter_type)
            
            # Limit number of articles
            if len(filtered_articles) > max_articles:
                filtered_articles = filtered_articles[:max_articles]
            
            # Add metadata
            for article in filtered_articles:
                article['collected_at'] = datetime.now().isoformat()
                article['id'] = self._generate_article_id(article)
            
            return filtered_articles
            
        except Exception as e:
            logger.error(f"Error processing articles: {e}")
            return articles[:max_articles] if articles else []
            
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
    
    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title and URL"""
        seen_titles = set()
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').strip().lower()
            url = article.get('url', '').strip()
            
            # Skip if we've seen this title or URL before
            if title in seen_titles or url in seen_urls:
                continue
            
            # Skip articles with empty titles or URLs
            if not title or not url:
                continue
            
            seen_titles.add(title)
            seen_urls.add(url)
            unique_articles.append(article)
        
        logger.info(f"Removed {len(articles) - len(unique_articles)} duplicate articles")
        return unique_articles

