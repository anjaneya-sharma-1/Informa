import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

import aiohttp
from newspaper import Article as NPArticle


logger = logging.getLogger(__name__)


class NewsCollectorAgent:
    """
    Simplified: NewsAPI-only collection with optional content enrichment.
    - Uses /top-headlines for category-aligned topics when possible, else /everything.
    - Ignores RSS/Reddit/HN/Tavily; keeps workflow-compatible method signature.
    """

    def __init__(self, config):
        self.config = config
        self.session = None
        self.last_run = None
        self.last_error = None

        # Config
        self.newsapi_key = config.get_newsapi_key() if config else None
        try:
            if hasattr(config, "get_endpoint"):
                self.base_url = config.get_endpoint("newsapi")
            elif hasattr(config, "get_api_endpoints"):
                self.base_url = (config.get_api_endpoints() or {}).get("newsapi", "https://newsapi.org/v2")
            else:
                self.base_url = "https://newsapi.org/v2"
        except Exception:
            self.base_url = "https://newsapi.org/v2"

        get_setting = getattr(config, "get_news_setting", lambda k, d=None: d)
        self.timeout = get_setting("timeout_seconds", 30)
        # Correct key: prefer explicit per-topic limit if present
        self.max_articles_per_topic = get_setting("max_articles_per_topic", get_setting("max_articles_per_source", 50))
        self.language = get_setting("language", "en")
        self.country = get_setting("country", "us")
        self.days_back = get_setting("days_back", 3)
        self.default_sort = get_setting("sort_by", "publishedAt")
        self.enrich_max = get_setting("enrich_max", 10)

        # Topic to NewsAPI category mapping
        self.topic_category_map = {
            "business": "business",
            "entertainment": "entertainment",
            "general": "general",
            "health": "health",
            "science": "science",
            "sports": "sports",
            "technology": "technology",
            # Map extra topics to closest category
            "world": "general",
            "politics": "general",
        }

    async def collect_news(
        self,
        topics: List[str],
        sources: List[str],  # ignored; kept for compatibility
        max_articles: int = 20,
        filter_type: str = "latest",
    ) -> List[Dict[str, Any]]:
        try:
            self.last_run = datetime.utcnow().isoformat()
            if not self.newsapi_key:
                logger.warning("NEWSAPI_KEY not set. Skipping NewsAPI collection.")
                return []

            headers = {
                "User-Agent": "Informa-NewsBot/1.0",
                "Accept": "application/json",
                "X-Api-Key": self.newsapi_key,
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            all_articles: List[Dict[str, Any]] = []

            per_topic_cap = max(1, min(self.max_articles_per_topic, max_articles))

            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                topic_list = topics or ["technology"]
                num_topics = len(topic_list)
                # Distribute allocation more evenly so first topic doesn't consume all slots
                base_alloc = max(1, max_articles // num_topics)
                remainder = max_articles % num_topics

                logger.debug(f"Topic allocation calculation: topics={topic_list}, max_articles={max_articles}, base_alloc={base_alloc}, remainder={remainder}, per_topic_cap={per_topic_cap}")
                for idx, topic in enumerate(topic_list):
                    remaining = max(0, max_articles - len(all_articles))
                    if remaining <= 0:
                        break
                    allocated = base_alloc + (1 if idx < remainder else 0)
                    per_topic_limit = min(per_topic_cap, allocated, remaining)

                    if per_topic_limit <= 0:
                        continue

                    category = self.topic_category_map.get(topic.lower())
                    if category:
                        articles = await self._collect_top_headlines(session, category, per_topic_limit)
                    else:
                        articles = await self._collect_everything(session, topic, per_topic_limit)

                    for a in articles:
                        a["topic"] = topic[:100]
                        a.setdefault("source_type", "newsapi")
                    if articles:
                        logger.debug(f"Collected {len(articles)} from topic '{topic}' (limit {per_topic_limit})")
                    all_articles.extend(articles)

            # Dedupe, sort, cap
            processed = self._process_collected_articles(all_articles, filter_type, max_articles)

            # Enrich content (best-effort, NP3k-only) for top N
            self._enrich_articles(processed[: self.enrich_max])

            logger.info(f"NewsAPI collection complete. Returned {len(processed)} articles.")
            return processed

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in NewsAPI-only collection: {e}")
            return []

    async def _collect_top_headlines(self, session: aiohttp.ClientSession, category: str, limit: int) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        page = 1
        collected = 0
        while collected < limit and page <= 5:
            page_size = min(100, limit - collected)
            params = {
                "category": category,
                "country": self.country,
                "pageSize": page_size,
                "page": page,
            }
            # Remove None values to avoid invalid query params
            params = {k: v for k, v in params.items() if v is not None}
            try:
                async with session.get(f"{self.base_url}/top-headlines", params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"top-headlines non-200 ({resp.status}) cat '{category}': {text[:200]}")
                        break
                    data = await resp.json()
                    items = (data or {}).get("articles", []) or []
                    if not items:
                        break
                    for item in items:
                        a = self._newsapi_item_to_article(item)
                        if a:
                            articles.append(a)
                            collected += 1
                            if collected >= limit:
                                break
                    total_results = (data or {}).get("totalResults", 0)
                    if page * page_size >= total_results:
                        break
                page += 1
            except Exception as e:
                logger.error(f"NewsAPI top-headlines error for category '{category}' page {page}: {e}")
                break
        return articles

    async def _collect_everything(self, session: aiohttp.ClientSession, query: str, limit: int) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        frm = (datetime.utcnow() - timedelta(days=int(self.days_back))).isoformat(timespec="seconds") + "Z"
        sort_by = self.default_sort
        page = 1
        collected = 0
        while collected < limit and page <= 5:
            page_size = min(100, limit - collected)
            params = {
                "q": query,
                "language": self.language,
                "pageSize": page_size,
                "page": page,
                "from": frm,
                "sortBy": sort_by,
            }
            # Remove None values to avoid invalid query params
            params = {k: v for k, v in params.items() if v is not None}
            try:
                async with session.get(f"{self.base_url}/everything", params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"everything non-200 ({resp.status}) q '{query}': {text[:200]}")
                        break
                    data = await resp.json()
                    items = (data or {}).get("articles", []) or []
                    if not items:
                        break
                    for item in items:
                        a = self._newsapi_item_to_article(item)
                        if a:
                            articles.append(a)
                            collected += 1
                            if collected >= limit:
                                break
                    total_results = (data or {}).get("totalResults", 0)
                    if page * page_size >= total_results:
                        break
                page += 1
            except Exception as e:
                logger.error(f"NewsAPI everything error for query '{query}' page {page}: {e}")
                break
        return articles

    def _newsapi_item_to_article(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url:
                return None
            description = (item.get("description") or "").strip()
            content = (item.get("content") or "").strip()
            published_at = (item.get("publishedAt") or "").strip()
            source_name = ((item.get("source") or {}).get("name") or "").strip()
            return {
                "id": self._generate_article_id_dict(title=title, url=url),
                "title": title[:1000],
                "content": (content or description or title)[:10000],
                "url": url[:500],
                "source": source_name or "newsapi",
                "published_at": published_at,
                "source_type": "newsapi",
                "collected_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to parse NewsAPI item: {e}")
            return None

    def _process_collected_articles(self, articles: List[Dict[str, Any]], filter_type: str, max_articles: int) -> List[Dict[str, Any]]:
        if not articles:
            return []
        unique = self._remove_duplicates(articles)
        sorted_articles = self._sort_articles(unique, filter_type)
        final_articles = sorted_articles[:max_articles]
        for a in final_articles:
            a.setdefault("collected_at", datetime.utcnow().isoformat())
            a.setdefault("source_type", "newsapi")
            a.setdefault("topic", "news")
            a.setdefault("published_at", "")
        return final_articles

    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_titles = set()
        seen_urls = set()
        unique: List[Dict[str, Any]] = []
        for a in articles:
            title = (a.get("title") or "").strip().lower()
            url = (a.get("url") or "").strip()
            if not title or not url:
                continue
            normalized = re.sub(r"[^\w\s]", "", title)
            normalized = " ".join(normalized.split())
            if normalized in seen_titles or url in seen_urls:
                continue
            seen_titles.add(normalized)
            seen_urls.add(url)
            unique.append(a)
        logger.info(f"Removed {len(articles) - len(unique)} duplicates")
        return unique

    def _sort_articles(self, articles: List[Dict[str, Any]], filter_type: str) -> List[Dict[str, Any]]:
        if not articles:
            return []
        if (filter_type or "").lower() == "latest":
            def parse_dt(s: str) -> float:
                try:
                    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
                except Exception:
                    return 0.0
            return sorted(articles, key=lambda a: parse_dt(a.get("published_at", "")), reverse=True)
        return sorted(articles, key=lambda a: (a.get("title") or "").lower())

    def _generate_article_id_dict(self, title: str, url: str) -> str:
        base = f"{title}::{url}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def _enrich_articles(self, articles: List[Dict[str, Any]]):
        if not articles:
            return
        for a in articles:
            try:
                # Skip if content already long enough
                if len(a.get("content", "")) >= 500:
                    continue
                url = a.get("url")
                if not url:
                    continue
                text = None
                try:
                    art = NPArticle(url)
                    art.download()
                    art.parse()
                    text = (art.text or "").strip()
                except Exception:
                    text = None
                if text:
                    a["content"] = (a.get("title", "") + "\n" + text)[:20000]
            except Exception:
                continue

