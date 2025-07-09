import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # API Keys (all free or not required for basic functionality)
        self.api_keys = {
            # These are optional - the system works without them
            'newsapi': os.getenv('NEWSAPI_KEY', ''),  # Free tier available
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID', ''),  # Not required for public access
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),  # Not required for public access
        }
        
        # Free API Endpoints
        self.api_endpoints = {
            'newsapi': 'https://newsapi.org/v2',
            'factcheck': 'https://factchecktools.googleapis.com/v1alpha1',
            'openai': 'https://api.openai.com/v1',
            'hackernews': 'https://hacker-news.firebaseio.com/v0',
            'reddit': 'https://www.reddit.com',
            'google_news_rss': 'https://news.google.com/rss',
            'bbc_rss': 'http://feeds.bbci.co.uk/news',
            'reuters_rss': 'https://www.reuters.com'
        }
        
        # Rate Limits (conservative to avoid issues)
        self.rate_limits = {
            'newsapi': {'requests_per_hour': 1000},
            'twitter': {'requests_per_15min': 300},
            'openai': {'requests_per_minute': 60},
            'rss_feeds': {'requests_per_minute': 30},
            'reddit': {'requests_per_minute': 60},
            'hackernews': {'requests_per_minute': 100},
            'web_scraping': {'requests_per_minute': 20}
        }
        
        # Analysis Settings
        self.analysis_settings = {
            'sentiment': {
                'confidence_threshold': 0.7,
                'batch_size': 10,
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
            },
            'bias': {
                'threshold': 0.5,
                'categories': ['political', 'sensational', 'emotional', 'loaded_language'],
                'confidence_threshold': 0.6
            },
            'fact_check': {
                'timeout': 30,
                'min_credibility': 0.3,
                'max_claims_per_article': 5
            },
            'rag': {
                'max_context_length': 4000,
                'max_sources': 5,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        }
        
        # Database Settings
        self.database_settings = {
            'persist_directory': './chroma_db',
            'collection_name': 'news_articles',
            'max_articles': 10000,
            'cleanup_interval_days': 30
        }
        
        # News Collection Settings
        self.news_settings = {
            'max_articles_per_source': 50,
            'max_articles_per_topic': 20,
            'update_interval_minutes': 15,
            'supported_languages': ['en'],
            'default_topics': ['technology', 'politics', 'health', 'business', 'science', 'world'],
            'default_sources': ['bbc', 'reuters', 'reddit', 'hackernews'],
            'timeout_seconds': 30
        }
        
        # Workflow Settings
        self.workflow_settings = {
            'max_concurrent_tasks': 5,
            'retry_attempts': 3,
            'retry_delay_seconds': 2,
            'enable_caching': True,
            'cache_ttl_hours': 1
        }
        
        # Logging Settings
        self.logging_settings = {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_logging': False,
            'log_file': 'news_analysis.log'
        }
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("Configuration initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.logging_settings['level'].upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=self.logging_settings['format'],
            handlers=[
                logging.StreamHandler()  # Console output
            ]
        )
        
        # Add file handler if enabled
        if self.logging_settings['file_logging']:
            file_handler = logging.FileHandler(self.logging_settings['log_file'])
            file_handler.setFormatter(logging.Formatter(self.logging_settings['format']))
            logging.getLogger().addHandler(file_handler)
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service"""
        return self.api_keys.get(service, '')
    
    def get_endpoint(self, service: str) -> str:
        """Get API endpoint for a specific service"""
        return self.api_endpoints.get(service, '')
    
    def get_rate_limit(self, service: str) -> Dict[str, int]:
        """Get rate limit settings for a specific service"""
        return self.rate_limits.get(service, {'requests_per_minute': 60})
    
    def get_analysis_setting(self, category: str, key: str = None) -> Any:
        """Get analysis configuration setting"""
        if key:
            return self.analysis_settings.get(category, {}).get(key)
        return self.analysis_settings.get(category, {})
    
    def get_database_setting(self, key: str) -> Any:
        """Get database configuration setting"""
        return self.database_settings.get(key)
    
    def get_news_setting(self, key: str) -> Any:
        """Get news collection setting"""
        return self.news_settings.get(key)
    
    def get_workflow_setting(self, key: str) -> Any:
        """Get workflow setting"""
        return self.workflow_settings.get(key)
    
    def update_setting(self, category: str, key: str, value: Any):
        """Update a configuration setting"""
        if hasattr(self, category):
            settings = getattr(self, category)
            if isinstance(settings, dict) and key in settings:
                settings[key] = value
                logger.info(f"Updated {category}.{key} = {value}")
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check if required directories exist or can be created
        try:
            persist_dir = self.database_settings['persist_directory']
            os.makedirs(persist_dir, exist_ok=True)
            validation_results['database_directory'] = True
        except Exception as e:
            validation_results['database_directory'] = False
            logger.error(f"Cannot create database directory: {e}")
        
        # Check API endpoints accessibility (basic validation)
        validation_results['endpoints_configured'] = bool(self.api_endpoints)
        
        # Check analysis settings
        validation_results['analysis_configured'] = bool(self.analysis_settings)
        
        # Check news sources
        validation_results['news_sources_configured'] = bool(self.news_settings)
        
        # Check workflow settings
        validation_results['workflow_configured'] = bool(self.workflow_settings)
        
        # Overall validation
        validation_results['overall'] = all(validation_results.values())
        
        if validation_results['overall']:
            logger.info("Configuration validation passed")
        else:
            logger.warning("Configuration validation failed for some components")
        
        return validation_results
    
    def get_model_settings(self) -> Dict[str, Any]:
        """Get Hugging Face model settings"""
        return {
            'sentiment_model': self.analysis_settings['sentiment']['model_name'],
            'embedding_model': self.analysis_settings['rag']['embedding_model'],
            'device': 'cuda' if os.getenv('USE_GPU', 'false').lower() == 'true' else 'cpu',
            'cache_dir': os.getenv('HF_CACHE_DIR', './hf_cache')
        }
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return os.getenv('ENVIRONMENT', 'production').lower() == 'development'
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags for enabling/disabling functionality"""
        return {
            'enable_fact_checking': os.getenv('ENABLE_FACT_CHECKING', 'true').lower() == 'true',
            'enable_bias_detection': os.getenv('ENABLE_BIAS_DETECTION', 'true').lower() == 'true',
            'enable_sentiment_analysis': os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true').lower() == 'true',
            'enable_rag': os.getenv('ENABLE_RAG', 'true').lower() == 'true',
            'enable_web_scraping': os.getenv('ENABLE_WEB_SCRAPING', 'false').lower() == 'true',
            'enable_caching': self.workflow_settings.get('enable_caching', True)
        }
