#!/usr/bin/env python3
"""
Setup script for the Real Multi-Agent News Analysis System
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "./chroma_db",
        "./logs",
        "./cache",
        "./hf_cache"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        logger.info("Creating .env file with default settings...")
        
        env_content = """# Multi-Agent News Analysis System Configuration

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Optional API Keys (system works without these)
NEWSAPI_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# Feature Flags
ENABLE_FACT_CHECKING=true
ENABLE_BIAS_DETECTION=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_RAG=true
ENABLE_WEB_SCRAPING=false
ENABLE_CACHING=true

# Performance Settings
USE_GPU=false
HF_CACHE_DIR=./hf_cache

# Database Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info(f"Created {env_file}")
        except Exception as e:
            logger.error(f"Failed to create {env_file}: {e}")
            return False
    else:
        logger.info(f"{env_file} already exists")
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'transformers',
        'sentence_transformers',
        'torch',
        'chromadb',
        'aiohttp',
        'requests',
        'beautifulsoup4',
        'feedparser',
        'langgraph',
        'langchain'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError as e:
            logger.error(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    logger.info("All required packages imported successfully")
    return True

def download_models():
    """Download required Hugging Face models"""
    logger.info("Downloading Hugging Face models...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        
        # Download sentiment analysis model
        logger.info("Downloading sentiment analysis model...")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        logger.info("✓ Sentiment analysis model downloaded")
        
        # Download embedding model
        logger.info("Downloading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Embedding model downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False

def run_health_check():
    """Run a basic health check"""
    logger.info("Running health check...")
    
    try:
        # Test ChromaDB
        import chromadb
        client = chromadb.Client()
        logger.info("✓ ChromaDB working")
        
        # Test basic functionality
        from utils.config import Config
        config = Config()
        logger.info("✓ Configuration loading")
        
        from utils.database import VectorDatabase
        db = VectorDatabase()
        logger.info("✓ Database initialization")
        
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting setup for Real Multi-Agent News Analysis System")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Installing requirements", install_requirements),
        ("Testing imports", test_imports),
        ("Downloading models", download_models),
        ("Running health check", run_health_check)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        if not step_func():
            logger.error(f"Setup failed at step: {step_name}")
            sys.exit(1)
        logger.info(f"✓ {step_name} completed")
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the application with: streamlit run app.py")

if __name__ == "__main__":
    main()
