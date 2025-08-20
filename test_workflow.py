#!/usr/bin/env python3
"""Simple test script to verify the news analysis workflow"""

import asyncio
import logging
from utils.config import AppConfig
from utils.database import VectorDatabase
from agents.workflow import NewsWorkflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_workflow():
    """Test the complete news analysis workflow"""
    try:
        logger.info("🔧 Initializing configuration...")
        config = AppConfig()
        
        logger.info("🗄️ Initializing database...")
        database = VectorDatabase(
            db_path="news_analysis.db",
            persist_directory="./chroma_db"
        )
        
        logger.info("🔄 Initializing workflow...")
        workflow = NewsWorkflow(config, database)
        
        logger.info("📰 Starting news collection workflow...")
        
        def progress_callback(message, progress):
            logger.info(f"Progress {progress}%: {message}")
        
        # Test with simple parameters
        articles = await workflow.execute_workflow(
            topics=["technology"],
            sources=["bbc"],
            max_articles=3,  # Small number for testing
            progress_callback=progress_callback
        )
        
        logger.info(f"✅ Workflow completed successfully!")
        logger.info(f"📊 Results: {len(articles)} articles processed")
        
        # Show a sample article
        if articles:
            sample = articles[0]
            logger.info("📄 Sample article:")
            logger.info(f"  Title: {sample.get('title', 'N/A')}")
            logger.info(f"  Source: {sample.get('source', 'N/A')}")
            logger.info(f"  Sentiment: {sample.get('sentiment_label', 'N/A')} ({sample.get('sentiment_score', 'N/A')})")
            logger.info(f"  Bias Score: {sample.get('bias_score', 'N/A')}")
            logger.info(f"  Quality Score: {sample.get('quality_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Informa workflow test...")
    
    # Run the async test
    success = asyncio.run(test_workflow())
    
    if success:
        logger.info("🎉 All tests passed! The system is working correctly.")
    else:
        logger.error("💥 Tests failed. Please check the logs for errors.")

if __name__ == "__main__":
    main()
