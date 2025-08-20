import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from utils.config import AppConfig

logger = logging.getLogger(__name__)

class LangSmithMonitor:
    """LangSmith monitoring and tracing integration"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.langsmith_settings = config.get_langsmith_settings()
        self.api_key = config.get_langsmith_api_key()
        self.enabled = False
        
        # Initialize LangSmith if available
        self._initialize_langsmith()
    
    def _initialize_langsmith(self):
        """Initialize LangSmith tracing"""
        try:
            # Check if LangSmith is enabled in config
            if not self.langsmith_settings.get('enabled', False):
                logger.info("LangSmith monitoring is disabled in configuration")
                return
            
            # Check if API key is available
            if not self.api_key:
                logger.warning("LangSmith API key not found. Monitoring will be disabled.")
                logger.info("To enable LangSmith monitoring:")
                logger.info("1. Get an API key from https://smith.langchain.com/")
                logger.info("2. Add LANGSMITH_API_KEY to your secrets.env or environment variables")
                logger.info("3. Add it to your Streamlit secrets for deployment")
                return
            
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_settings.get('project_name', 'informa-news-analysis')
            
            # Set optional session name
            session_name = self.langsmith_settings.get('session_name')
            if session_name:
                os.environ["LANGCHAIN_SESSION"] = session_name
            else:
                # Auto-generate session name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.environ["LANGCHAIN_SESSION"] = f"informa_session_{timestamp}"
            
            self.enabled = True
            logger.info(f"LangSmith monitoring enabled for project: {os.environ['LANGCHAIN_PROJECT']}")
            logger.info(f"Session: {os.environ.get('LANGCHAIN_SESSION', 'default')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if LangSmith monitoring is enabled"""
        return self.enabled
    
    def create_run_metadata(self, workflow_type: str, **kwargs) -> Dict[str, Any]:
        """Create metadata for a LangSmith run"""
        metadata = {
            "workflow_type": workflow_type,
            "timestamp": datetime.now().isoformat(),
            "app_version": "1.0.0",
            "environment": "development" if self.config.is_development_mode() else "production"
        }
        
        # Add any additional metadata
        metadata.update(kwargs)
        
        return metadata
    
    def log_workflow_start(self, workflow_type: str, inputs: Dict[str, Any]) -> Optional[str]:
        """Log the start of a workflow execution"""
        if not self.enabled:
            return None
        
        try:
            metadata = self.create_run_metadata(
                workflow_type=workflow_type,
                topics=inputs.get('topics', []),
                sources=inputs.get('sources', []),
                max_articles=inputs.get('max_articles', 0)
            )
            
            logger.info(f"LangSmith: Starting {workflow_type} workflow with metadata: {metadata}")
            return metadata.get('timestamp')
            
        except Exception as e:
            logger.error(f"Failed to log workflow start: {e}")
            return None
    
    def log_workflow_end(self, workflow_type: str, outputs: Dict[str, Any], run_id: Optional[str] = None):
        """Log the end of a workflow execution"""
        if not self.enabled:
            return
        
        try:
            metadata = self.create_run_metadata(
                workflow_type=workflow_type,
                articles_processed=len(outputs.get('stored_articles', [])),
                errors_count=len(outputs.get('errors', [])),
                run_id=run_id
            )
            
            logger.info(f"LangSmith: Completed {workflow_type} workflow with metadata: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to log workflow end: {e}")
    
    def log_agent_execution(self, agent_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], workflow_type: str = "news_analysis"):
        """Log individual agent execution"""
        if not self.enabled:
            return
        
        try:
            metadata = self.create_run_metadata(
                workflow_type=workflow_type,
                agent=agent_name,
                input_size=len(str(inputs)),
                output_size=len(str(outputs)),
                success=not outputs.get('error')
            )
            
            logger.debug(f"LangSmith: Agent {agent_name} executed with metadata: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to log agent execution: {e}")
    
    def add_feedback(self, run_id: str, score: float, comment: str = ""):
        """Add feedback to a LangSmith run"""
        if not self.enabled:
            return
        
        try:
            # This would typically use the LangSmith SDK to add feedback
            # For now, we'll just log it
            logger.info(f"LangSmith feedback for run {run_id}: score={score}, comment='{comment}'")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
    
    def get_monitoring_url(self) -> Optional[str]:
        """Get the LangSmith monitoring dashboard URL"""
        if not self.enabled:
            return None
        
        project_name = os.environ.get('LANGCHAIN_PROJECT', 'informa-news-analysis')
        return f"https://smith.langchain.com/projects/{project_name}"
    
    def get_session_url(self) -> Optional[str]:
        """Get the current session URL in LangSmith"""
        if not self.enabled:
            return None
        
        project_name = os.environ.get('LANGCHAIN_PROJECT', 'informa-news-analysis')
        session_name = os.environ.get('LANGCHAIN_SESSION', 'default')
        return f"https://smith.langchain.com/projects/{project_name}/sessions/{session_name}"

def setup_langsmith_monitoring(config: AppConfig) -> LangSmithMonitor:
    """Setup and return LangSmith monitor instance"""
    monitor = LangSmithMonitor(config)
    
    if monitor.is_enabled():
        logger.info("✅ LangSmith monitoring is active")
        logger.info(f"📊 Dashboard: {monitor.get_monitoring_url()}")
        logger.info(f"🔗 Session: {monitor.get_session_url()}")
    else:
        logger.info("ℹ️ LangSmith monitoring is disabled")
    
    return monitor
