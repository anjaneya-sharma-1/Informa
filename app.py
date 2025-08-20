import streamlit as st
import asyncio
import logging
from datetime import datetime
import json
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from utils.database import VectorDatabase
from utils.config import AppConfig
from agents.workflow import NewsWorkflow
from agents.chat_agent import ChatAgent
from agents.fact_checker import FactCheckerAgent

# Page config
st.set_page_config(
    page_title="Informa - AI News Analysis & Fact Checking",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InformaApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = AppConfig()
        self.db = VectorDatabase()
        self.workflow = NewsWorkflow(self.config, self.db)
        self.chat_agent = ChatAgent(self.config, self.db)
        self.fact_checker = FactCheckerAgent(self.config)
    
    def run(self):
        """Main application entry point"""
        st.title("🌍 Informa - AI News Analysis & Fact Checking")
        st.markdown("*Multi-agent RAG system for news analysis and fact verification*")
        
        # Initialize session state
        self._init_session_state()
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📰 News Collection & Analysis", 
            "💬 News Chat", 
            "🔍 Fact Checking",
            "📊 Database Statistics"
        ])
        
        with tab1:
            self._render_news_collection_tab()
        
        with tab2:
            self._render_chat_tab()
        
        with tab3:
            self._render_fact_checking_tab()
        
        with tab4:
            self._render_statistics_tab()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'collected_articles' not in st.session_state:
            st.session_state.collected_articles = []
        
        if 'fact_check_results' not in st.session_state:
            st.session_state.fact_check_results = []
        
        if 'workflow_running' not in st.session_state:
            st.session_state.workflow_running = False
    
    def _render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("🔧 Configuration")
        
        # News collection settings
        st.sidebar.subheader("News Collection")
        
        # Topic selection
        available_topics = [
            "technology", "politics", "health", "business", 
            "science", "world", "entertainment", "sports"
        ]
        
        selected_topics = st.sidebar.multiselect(
            "Select Topics",
            available_topics,
            default=["technology", "politics", "health"],
            help="Choose news topics to collect"
        )
        
        # Source selection
        available_sources = [
            "bbc", "reuters", "google_news", "reddit", "hackernews"
        ]
        
        selected_sources = st.sidebar.multiselect(
            "Select Sources",
            available_sources,
            default=["bbc", "reuters"],
            help="Choose news sources to collect from"
        )
        
        # Filter preferences
        st.sidebar.subheader("Content Filters")
        
        sentiment_filter = st.sidebar.selectbox(
            "Sentiment Filter",
            ["all", "positive", "neutral", "negative"],
            help="Filter articles by sentiment"
        )
        
        bias_threshold = st.sidebar.slider(
            "Bias Threshold",
            0.0, 1.0, 0.5,
            help="Filter articles with bias score below threshold"
        )
        
        max_articles = st.sidebar.number_input(
            "Max Articles per Collection",
            min_value=5,
            max_value=100,
            value=20,
            help="Maximum number of articles to collect"
        )
        
        # Store in session state
        st.session_state.selected_topics = selected_topics
        st.session_state.selected_sources = selected_sources
        st.session_state.sentiment_filter = sentiment_filter
        st.session_state.bias_threshold = bias_threshold
        st.session_state.max_articles = max_articles
        
        # API Status
        st.sidebar.subheader("🔌 API Status")
        hf_key = self.config.get_huggingface_key()
        if hf_key:
            st.sidebar.success("✅ Hugging Face API Connected")
        else:
            st.sidebar.error("❌ Hugging Face API Key Missing")
            st.sidebar.info("Add HF_API_KEY to secrets.env file")
        
        # Database info
        total_articles = self.db.get_count()
        st.sidebar.info(f"📚 Total Articles in DB: {total_articles}")
        
        # Clear database button
        if st.sidebar.button("🗑️ Clear Database", type="secondary"):
            if st.sidebar.confirm("Are you sure you want to clear all articles?"):
                self.db.clear_database()
                st.sidebar.success("Database cleared!")
                st.rerun()
    
    def _render_news_collection_tab(self):
        """Render news collection and analysis tab"""
        st.header("📰 News Collection & Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Instructions:**
            1. Configure your preferences in the sidebar
            2. Click 'Start News Collection' to fetch and analyze articles
            3. Articles will be automatically analyzed for sentiment and bias
            4. Results are stored in the vector database for chat queries
            """)
        
        with col2:
            if not st.session_state.workflow_running:
                if st.button("🚀 Start News Collection", type="primary"):
                    self._run_news_collection_workflow()
            else:
                st.warning("⏳ Collection in progress...")
                if st.button("⏹️ Stop Collection"):
                    st.session_state.workflow_running = False
                    st.rerun()
        
        # Display recent collection results
        if st.session_state.collected_articles:
            st.subheader("📋 Recently Collected Articles")
            self._display_article_results(st.session_state.collected_articles)
        
        # Display recent articles from database
        st.subheader("🗂️ Recent Articles in Database")
        recent_articles = self.db.get_recent_articles(limit=10)
        if recent_articles:
            self._display_article_results(recent_articles)
        else:
            st.info("No articles in database. Start collecting news above!")
    
    def _render_chat_tab(self):
        """Render news chat interface"""
        st.header("💬 News Chat Interface")
        
        st.markdown("""
        **Ask questions about the news!**
        
        Examples:
        - "What are the latest technology developments?"
        - "Tell me about recent political news with positive sentiment"
        - "What health news has been reported recently?"
        - "Summarize the most credible business news"
        """)
        
        # Chat input
        user_query = st.chat_input("Ask about news articles...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process query with chat agent
            with st.spinner("🤔 Thinking..."):
                response = asyncio.run(self._process_chat_query(user_query))
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    def _render_fact_checking_tab(self):
        """Render fact checking interface"""
        st.header("🔍 Fact Checking & Trustworthiness Analysis")
        
        st.markdown("""
        **Submit any claim or headline for fact-checking:**
        
        The system will:
        - Search multiple fact-checking sources
        - Analyze text characteristics and credibility indicators
        - Provide a trustworthiness score with evidence
        - Cross-reference with reliable news sources
        """)
        
        # Fact check input
        claim_input = st.text_area(
            "Enter claim or headline to fact-check:",
            placeholder="Example: 'Scientists discover cure for cancer using AI technology'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Fact Check", type="primary", disabled=not claim_input.strip()):
                self._run_fact_check(claim_input.strip())
        
        with col2:
            if st.button("📋 Use Sample Claims"):
                sample_claims = [
                    "Climate change is caused by solar radiation, not human activity",
                    "Vaccines contain microchips for tracking people",
                    "The 2024 Olympics will be held in Paris, France",
                    "Drinking lemon water can cure diabetes"
                ]
                selected_sample = st.selectbox("Choose a sample claim:", sample_claims)
                if st.button("Use Selected Sample"):
                    st.session_state.sample_claim = selected_sample
                    st.rerun()
        
        # Use sample claim if selected
        if hasattr(st.session_state, 'sample_claim'):
            claim_input = st.session_state.sample_claim
            del st.session_state.sample_claim
        
        # Display fact check results
        if st.session_state.fact_check_results:
            st.subheader("🔬 Fact Check Results")
            for result in reversed(st.session_state.fact_check_results[-5:]):  # Show last 5
                self._display_fact_check_result(result)
    
    def _render_statistics_tab(self):
        """Render database statistics"""
        st.header("📊 Database Statistics")
        
        # Get statistics
        stats = self.db.get_statistics()
        
        if stats and stats.get('total_articles', 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Articles", stats['total_articles'])
            
            with col2:
                avg_credibility = stats.get('avg_credibility', 0)
                st.metric("Avg Credibility", f"{avg_credibility:.2f}")
            
            with col3:
                sentiment_dist = stats.get('sentiment_distribution', {})
                most_common_sentiment = max(sentiment_dist.keys(), key=lambda k: sentiment_dist[k]) if sentiment_dist else "N/A"
                st.metric("Top Sentiment", most_common_sentiment)
            
            with col4:
                sources = stats.get('sources', {})
                top_source = max(sources.keys(), key=lambda k: sources[k]) if sources else "N/A"
                st.metric("Top Source", top_source)
            
            # Detailed breakdowns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📺 Sources")
                sources = stats.get('sources', {})
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{source}**: {count} articles")
            
            with col2:
                st.subheader("🏷️ Topics")
                topics = stats.get('topics', {})
                for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{topic}**: {count} articles")
            
            # Sentiment distribution
            st.subheader("😊 Sentiment Distribution")
            sentiment_dist = stats.get('sentiment_distribution', {})
            if sentiment_dist:
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame(list(sentiment_dist.items()), columns=['Sentiment', 'Count'])
                fig = px.pie(df, values='Count', names='Sentiment', title="Article Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No articles in database yet. Start collecting news to see statistics!")
    
    def _run_news_collection_workflow(self):
        """Run the news collection workflow"""
        st.session_state.workflow_running = True
        
        try:
            # Get configuration from session state
            topics = st.session_state.get('selected_topics', ['technology'])
            sources = st.session_state.get('selected_sources', ['bbc'])
            max_articles = st.session_state.get('max_articles', 20)
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run workflow
                status_text.text("🔍 Starting news collection...")
                progress_bar.progress(10)
                
                # Execute workflow asynchronously
                articles = asyncio.run(self.workflow.execute_workflow(
                    topics=topics,
                    sources=sources,
                    max_articles=max_articles,
                    progress_callback=lambda msg, pct: self._update_progress(status_text, progress_bar, msg, pct)
                ))
                
                progress_bar.progress(100)
                status_text.text(f"✅ Collection complete! Processed {len(articles)} articles")
                
                # Store results
                st.session_state.collected_articles = articles
                
        except Exception as e:
            st.error(f"Error during news collection: {str(e)}")
            logger.error(f"Workflow error: {e}")
        
        finally:
            st.session_state.workflow_running = False
    
    def _update_progress(self, status_text, progress_bar, message, percentage):
        """Update progress display"""
        status_text.text(message)
        progress_bar.progress(percentage)
    
    async def _process_chat_query(self, query: str) -> str:
        """Process chat query with the chat agent"""
        try:
            response = await self.chat_agent.process_query(query)
            return response.get('response', 'Sorry, I could not process your query.')
        except Exception as e:
            logger.error(f"Chat query error: {e}")
            return f"Error processing query: {str(e)}"
    
    def _run_fact_check(self, claim: str):
        """Run fact checking on a claim"""
        try:
            with st.spinner("🔍 Fact-checking claim..."):
                result = asyncio.run(self.fact_checker.check_claim(claim))
                
                # Store result
                st.session_state.fact_check_results.append(result)
                
                st.success("Fact check complete!")
        
        except Exception as e:
            st.error(f"Error during fact checking: {str(e)}")
            logger.error(f"Fact check error: {e}")
    
    def _display_article_results(self, articles: List[Dict[str, Any]]):
        """Display article results in a formatted way"""
        for article in articles:
            with st.expander(f"📄 {article.get('title', 'Untitled')[:100]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    st.write(f"**Topic:** {article.get('topic', 'Unknown')}")
                    
                    description = article.get('content', '') or article.get('description', '')
                    if description:
                        st.write(f"**Description:** {description[:300]}...")
                    
                    if article.get('url'):
                        st.write(f"**URL:** {article['url']}")
                
                with col2:
                    # Sentiment
                    sentiment = article.get('sentiment_label', 'neutral')
                    sentiment_score = article.get('sentiment_score', 0.0)
                    
                    if sentiment == 'positive':
                        st.success(f"😊 {sentiment.title()}")
                    elif sentiment == 'negative':
                        st.error(f"😞 {sentiment.title()}")
                    else:
                        st.info(f"😐 {sentiment.title()}")
                    
                    st.write(f"Score: {sentiment_score:.2f}")
                    
                    # Bias
                    bias_score = article.get('bias_score', 0.0)
                    if bias_score > 0.7:
                        st.warning(f"⚠️ High Bias: {bias_score:.2f}")
                    elif bias_score > 0.4:
                        st.info(f"⚡ Medium Bias: {bias_score:.2f}")
                    else:
                        st.success(f"✅ Low Bias: {bias_score:.2f}")
                    
                    # Credibility
                    credibility = article.get('credibility_score', 0.5)
                    st.write(f"🔍 Credibility: {credibility:.2f}")
    
    def _display_fact_check_result(self, result: Dict[str, Any]):
        """Display fact check result"""
        with st.expander(f"🔍 {result.get('original_claim', '')[:100]}..."):
            
            # Credibility score with color coding
            credibility = result.get('credibility_score', 0.5)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Original Claim:** {result.get('original_claim', '')}")
                
                analysis = result.get('analysis', {})
                if analysis:
                    st.write(f"**Analysis:** {analysis.get('summary', 'No analysis available')}")
                
                # Evidence
                evidence = result.get('evidence', [])
                if evidence:
                    st.write("**Supporting Evidence:**")
                    for item in evidence[:3]:  # Show top 3
                        st.write(f"- {item}")
                
                contradictions = result.get('contradictions', [])
                if contradictions:
                    st.write("**Contradicting Evidence:**")
                    for item in contradictions[:3]:  # Show top 3
                        st.write(f"- {item}")
            
            with col2:
                # Credibility score
                if credibility >= 0.7:
                    st.success(f"✅ High Credibility: {credibility:.2f}")
                elif credibility >= 0.4:
                    st.warning(f"⚠️ Medium Credibility: {credibility:.2f}")
                else:
                    st.error(f"❌ Low Credibility: {credibility:.2f}")
                
                # Confidence
                confidence = result.get('confidence', 'medium')
                st.write(f"**Confidence:** {confidence}")
                
                # Sources checked
                sources = result.get('sources_checked', [])
                if sources:
                    st.write(f"**Sources:** {len(sources)} checked")

def main():
    """Main application entry point"""
    try:
        app = InformaApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()