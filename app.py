import streamlit as st
import asyncio
from datetime import datetime
from agents.workflow import NewsAnalysisWorkflow
from utils.database import VectorDatabase
from utils.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Informa",
    page_icon="📰",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.workflow = None
    st.session_state.chat_messages = []
    st.session_state.articles = []

@st.cache_resource
def initialize_system():
    """Initialize the system"""
    try:
        config = Config()
        database = VectorDatabase()
        workflow = NewsAnalysisWorkflow(config, database)
        return workflow
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None

def main():
    st.title("📰 Informa")
    st.markdown("*Your AI-powered news companion*")
    
    # Initialize system
    if not st.session_state.initialized:
        workflow = initialize_system()
        if workflow:
            st.session_state.workflow = workflow
            st.session_state.initialized = True
        else:
            st.error("System initialization failed. Please check your configuration.")
            st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # News preferences
        st.subheader("📰 News Preferences")
        topics = st.multiselect(
            "Topics of Interest",
            ['technology', 'politics', 'health', 'business', 'sports', 'science', 'world', 'entertainment'],
            default=['technology', 'politics']
        )
        
        bias_preference = st.selectbox(
            "Bias Preference",
            ['All sources', 'Low bias only', 'Balanced coverage']
        )
        
        sentiment_preference = st.selectbox(
            "Sentiment Preference",
            ['All sentiments', 'Positive news', 'Neutral news', 'Negative news']
        )
        
        max_articles = st.slider("Max Articles to Fetch", 5, 20, 10)
        
        # Store preferences
        st.session_state.preferences = {
            'topics': topics,
            'bias_preference': bias_preference,
            'sentiment_preference': sentiment_preference,
            'max_articles': max_articles
        }
        
        # Fetch news button
        st.divider()
        if st.button("📰 Fetch Latest News", type="primary", use_container_width=True):
            fetch_news()
    
    # Main content - two tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Fact Check"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        fact_check_interface()

def fetch_news():
    """Fetch news articles and store in RAG"""
    try:
        workflow = st.session_state.workflow
        preferences = st.session_state.preferences
        
        with st.spinner("Fetching latest news..."):
            result = asyncio.run(workflow.run_complete_analysis({
                'topics': preferences['topics'],
                'sources': ['bbc', 'reuters', 'reddit'],
                'max_articles': preferences['max_articles'],
                'filter_type': 'latest'
            }))
            
            if result and result.get('success'):
                articles = result.get('articles', [])
                st.session_state.articles = articles
                
                # Add system message about fetched articles
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": f"📰 I've fetched {len(articles)} articles on your preferred topics! You can now ask me questions about the news."
                })
                
                st.success(f"✅ Successfully fetched {len(articles)} articles!")
                st.rerun()
            else:
                st.error("❌ Failed to fetch news. Please try again.")
    
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        logger.error(f"News fetching error: {e}")

def chat_interface():
    st.header("💬 Chat with Informa")
    st.markdown("Ask me anything about news, topics, or current events!")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_chat_message(prompt)
                st.write(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

def fact_check_interface():
    st.header("🔍 Fact Check & Analysis")
    st.markdown("Enter any claim, news, or statement to analyze its sentiment, bias, and truthfulness.")
    
    # Text input for fact checking
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Enter a claim, news snippet, or statement to analyze...",
        height=150
    )
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True) and user_input:
        with st.spinner("Analyzing..."):
            result = analyze_text(user_input)
            
            # Display results in natural language
            st.subheader("📋 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Truthfulness:** {result['truthfulness']}")
            
            with col2:
                st.info(f"**Sentiment:** {result['sentiment']}")
            
            with col3:
                st.info(f"**Bias Level:** {result['bias']}")
            
            # Detailed analysis
            st.subheader("📝 Detailed Analysis")
            st.write(result['analysis'])

def process_chat_message(message: str) -> str:
    """Process chat messages using RAG"""
    try:
        workflow = st.session_state.workflow
        
        # Use RAG to answer questions
        result = asyncio.run(workflow.answer_question(message))
        
        if result and result.get('answer'):
            return result['answer']
        else:
            return "I'm not sure about that. Could you try asking in a different way?"
    
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return "Sorry, I encountered an error. Please try again."

def analyze_text(text: str) -> dict:
    """Analyze text for sentiment, bias, and truthfulness"""
    try:
        workflow = st.session_state.workflow
        
        # Fact check
        fact_result = asyncio.run(workflow.fact_check_claim(text))
        
        # Sentiment analysis
        sentiment_result = asyncio.run(workflow.sentiment_analyzer.analyze(text))
        
        # Bias analysis
        bias_result = asyncio.run(workflow.bias_detector.analyze_bias(text, "user_input"))
        
        # Convert to natural language
        credibility = fact_result.get('credibility_score', 0.5) if fact_result else 0.5
        sentiment_label = sentiment_result.get('label', 'neutral') if sentiment_result else 'neutral'
        bias_score = bias_result.get('overall_bias_score', 0.5) if bias_result else 0.5
        
        # Truthfulness in natural language
        if credibility > 0.8:
            truthfulness = "This appears to be true"
        elif credibility > 0.6:
            truthfulness = "This appears to be mostly true"
        elif credibility > 0.4:
            truthfulness = "This appears to be partially true"
        else:
            truthfulness = "This appears to be false or misleading"
        
        # Sentiment in natural language
        if sentiment_label == 'positive':
            sentiment = "Positive and optimistic"
        elif sentiment_label == 'negative':
            sentiment = "Negative and pessimistic"
        else:
            sentiment = "Neutral and balanced"
        
        # Bias in natural language
        if bias_score < 0.3:
            bias = "Low bias, fairly objective"
        elif bias_score < 0.6:
            bias = "Moderate bias, some subjectivity"
        else:
            bias = "High bias, strongly opinionated"
        
        # Generate analysis
        analysis = f"Based on my analysis, this text appears to be {truthfulness.lower()}. "
        analysis += f"The tone is {sentiment.lower()}, and the content shows {bias.lower()}. "
        
        if fact_result and fact_result.get('analysis'):
            analysis += f"\n\n{fact_result['analysis']}"
        
        return {
            'truthfulness': truthfulness,
            'sentiment': sentiment,
            'bias': bias,
            'analysis': analysis
        }
    
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return {
            'truthfulness': "Unable to determine",
            'sentiment': "Unable to determine", 
            'bias': "Unable to determine",
            'analysis': "Sorry, I encountered an error while analyzing this text."
        }

if __name__ == "__main__":
    main()
