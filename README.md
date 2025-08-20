# 📰 Informa

**Your AI-powered news companion** - A clean, simple multi-agent system for news analysis and fact-checking.

## Features

### 💬 Chat Interface
- **Natural Conversation**: Ask anything about news, topics, or current events
- **Fetch News Button**: Click to collect latest articles on your preferred topics
- **RAG-powered Responses**: Intelligent answers based on collected articles
- **Automatic Learning**: When you ask about topics not in the database, it searches and stores new information

### 🔍 Fact Check & Analysis
- **Simple Text Input**: Enter any claim, news, or statement
- **Natural Language Results**: Get sentiment, bias, and truthfulness in plain English
- **No Quantifications**: Results are descriptive, not numerical
- **Comprehensive Analysis**: Detailed breakdown of the text

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Configure Settings** (in sidebar):
   - Select your preferred news topics
   - Choose bias preference (All sources, Low bias only, Balanced coverage)
   - Set sentiment preference (All sentiments, Positive, Neutral, Negative)
   - Set maximum articles to fetch

## How to Use

### Chat Section
1. Go to the "💬 Chat" tab
2. **Fetch News**: Click the "📰 Fetch Latest News" button in the sidebar
3. **Ask Questions**: Chat naturally about any topic or news
4. **Automatic Learning**: The system searches for new information when needed

### Fact Check Section
1. Go to the "🔍 Fact Check" tab
2. Enter any text in the text area
3. Click "🔍 Analyze"
4. Get results showing:
   - **Truthfulness**: "This appears to be true/false/partially true"
   - **Sentiment**: "Positive and optimistic/Neutral and balanced/Negative and pessimistic"
   - **Bias Level**: "Low bias, fairly objective/Moderate bias, some subjectivity/High bias, strongly opinionated"
   - **Detailed Analysis**: Comprehensive breakdown in natural language

## Architecture

The system uses multiple AI agents working together:
- **News Collector**: Fetches articles from multiple sources
- **Sentiment Analyzer**: Analyzes emotional tone
- **Bias Detector**: Identifies potential bias
- **Fact Checker**: Verifies claims against multiple sources
- **RAG Agent**: Answers questions using retrieved articles

## Technology Stack

- **Frontend**: Streamlit
- **AI Framework**: LangGraph + LangChain
- **ML Models**: Hugging Face Transformers
- **Vector Database**: ChromaDB
- **News Sources**: Free APIs and web scraping

## Clean & Simple Design

- **Two sections only**: Chat and Fact Check
- **Sidebar controls**: Essential preferences only
- **Natural language**: All responses in plain English
- **No complex metrics**: Focus on understanding, not numbers
- **Intuitive workflow**: Fetch news, then chat about it

---

*Built with ❤️ using Python and Streamlit*
