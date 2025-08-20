# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be in a public GitHub reposi4. **Memory issues**:
   - Streamlit Cloud has memory limits
   - The app is optimized for cloud deployment

5. **LangSmith monitoring not showing**:
   - Verify LANGSMITH_API_KEY is correctly set
   - Check project name in LangSmith dashboard
   - Ensure you have access to the LangSmith project

## 📊 LangSmith Monitoring

### What is LangSmith?
LangSmith is a platform for debugging, testing, and monitoring LLM applications. It provides:
- **Tracing**: Track every step of your multi-agent workflow
- **Debugging**: Identify bottlenecks and errors in real-time
- **Performance Monitoring**: Monitor latency, token usage, and success rates
- **Feedback Collection**: Gather user feedback on AI responses

### Setting up LangSmith Monitoring

1. **Create a LangSmith account** at https://smith.langchain.com/
2. **Generate an API key** in your LangSmith settings
3. **Add the API key** to your secrets:
   - Local: Add `LANGSMITH_API_KEY=ls_your_key_here` to `secrets.env`
   - Streamlit Cloud: Add to the secrets management interface

### Monitoring Features in Informa

Once enabled, you can monitor:
- **News Collection**: Track article collection from different sources
- **Sentiment Analysis**: Monitor sentiment analysis performance
- **Bias Detection**: Track bias detection accuracy
- **Fact Checking**: Monitor fact-checking processes
- **Chat Interactions**: Track user queries and responses

### Accessing Your Monitoring Dashboard

1. **Dashboard URL**: Automatically displayed in the Streamlit sidebar
2. **Project**: `informa-news-analysis` (customizable in config)
3. **Sessions**: Auto-generated with timestamps for each app run

### Performance Tips
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Hugging Face API Key**: Get a free API key from [huggingface.co](https://huggingface.co/settings/tokens)

## Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Informa
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets** (choose one method):

   **Method A: Using .streamlit/secrets.toml (recommended)**
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` and add your actual API key:
   ```toml
   [api_keys]
   HF_API_KEY = "hf_your_actual_api_key_here"
   ```

   **Method B: Using environment variables**
   ```bash
   export HF_API_KEY="hf_your_actual_api_key_here"
   ```

   **Method C: Using secrets.env file**
   ```bash
   cp secrets.env.template secrets.env
   ```
   Then edit `secrets.env` and add your actual API key.

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. **Ensure secrets are not committed**:
   ```bash
   # Check if .gitignore is working
   git status
   # Should NOT show secrets.env or .streamlit/secrets.toml
   ```

2. **Commit and push your code**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Click "New app"**
3. **Connect your GitHub repository**
4. **Configure the app**:
   - Repository: `your-username/Informa`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom URL (optional)

### Step 3: Add Secrets

1. **In the Streamlit Cloud dashboard**, click on your app
2. **Go to "Manage app" → "Secrets"**
3. **Add your secrets in TOML format**:
   ```toml
   [api_keys]
   HF_API_KEY = "hf_your_actual_api_key_here"
   NEWSAPI_KEY = ""  # Optional
   REDDIT_CLIENT_ID = ""  # Optional
   REDDIT_CLIENT_SECRET = ""  # Optional
   LANGSMITH_API_KEY = ""  # Optional - for workflow monitoring
   
   [configuration]
   debug_mode = false
   ```

### Step 4: Deploy

1. **Click "Deploy"**
2. **Wait for the build to complete** (usually 2-5 minutes)
3. **Your app will be available at the provided URL**

## Environment Variables (Alternative to Secrets)

If you prefer environment variables over secrets.toml, you can set them in Streamlit Cloud:

1. **In your app dashboard**, go to "Advanced settings"
2. **Add environment variables**:
   - `HF_API_KEY`: Your Hugging Face API key
   - `NEWSAPI_KEY`: Your NewsAPI key (optional)
   - `REDDIT_CLIENT_ID`: Your Reddit client ID (optional)
   - `REDDIT_CLIENT_SECRET`: Your Reddit client secret (optional)
   - `LANGSMITH_API_KEY`: Your LangSmith API key (optional)

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**:
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **"API key not found"**:
   - Verify secrets are properly set in Streamlit Cloud
   - Check the secrets format (TOML syntax)

3. **"ChromaDB permission errors"**:
   - The app will automatically handle database initialization
   - No additional setup required for ChromaDB

4. **"Memory issues"**:
   - Streamlit Cloud has memory limits
   - The app is optimized for cloud deployment

### Performance Tips

1. **Database persistence**: ChromaDB will be recreated on each deployment
2. **Caching**: The app uses Streamlit's caching for better performance
3. **API rate limits**: Built-in rate limiting prevents API quota issues

## API Keys Required

### Essential (Free)
- **Hugging Face API**: Required for AI analysis features
  - Get it free at: https://huggingface.co/settings/tokens
  - Free tier includes generous usage limits

### Optional (Free/Paid)
- **LangSmith API**: Optional for workflow monitoring and debugging
  - Get it at: https://smith.langchain.com/
  - Free tier available with usage limits
- **NewsAPI**: Enhances news collection (free tier: 1000 requests/day)
  - Get it at: https://newsapi.org/
- **Reddit API**: Enables Reddit news collection
  - Get it at: https://www.reddit.com/prefs/apps

## Support

If you encounter any issues:

1. **Check the Streamlit Cloud logs** in your app dashboard
2. **Verify all secrets are properly formatted**
3. **Ensure your repository is public** (or upgrade to Streamlit Cloud Pro)
4. **Test locally first** to ensure everything works

## Security Notes

- ✅ All sensitive data is properly excluded from git
- ✅ API keys are managed through Streamlit's secure secrets system
- ✅ No hardcoded credentials in the codebase
- ✅ Environment-based configuration for different deployment scenarios
