# API Setup Guide

## Overview
This guide explains how to set up API keys for the Netflix Churn Prediction System. The system supports multiple foundation model providers through the DSPy framework.

## Supported Providers

### Google Gemini API (Recommended)
1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Click on "Get API key" in your account
4. Copy the API key
5. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

### OpenAI API (Alternative)
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to API Keys
4. Create a new secret key
5. Copy the API key
6. Add it to your `.env` file:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

### Anthropic Claude API (Alternative)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to API Keys
4. Create a new API key
5. Copy the API key
6. Add it to your `.env` file:
   ```
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   ```

## Environment Setup

### Creating .env File
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual API keys:
   ```bash
   nano .env
   ```

3. Add your API keys (replace placeholders with actual keys)

### Loading Environment Variables
The system automatically loads environment variables from `.env` file. Make sure the file is in the root directory.

## Testing API Configuration

### Google Gemini API Test
```python
from src.utils.api_test import test_gemini_api
test_gemini_api()
```

### OpenAI API Test
```python
from src.utils.api_test import test_openai_api
test_openai_api()
```

### Anthropic API Test
```python
from src.utils.api_test import test_anthropic_api
test_anthropic_api()
```

## Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Error: "403 Forbidden" or "Invalid API key"
   - Solution: Verify your API key is correct and active

2. **Quota Exceeded**
   - Error: "429 Too Many Requests" or quota exceeded
   - Solution: Wait for quota reset or upgrade your plan

3. **Network Issues**
   - Error: "Connection refused" or timeout
   - Solution: Check internet connection and firewall settings

### API Rate Limits

#### Google Gemini
- Free tier: 60 requests/minute
- Pro tier: Higher limits based on plan

#### OpenAI
- Varies by model and pricing tier
- Typically 3000-10000 RPM

#### Anthropic
- Varies by model and pricing tier
- Typically 1000-4000 RPM

## Best Practices

### Security
1. Never commit API keys to version control
2. Use environment variables or secure vaults
3. Rotate keys regularly
4. Monitor usage and costs

### Cost Management
1. Set up billing alerts
2. Monitor usage patterns
3. Use appropriate models for tasks
4. Implement caching for repeated requests

### Error Handling
1. Implement retry logic with exponential backoff
2. Handle rate limiting gracefully
3. Provide informative error messages
4. Fall back to alternative approaches when needed

## Example Usage

```python
import os
from src.dspy_integration import DSPySentimentAnalyzer

# Initialize analyzer with environment variables
analyzer = DSPySentimentAnalyzer()

# Analyze sentiment (uses configured provider)
sentiment_score = analyzer.analyze_sentiment("This movie is amazing!")

print(f"Sentiment score: {sentiment_score}")
```

## Further Reading

- [Google AI Studio Documentation](https://developers.generativeai.google/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [DSPy Documentation](https://github.com/stanfordnlp/dspy)