# test_correct_gemini_model.py

import os
import google.generativeai as genai

print("Testing with correct Gemini model...")

# Set the API key
api_key = "YOUR_GOOGLE_API_KEY_HERE"
os.environ['GOOGLE_API_KEY'] = api_key

try:
    # Configure the Google Generative AI client
    genai.configure(api_key=api_key)
    
    # Try to use the gemini-1.5-pro model (one of the available models)
    model = genai.GenerativeModel('gemini-1.5-pro')
    print("Gemini 1.5 Pro model loaded successfully")
    
    # Test with a simple prompt
    response = model.generate_content("Say hello world in one word")
    print(f"Response: {response.text}")
    
    # Test with a sentiment analysis prompt
    sentiment_prompt = """Analyze the sentiment of this Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "This show is absolutely amazing! I've been binge-watching it every day."

Sentiment score:"""
    
    sentiment_response = model.generate_content(sentiment_prompt)
    print(f"Sentiment Response: {sentiment_response.text}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()