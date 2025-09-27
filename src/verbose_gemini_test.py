# verbose_gemini_test.py

import os
import google.generativeai as genai

print("Starting Gemini API test...")

# Set the API key
api_key = "YOUR_GOOGLE_API_KEY_HERE"
os.environ['GOOGLE_API_KEY'] = api_key

print("API Key set:", "Yes" if os.environ.get('GOOGLE_API_KEY') else "No")

try:
    # Configure the Google Generative AI client
    print("Configuring Google Generative AI...")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully")
    
    # Try to use the gemini-pro model
    print("Loading Gemini Pro model...")
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini Pro model loaded successfully")
    
    # Test with a simple prompt
    print("Sending test prompt...")
    response = model.generate_content("Say hello world in one word")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()