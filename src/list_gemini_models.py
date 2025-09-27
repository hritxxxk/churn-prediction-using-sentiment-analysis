# list_gemini_models.py

import os
import google.generativeai as genai

print("Listing available Gemini models...")

# Set the API key
api_key = "YOUR_GOOGLE_API_KEY_HERE"
os.environ['GOOGLE_API_KEY'] = api_key

try:
    # Configure the Google Generative AI client
    genai.configure(api_key=api_key)
    
    # List available models
    print("\nAvailable models:")
    for model in genai.list_models():
        print(f"  - {model.name}: {model.display_name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"    Supported methods: {model.supported_generation_methods}")
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()