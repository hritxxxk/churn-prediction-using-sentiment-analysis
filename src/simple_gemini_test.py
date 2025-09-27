# simple_gemini_test.py

import os

# Set the API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

print("API Key set:", os.environ.get('GOOGLE_API_KEY', 'Not set'))

# Try importing and using the model
try:
    import dspy
    print("DSPy imported successfully")
    
    # Try to create the language model
    lm = dspy.LM('google/gemini-pro')
    print("Language model created successfully")
    
    # Try to configure
    dspy.configure(lm=lm)
    print("DSPy configured successfully")
    
    # Try a simple completion
    response = lm("Say hello world")
    print("Response:", response)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()