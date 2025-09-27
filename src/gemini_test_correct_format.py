# gemini_test_correct_format.py

import os

# Set the API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

print("API Key set:", os.environ.get('GOOGLE_API_KEY', 'Not set'))

# Try importing and using the model with correct format
try:
    import dspy
    print("DSPy imported successfully")
    
    # Try different model formats
    models_to_try = [
        'gemini/gemini-pro',
        'google/gemini-pro',
        'vertex_ai/gemini-pro',
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\nTrying model: {model_name}")
            lm = dspy.LM(model_name)
            print(f"Language model {model_name} created successfully")
            
            # Try to configure
            dspy.configure(lm=lm)
            print(f"DSPy configured with {model_name} successfully")
            
            # Try a simple completion
            response = lm("Say hello world in one word")
            print(f"Response: {response}")
            break  # If successful, break out of loop
        except Exception as e:
            print(f"Failed with {model_name}: {e}")
            continue
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()