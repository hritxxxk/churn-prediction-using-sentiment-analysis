# test_dspy_direct.py

import dspy

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Test direct prompt
sample_review = "This show is absolutely amazing! I've been binge-watching it every day."

# Create a direct prompt
prompt = f"Analyze the sentiment of this Netflix review and respond ONLY with a number between -1.0 and 1.0 where -1.0 is very negative and 1.0 is very positive:\n\n{sample_review}\n\nSentiment score:"

print("Prompt:", prompt)

try:
    # Call the LM directly
    response = lm(prompt)
    
    print("Raw response:", response)
    print("Type of response:", type(response))
    
    # If response is a list, get the first element
    if isinstance(response, list):
        response_text = response[0]
    else:
        response_text = response
    
    print("Response text:", response_text)
    
    # Try to extract a number
    import re
    numbers = re.findall(r'-?\d+\.?\d*', str(response_text))
    if numbers:
        score = float(numbers[0])
        print("Extracted score:", score)
    else:
        print("No numbers found in response")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()