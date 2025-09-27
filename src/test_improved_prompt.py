# test_improved_prompt.py

import dspy
import re

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

def analyze_sentiment_improved(text: str) -> float:
    """
    Improved sentiment analysis with better prompting.
    """
    # Clean the text
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Create a more explicit prompt
    prompt = f"""Analyze the sentiment of the following Netflix review. 
Respond ONLY with a single decimal number between -1.0 and 1.0 where:
- -1.0 means very negative sentiment
- 0.0 means neutral sentiment
- 1.0 means very positive sentiment

Review: "{cleaned_text}"

Sentiment score:"""
    
    try:
        # Call the LM directly
        response = lm(prompt, max_tokens=10)
        
        # If response is a list, get the first element
        if isinstance(response, list):
            response_text = response[0]
        else:
            response_text = response
        
        print(f"Raw response: {response_text}")
        
        # Try to extract a number
        numbers = re.findall(r'-?\d+\.?\d*', str(response_text))
        if numbers:
            score = float(numbers[0])
            # Clamp to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            return score
        else:
            # Fallback to neutral sentiment
            return 0.0
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Fallback to neutral sentiment
        return 0.0

# Test with a few sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

print("Testing improved prompt approach:")
for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print(f"Input review: {sample_review}")
    
    score = analyze_sentiment_improved(sample_review)
    print(f"Sentiment score: {score}")