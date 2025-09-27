# test_few_shot.py

import dspy
import re

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

def analyze_sentiment_few_shot(text: str) -> float:
    """
    Few-shot sentiment analysis with examples.
    """
    # Clean the text
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Create a few-shot prompt with examples
    prompt = f"""Analyze the sentiment of Netflix reviews. Respond ONLY with a single decimal number between -1.0 and 1.0.

Examples:
Review: "This movie is absolutely fantastic! I love it so much!"
Sentiment: 0.9

Review: "Terrible film. Complete waste of time and money."
Sentiment: -0.9

Review: "The movie was okay, nothing special."
Sentiment: 0.0

Review: "Outstanding performance by all actors. Highly recommended!"
Sentiment: 1.0

Review: "Not worth watching. Poor plot and bad acting."
Sentiment: -1.0

Now analyze this review:
Review: "{cleaned_text}"
Sentiment:"""
    
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

# Test with sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

print("Testing few-shot approach:")
for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print(f"Input review: {sample_review}")
    
    score = analyze_sentiment_few_shot(sample_review)
    print(f"Sentiment score: {score}")