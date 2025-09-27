# test_dspy_custom.py

import dspy
import re

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

class CustomSentimentAnalyzer(dspy.Module):
    # Custom sentiment analyzer that directly calls the LM with a constrained prompt.
    
    def __init__(self):
        super().__init__()
    
    def forward(self, review_content):
        # Create a direct prompt
        prompt = f"Analyze the sentiment of this Netflix review and respond ONLY with a number between -1.0 and 1.0 where -1.0 is very negative and 1.0 is very positive:\n\n{review_content}\n\nSentiment score:"
        
        # Call the LM directly
        response = lm(prompt)
        
        # If response is a list, get the first element
        if isinstance(response, list):
            response_text = response[0]
        else:
            response_text = response
        
        # Try to extract a number
        numbers = re.findall(r'-?\d+\.?\d*', str(response_text))
        if numbers:
            score = float(numbers[0])
            # Clamp to [-1, 1] range
            score = max(-1.0, min(1.0, score))
        else:
            # Fallback to neutral sentiment
            score = 0.0
            
        # Return a prediction object with the score
        return dspy.Prediction(sentiment_score=str(score))

# Test with different sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

# Create the analyzer
analyzer = CustomSentimentAnalyzer()

print("Testing with CustomSentimentAnalyzer:")
for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print("Input review:", sample_review)
    
    # Make prediction
    try:
        result = analyzer(review_content=sample_review)
        print("Raw result:", result)
        print("Sentiment score text:", result.sentiment_score)
        
        # Try to parse as float
        try:
            score = float(result.sentiment_score.strip())
            print("Parsed score:", score)
        except ValueError:
            print("Could not parse as float")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()