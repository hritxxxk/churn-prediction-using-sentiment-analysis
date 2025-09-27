# test_dspy_cot.py

import dspy

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Define the signature for sentiment analysis with clearer instructions
class SentimentSignature(dspy.Signature):
    review_content = dspy.InputField(desc="Netflix review content")
    sentiment_score = dspy.OutputField(desc="Sentiment score as a decimal number from -1.0 (very negative) to 1.0 (very positive). Respond with only the number, nothing else.")

# Try using ChainOfThought instead of Predict
from dspy.predict import ChainOfThought

# Create the prediction module
sentiment_predictor = ChainOfThought(SentimentSignature)

# Test with different sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print("Input review:", sample_review)
    
    # Make prediction
    try:
        result = sentiment_predictor(review_content=sample_review)
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