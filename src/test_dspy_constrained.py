# test_dspy_constrained.py

import dspy

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Define the signature with more explicit constraints
class SentimentSignature(dspy.Signature):
    # Analyze sentiment of a Netflix review and return a numeric score.
    
    review_content = dspy.InputField()
    sentiment_score = dspy.OutputField(
        desc="A decimal number between -1.0 and 1.0 representing sentiment (-1.0=very negative, 0.0=neutral, 1.0=very positive)",
        format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
    )

# Try a different approach with explicit instructions in the signature
class ExplicitSentimentSignature(dspy.Signature):
    # Analyze sentiment of Netflix reviews.
    
    review = dspy.InputField(desc="A Netflix review text")
    score = dspy.OutputField(desc="Sentiment score: Respond ONLY with a number between -1.0 and 1.0")

# Test with Predict
sentiment_predictor = dspy.Predict(ExplicitSentimentSignature)

# Test with different sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

print("Testing with ExplicitSentimentSignature:")
for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print("Input review:", sample_review)
    
    # Make prediction
    try:
        result = sentiment_predictor(review=sample_review)
        print("Raw result:", result)
        print("Score text:", result.score)
        
        # Try to parse as float
        try:
            score = float(result.score.strip())
            # Clamp to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            print("Parsed score:", score)
        except ValueError:
            print("Could not parse as float")
    except Exception as e:
        print(f"Error: {e}")