# test_dspy_output.py

import dspy

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Define the signature for sentiment analysis
class SentimentSignature(dspy.Signature):
    """Signature for sentiment analysis using DSPy."""
    review_content = dspy.InputField(desc="Netflix review content")
    sentiment_score = dspy.OutputField(desc="Sentiment score from -1 (very negative) to 1 (very positive)")

# Create the prediction module
sentiment_predictor = dspy.Predict(SentimentSignature)

# Test with a sample review
sample_review = "This show is absolutely amazing! I've been binge-watching it every day."
print("Input review:", sample_review)

# Make prediction
result = sentiment_predictor(review_content=sample_review)
print("Raw result:", result)
print("Sentiment score text:", result.sentiment_score)
print("Type of result:", type(result.sentiment_score))