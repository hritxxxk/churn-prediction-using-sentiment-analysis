# test_improved_dspy.py

import dspy
from dspy_integration import DSPySentimentAnalyzer

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Create the analyzer
analyzer = DSPySentimentAnalyzer()

# Test with different sample reviews
sample_reviews = [
    "This show is absolutely amazing! I've been binge-watching it every day.",
    "Terrible show. Waste of time. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either."
]

print("Testing improved DSPySentimentAnalyzer:")
for i, sample_review in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print("Input review:", sample_review)
    
    # Make prediction
    try:
        score = analyzer.analyze_sentiment(sample_review)
        print("Sentiment score:", score)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()