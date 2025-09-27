# test_broad_dspy.py

import dspy
from dspy_integration import DSPySentimentAnalyzer

# Configure DSPy with the local model
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Create the analyzer
analyzer = DSPySentimentAnalyzer()

# Test with a broader set of sample reviews
sample_reviews = [
    ("This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging.", 0.9),
    ("Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode.", -0.9),
    ("The movie was okay, nothing special but not bad either. It was entertaining enough to pass the time.", 0.1),
    ("Outstanding performance by all actors. The storyline kept me hooked from beginning to end. A masterpiece!", 0.95),
    ("Not worth watching. The story doesn't make sense and the characters are poorly written. Save your time and money.", -0.85),
    ("It's an average film. Some parts were good, others not so much. Overall, it's watchable but forgettable.", 0.0),
    ("Absolutely loved it! Best series I've watched in years. Can't wait for the next season.", 0.98),
    ("Worst movie ever. Complete garbage. Don't waste your time or money on this trash.", -1.0),
    ("Meh. It was fine. Not great, not terrible. Just average entertainment.", 0.0),
    ("Fantastic cinematography and brilliant acting. This is how you make a movie.", 0.9)
]

print("Testing improved DSPySentimentAnalyzer with broader dataset:")
print("=" * 60)

total_error = 0
correct_direction = 0
total_tests = len(sample_reviews)

for i, (sample_review, expected_score) in enumerate(sample_reviews):
    print(f"\nTest {i+1}:")
    print(f"Input review: {sample_review[:50]}...")
    print(f"Expected score: {expected_score}")
    
    # Make prediction
    try:
        predicted_score = analyzer.analyze_sentiment(sample_review)
        print(f"Predicted score: {predicted_score}")
        
        # Calculate error
        error = abs(predicted_score - expected_score)
        total_error += error
        print(f"Error: {error:.2f}")
        
        # Check if direction (positive/negative) is correct
        if (predicted_score >= 0 and expected_score >= 0) or (predicted_score < 0 and expected_score < 0):
            correct_direction += 1
            print("Direction: CORRECT")
        else:
            print("Direction: INCORRECT")
            
    except Exception as e:
        print(f"Error: {e}")

# Calculate statistics
avg_error = total_error / total_tests
direction_accuracy = correct_direction / total_tests * 100

print("\n" + "=" * 60)
print("SUMMARY STATISTICS:")
print(f"Average absolute error: {avg_error:.3f}")
print(f"Direction accuracy: {direction_accuracy:.1f}% ({correct_direction}/{total_tests})")
print("=" * 60)