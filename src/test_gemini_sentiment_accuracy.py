# test_gemini_sentiment_accuracy.py

import os
import google.generativeai as genai

print("Testing Gemini Flash model for sentiment analysis accuracy...")

# Set the API key
api_key = "YOUR_GOOGLE_API_KEY_HERE"
os.environ['GOOGLE_API_KEY'] = api_key

# Test reviews with expected sentiment scores
test_reviews = [
    ("This show is absolutely amazing! I've been binge-watching it every day.", 0.9),
    ("Terrible show. Waste of time. Poor acting and boring plot.", -0.9),
    ("The movie was okay, nothing special but not bad either.", 0.0),
    ("Outstanding performance by all actors. Highly recommended!", 1.0),
    ("Not worth watching. Poor plot and bad acting.", -0.8)
]

try:
    # Configure the Google Generative AI client
    genai.configure(api_key=api_key)
    
    # Use the gemini-1.5-flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini 1.5 Flash model loaded successfully")
    
    total_error = 0
    correct_direction = 0
    
    for i, (review, expected_score) in enumerate(test_reviews):
        # Create the sentiment analysis prompt
        sentiment_prompt = f"""Analyze the sentiment of this Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "{review}"

Sentiment score:"""
        
        # Get the response
        response = model.generate_content(sentiment_prompt)
        predicted_text = response.text.strip()
        
        # Try to parse the predicted score
        try:
            # Extract numeric value from response
            import re
            numbers = re.findall(r'-?\d+\.?\d*', predicted_text)
            if numbers:
                predicted_score = float(numbers[0])
                # Clamp to [-1, 1] range
                predicted_score = max(-1.0, min(1.0, predicted_score))
            else:
                predicted_score = 0.0
        except:
            predicted_score = 0.0
        
        # Calculate error
        error = abs(predicted_score - expected_score)
        total_error += error
        
        # Check direction accuracy
        if (predicted_score >= 0) == (expected_score >= 0):
            correct_direction += 1
        
        print(f"\nTest {i+1}:")
        print(f"  Review: {review[:50]}...")
        print(f"  Expected: {expected_score:.1f}")
        print(f"  Predicted: {predicted_score:.1f}")
        print(f"  Error: {error:.2f}")
    
    # Calculate statistics
    avg_error = total_error / len(test_reviews)
    direction_accuracy = correct_direction / len(test_reviews) * 100
    
    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"Average Absolute Error: {avg_error:.3f}")
    print(f"Direction Accuracy: {direction_accuracy:.1f}%")
    print("="*50)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()