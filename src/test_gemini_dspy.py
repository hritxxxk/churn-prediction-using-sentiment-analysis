# test_gemini_dspy.py

import dspy
import os

# This script shows how to use the Gemini API with DSPy once you have your API key
# To use this, you'll need to:
# 1. Get a Google AI API key from https://ai.google.dev/
# 2. Set it as an environment variable: export GOOGLE_API_KEY=your_key_here

def setup_gemini_api():
    \"\"\"Setup Gemini API with DSPy.\"\"\"
    # Check if API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Google API key not found. Please set it with:")
        print("  export GOOGLE_API_KEY=your_actual_api_key_here")
        print("Get your API key from: https://ai.google.dev/")
        return None
    
    try:
        # Configure DSPy with Gemini Pro
        lm = dspy.LM('google/gemini-pro')
        dspy.configure(lm=lm)
        print("Successfully configured DSPy with Google Gemini Pro")
        return lm
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return None

def test_gemini_sentiment_analysis():
    \"\"\"Test sentiment analysis with Gemini API.\"\"\"
    lm = setup_gemini_api()
    if not lm:
        print("Could not setup Gemini API. Using fallback.")
        return
    
    # Test reviews
    test_reviews = [
        "This show is absolutely amazing! I've been binge-watching it every day.",
        "Terrible show. Waste of time. Poor acting and boring plot.",
        "The movie was okay, nothing special but not bad either."
    ]
    
    from dspy_integration import DSPySentimentAnalyzer
    analyzer = DSPySentimentAnalyzer()
    
    print("\nTesting sentiment analysis with Gemini API:")
    print("=" * 50)
    
    for i, review in enumerate(test_reviews):
        print(f"\nTest {i+1}:")
        print(f"Review: {review}")
        
        score = analyzer.analyze_sentiment(review)
        print(f"Sentiment Score: {score}")

def main():
    \"\"\"Main function to demonstrate Gemini API usage.\"\"\"
    print("Netflix Churn Prediction - Gemini API Integration")
    print("=" * 50)
    
    # Show instructions
    print("To use the Gemini API:")
    print("1. Get an API key from https://ai.google.dev/")
    print("2. Set the environment variable:")
    print("   export GOOGLE_API_KEY=your_actual_api_key_here")
    print("3. Run this script again")
    
    # Test if API key is available
    if os.getenv("GOOGLE_API_KEY"):
        test_gemini_sentiment_analysis()
    else:
        print("\nNo API key found. Please set your GOOGLE_API_KEY environment variable.")

if __name__ == "__main__":
    main()