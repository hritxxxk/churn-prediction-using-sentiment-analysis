# modular_version/test_dspy.py
"""
Test script for DSPy integration in Netflix Churn Prediction
==========================================================

This script tests the DSPy integration components to ensure they work correctly.

To run this test:
    python test_dspy.py
"""

from modular_version.dspy_integration import (
    DSPySentimentAnalyzer, 
    DSPyFeatureEngineer
)
import pandas as pd

def test_dspy_sentiment_analyzer():
    """Test the DSPy sentiment analyzer."""
    print("Testing DSPy Sentiment Analyzer...")
    
    # Initialize the analyzer
    analyzer = DSPySentimentAnalyzer()
    
    # Test cases
    test_cases = [
        ("This movie is absolutely fantastic! I love it!", 0.8),
        ("This is the worst movie I've ever seen. Terrible!", -0.8),
        ("The movie was okay, nothing special.", 0.0),
        ("", 0.0)  # Edge case: empty string
    ]
    
    for text, expected_sentiment in test_cases:
        try:
            sentiment = analyzer.analyze_sentiment(text)
            print(f"Text: '{text}'")
            print(f"Sentiment: {sentiment:.2f} (expected around {expected_sentiment})")
            print("---")
        except Exception as e:
            print(f"Error analyzing sentiment for '{text}': {e}")
            print("---")

def test_dspy_feature_engineer():
    """Test the DSPy feature engineer."""
    print("\nTesting DSPy Feature Engineer...")
    
    # Initialize the feature engineer
    feature_engineer = DSPyFeatureEngineer()
    
    # Create a sample dataframe
    sample_data = {
        'content': [
            "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging. I would definitely recommend it to my friends.",
            "Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode."
        ],
        'score': [5, 1]
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Engineer features
        enhanced_df = feature_engineer.create_advanced_features(df)
        
        print("Feature engineering completed successfully!")
        print(f"Original columns: {list(df.columns)}")
        print(f"Enhanced columns: {list(enhanced_df.columns)}")
        
        # Show results for first row
        print("\nSample results:")
        print(f"Engagement description: {enhanced_df.iloc[0]['dspy_engagement_description']}")
        print(f"Satisfaction description: {enhanced_df.iloc[0]['dspy_satisfaction_description']}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")

def main():
    """Main test function."""
    print("Netflix Churn Prediction - DSPy Integration Tests")
    print("=" * 50)
    
    test_dspy_sentiment_analyzer()
    test_dspy_feature_engineer()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()
