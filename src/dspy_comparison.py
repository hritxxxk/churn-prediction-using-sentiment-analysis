# modular_version/dspy_comparison.py

import pandas as pd
import time
from textblob import TextBlob
import dspy

def traditional_sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

def dspy_sentiment_analysis(text):
    """DSPy-based sentiment analysis."""
    try:
        # Define the signature for sentiment analysis
        class SentimentSignature(dspy.Signature):
            review_content = dspy.InputField(desc="Netflix review content")
            sentiment_score = dspy.OutputField(desc="Sentiment score from -1 (very negative) to 1 (very positive)")

        # Create the prediction module
        sentiment_predictor = dspy.Predict(SentimentSignature)
        
        # Make prediction
        result = sentiment_predictor(review_content=text)
        
        # Extract and convert sentiment score
        sentiment_text = result.sentiment_score
        
        # Try to extract a numeric value
        try:
            # Look for explicit numeric values
            import re
            numbers = re.findall(r'-?\d+\.?\d*', sentiment_text)
            if numbers:
                score = float(numbers[0])
                return max(-1.0, min(1.0, score))
        except:
            pass
        
        # Simple keyword-based approach as fallback
        positive_keywords = ['positive', 'good', 'great', 'excellent', 'fantastic', 'love', 'amazing']
        negative_keywords = ['negative', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        sentiment_text_lower = sentiment_text.lower()
        positive_count = sum(1 for word in positive_keywords if word in sentiment_text_lower)
        negative_count = sum(1 for word in negative_keywords if word in sentiment_text_lower)
        
        if positive_count > negative_count:
            return 0.5
        elif negative_count > positive_count:
            return -0.5
        else:
            return 0.0
    except Exception as e:
        print(f"Error in DSPy sentiment analysis: {e}")
        return 0.0

def compare_sentiment_approaches():
    print("Netflix Churn Prediction - Sentiment Analysis Comparison")
    print("=" * 58)
    
    # Sample reviews
    sample_reviews = [
        "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging. I would definitely recommend it to my friends.",
        "Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode.",
        "The movie was okay, nothing special. It was entertaining enough to pass the time but I wouldn't watch it again.",
        "Outstanding performance by all actors. The storyline kept me hooked from beginning to end. A masterpiece!",
        "Not worth watching. The story doesn't make sense and the characters are poorly written. Save your time and money."
    ]
    
    print("\nSentiment Analysis Comparison:")
    print("-" * 30)
    print(f"{'Review':<15} | {'Traditional':<12} | {'DSPy':<8} | {'Difference':<10}")
    print("-" * 60)
    
    for i, review in enumerate(sample_reviews):
        # Traditional approach
        traditional_score = traditional_sentiment_analysis(review)
        
        # DSPy approach
        dspy_score = dspy_sentiment_analysis(review)
        
        # Calculate difference
        difference = abs(traditional_score - dspy_score)
        
        # Truncate review for display
        review_display = review[:12] + "..." if len(review) > 15 else review
        
        print(f"{review_display:<15} | {traditional_score:<12.3f} | {dspy_score:<8.3f} | {difference:<10.3f}")

def demonstrate_performance_impact():
    print("\n\nPerformance Impact Analysis:")
    print("-" * 28)
    
    # Sample data
    sample_data = {
        'content': [
            "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging. I would definitely recommend it to my friends.",
            "Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode.",
            "The movie was okay, nothing special. It was entertaining enough to pass the time but I wouldn't watch it again.",
            "Outstanding performance by all actors. The storyline kept me hooked from beginning to end. A masterpiece!",
            "Not worth watching. The story doesn't make sense and the characters are poorly written. Save your time and money."
        ],
        'score': [5, 1, 3, 5, 1]  # Star ratings
    }
    
    df = pd.DataFrame(sample_data)
    
    # Traditional feature engineering
    print("Traditional Feature Engineering:")
    start_time = time.time()
    
    df['traditional_sentiment'] = df['content'].apply(traditional_sentiment_analysis)
    df['traditional_satisfaction'] = (
        df['score'] * 0.4 +
        (df['traditional_sentiment'] + 1) * 30
    ).clip(0, 100)
    
    traditional_time = time.time() - start_time
    print(f"  Time taken: {traditional_time:.4f} seconds")
    print(f"  Sample satisfaction scores: {df['traditional_satisfaction'].tolist()}")
    
    # DSPy-based feature engineering
    print("\nDSPy-based Feature Engineering:")
    start_time = time.time()
    
    df['dspy_sentiment'] = df['content'].apply(dspy_sentiment_analysis)
    df['dspy_satisfaction'] = (
        df['score'] * 0.4 +
        (df['dspy_sentiment'] + 1) * 30
    ).clip(0, 100)
    
    dspy_time = time.time() - start_time
    print(f"  Time taken: {dspy_time:.4f} seconds")
    print(f"  Sample satisfaction scores: {df['dspy_satisfaction'].tolist()}")
    
    # Compare accuracy (simulated)
    print("\nQuality Comparison (simulated):")
    print("  Traditional approach: Good baseline accuracy")
    print("  DSPy approach: Potentially higher accuracy with nuanced understanding")
    print("  Trade-off: Computational cost vs. accuracy improvement")

def main():
    print("Netflix Churn Prediction System - Approach Comparison")
    print("=" * 55)
    
    # First configure DSPy with local model
    try:
        lm = dspy.LM('ollama/gemma3:270m')
        dspy.configure(lm=lm)
        print("DSPy configured with Gemma3 model")
    except Exception as e:
        print(f"Could not configure DSPy: {e}")
        print("Falling back to neutral sentiment scores")
    
    compare_sentiment_approaches()
    demonstrate_performance_impact()
    
    print("\n" + "=" * 55)
    print("Comparison completed!")
    print("\nKey Takeaways:")
    print("1. Traditional approaches (TextBlob) are fast and lightweight")
    print("2. DSPy-based approaches may provide more nuanced analysis")
    print("3. The choice depends on your accuracy requirements vs. computational budget")
    print("4. DSPy allows for automatic optimization of language model usage")

if __name__ == "__main__":
    main()