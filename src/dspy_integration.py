"""
DSPy Integration Module for Netflix Churn Prediction
==================================================

This module provides classes and functions for integrating DSPy (Declarative Self-improving Programs)
into the Netflix churn prediction pipeline. It demonstrates how to use foundation models for enhanced
sentiment analysis, feature engineering, and churn prediction.

The module includes:
- Advanced sentiment analysis using DSPy-optimized prompts
- Feature engineering with foundation models
- Churn prediction using programmatic prompting
- Automatic optimization of language model usage

Example:
    >>> from modular_version.dspy_integration import DSPySentimentAnalyzer
    >>> analyzer = DSPySentimentAnalyzer()
    >>> sentiment_score = analyzer.analyze_sentiment("This movie was absolutely fantastic!")
    
Classes:
    DSPySentimentAnalyzer: Enhanced sentiment analysis using DSPy
    DSPyFeatureEngineer: Feature engineering with foundation models
    DSPyChurnPredictor: Churn prediction using programmatic prompting
"""

import dspy
import pandas as pd
from typing import List, Dict
import re

# Import our optimized prompt for Gemini
try:
    from modular_version.gemini_optimized_prompt import GEMINI_OPTIMIZED_PROMPT
    GEMINI_PROMPT_AVAILABLE = True
except ImportError:
    GEMINI_PROMPT_AVAILABLE = False
    # Fallback prompt
    GEMINI_OPTIMIZED_PROMPT = """Analyze the sentiment of the following Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "{review}"

Sentiment score:"""

# Configure DSPy with a local model (for demonstration)
# In production, you might use OpenAI, Claude, or other APIs
# For now, we'll use a lightweight setup for demonstration
try:
    # Try to configure with a lightweight model or fallback
    # To use with OpenAI, configure as follows:
    # dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
    # For Anthropic: dspy.configure(lm=dspy.LM('anthropic/claude-3-haiku'))
    dspy.settings.configure()
except Exception as e:
    print(f"Could not configure DSPy: {e}")

class SentimentSignature(dspy.Signature):
    """Signature for sentiment analysis using DSPy."""
    review_content = dspy.InputField(desc="Netflix review content")
    sentiment_score = dspy.OutputField(desc="Sentiment score from -1 (very negative) to 1 (very positive)")

class EnhancedSentimentAnalyzer(dspy.Module):
    """DSPy module for enhanced sentiment analysis."""
    
    def __init__(self):
        super().__init__()
        try:
            self.predict_sentiment = dspy.Predict(SentimentSignature)
        except Exception as e:
            print(f"Could not initialize DSPy sentiment predictor: {e}")
            self.predict_sentiment = None
    
    def forward(self, review_content):
        """Analyze sentiment of a Netflix review."""
        if self.predict_sentiment is None:
            return None
        prediction = self.predict_sentiment(review_content=review_content)
        return prediction

class DSPySentimentAnalyzer:
    """
    Enhanced sentiment analyzer using Google Generative AI for improved accuracy.
    
    This class uses Google's Gemini models for sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.model = None
        try:
            import google.generativeai as genai
            import os
            
            # Check if Google API key is available
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                # Use gemini-2.0-flash-001 as specified
                self.model = genai.GenerativeModel('gemini-2.0-flash-001')
                print("Google Generative AI configured successfully")
            else:
                print("No Google API key found. Will fall back to neutral sentiment.")
        except Exception as e:
            print(f"Could not initialize Google Generative AI: {e}")
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using Google Generative AI.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Sentiment score ranging from -1 (negative) to 1 (positive)
        """
        # If no model is configured, return neutral sentiment
        if self.model is None:
            return 0.0
            
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Use the optimized prompt for better results
            if GEMINI_PROMPT_AVAILABLE:
                prompt = GEMINI_OPTIMIZED_PROMPT.format(review=cleaned_text)
            else:
                # Fallback to basic prompt optimized for Gemini
                prompt = f"""Analyze the sentiment of this Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "{cleaned_text}"

Sentiment score:"""
            
            # Call the model directly
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract a number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                # Clamp to [-1, 1] range
                score = max(-1.0, min(1.0, score))
                return score
            else:
                # Fallback to neutral sentiment
                return 0.0
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Fallback to neutral sentiment
            return 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.strip()
    
    def _extract_sentiment_score(self, sentiment_text: str) -> float:
        """
        Extract numeric sentiment score from text description.
        
        This is a simplified implementation - in practice, you would
        want more sophisticated parsing or a signature that outputs
        directly numeric values.
        """
        # If this method is called, it means we're using the old approach
        # Look for explicit numeric values
        import re
        numbers = re.findall(r'-?\d+\.?\d*', str(sentiment_text))
        if numbers:
            # Return first number clamped to [-1, 1] range
            score = float(numbers[0])
            return max(-1.0, min(1.0, score))
        
        # Simple keyword-based approach as fallback
        positive_keywords = ['positive', 'good', 'great', 'excellent', 'fantastic', 'love', 'amazing']
        negative_keywords = ['negative', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        sentiment_text_lower = str(sentiment_text).lower()
        positive_count = sum(1 for word in positive_keywords if word in sentiment_text_lower)
        negative_count = sum(1 for word in negative_keywords if word in sentiment_text_lower)
        
        if positive_count > negative_count:
            return 0.5
        elif negative_count > positive_count:
            return -0.5
        else:
            return 0.0

class FeatureEngineeringSignature(dspy.Signature):
    """Signature for feature engineering using DSPy."""
    review_content = dspy.InputField(desc="Netflix review content")
    review_length = dspy.OutputField(desc="Length of the review in characters")
    engagement_indicators = dspy.OutputField(desc="Description of engagement indicators like enthusiasm, detail level, etc.")
    satisfaction_indicators = dspy.OutputField(desc="Description of satisfaction indicators like praise, criticism, recommendations, etc.")

class DSPyFeatureEngineer:
    """
    Feature engineering using DSPy and foundation models.
    
    This class uses programmatic prompting to extract sophisticated
    features from Netflix reviews that can improve churn prediction.
    
    To use with a language model, configure DSPy before initializing:
    ```python
    import dspy
    dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
    engineer = DSPyFeatureEngineer()
    ```
    """
    
    def __init__(self):
        """Initialize the DSPy feature engineer."""
        try:
            self.feature_predictor = dspy.Predict(FeatureEngineeringSignature)
        except Exception as e:
            print(f"Could not initialize DSPy feature predictor: {e}")
            self.feature_predictor = None
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features using DSPy-optimized prompting.
        
        Args:
            df (pd.DataFrame): Input dataframe with Netflix reviews
            
        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        # Check if DSPy is properly configured with an LM
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            print("No language model configured for DSPy. Please configure with:")
            print("  import dspy")
            print("  dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))  # or another provider")
            print("Returning original dataframe without advanced features.")
            return df
            
        # If DSPy is not available, return the original dataframe
        if self.feature_predictor is None:
            return df
            
        # Add new columns for advanced features
        df = df.copy()
        df['dspy_engagement_description'] = ""
        df['dspy_satisfaction_description'] = ""
        
        # Process a sample of reviews for demonstration
        # (In practice, you'd process all reviews)
        sample_size = min(5, len(df))  # Limit for cost/efficiency
        
        for i in range(sample_size):
            content = df.iloc[i]['content'] if 'content' in df.columns else df.iloc[i]['cleaned_content']
            
            try:
                # Use DSPy to extract advanced features
                result = self.feature_predictor(review_content=str(content))
                
                # Store the descriptive features
                df.loc[i, 'dspy_engagement_description'] = result.engagement_indicators
                df.loc[i, 'dspy_satisfaction_description'] = result.satisfaction_indicators
                
            except Exception as e:
                print(f"Error processing review {i}: {e}")
        
        return df

class ChurnPredictionSignature(dspy.Signature):
    """Signature for churn prediction using DSPy."""
    review_content = dspy.InputField(desc="Netflix review content")
    sentiment_score = dspy.InputField(desc="Sentiment score from -1 to 1")
    review_rating = dspy.InputField(desc="Numerical rating from 1 to 5")
    engagement_description = dspy.InputField(desc="Description of user engagement with the content")
    churn_likelihood = dspy.OutputField(desc="Churn likelihood from 0 (no churn) to 1 (definite churn)")

class DSPyChurnPredictor(dspy.Module):
    """DSPy module for churn prediction."""
    
    def __init__(self):
        super().__init__()
        try:
            self.predict_churn = dspy.Predict(ChurnPredictionSignature)
        except Exception as e:
            print(f"Could not initialize churn predictor: {e}")
            self.predict_churn = None
    
    def forward(self, review_content, sentiment_score, review_rating, engagement_description):
        """Predict churn likelihood based on review features."""
        if self.predict_churn is None:
            return None
        prediction = self.predict_churn(
            review_content=review_content,
            sentiment_score=sentiment_score,
            review_rating=review_rating,
            engagement_description=engagement_description
        )
        return prediction

# Example usage function
def demonstrate_dspy_integration():
    """
    Demonstrate the DSPy integration with a simple example.
    
    This function shows how to use the DSPy components together.
    """
    print("=== DSPy Integration Demo ===")
    
    # Check if DSPy is configured with an LM
    if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
        print("Note: DSPy is not configured with a language model.")
        print("To enable full functionality, configure DSPy with:")
        print("  import dspy")
        print("  dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))  # or another provider")
        print()
    
    # Initialize components
    sentiment_analyzer = DSPySentimentAnalyzer()
    feature_engineer = DSPyFeatureEngineer()
    
    # Example review
    sample_review = "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging. I would definitely recommend it to my friends."
    
    # Analyze sentiment
    sentiment_score = sentiment_analyzer.analyze_sentiment(sample_review)
    print(f"Sentiment score: {sentiment_score}")
    
    # Create a sample dataframe
    sample_df = pd.DataFrame({
        'content': [sample_review],
        'score': [5]  # 5-star rating
    })
    
    # Engineer features
    enhanced_df = feature_engineer.create_advanced_features(sample_df)
    print(f"Engagement description: {enhanced_df.iloc[0]['dspy_engagement_description']}")
    print(f"Satisfaction description: {enhanced_df.iloc[0]['dspy_satisfaction_description']}")

if __name__ == "__main__":
    demonstrate_dspy_integration()