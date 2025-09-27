# modular_version/dspy_demo.py
"""
DSPy Integration Demo for Netflix Churn Prediction
================================================

This script demonstrates how to integrate DSPy into the Netflix churn prediction pipeline.
It shows enhanced sentiment analysis, feature engineering, and churn prediction using
foundation models with programmatic prompting.

To run this demo:
    python dspy_demo.py
"""

from modular_version.dspy_integration import (
    DSPySentimentAnalyzer, 
    DSPyFeatureEngineer,
    demonstrate_dspy_integration
)
import pandas as pd

def main():
    """Main function to run the DSPy integration demo."""
    print("Netflix Churn Prediction - DSPy Integration Demo")
    print("=" * 50)
    
    # Run the integrated demo
    demonstrate_dspy_integration()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()