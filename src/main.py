"""
Main entry point for the Netflix Churn Prediction System.
"""

from analysis.analyzer import NetflixAnalyzer


def main():
    """Main function to run the analysis."""
    print("Netflix Churn Prediction System")
    print("=" * 40)
    
    # Initialize analyzer with sample data
    analyzer = NetflixAnalyzer("../data/small_sample.csv")
    
    # Run analysis pipeline
    results = analyzer.run_analysis(
        use_advanced_nlp=False,
        use_advanced_sentiment=False,
        use_dspy_sentiment=False
    )
    
    # Save preprocessed data
    analyzer.save_preprocessed_data("../output/results.csv")
    
    return results


if __name__ == "__main__":
    results = main()