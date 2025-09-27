"""
Netflix Churn Prediction System
==============================

This module provides a comprehensive solution for predicting customer churn
based on sentiment analysis of Netflix reviews.

The system includes:
- Data preprocessing and feature engineering
- Advanced NLP techniques for sentiment analysis
- Clustering analysis for customer segmentation
- Machine learning models for churn prediction
- Model interpretability with SHAP and LIME
- Optional DSPy integration for enhanced foundation model usage

Example:
    >>> from src.analysis.analyzer import NetflixAnalyzer
    >>> analyzer = NetflixAnalyzer("data.csv")
    >>> results = analyzer.run_analysis()
    >>> analyzer.save_preprocessed_data("output.csv")

Classes:
    NetflixAnalyzer: Main class orchestrating the analysis pipeline
"""

from src.data_processing.data_processor import DataProcessor
from src.model_training.model_trainer import ModelTrainer
from src.visualization.explainer import Explainer
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, ttest_ind

# Optional DSPy integration
try:
    from modular_version.dspy_integration import DSPySentimentAnalyzer
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("DSPy not available. Install with 'pip install dspy-ai' for enhanced sentiment analysis.")

class NetflixAnalyzer:
    """
    Main class that orchestrates the Netflix churn prediction analysis pipeline.
    
    This class integrates data processing, model training, and model explanation
    components to provide a complete solution for churn prediction.
    
    Attributes:
        file_path (str): Path to the input CSV file
        df (pd.DataFrame): Raw input data
        processed_df (pd.DataFrame): Cleaned and processed data
        data_processor (DataProcessor): Instance for data processing tasks
        model_trainer (ModelTrainer): Instance for model training tasks
        explainer (Explainer): Instance for model explanation tasks
        model: Trained machine learning model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
    """
    
    def __init__(self, file_path):
        """
        Initialize the Netflix analyzer.
        
        Args:
            file_path (str): Path to the CSV file containing Netflix reviews
        """
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.explainer = Explainer()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def run_analysis(self, use_advanced_nlp=False, use_advanced_sentiment=False, use_dspy_sentiment=False):
        """
        Run the complete analysis pipeline.
        
        This method orchestrates the entire analysis process:
        1. Load and clean data
        2. Preprocess text
        3. Create features
        4. Perform clustering analysis
        5. Conduct hypothesis testing
        6. Train predictive model
        7. Explain model predictions
        
        Args:
            use_advanced_nlp (bool): Whether to use lemmatization for text preprocessing
            use_advanced_sentiment (bool): Whether to use transformer-based sentiment analysis
            use_dspy_sentiment (bool): Whether to use DSPy-based sentiment analysis
            
        Returns:
            dict: Dictionary containing analysis results and components
        """
        # Load and clean data
        self.df = self.data_processor.load_data(self.file_path)
        self.processed_df = self.data_processor.clean_data(self.df)
        
        # Text cleaning
        print("\nCleaning text data...")
        self.processed_df['cleaned_content'] = self.processed_df['content'].apply(
            lambda x: self.data_processor.clean_text(x, use_advanced_nlp)
        )
        
        # Feature engineering
        print("Creating features...")
        self.processed_df = self.data_processor.create_features(
            self.processed_df, use_advanced_sentiment, use_dspy_sentiment
        )
        
        print(f"\nFinal shape after preprocessing: {self.processed_df.shape}")
        
        # Display sample of processed data
        print("\nSample of processed features:")
        print(self.processed_df[['sentiment_score', 'satisfaction_score', 'engagement_score']].describe())
        
        # Clustering analysis
        self.processed_df = self.model_trainer.perform_clustering(self.processed_df)
        
        # Hypothesis testing
        self.perform_hypothesis_tests()
        
        # Model training
        (self.model, self.X_train, self.X_test, 
         self.y_train, self.y_test) = self.model_trainer.train_model(self.processed_df)
        
        # Model explanation
        explainer, shap_values = self.explainer.explain_with_shap(
            self.model, self.X_train, self.X_test
        )
        lime_explanation = self.explainer.explain_with_lime(
            self.model, self.X_train, self.X_test, instance_index=0
        )
        
        return {
            'analyzer': self,
            'model': self.model,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'explainer': explainer,
            'shap_values': shap_values,
            'lime_explanation': lime_explanation
        }
    
    def save_preprocessed_data(self, output_path):
        """
        Save preprocessed data to CSV.
        
        Args:
            output_path (str): Path to save the processed data
        """
        if self.processed_df is not None:
            self.processed_df.to_csv(output_path, index=False)
            print(f"Preprocessed data saved to {output_path}")
        else:
            print("Data has not been preprocessed. Please run analysis first.")
    
    def perform_hypothesis_tests(self):
        """
        Conduct hypothesis testing on the processed data.
        
        This method performs four statistical tests:
        1. Satisfaction score impact on churn (t-test)
        2. Engagement-sentiment correlation (Spearman)
        3. Cluster-churn association (Chi-square)
        4. Review length-satisfaction correlation (Spearman)
        """
        print("\n=== Hypothesis Testing ===")
        
        tests = [
            self._test_satisfaction_churn(),
            self._test_engagement_sentiment(),
            self._test_cluster_churn(),
            self._test_length_satisfaction()
        ]
        
        for test in tests:
            print(f"\n{test['name']}:")
            print(f"Statistic: {test['statistic']:.4f}")
            print(f"P-value: {test['p_value']:.4e}")
            print(f"Conclusion: {test['conclusion']}")
    
    def _test_satisfaction_churn(self):
        """
        Test relationship between satisfaction and churn using t-test.
        
        Returns:
            dict: Test results including statistic, p-value, and conclusion
        """
        churned = self.processed_df[self.processed_df['churn_risk'] == 1]['satisfaction_score']
        retained = self.processed_df[self.processed_df['churn_risk'] == 0]['satisfaction_score']
        stat, p_value = ttest_ind(churned, retained)
        
        return {
            'name': 'H1: Satisfaction Score Impact on Churn',
            'statistic': stat,
            'p_value': p_value,
            'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }
    
    def _test_engagement_sentiment(self):
        """
        Test correlation between engagement and sentiment using Spearman correlation.
        
        Returns:
            dict: Test results including statistic, p-value, and conclusion
        """
        # Check if either array is constant to avoid Spearman correlation warning
        engagement_vals = self.processed_df['engagement_score']
        sentiment_vals = self.processed_df['sentiment_score']
        
        if engagement_vals.std() == 0 or sentiment_vals.std() == 0:
            # If either array is constant, correlation is undefined
            return {
                'name': 'H2: Engagement-Sentiment Correlation',
                'statistic': float('nan'),
                'p_value': float('nan'),
                'conclusion': 'No significant correlation (one array is constant)'
            }
        
        correlation, p_value = spearmanr(engagement_vals, sentiment_vals)
        
        return {
            'name': 'H2: Engagement-Sentiment Correlation',
            'statistic': correlation,
            'p_value': p_value,
            'conclusion': 'Significant correlation' if p_value < 0.05 else 'No significant correlation'
        }
    
    def _test_cluster_churn(self):
        """
        Test association between clusters and churn risk using Chi-square test.
        
        Returns:
            dict: Test results including statistic, p-value, and conclusion
        """
        contingency = pd.crosstab(self.processed_df['cluster'], self.processed_df['churn_risk'])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        return {
            'name': 'H3: Cluster-Churn Association',
            'statistic': chi2,
            'p_value': p_value,
            'conclusion': 'Significant association' if p_value < 0.05 else 'No significant association'
        }
    
    def _test_length_satisfaction(self):
        """
        Test correlation between review length and satisfaction using Spearman correlation.
        
        Returns:
            dict: Test results including statistic, p-value, and conclusion
        """
        # Check if either array is constant to avoid Spearman correlation warning
        length_vals = self.processed_df['review_length']
        satisfaction_vals = self.processed_df['satisfaction_score']
        
        if length_vals.std() == 0 or satisfaction_vals.std() == 0:
            # If either array is constant, correlation is undefined
            return {
                'name': 'H4: Review Length-Satisfaction Correlation',
                'statistic': float('nan'),
                'p_value': float('nan'),
                'conclusion': 'No significant correlation (one array is constant)'
            }
        
        correlation, p_value = spearmanr(length_vals, satisfaction_vals)
        
        return {
            'name': 'H4: Review Length-Satisfaction Correlation',
            'statistic': correlation,
            'p_value': p_value,
            'conclusion': 'Significant correlation' if p_value < 0.05 else 'No significant correlation'
        }

def main():
    """
    Main function to run the Netflix churn prediction analysis.
    
    This function demonstrates the usage of the NetflixAnalyzer class
    with a small sample dataset and advanced sentiment analysis.
    
    Returns:
        dict: Analysis results
    """
    # Initialize analyzer with small sample data for testing
    analyzer = NetflixAnalyzer("small_sample.csv")
    
    # Run analysis pipeline with advanced sentiment analysis
    results = analyzer.run_analysis(use_advanced_nlp=False, use_advanced_sentiment=True)
    
    # Save preprocessed data
    analyzer.save_preprocessed_data("processed_netflix_reviews_output.csv")
    
    return results

if __name__ == "__main__":
    results = main()