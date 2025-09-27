"""
Model Training Module for Netflix Churn Prediction
=================================================

This module provides classes and functions for training machine learning models
and performing clustering analysis on Netflix review data.

The module includes:
- Clustering analysis with PCA dimensionality reduction
- Predictive modeling with Random Forest
- Comprehensive model evaluation metrics
- Cross-validation for robust performance estimation

Example:
    >>> from src.model_training.model_trainer import ModelTrainer
    >>> trainer = ModelTrainer()
    >>> df = trainer.perform_clustering(df)
    >>> model, X_train, X_test, y_train, y_test = trainer.train_model(df)

Classes:
    ModelTrainer: Main class for model training and evaluation tasks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score)

class ModelTrainer:
    """
    Handles model training, evaluation, and clustering analysis.
    
    This class provides methods for training predictive models, performing
    clustering analysis, and evaluating model performance with comprehensive metrics.
    
    Attributes:
        model: Trained machine learning model
        scaler (StandardScaler): Scaler for feature standardization
        pca (PCA): PCA transformer for dimensionality reduction
    """
    
    def __init__(self):
        """
        Initialize the model trainer.
        
        Sets up initial values for model components.
        """
        self.model = None
        self.scaler = None
        self.pca = None
    
    def perform_clustering(self, df):
        """
        Perform clustering analysis with PCA dimensionality reduction.
        
        This method performs customer segmentation using:
        1. Principal Component Analysis for dimensionality reduction
        2. K-means clustering to identify customer segments
        3. Analysis of cluster characteristics
        
        Args:
            df (pd.DataFrame): Input dataframe with engineered features
            
        Returns:
            pd.DataFrame: Dataframe with cluster assignments
        """
        print("\n=== Clustering Analysis ===")
        
        # Select features for clustering
        features = [
            'satisfaction_score', 'sentiment_score', 'engagement_score',
            'review_length', 'word_count', 'score'
        ]
        X = df[features].values
        
        # Standardize and perform PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of components
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate explained variance ratio
        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance >= 0.95) + 1
        
        print(f"\nOptimal number of PCA components: {n_components}")
        print("Explained variance ratio:", explained_variance[:n_components])
        
        # Perform clustering on reduced dimensions
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_pca)
        
        # Analyze clusters
        cluster_stats = df.groupby('cluster')[features].mean()
        print("\nCluster Profiles:")
        print(cluster_stats)
        
        return df
    
    def train_model(self, df):
        """
        Train an optimized predictive model for churn prediction.
        
        This method trains a Random Forest classifier with:
        - Cross-validation for performance estimation
        - Comprehensive evaluation metrics
        - Feature importance analysis
        - Confusion matrix and classification report
        
        Args:
            df (pd.DataFrame): Input dataframe with engineered features and target
            
        Returns:
            tuple: (model, X_train, X_test, y_train, y_test)
        """
        print("\n===  Predictive Modeling ===")
        
        # Prepare features - FIXED DATA LEAKAGE ISSUE
        # Removed features that are directly used in calculating churn_risk to prevent data leakage
        # Original features that caused data leakage:
        # ['satisfaction_score', 'sentiment_score', 'engagement_score', 'score']
        # Keeping only features that are less directly related to churn definition:
        features = [
            'review_length', 'word_count', 'avg_word_length', 'subjectivity_score'
        ]
        X = df[features]
        y = df['churn_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train optimized random forest
        self.model = RandomForestClassifier(
            n_estimators=35,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model and evaluate
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Additional metrics
        self._print_additional_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        return self.model, X_train, X_test, y_train, y_test
    
    def _print_additional_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Print additional evaluation metrics for the trained model.
        
        This method calculates and displays:
        - Accuracy, precision, recall, F1-score
        - ROC AUC and average precision
        - Confusion matrix
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_pred_proba (np.array): Predicted probabilities
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        print("\nAdditional Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 0     1")
        print(f"Actual 0:     {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       1:     {cm[1,0]:4d}  {cm[1,1]:4d}")