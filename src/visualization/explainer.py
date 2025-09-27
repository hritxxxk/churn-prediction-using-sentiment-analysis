"""
Model Explanation Module for Netflix Churn Prediction
====================================================

This module provides classes and functions for explaining machine learning models
using SHAP and LIME interpretability techniques.

The module includes:
- SHAP explanations for global model interpretability
- LIME explanations for local instance understanding
- Feature importance analysis

Example:
    >>> from src.visualization.explainer import Explainer
    >>> explainer = Explainer()
    >>> explainer_obj, shap_values = explainer.explain_with_shap(model, X_train, X_test)
    >>> lime_explanation = explainer.explain_with_lime(model, X_train, X_test)

Classes:
    Explainer: Main class for model explanation tasks
"""

import shap
import lime
from lime import lime_tabular

class Explainer:
    """
    Provides model interpretability using SHAP and LIME.
    
    This class offers methods to explain machine learning model predictions
    using two popular interpretability techniques:
    - SHAP (SHapley Additive exPlanations) for global model understanding
    - LIME (Local Interpretable Model-agnostic Explanations) for local explanations
    
    Attributes:
        None
    """
    
    def __init__(self):
        """
        Initialize the explainer.
        
        No initialization required for this class.
        """
        pass
    
    def explain_with_shap(self, model, X_train, X_test):
        """
        Use SHAP to explain model predictions globally.
        
        SHAP (SHapley Additive exPlanations) is a game theoretic approach to 
        explain the output of any machine learning model. This method provides
        global interpretability by showing how each feature contributes to
        predictions across the entire dataset.
        
        Args:
            model: Trained machine learning model
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            tuple: (explainer, shap_values)
        """
        print("\n=== SHAP Model Explanation ===")
        
        # Create SHAP explainer
        # Ensure that X_test preserves feature names to avoid sklearn warnings
        # Check if X_test is a DataFrame and has feature names
        try:
            # Use the feature names from X_train to maintain consistency
            if hasattr(X_train, 'columns') and X_train.columns is not None:
                feature_names = list(X_train.columns)
            else:
                feature_names = None
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, shap_values is a list with two elements
            # We'll use the values for the positive class (index 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
            # Return default values if SHAP fails
            shap_values = None
            explainer = None
        
        # Note: Skipping SHAP visualization due to Qt issues
        
        # Return explainer and values for further analysis
        return explainer, shap_values
    
    def explain_with_lime(self, model, X_train, X_test, instance_index=0):
        """
        Use LIME to explain individual predictions locally.
        
        LIME (Local Interpretable Model-agnostic Explanations) explains
        individual predictions by approximating the model locally with an
        interpretable model (like linear regression).
        
        Args:
            model: Trained machine learning model
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            instance_index (int): Index of instance to explain
            
        Returns:
            lime explanation object
        """
        print("\n=== LIME Local Explanation ===")
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['Not Churn', 'Churn'],
            mode='classification'
        )
        
        # Explain a specific instance
        instance = X_test.iloc[instance_index].values
        exp = explainer.explain_instance(
            instance, 
            model.predict_proba, 
            num_features=6
        )
        
        # Show explanation
        print(f"\nLIME explanation for instance {instance_index}:")
        # Note: Skipping LIME visualization due to Qt issues
        # Print the explanation instead
        print(exp.as_list())
        
        # Return explanation for further analysis
        return exp