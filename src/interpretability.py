"""
Model Interpretability Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class ModelInterpreter:
    """
    Model interpretability using SHAP values and feature importance.
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained model
            X_train: Training data
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names is not None else [
            f'feature_{i}' for i in range(X_train.shape[1])
        ]
        self.shap_explainer = None
        self.shap_values = None
        
    def get_feature_importance(self, X_test=None, y_test=None, method='native'):
        """
        Get feature importance.
        
        Args:
            X_test: Test data for permutation importance
            y_test: Test labels for permutation importance
            method: Method ('native', 'permutation')
            
        Returns:
            DataFrame with feature importance
        """
        if method == 'native' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif method == 'permutation' and X_test is not None and y_test is not None:
            perm_importance = permutation_importance(
                self.model, X_test, y_test, n_repeats=10, random_state=42
            )
            importances = perm_importance.importances_mean
        else:
            raise ValueError("Invalid method or missing data for permutation importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction_shap(self, X_explain, max_evals=100):
        """
        Explain predictions using SHAP values.
        
        Args:
            X_explain: Data to explain
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            SHAP values and explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            if self.shap_explainer is None:
                # Use TreeExplainer for tree-based models
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except:
                    # Fallback to KernelExplainer
                    sample_data = shap.sample(self.X_train, min(100, len(self.X_train)))
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, sample_data
                    )
            
            # Calculate SHAP values
            if isinstance(self.shap_explainer, shap.TreeExplainer):
                shap_values = self.shap_explainer.shap_values(X_explain)
            else:
                shap_values = self.shap_explainer.shap_values(
                    X_explain, nsamples=max_evals
                )
            
            # Handle binary classification
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Use positive class
            
            self.shap_values = shap_values
            return shap_values, self.shap_explainer
        else:
            raise ValueError("Model must have predict_proba method for SHAP")
    
    def plot_feature_importance(self, importance_df=None, top_n=20, save_path=None):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with importance (if None, calculates it)
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        if importance_df is None:
            importance_df = self.get_feature_importance()
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shap_summary(self, shap_values=None, X_explain=None, save_path=None):
        """
        Plot SHAP summary.
        
        Args:
            shap_values: SHAP values (if None, calculates them)
            X_explain: Data to explain
            save_path: Path to save the plot
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        if shap_values is None:
            if X_explain is None:
                X_explain = self.X_train[:100]  # Sample for speed
            shap_values, _ = self.explain_prediction_shap(X_explain)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def explain_instance(self, instance, shap_values=None, top_features=10):
        """
        Explain a single instance prediction.
        
        Args:
            instance: Single instance to explain
            shap_values: SHAP values (if None, calculates them)
            top_features: Number of top features to return
            
        Returns:
            Dictionary with explanation
        """
        if not SHAP_AVAILABLE:
            # Fallback to feature importance
            importance_df = self.get_feature_importance()
            top_feat = importance_df.head(top_features)
            return {
                'top_features': dict(zip(top_feat['feature'], top_feat['importance'])),
                'method': 'feature_importance'
            }
        
        if shap_values is None:
            shap_values, _ = self.explain_prediction_shap(instance.reshape(1, -1))
        
        # Get feature contributions
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            feature_contributions[feature] = float(shap_values[0][i])
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_features]
        
        return {
            'top_features': dict(sorted_features),
            'method': 'shap',
            'prediction_contributions': feature_contributions
        }

