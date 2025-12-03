"""
Cross-Validation and Model Selection Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    make_scorer, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score
)
from src.models import FraudDetectionModel
from src.evaluation import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """
    Advanced cross-validation for fraud detection models.
    """
    
    def __init__(self, cv=5, scoring='roc_auc', random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            cv: Number of folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.cv_splits = None
        
    def create_cv_splits(self, X, y, method='stratified', shuffle=True):
        """
        Create cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: CV method ('stratified', 'kfold', 'timeseries')
            shuffle: Whether to shuffle data
            
        Returns:
            CV split generator
        """
        if method == 'stratified':
            self.cv_splits = StratifiedKFold(
                n_splits=self.cv, shuffle=shuffle, random_state=self.random_state
            )
        elif method == 'kfold':
            self.cv_splits = KFold(
                n_splits=self.cv, shuffle=shuffle, random_state=self.random_state
            )
        elif method == 'timeseries':
            self.cv_splits = TimeSeriesSplit(n_splits=self.cv)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.cv_splits.split(X, y)
    
    def cross_validate_model(self, model, X, y, cv_splits=None, metrics=None):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            cv_splits: CV splits (if None, creates new)
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with CV results
        """
        if metrics is None:
            metrics = ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']
        
        scoring_dict = {}
        for metric in metrics:
            if metric == 'roc_auc':
                scoring_dict[metric] = make_scorer(roc_auc_score, needs_proba=True)
            elif metric == 'precision':
                scoring_dict[metric] = make_scorer(precision_score, zero_division=0)
            elif metric == 'recall':
                scoring_dict[metric] = make_scorer(recall_score, zero_division=0)
            elif metric == 'f1':
                scoring_dict[metric] = make_scorer(f1_score, zero_division=0)
            elif metric == 'accuracy':
                scoring_dict[metric] = make_scorer(accuracy_score)
        
        if cv_splits is None:
            cv_splits = self.create_cv_splits(X, y)
        
        cv_results = cross_validate(
            model, X, y, cv=cv_splits, scoring=scoring_dict,
            return_train_score=True, n_jobs=-1
        )
        
        return cv_results
    
    def evaluate_cv_results(self, cv_results):
        """
        Evaluate and summarize CV results.
        
        Args:
            cv_results: CV results dictionary
            
        Returns:
            DataFrame with summary statistics
        """
        summary = {}
        
        for metric in cv_results.keys():
            if 'test_' in metric:
                metric_name = metric.replace('test_', '')
                scores = cv_results[metric]
                summary[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                }
        
        return pd.DataFrame(summary).T


class ModelSelector:
    """
    Automated model selection with cross-validation.
    """
    
    def __init__(self, cv=5, scoring='roc_auc'):
        """
        Initialize model selector.
        
        Args:
            cv: Number of CV folds
            scoring: Scoring metric
        """
        self.cv = cv
        self.scoring = scoring
        self.cv_validator = CrossValidator(cv=cv, scoring=scoring)
        self.results = {}
        
    def select_best_model(self, X_train, y_train, model_types=None):
        """
        Select best model from multiple types.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_types: List of model types to try
            
        Returns:
            Best model and results
        """
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'neural_network']
        
        best_score = -np.inf
        best_model = None
        best_model_type = None
        
        for model_type in model_types:
            print(f"\nEvaluating {model_type}...")
            
            model = FraudDetectionModel(model_type=model_type)
            
            # Perform cross-validation
            cv_results = self.cv_validator.cross_validate_model(
                model, X_train, y_train
            )
            
            # Get mean test score
            test_score_key = f'test_{self.scoring}'
            if test_score_key in cv_results:
                mean_score = np.mean(cv_results[test_score_key])
                self.results[model_type] = {
                    'mean_score': mean_score,
                    'std_score': np.std(cv_results[test_score_key]),
                    'cv_results': cv_results
                }
                
                print(f"  Mean {self.scoring}: {mean_score:.4f} (+/- {np.std(cv_results[test_score_key]):.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_type = model_type
        
        print(f"\nBest model: {best_model_type} with score: {best_score:.4f}")
        
        return best_model, best_model_type, self.results
    
    def compare_models(self, models_dict, X_train, y_train):
        """
        Compare multiple models.
        
        Args:
            models_dict: Dictionary of {name: model}
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Comparison DataFrame
        """
        comparison_results = {}
        
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            cv_results = self.cv_validator.cross_validate_model(
                model, X_train, y_train
            )
            
            test_score_key = f'test_{self.scoring}'
            if test_score_key in cv_results:
                comparison_results[name] = {
                    'mean': np.mean(cv_results[test_score_key]),
                    'std': np.std(cv_results[test_score_key])
                }
        
        return pd.DataFrame(comparison_results).T

