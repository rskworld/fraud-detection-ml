"""
Real-time Fraud Scoring Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
from src.models import FraudDetectionModel
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudScorer:
    """
    Real-time fraud scoring system.
    Handles preprocessing, feature engineering, and prediction for live transactions.
    """
    
    def __init__(self, model_path=None, model_type='ensemble', threshold=0.5):
        """
        Initialize the fraud scorer.
        
        Args:
            model_path: Path to saved model
            model_type: Type of model to use
            threshold: Fraud detection threshold
        """
        self.model_type = model_type
        self.threshold = threshold
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = FraudDetectionModel(model_type=model_type)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load pre-trained model and preprocessors.
        
        Args:
            model_path: Path to saved model
        """
        try:
            self.model.load_model(model_path)
            # Try to load preprocessor if saved
            try:
                self.preprocessor = joblib.load(f"{model_path}_preprocessor.pkl")
                self.feature_engineer = joblib.load(f"{model_path}_feature_engineer.pkl")
            except:
                pass  # Use default preprocessors
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, transaction_data, is_training=False):
        """
        Prepare features from raw transaction data.
        
        Args:
            transaction_data: Dictionary or DataFrame with transaction data
            is_training: Whether this is training data
            
        Returns:
            Processed feature array
        """
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Feature engineering
        df, _ = self.feature_engineer.engineer_features(
            df, 
            amount_col='amount' if 'amount' in df.columns else None,
            user_col='user_id' if 'user_id' in df.columns else None,
            date_col='timestamp' if 'timestamp' in df.columns else None
        )
        
        # Preprocessing
        if is_training:
            X, _, _, _ = self.preprocessor.preprocess(df)
        else:
            X = self.preprocessor.transform(df)
        
        return X
    
    def predict(self, transaction_data, return_probability=True):
        """
        Predict fraud for a single transaction or batch.
        
        Args:
            transaction_data: Dictionary or DataFrame with transaction data
            return_probability: Whether to return probability or binary prediction
            
        Returns:
            Fraud score (probability) or binary prediction
        """
        if not self.model.is_trained:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Prepare features
        X = self.prepare_features(transaction_data, is_training=False)
        
        # Predict
        fraud_probability = self.model.predict(X)
        
        if return_probability:
            return fraud_probability[0] if len(fraud_probability) == 1 else fraud_probability
        else:
            predictions = (fraud_probability >= self.threshold).astype(int)
            return predictions[0] if len(predictions) == 1 else predictions
    
    def score_batch(self, transactions_df):
        """
        Score multiple transactions at once.
        
        Args:
            transactions_df: DataFrame with multiple transactions
            
        Returns:
            DataFrame with fraud scores added
        """
        fraud_scores = self.predict(transactions_df, return_probability=True)
        fraud_predictions = (fraud_scores >= self.threshold).astype(int)
        
        result_df = transactions_df.copy()
        result_df['fraud_probability'] = fraud_scores
        result_df['fraud_prediction'] = fraud_predictions
        result_df['risk_level'] = pd.cut(
            fraud_scores, 
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return result_df
    
    def get_fraud_explanation(self, transaction_data, top_features=5):
        """
        Get explanation for fraud prediction (feature importance).
        
        Args:
            transaction_data: Dictionary or DataFrame with transaction data
            top_features: Number of top features to return
            
        Returns:
            Dictionary with fraud score and top contributing features
        """
        fraud_score = self.predict(transaction_data, return_probability=True)
        
        # Get feature importance if available
        explanation = {
            'fraud_score': float(fraud_score),
            'fraud_prediction': int(fraud_score >= self.threshold),
            'risk_level': 'High' if fraud_score >= 0.7 else ('Medium' if fraud_score >= 0.3 else 'Low')
        }
        
        # Try to get feature importance from model
        if hasattr(self.model.model, 'feature_importances_'):
            X = self.prepare_features(transaction_data, is_training=False)
            importances = self.model.model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            top_indices = np.argsort(importances)[-top_features:][::-1]
            explanation['top_features'] = {
                feature_names[i]: float(importances[i]) 
                for i in top_indices
            }
        
        return explanation
    
    def update_threshold(self, new_threshold):
        """
        Update the fraud detection threshold.
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
        else:
            raise ValueError("Threshold must be between 0 and 1")

