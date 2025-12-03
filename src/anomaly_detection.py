"""
Anomaly Detection Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Anomaly detection for fraud detection using multiple algorithms.
    """
    
    def __init__(self, method='isolation_forest', contamination=0.05):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'elliptic_envelope', 'lof')
            contamination: Expected proportion of outliers
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_model(self):
        """Create the anomaly detection model."""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, X):
        """
        Fit the anomaly detector.
        
        Args:
            X: Feature matrix
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._create_model()
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X):
        """
        Get anomaly scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_scaled)
            # Normalize to 0-1 (1 = most anomalous)
            scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores_normalized
        elif hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(X_scaled)
            scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores_normalized
        else:
            predictions = self.predict(X)
            return (predictions == -1).astype(float)
    
    def detect_anomalies(self, X, threshold=0.5):
        """
        Detect anomalies with threshold.
        
        Args:
            X: Feature matrix
            threshold: Anomaly score threshold
            
        Returns:
            Dictionary with predictions and scores
        """
        predictions = self.predict(X)
        scores = self.predict_proba(X)
        is_anomaly = scores >= threshold
        
        return {
            'predictions': predictions,
            'scores': scores,
            'is_anomaly': is_anomaly,
            'anomaly_count': is_anomaly.sum()
        }


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection methods.
    """
    
    def __init__(self, contamination=0.05):
        """
        Initialize ensemble anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.detectors = {
            'isolation_forest': AnomalyDetector('isolation_forest', contamination),
            'elliptic_envelope': AnomalyDetector('elliptic_envelope', contamination),
            'lof': AnomalyDetector('lof', contamination)
        }
        self.is_fitted = False
    
    def fit(self, X):
        """
        Fit all detectors.
        
        Args:
            X: Feature matrix
        """
        for name, detector in self.detectors.items():
            detector.fit(X)
        self.is_fitted = True
    
    def predict(self, X, voting='majority'):
        """
        Predict using ensemble voting.
        
        Args:
            X: Feature matrix
            voting: Voting method ('majority', 'average')
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_predictions = []
        all_scores = []
        
        for detector in self.detectors.values():
            pred = detector.predict(X)
            scores = detector.predict_proba(X)
            all_predictions.append(pred)
            all_scores.append(scores)
        
        if voting == 'majority':
            # Majority vote
            predictions_array = np.array(all_predictions)
            ensemble_pred = np.sign(np.sum(predictions_array, axis=0))
            ensemble_pred = np.where(ensemble_pred < 0, -1, 1)
        else:
            # Average scores
            avg_scores = np.mean(all_scores, axis=0)
            ensemble_pred = np.where(avg_scores >= 0.5, -1, 1)
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """
        Get average anomaly scores from all detectors.
        
        Args:
            X: Feature matrix
            
        Returns:
            Average anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_scores = []
        for detector in self.detectors.values():
            scores = detector.predict_proba(X)
            all_scores.append(scores)
        
        return np.mean(all_scores, axis=0)

