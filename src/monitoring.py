"""
Model Monitoring and Logging Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelMonitor:
    """
    Monitor model performance and predictions over time.
    """
    
    def __init__(self, log_dir='logs/', model_name='fraud_detection'):
        """
        Initialize model monitor.
        
        Args:
            log_dir: Directory for logs
            model_name: Name of the model
        """
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.predictions_log = self.log_dir / f'{model_name}_predictions.jsonl'
        self.performance_log = self.log_dir / f'{model_name}_performance.jsonl'
        self.alerts_log = self.log_dir / f'{model_name}_alerts.jsonl'
    
    def log_prediction(self, transaction_id, features, prediction, probability, 
                      actual_label=None, timestamp=None):
        """
        Log a prediction.
        
        Args:
            transaction_id: Unique transaction ID
            features: Transaction features (dict)
            prediction: Fraud prediction (0 or 1)
            probability: Fraud probability
            actual_label: Actual label if known
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'transaction_id': transaction_id,
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
            'actual_label': int(actual_label) if actual_label is not None else None,
            'risk_level': self._get_risk_level(probability)
        }
        
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_performance(self, metrics, model_version=None, timestamp=None):
        """
        Log model performance metrics.
        
        Args:
            metrics: Dictionary of metrics
            model_version: Model version
            timestamp: Evaluation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'model_version': model_version,
            'metrics': metrics
        }
        
        with open(self.performance_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_alert(self, alert_type, message, transaction_id=None, severity='medium'):
        """
        Log an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            transaction_id: Related transaction ID
            severity: Alert severity ('low', 'medium', 'high', 'critical')
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'message': message,
            'transaction_id': transaction_id,
            'severity': severity
        }
        
        with open(self.alerts_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Print critical alerts
        if severity == 'critical':
            print(f"🚨 CRITICAL ALERT: {message}")
    
    def get_prediction_stats(self, hours=24):
        """
        Get prediction statistics for last N hours.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not self.predictions_log.exists():
            return {}
        
        # Read recent predictions
        predictions = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with open(self.predictions_log, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                if entry_time >= cutoff_time:
                    predictions.append(entry)
        
        if not predictions:
            return {}
        
        df = pd.DataFrame(predictions)
        
        stats = {
            'total_predictions': len(df),
            'fraud_predictions': int(df['prediction'].sum()),
            'fraud_rate': float(df['prediction'].mean()),
            'avg_probability': float(df['probability'].mean()),
            'high_risk_count': int((df['probability'] >= 0.7).sum()),
            'medium_risk_count': int(((df['probability'] >= 0.3) & (df['probability'] < 0.7)).sum()),
            'low_risk_count': int((df['probability'] < 0.3).sum())
        }
        
        return stats
    
    def check_model_drift(self, current_metrics, baseline_metrics, threshold=0.1):
        """
        Check for model performance drift.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            threshold: Threshold for drift detection
            
        Returns:
            Dictionary with drift information
        """
        drift_detected = False
        drift_details = {}
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                change = abs(current_value - baseline_value) / baseline_value
                
                if change > threshold:
                    drift_detected = True
                    drift_details[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': change * 100
                    }
        
        if drift_detected:
            self.log_alert(
                'model_drift',
                f'Model drift detected: {drift_details}',
                severity='high'
            )
        
        return {
            'drift_detected': drift_detected,
            'details': drift_details
        }
    
    def _get_risk_level(self, probability):
        """Get risk level from probability."""
        if probability >= 0.7:
            return 'high'
        elif probability >= 0.3:
            return 'medium'
        else:
            return 'low'


class AlertSystem:
    """
    Alert system for high-risk transactions.
    """
    
    def __init__(self, monitor=None):
        """
        Initialize alert system.
        
        Args:
            monitor: ModelMonitor instance
        """
        self.monitor = monitor
        self.alert_rules = []
    
    def add_rule(self, rule_name, condition, severity='medium', action=None):
        """
        Add an alert rule.
        
        Args:
            rule_name: Name of the rule
            condition: Function that returns True if alert should trigger
            severity: Alert severity
            action: Function to execute when alert triggers
        """
        self.alert_rules.append({
            'name': rule_name,
            'condition': condition,
            'severity': severity,
            'action': action
        })
    
    def check_alerts(self, transaction_data, prediction, probability):
        """
        Check all alert rules for a transaction.
        
        Args:
            transaction_data: Transaction data
            prediction: Fraud prediction
            probability: Fraud probability
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if rule['condition'](transaction_data, prediction, probability):
                alert = {
                    'rule': rule['name'],
                    'severity': rule['severity'],
                    'message': f"Alert triggered: {rule['name']}"
                }
                triggered_alerts.append(alert)
                
                if self.monitor:
                    self.monitor.log_alert(
                        rule['name'],
                        alert['message'],
                        transaction_data.get('transaction_id'),
                        rule['severity']
                    )
                
                if rule['action']:
                    rule['action'](transaction_data, prediction, probability)
        
        return triggered_alerts
    
    def add_default_rules(self):
        """Add default alert rules."""
        # High probability fraud
        self.add_rule(
            'high_probability_fraud',
            lambda data, pred, prob: prob >= 0.9,
            severity='critical',
            action=lambda data, pred, prob: print(f"🚨 CRITICAL: High probability fraud detected: {prob:.2%}")
        )
        
        # High amount transaction
        self.add_rule(
            'high_amount_transaction',
            lambda data, pred, prob: data.get('amount', 0) > 10000 and prob >= 0.5,
            severity='high'
        )
        
        # Unusual time transaction
        self.add_rule(
            'unusual_time',
            lambda data, pred, prob: self._is_unusual_time(data) and prob >= 0.6,
            severity='medium'
        )
    
    def _is_unusual_time(self, data):
        """Check if transaction is at unusual time."""
        # Implement time-based logic
        return False  # Placeholder

