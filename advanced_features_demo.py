"""
Advanced Features Demo for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
from src.anomaly_detection import AnomalyDetector, EnsembleAnomalyDetector
from src.interpretability import ModelInterpreter
from src.data_augmentation import DataAugmenter, AdvancedAugmenter
from src.cross_validation import CrossValidator, ModelSelector
from src.monitoring import ModelMonitor, AlertSystem
from src.time_series_analysis import TimeSeriesAnalyzer
from src.model_versioning import ModelVersionManager, ModelComparator
from src.models import FraudDetectionModel
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')


def demo_anomaly_detection():
    """Demo: Anomaly Detection"""
    print("="*60)
    print("Demo 1: Anomaly Detection")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    
    # Add some anomalies
    X[-50:] += np.random.randn(50, 10) * 3
    
    # Use Isolation Forest
    detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
    detector.fit(X)
    
    results = detector.detect_anomalies(X, threshold=0.5)
    print(f"Detected {results['anomaly_count']} anomalies out of {len(X)} samples")
    print(f"Anomaly rate: {results['anomaly_count']/len(X)*100:.2f}%")
    
    # Ensemble detector
    print("\nUsing Ensemble Anomaly Detector...")
    ensemble_detector = EnsembleAnomalyDetector(contamination=0.05)
    ensemble_detector.fit(X)
    ensemble_scores = ensemble_detector.predict_proba(X)
    print(f"Average anomaly score: {ensemble_scores.mean():.4f}")


def demo_data_augmentation():
    """Demo: Data Augmentation"""
    print("\n" + "="*60)
    print("Demo 2: Data Augmentation with SMOTE")
    print("="*60)
    
    # Generate imbalanced data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    print(f"Original class distribution:")
    print(f"  Class 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  Class 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    # Apply SMOTE
    augmenter = DataAugmenter(method='smote')
    X_resampled, y_resampled = augmenter.fit_resample(X, y)
    
    print(f"\nAfter SMOTE:")
    print(f"  Class 0: {(y_resampled == 0).sum()} ({(y_resampled == 0).mean()*100:.1f}%)")
    print(f"  Class 1: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean()*100:.1f}%)")
    print(f"  Total samples: {len(X_resampled)} (increased from {len(X)})")


def demo_cross_validation():
    """Demo: Cross-Validation"""
    print("\n" + "="*60)
    print("Demo 3: Cross-Validation")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Create model
    model = FraudDetectionModel(model_type='random_forest')
    
    # Cross-validation
    cv = CrossValidator(cv=5, scoring='roc_auc')
    cv_results = cv.cross_validate_model(model, X, y)
    
    # Evaluate results
    summary = cv.evaluate_cv_results(cv_results)
    print("\nCross-Validation Results:")
    print(summary[['mean', 'std']].round(4))


def demo_model_selection():
    """Demo: Automated Model Selection"""
    print("\n" + "="*60)
    print("Demo 4: Automated Model Selection")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 8)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Model selection
    selector = ModelSelector(cv=3, scoring='roc_auc')
    best_model, best_type, results = selector.select_best_model(
        X, y, model_types=['random_forest', 'xgboost']
    )
    
    print(f"\nBest model type: {best_type}")


def demo_monitoring():
    """Demo: Model Monitoring"""
    print("\n" + "="*60)
    print("Demo 5: Model Monitoring and Logging")
    print("="*60)
    
    # Initialize monitor
    monitor = ModelMonitor(log_dir='logs/demo/', model_name='demo_model')
    
    # Log some predictions
    for i in range(10):
        monitor.log_prediction(
            transaction_id=f'txn_{i}',
            features={'amount': np.random.uniform(10, 1000)},
            prediction=np.random.choice([0, 1]),
            probability=np.random.uniform(0, 1)
        )
    
    # Get statistics
    stats = monitor.get_prediction_stats(hours=24)
    print("\nPrediction Statistics (last 24 hours):")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Alert system
    alert_system = AlertSystem(monitor)
    alert_system.add_default_rules()
    
    # Check alerts
    transaction = {'amount': 15000, 'transaction_id': 'txn_123'}
    alerts = alert_system.check_alerts(transaction, prediction=1, probability=0.95)
    print(f"\nAlerts triggered: {len(alerts)}")


def demo_time_series_analysis():
    """Demo: Time Series Analysis"""
    print("\n" + "="*60)
    print("Demo 6: Time Series Analysis")
    print("="*60)
    
    # Generate time series data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'amount': np.random.lognormal(3, 1, 1000),
        'user_id': np.random.randint(1, 100, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Time series analyzer
    analyzer = TimeSeriesAnalyzer(time_col='timestamp')
    
    # Extract temporal features
    df_features = analyzer.extract_temporal_features(df)
    print(f"Extracted {len([c for c in df_features.columns if c not in df.columns])} temporal features")
    
    # Calculate velocity features
    df_velocity = analyzer.calculate_velocity_features(
        df_features, user_col='user_id', value_col='amount'
    )
    print(f"Added velocity features for transaction patterns")
    
    # Detect fraud patterns
    df_patterns = analyzer.detect_fraud_patterns(df_velocity, user_col='user_id')
    print(f"Detected fraud patterns in time series")


def demo_model_versioning():
    """Demo: Model Versioning"""
    print("\n" + "="*60)
    print("Demo 7: Model Versioning")
    print("="*60)
    
    # Initialize version manager
    version_manager = ModelVersionManager(model_dir='data/models/versions/')
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], 100, p=[0.9, 0.1])
    
    # Train and register models
    for version in ['1.0.0', '1.1.0', '2.0.0']:
        model = FraudDetectionModel(model_type='random_forest')
        model.train(X, y)
        
        metrics = {'accuracy': np.random.uniform(0.85, 0.95)}
        
        version_manager.register_model(
            model.model,
            model_name='fraud_detector',
            version=version,
            metrics=metrics,
            description=f'Model version {version}'
        )
        print(f"Registered version {version}")
    
    # Compare versions
    comparison = version_manager.compare_versions('fraud_detector')
    print("\nVersion Comparison:")
    print(comparison[['version', 'created_at', 'accuracy']])


def demo_interpretability():
    """Demo: Model Interpretability"""
    print("\n" + "="*60)
    print("Demo 8: Model Interpretability")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.random.choice([0, 1], 200, p=[0.9, 0.1])
    
    # Train model
    model = FraudDetectionModel(model_type='random_forest')
    model.train(X, y)
    
    # Create interpreter
    feature_names = [f'feature_{i}' for i in range(5)]
    interpreter = ModelInterpreter(model.model, X, feature_names=feature_names)
    
    # Get feature importance
    importance_df = interpreter.get_feature_importance()
    print("\nTop 5 Most Important Features:")
    print(importance_df.head(5))
    
    # Explain instance (if SHAP available)
    try:
        instance = X[0:1]
        explanation = interpreter.explain_instance(instance)
        print(f"\nExplanation method: {explanation['method']}")
        print("Top contributing features:")
        for feature, contribution in list(explanation['top_features'].items())[:3]:
            print(f"  {feature}: {contribution:.4f}")
    except Exception as e:
        print(f"\nSHAP explanation not available: {e}")


if __name__ == '__main__':
    print("Fraud Detection System - Advanced Features Demo")
    print("Developer: Molla Samser (Founder) - https://rskworld.in")
    print("="*60)
    
    try:
        demo_anomaly_detection()
        demo_data_augmentation()
        demo_cross_validation()
        demo_model_selection()
        demo_monitoring()
        demo_time_series_analysis()
        demo_model_versioning()
        demo_interpretability()
        
        print("\n" + "="*60)
        print("All advanced feature demos completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

