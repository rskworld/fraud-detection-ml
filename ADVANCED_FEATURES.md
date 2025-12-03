# Advanced Features Guide - Fraud Detection ML System

<!--
Project: Fraud Detection System using ML
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
-->

## Overview

This document describes the advanced features available in the Fraud Detection ML System.

## 1. Anomaly Detection

Detect outliers and anomalies using multiple algorithms.

### Usage

```python
from src.anomaly_detection import AnomalyDetector, EnsembleAnomalyDetector

# Single detector
detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
detector.fit(X_train)
results = detector.detect_anomalies(X_test, threshold=0.5)

# Ensemble detector (combines multiple methods)
ensemble = EnsembleAnomalyDetector(contamination=0.05)
ensemble.fit(X_train)
scores = ensemble.predict_proba(X_test)
```

### Methods Available
- **Isolation Forest**: Tree-based anomaly detection
- **Elliptic Envelope**: Statistical outlier detection
- **Local Outlier Factor (LOF)**: Density-based detection
- **Ensemble**: Combines all methods for robust detection

## 2. Model Interpretability

Understand model predictions using SHAP values and feature importance.

### Usage

```python
from src.interpretability import ModelInterpreter

interpreter = ModelInterpreter(model, X_train, feature_names)
importance_df = interpreter.get_feature_importance()

# Explain single prediction
explanation = interpreter.explain_instance(instance, top_features=10)

# SHAP values (if SHAP installed)
shap_values, explainer = interpreter.explain_prediction_shap(X_explain)
interpreter.plot_shap_summary(shap_values, X_explain)
```

### Features
- Native feature importance
- Permutation importance
- SHAP values (TreeExplainer, KernelExplainer)
- Feature contribution analysis
- Visualization tools

## 3. Data Augmentation

Handle imbalanced datasets with various oversampling and undersampling techniques.

### Usage

```python
from src.data_augmentation import DataAugmenter, AdvancedAugmenter

# SMOTE
augmenter = DataAugmenter(method='smote')
X_resampled, y_resampled = augmenter.fit_resample(X, y)

# Balance to target ratio
X_balanced, y_balanced = augmenter.balance_data(X, y, target_ratio=0.5)

# Advanced augmentation
advanced = AdvancedAugmenter()
X_aug, y_aug = advanced.augment_with_noise(X, y, noise_factor=0.01)
```

### Methods Available
- **SMOTE**: Synthetic Minority Oversampling
- **ADASYN**: Adaptive Synthetic Sampling
- **BorderlineSMOTE**: Borderline samples focus
- **SMOTETomek**: SMOTE + Tomek links
- **SMOTEENN**: SMOTE + Edited Nearest Neighbours
- **Random Undersampling**: Reduce majority class

## 4. Cross-Validation

Robust model evaluation with multiple cross-validation strategies.

### Usage

```python
from src.cross_validation import CrossValidator, ModelSelector

# Cross-validation
cv = CrossValidator(cv=5, scoring='roc_auc')
cv_results = cv.cross_validate_model(model, X, y)
summary = cv.evaluate_cv_results(cv_results)

# Automated model selection
selector = ModelSelector(cv=5, scoring='roc_auc')
best_model, best_type, results = selector.select_best_model(
    X_train, y_train, 
    model_types=['random_forest', 'xgboost', 'neural_network']
)
```

### CV Methods
- **Stratified K-Fold**: Maintains class distribution
- **K-Fold**: Standard cross-validation
- **Time Series Split**: For temporal data

## 5. Model Monitoring

Track model performance and detect drift over time.

### Usage

```python
from src.monitoring import ModelMonitor, AlertSystem

# Initialize monitor
monitor = ModelMonitor(log_dir='logs/', model_name='fraud_model')

# Log predictions
monitor.log_prediction(
    transaction_id='txn_123',
    features={'amount': 1500},
    prediction=1,
    probability=0.85
)

# Get statistics
stats = monitor.get_prediction_stats(hours=24)

# Check for drift
drift_info = monitor.check_model_drift(current_metrics, baseline_metrics)

# Alert system
alert_system = AlertSystem(monitor)
alert_system.add_default_rules()
alerts = alert_system.check_alerts(transaction_data, prediction, probability)
```

### Features
- Prediction logging
- Performance tracking
- Drift detection
- Configurable alert rules
- Real-time statistics

## 6. Time Series Analysis

Extract temporal patterns and velocity features.

### Usage

```python
from src.time_series_analysis import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer(time_col='timestamp')

# Extract temporal features
df_features = analyzer.extract_temporal_features(df)

# Calculate velocity features
df_velocity = analyzer.calculate_velocity_features(
    df, user_col='user_id', value_col='amount', windows=[1, 6, 24]
)

# Detect anomalous timing
df_anomalous = analyzer.detect_anomalous_timing(df, user_col='user_id')

# Detect fraud patterns
df_patterns = analyzer.detect_fraud_patterns(df, user_col='user_id')
```

### Features
- Temporal feature extraction (hour, day, month, cyclical encoding)
- Transaction velocity (counts, sums, averages over time windows)
- Anomalous timing detection
- Fraud pattern detection (rapid transactions, unusual sequences)

## 7. Model Versioning

Manage and compare multiple model versions.

### Usage

```python
from src.model_versioning import ModelVersionManager, ModelComparator

# Version management
version_manager = ModelVersionManager(model_dir='data/models/versions/')

# Register model
version_manager.register_model(
    model, 
    model_name='fraud_detector',
    version='1.0.0',
    metrics={'accuracy': 0.95, 'roc_auc': 0.92},
    description='Initial model'
)

# Set active version
version_manager.set_active_version('fraud_detector', '1.0.0')

# Compare versions
comparison = version_manager.compare_versions('fraud_detector')

# Load model
model = version_manager.load_model('fraud_detector', version='1.0.0')

# Compare multiple models
comparator = ModelComparator()
comparison_df = comparator.compare_models(
    {'RF': model1, 'XGB': model2, 'NN': model3},
    X_test, y_test
)
```

### Features
- Model version tracking
- Metadata management
- Version comparison
- Active version management
- Model loading by version

## 8. Complete Workflow Example

```python
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.data_augmentation import DataAugmenter
from src.cross_validation import ModelSelector
from src.models import FraudDetectionModel
from src.anomaly_detection import AnomalyDetector
from src.interpretability import ModelInterpreter
from src.monitoring import ModelMonitor
from src.model_versioning import ModelVersionManager

# 1. Load and preprocess
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df, target_col='is_fraud')

# 2. Feature engineering
fe = FeatureEngineer()
X_train, _ = fe.engineer_features(X_train, target_col='is_fraud')
X_test, _ = fe.engineer_features(X_test, target_col='is_fraud')

# 3. Handle imbalanced data
augmenter = DataAugmenter(method='smote')
X_train, y_train = augmenter.fit_resample(X_train, y_train)

# 4. Anomaly detection
anomaly_detector = AnomalyDetector()
anomaly_detector.fit(X_train)
anomaly_results = anomaly_detector.detect_anomalies(X_test)

# 5. Model selection
selector = ModelSelector(cv=5)
best_model, best_type, _ = selector.select_best_model(
    X_train, y_train,
    model_types=['random_forest', 'xgboost']
)

# 6. Train final model
final_model = FraudDetectionModel(model_type=best_type)
final_model.train(X_train, y_train, X_test, y_test)

# 7. Interpretability
interpreter = ModelInterpreter(final_model.model, X_train)
importance_df = interpreter.get_feature_importance()

# 8. Monitoring
monitor = ModelMonitor()
monitor.log_performance(metrics, model_version='1.0.0')

# 9. Versioning
version_manager = ModelVersionManager()
version_manager.register_model(
    final_model.model,
    model_name='fraud_detector',
    version='1.0.0',
    metrics=metrics
)
```

## Running Advanced Features Demo

```bash
python advanced_features_demo.py
```

This will demonstrate all advanced features with sample data.

## Contact

**Founder**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in, support@rskworld.in  
**Phone**: +91 93305 39277

© 2025 RSK World. All rights reserved.

