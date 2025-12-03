# Fraud Detection ML Project - Summary

<!--
Project: Fraud Detection System using ML
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
-->

## Project Overview

This is an advanced fraud detection system using machine learning with comprehensive features for identifying fraudulent transactions and activities.

## Project Structure

### Core Modules (`src/`)

1. **`preprocessing.py`** - Data preprocessing module
   - Missing value handling
   - Categorical encoding
   - Feature scaling
   - Outlier removal
   - Train/test splitting

2. **`feature_engineering.py`** - Advanced feature engineering
   - Temporal features (hour, day, month, weekend detection)
   - Statistical aggregation features
   - Interaction features
   - Rolling window statistics
   - Fraud-specific indicators
   - Feature selection

3. **`models.py`** - Machine learning models
   - Random Forest Classifier
   - XGBoost Classifier
   - Neural Network (TensorFlow/Keras)
   - Ensemble model (combines all three)
   - Hyperparameter tuning support
   - Model saving/loading

4. **`evaluation.py`** - Model evaluation
   - Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Confusion matrix visualization
   - ROC curve plotting
   - Precision-Recall curve
   - Calibration curve
   - Optimal threshold finding

5. **`fraud_scorer.py`** - Real-time fraud scoring
   - Single transaction scoring
   - Batch transaction scoring
   - Fraud explanation/interpretation
   - Risk level classification
   - Threshold management

### Jupyter Notebooks (`notebooks/`)

1. **`01_data_exploration.ipynb`** - Data exploration and analysis
2. **`02_feature_engineering.ipynb`** - Feature engineering workflow
3. **`03_model_training.ipynb`** - Model training pipeline
4. **`04_model_evaluation.ipynb`** - Model evaluation and comparison

### API and Scripts

1. **`app.py`** - Flask REST API for real-time fraud detection
   - `/predict` - Single transaction prediction
   - `/predict/batch` - Batch prediction
   - `/explain` - Get fraud explanation
   - `/threshold` - Manage detection threshold
   - `/health` - Health check

2. **`main.py`** - Complete training pipeline script

3. **`example_usage.py`** - Usage examples and demos

4. **`generate_sample_data.py`** - Sample data generator

### Configuration

- **`config/config.yaml`** - Project configuration file
- **`requirements.txt`** - Python dependencies
- **`setup.py`** - Package setup script

### Documentation

- **`README.md`** - Main project documentation
- **`QUICKSTART.md`** - Quick start guide
- **`LICENSE`** - MIT License
- **`.gitignore`** - Git ignore rules

## Key Features Implemented

✅ **Transaction Data Preprocessing**
- Comprehensive data cleaning
- Missing value imputation
- Categorical encoding
- Feature scaling (RobustScaler)
- Outlier detection and removal

✅ **Feature Engineering**
- Temporal features (time-based patterns)
- Statistical aggregations
- Interaction features
- Rolling window statistics
- Fraud-specific indicators
- Feature selection (mutual information, f-classif)

✅ **Multiple ML Algorithms**
- Random Forest (with class balancing)
- XGBoost (with early stopping)
- Neural Network (deep learning with dropout)
- Ensemble model (combines all three)

✅ **Real-time Fraud Scoring**
- Single transaction API endpoint
- Batch processing API endpoint
- Fraud probability scores
- Risk level classification (Low/Medium/High)
- Feature importance explanation

✅ **Performance Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Average Precision
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Calibration Curve
- Optimal threshold finding

## Advanced Features

1. **Ensemble Learning**: Combines predictions from multiple models for better accuracy
2. **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV support
3. **Imbalanced Data Handling**: Class weights and scale_pos_weight for fraud detection
4. **Real-time API**: Flask-based REST API for production deployment
5. **Comprehensive Evaluation**: Multiple metrics and visualizations
6. **Feature Importance**: Model interpretability
7. **Batch Processing**: Efficient scoring of multiple transactions
8. **Threshold Optimization**: Find optimal threshold based on different metrics

## Usage Examples

### Training a Model
```python
from src.models import FraudDetectionModel
from src.preprocessing import DataPreprocessor

# Preprocess data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df, target_col='is_fraud')

# Train model
model = FraudDetectionModel(model_type='ensemble')
model.train(X_train, y_train, X_test, y_test)
```

### Real-time Scoring
```python
from src.fraud_scorer import FraudScorer

scorer = FraudScorer(model_path='data/models/fraud_model.pkl')
fraud_score = scorer.predict(transaction_data)
```

### API Usage
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "user_id": 123, "timestamp": "2024-01-15 14:30:00"}'
```

## Contact Information

**Founder**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in, support@rskworld.in  
**Phone**: +91 93305 39277  
**Location**: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147

© 2025 RSK World. All rights reserved.

