"""
Example Usage of Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import FraudDetectionModel
from src.fraud_scorer import FraudScorer
from src.evaluation import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')


def example_1_basic_training():
    """Example 1: Basic model training."""
    print("="*60)
    print("Example 1: Basic Model Training")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'user_id': np.random.randint(1, 500, size=n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    })
    
    # Feature engineering
    fe = FeatureEngineer()
    X, y = fe.engineer_features(
        df, target_col='is_fraud', amount_col='amount',
        user_col='user_id', date_col='timestamp'
    )
    
    # Preprocessing
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        pd.concat([X, pd.Series(y, name='is_fraud')], axis=1),
        target_col='is_fraud'
    )
    
    # Train model
    model = FraudDetectionModel(model_type='random_forest')
    model.train(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    y_pred_proba = model.predict(X_test)
    y_pred = model.predict_class(X_test, threshold=0.5)
    
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba, model_name='Random Forest')
    
    return model, metrics


def example_2_real_time_scoring():
    """Example 2: Real-time fraud scoring."""
    print("\n" + "="*60)
    print("Example 2: Real-time Fraud Scoring")
    print("="*60)
    
    # Initialize scorer (in production, load trained model)
    scorer = FraudScorer(model_type='random_forest', threshold=0.5)
    
    # Note: In real usage, you would load a pre-trained model:
    # scorer = FraudScorer(model_path='data/models/fraud_model.pkl')
    
    # Sample transaction
    transaction = {
        'amount': 1500.00,
        'user_id': 123,
        'timestamp': '2024-01-15 14:30:00',
        'merchant_category': 'online'
    }
    
    # Score transaction
    fraud_score = scorer.predict(transaction, return_probability=True)
    fraud_prediction = scorer.predict(transaction, return_probability=False)
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${transaction['amount']:.2f}")
    print(f"  User ID: {transaction['user_id']}")
    print(f"  Timestamp: {transaction['timestamp']}")
    
    print(f"\nFraud Detection Results:")
    print(f"  Fraud Probability: {fraud_score:.4f}")
    print(f"  Fraud Prediction: {'FRAUD' if fraud_prediction == 1 else 'LEGITIMATE'}")
    print(f"  Risk Level: {'High' if fraud_score >= 0.7 else ('Medium' if fraud_score >= 0.3 else 'Low')}")
    
    return fraud_score, fraud_prediction


def example_3_batch_scoring():
    """Example 3: Batch transaction scoring."""
    print("\n" + "="*60)
    print("Example 3: Batch Transaction Scoring")
    print("="*60)
    
    # Create sample transactions
    transactions = pd.DataFrame({
        'transaction_id': [1, 2, 3, 4, 5],
        'amount': [50.00, 5000.00, 25.00, 10000.00, 100.00],
        'user_id': [101, 102, 103, 104, 105],
        'timestamp': pd.date_range('2024-01-15 10:00:00', periods=5, freq='1H'),
        'merchant_category': ['retail', 'online', 'gas', 'online', 'restaurant']
    })
    
    print(f"\nScoring {len(transactions)} transactions...")
    
    # Note: In production, use trained model
    scorer = FraudScorer(model_type='random_forest')
    
    # Score batch
    results = scorer.score_batch(transactions)
    
    print("\nResults:")
    print(results[['transaction_id', 'amount', 'fraud_probability', 'fraud_prediction', 'risk_level']])
    
    return results


def example_4_ensemble_model():
    """Example 4: Using ensemble model."""
    print("\n" + "="*60)
    print("Example 4: Ensemble Model Training")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 3000
    
    df = pd.DataFrame({
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'user_id': np.random.randint(1, 300, size=n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    })
    
    # Feature engineering and preprocessing
    fe = FeatureEngineer()
    X, y = fe.engineer_features(df, target_col='is_fraud', amount_col='amount',
                                user_col='user_id', date_col='timestamp')
    
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        pd.concat([X, pd.Series(y, name='is_fraud')], axis=1),
        target_col='is_fraud'
    )
    
    # Train ensemble model
    print("Training ensemble model (this may take a few minutes)...")
    model = FraudDetectionModel(model_type='ensemble')
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    y_pred_proba = model.predict(X_test)
    y_pred = model.predict_class(X_test, threshold=0.5)
    
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba, model_name='Ensemble Model')
    
    return model, metrics


if __name__ == '__main__':
    print("Fraud Detection System - Example Usage")
    print("Developer: Molla Samser (Founder) - https://rskworld.in")
    print("="*60)
    
    # Run examples
    try:
        # Example 1: Basic training
        model1, metrics1 = example_1_basic_training()
        
        # Example 2: Real-time scoring
        example_2_real_time_scoring()
        
        # Example 3: Batch scoring
        example_3_batch_scoring()
        
        # Example 4: Ensemble (commented out as it takes longer)
        # model4, metrics4 = example_4_ensemble_model()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

