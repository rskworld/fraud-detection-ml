"""
Main Training Script for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
import os
import sys
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import FraudDetectionModel
from src.evaluation import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main training pipeline."""
    print("="*60)
    print("Fraud Detection ML System - Training Pipeline")
    print("Developer: Molla Samser (Founder) - https://rskworld.in")
    print("="*60)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load or generate sample data
    data_path = 'data/raw/transactions.csv'
    if os.path.exists(data_path):
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("\nGenerating sample data...")
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            'transaction_id': range(1, n_samples + 1),
            'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
            'user_id': np.random.randint(1, 1000, size=n_samples),
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'merchant_category': np.random.choice(['retail', 'online', 'gas', 'restaurant'], n_samples),
            'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        })
        df.to_csv(data_path, index=False)
        print(f"Sample data saved to {data_path}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # Feature Engineering
    print("\n" + "="*60)
    print("Step 1: Feature Engineering")
    print("="*60)
    fe = FeatureEngineer()
    X, y = fe.engineer_features(
        df,
        target_col='is_fraud',
        amount_col='amount',
        user_col='user_id',
        date_col='timestamp'
    )
    print(f"Engineered features shape: {X.shape}")
    
    # Preprocessing
    print("\n" + "="*60)
    print("Step 2: Data Preprocessing")
    print("="*60)
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        pd.concat([X, pd.Series(y, name='is_fraud')], axis=1),
        target_col='is_fraud'
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Model Training
    print("\n" + "="*60)
    print("Step 3: Model Training")
    print("="*60)
    
    # Train Ensemble Model
    print("Training Ensemble Model...")
    model = FraudDetectionModel(model_type='ensemble')
    model.train(X_train, y_train, X_test, y_test)
    print("Training complete!")
    
    # Evaluation
    print("\n" + "="*60)
    print("Step 4: Model Evaluation")
    print("="*60)
    evaluator = ModelEvaluator()
    
    y_pred_proba = model.predict(X_test)
    y_pred = model.predict_class(X_test, threshold=0.5)
    
    metrics = evaluator.evaluate(
        y_test, y_pred, y_pred_proba,
        model_name='Ensemble Model',
        save_plots=True,
        plot_dir='./plots/'
    )
    
    # Save model
    print("\n" + "="*60)
    print("Step 5: Saving Model")
    print("="*60)
    model_path = 'data/models/fraud_model'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

