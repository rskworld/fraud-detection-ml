"""
Sample Data Generator for Fraud Detection System
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

def generate_fraud_data(n_samples=10000, fraud_rate=0.05, random_state=42):
    """
    Generate synthetic fraud detection dataset.
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions
        random_state: Random seed
        
    Returns:
        DataFrame with transaction data
    """
    np.random.seed(random_state)
    
    # Generate base transaction data
    df = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'user_id': np.random.randint(1, 1000, size=n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'merchant_category': np.random.choice(
            ['retail', 'online', 'gas', 'restaurant', 'grocery', 'entertainment'], 
            n_samples
        ),
        'merchant_id': np.random.randint(1, 500, size=n_samples),
        'card_type': np.random.choice(['credit', 'debit', 'prepaid'], n_samples),
    })
    
    # Generate fraud labels with some patterns
    fraud_indices = np.random.choice(
        n_samples, 
        size=int(n_samples * fraud_rate), 
        replace=False
    )
    
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Make fraud transactions have different patterns
    # Higher amounts for some fraud
    fraud_mask = df['is_fraud'] == 1
    high_amount_fraud = np.random.choice(
        fraud_indices, 
        size=int(len(fraud_indices) * 0.3), 
        replace=False
    )
    df.loc[high_amount_fraud, 'amount'] = np.random.lognormal(mean=5, sigma=1.5, size=len(high_amount_fraud))
    
    # Unusual hours for fraud
    unusual_hour_fraud = np.random.choice(
        fraud_indices, 
        size=int(len(fraud_indices) * 0.4), 
        replace=False
    )
    df.loc[unusual_hour_fraud, 'timestamp'] = pd.to_datetime(df.loc[unusual_hour_fraud, 'timestamp']) + pd.Timedelta(hours=np.random.choice([22, 23, 0, 1, 2, 3], size=len(unusual_hour_fraud)))
    
    # Add some additional features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    
    return df

if __name__ == '__main__':
    print("Generating sample fraud detection dataset...")
    print("Developer: Molla Samser (Founder) - https://rskworld.in")
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate data
    df = generate_fraud_data(n_samples=10000, fraud_rate=0.05)
    
    # Save to CSV
    output_path = 'data/raw/transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Saved to: {output_path}")
    print(f"\nFirst few rows:")
    print(df.head())

