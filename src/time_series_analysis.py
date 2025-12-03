"""
Time Series Analysis Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Time series analysis for fraud detection patterns.
    """
    
    def __init__(self, time_col='timestamp'):
        """
        Initialize time series analyzer.
        
        Args:
            time_col: Name of timestamp column
        """
        self.time_col = time_col
    
    def extract_temporal_features(self, df):
        """
        Extract comprehensive temporal features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        df_features = df.copy()
        df_features[self.time_col] = pd.to_datetime(df_features[self.time_col])
        
        # Basic temporal features
        df_features['hour'] = df_features[self.time_col].dt.hour
        df_features['day_of_week'] = df_features[self.time_col].dt.dayofweek
        df_features['day_of_month'] = df_features[self.time_col].dt.day
        df_features['month'] = df_features[self.time_col].dt.month
        df_features['quarter'] = df_features[self.time_col].dt.quarter
        df_features['year'] = df_features[self.time_col].dt.year
        
        # Derived features
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_month_start'] = (df_features['day_of_month'] <= 3).astype(int)
        df_features['is_month_end'] = (df_features['day_of_month'] >= 28).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
        df_features['is_business_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        return df_features
    
    def calculate_time_based_statistics(self, df, group_col=None, value_col='amount'):
        """
        Calculate time-based statistics.
        
        Args:
            df: DataFrame
            group_col: Column to group by (e.g., 'user_id')
            value_col: Column to aggregate
            
        Returns:
            DataFrame with statistics
        """
        df_features = df.copy()
        df_features[self.time_col] = pd.to_datetime(df_features[self.time_col])
        
        if group_col:
            # Per-group statistics
            for period in ['hour', 'day', 'week', 'month']:
                if period == 'hour':
                    df_features['period'] = df_features[self.time_col].dt.floor('H')
                elif period == 'day':
                    df_features['period'] = df_features[self.time_col].dt.date
                elif period == 'week':
                    df_features['period'] = df_features[self.time_col].dt.to_period('W')
                elif period == 'month':
                    df_features['period'] = df_features[self.time_col].dt.to_period('M')
                
                stats = df_features.groupby([group_col, 'period'])[value_col].agg([
                    'mean', 'std', 'count', 'sum'
                ]).reset_index()
                
                stats.columns = [group_col, 'period', 
                               f'{value_col}_mean_{period}', 
                               f'{value_col}_std_{period}',
                               f'{value_col}_count_{period}',
                               f'{value_col}_sum_{period}']
                
                df_features = df_features.merge(
                    stats, on=[group_col, 'period'], how='left'
                )
        
        return df_features
    
    def detect_anomalous_timing(self, df, user_col='user_id', window_hours=24):
        """
        Detect transactions at anomalous times for users.
        
        Args:
            df: DataFrame
            user_col: User identifier column
            window_hours: Time window for analysis
            
        Returns:
            DataFrame with anomalous timing flags
        """
        df_features = df.copy()
        df_features[self.time_col] = pd.to_datetime(df_features[self.time_col])
        df_features = df_features.sort_values([user_col, self.time_col])
        
        # Calculate time since last transaction
        df_features['time_since_last'] = (
            df_features.groupby(user_col)[self.time_col].diff().dt.total_seconds() / 3600
        )
        
        # Calculate typical transaction hours per user
        user_hours = df_features.groupby(user_col)['hour'].agg(['mean', 'std']).reset_index()
        user_hours.columns = [user_col, 'typical_hour_mean', 'typical_hour_std']
        df_features = df_features.merge(user_hours, on=user_col, how='left')
        
        # Flag anomalous timing
        df_features['hour_deviation'] = abs(
            df_features['hour'] - df_features['typical_hour_mean']
        )
        df_features['is_anomalous_time'] = (
            (df_features['hour_deviation'] > 6) | 
            (df_features['time_since_last'] < 0.1)  # Very quick successive transactions
        ).astype(int)
        
        return df_features
    
    def calculate_velocity_features(self, df, user_col='user_id', 
                                   value_col='amount', windows=[1, 6, 24]):
        """
        Calculate transaction velocity features.
        
        Args:
            df: DataFrame
            user_col: User identifier column
            value_col: Value column to analyze
            windows: List of time windows in hours
            
        Returns:
            DataFrame with velocity features
        """
        df_features = df.copy()
        df_features[self.time_col] = pd.to_datetime(df_features[self.time_col])
        df_features = df_features.sort_values([user_col, self.time_col])
        
        for window in windows:
            # Rolling window calculations
            df_features[f'transactions_last_{window}h'] = (
                df_features.groupby(user_col)[self.time_col]
                .transform(lambda x: x.rolling(f'{window}H', on=self.time_col).count())
            )
            
            df_features[f'amount_sum_last_{window}h'] = (
                df_features.groupby(user_col)[[self.time_col, value_col]]
                .apply(lambda x: x.set_index(self.time_col)[value_col]
                      .rolling(f'{window}H').sum().values)
                .reset_index(level=0, drop=True)
            )
            
            df_features[f'amount_avg_last_{window}h'] = (
                df_features[f'amount_sum_last_{window}h'] / 
                (df_features[f'transactions_last_{window}h'] + 1e-6)
            )
        
        return df_features
    
    def detect_fraud_patterns(self, df, user_col='user_id'):
        """
        Detect common fraud patterns in time series.
        
        Args:
            df: DataFrame
            user_col: User identifier column
            
        Returns:
            DataFrame with fraud pattern flags
        """
        df_features = df.copy()
        df_features[self.time_col] = pd.to_datetime(df_features[self.time_col])
        
        # Pattern 1: Rapid successive transactions
        df_features = df_features.sort_values([user_col, self.time_col])
        df_features['time_diff'] = (
            df_features.groupby(user_col)[self.time_col].diff().dt.total_seconds() / 60
        )
        df_features['rapid_transactions'] = (df_features['time_diff'] < 5).astype(int)
        
        # Pattern 2: Unusual amount sequences
        df_features['amount_diff'] = df_features.groupby(user_col)['amount'].diff()
        df_features['amount_ratio'] = (
            df_features['amount'] / 
            (df_features.groupby(user_col)['amount'].shift(1) + 1e-6)
        )
        df_features['unusual_amount_change'] = (
            (df_features['amount_ratio'] > 10) | (df_features['amount_ratio'] < 0.1)
        ).astype(int)
        
        # Pattern 3: Time clustering
        df_features['hour_cluster'] = (
            df_features.groupby([user_col, df_features[self.time_col].dt.date])['hour']
            .transform('nunique')
        )
        df_features['multiple_hours_same_day'] = (df_features['hour_cluster'] > 8).astype(int)
        
        return df_features

