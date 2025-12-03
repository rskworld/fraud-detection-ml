"""
Feature Engineering Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection.
    Creates statistical, temporal, and behavioral features.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        
    def create_temporal_features(self, df, date_col=None):
        """
        Create temporal features from date/time columns.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with temporal features
        """
        df_features = df.copy()
        
        if date_col and date_col in df.columns:
            df_features[date_col] = pd.to_datetime(df_features[date_col])
            df_features['hour'] = df_features[date_col].dt.hour
            df_features['day_of_week'] = df_features[date_col].dt.dayofweek
            df_features['day_of_month'] = df_features[date_col].dt.day
            df_features['month'] = df_features[date_col].dt.month
            df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
            df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
        
        return df_features
    
    def create_statistical_features(self, df, group_cols=None, value_cols=None):
        """
        Create statistical aggregation features.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            value_cols: Columns to aggregate
            
        Returns:
            DataFrame with statistical features
        """
        df_features = df.copy()
        
        if group_cols and value_cols:
            for group_col in group_cols:
                if group_col in df.columns:
                    for value_col in value_cols:
                        if value_col in df.columns:
                            # Mean, std, count per group
                            stats = df.groupby(group_col)[value_col].agg([
                                'mean', 'std', 'count'
                            ]).reset_index()
                            stats.columns = [group_col, f'{value_col}_mean_by_{group_col}',
                                           f'{value_col}_std_by_{group_col}',
                                           f'{value_col}_count_by_{group_col}']
                            df_features = df_features.merge(stats, on=group_col, how='left')
                            
                            # Ratio to mean
                            df_features[f'{value_col}_ratio_to_mean_{group_col}'] = (
                                df_features[value_col] / 
                                (df_features[f'{value_col}_mean_by_{group_col}'] + 1e-6)
                            )
        
        return df_features
    
    def create_interaction_features(self, df, feature_pairs=None):
        """
        Create interaction features between pairs of features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples (col1, col2) for interactions
            
        Returns:
            DataFrame with interaction features
        """
        df_features = df.copy()
        
        if feature_pairs:
            for col1, col2 in feature_pairs:
                if col1 in df.columns and col2 in df.columns:
                    df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
                    df_features[f'{col1}_div_{col2}'] = (
                        df_features[col1] / (df_features[col2] + 1e-6)
                    )
        
        return df_features
    
    def create_rolling_features(self, df, value_cols=None, windows=[3, 7, 30]):
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            value_cols: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_features = df.copy()
        
        if value_cols:
            for col in value_cols:
                if col in df.columns:
                    for window in windows:
                        df_features[f'{col}_rolling_mean_{window}'] = (
                            df_features[col].rolling(window=window, min_periods=1).mean()
                        )
                        df_features[f'{col}_rolling_std_{window}'] = (
                            df_features[col].rolling(window=window, min_periods=1).std()
                        )
        
        return df_features
    
    def create_fraud_indicators(self, df, amount_col='amount', 
                               time_col=None, user_col=None):
        """
        Create fraud-specific indicator features.
        
        Args:
            df: Input DataFrame
            amount_col: Name of amount column
            time_col: Name of time column
            user_col: Name of user/account column
            
        Returns:
            DataFrame with fraud indicators
        """
        df_features = df.copy()
        
        # Amount-based features
        if amount_col in df.columns:
            df_features['amount_log'] = np.log1p(df_features[amount_col])
            df_features['amount_squared'] = df_features[amount_col] ** 2
            df_features['is_high_amount'] = (
                df_features[amount_col] > df_features[amount_col].quantile(0.95)
            ).astype(int)
            df_features['is_low_amount'] = (
                df_features[amount_col] < df_features[amount_col].quantile(0.05)
            ).astype(int)
        
        # User behavior features
        if user_col and user_col in df.columns:
            # Transaction frequency
            user_counts = df_features.groupby(user_col).size().reset_index(name='user_txn_count')
            df_features = df_features.merge(user_counts, on=user_col, how='left')
            
            # Average amount per user
            if amount_col in df.columns:
                user_avg_amount = df_features.groupby(user_col)[amount_col].mean().reset_index(
                    name='user_avg_amount'
                )
                df_features = df_features.merge(user_avg_amount, on=user_col, how='left')
                df_features['amount_diff_from_user_avg'] = (
                    df_features[amount_col] - df_features['user_avg_amount']
                )
        
        # Time-based fraud indicators
        if time_col and time_col in df.columns:
            df_features[time_col] = pd.to_datetime(df_features[time_col])
            # Time since last transaction
            if user_col and user_col in df.columns:
                df_features = df_features.sort_values([user_col, time_col])
                df_features['time_since_last_txn'] = (
                    df_features.groupby(user_col)[time_col].diff().dt.total_seconds() / 3600
                )
                df_features['time_since_last_txn'].fillna(0, inplace=True)
        
        return df_features
    
    def select_features(self, X, y, method='mutual_info', k=50):
        """
        Select best features using statistical methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('mutual_info' or 'f_classif')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        
        self.feature_selector = selector
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)
        
        # Return as DataFrame with selected column names
        selected_cols = X.columns[self.selected_features]
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
    
    def apply_pca(self, X, n_components=0.95):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components or variance to retain
            
        Returns:
            DataFrame with PCA features
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    
    def engineer_features(self, df, target_col='is_fraud', 
                         amount_col='amount', user_col=None, date_col=None):
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            amount_col: Name of amount column
            user_col: Name of user/account column
            date_col: Name of date column
            
        Returns:
            Tuple of (X, y) with engineered features
        """
        df_features = df.copy()
        
        # Create temporal features
        if date_col:
            df_features = self.create_temporal_features(df_features, date_col)
        
        # Create fraud indicators
        df_features = self.create_fraud_indicators(
            df_features, amount_col=amount_col, 
            time_col=date_col, user_col=user_col
        )
        
        # Create interaction features (example)
        if amount_col in df_features.columns:
            numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_cols) > 1:
                # Create some interaction features
                for col in numerical_cols[:3]:  # Limit to avoid too many features
                    if col != amount_col:
                        df_features[f'{amount_col}_x_{col}'] = (
                            df_features[amount_col] * df_features[col]
                        )
        
        # Separate features and target
        if target_col in df_features.columns:
            X = df_features.drop(columns=[target_col])
            y = df_features[target_col]
        else:
            X = df_features
            y = None
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X, y

