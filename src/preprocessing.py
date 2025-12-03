"""
Data Preprocessing Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for fraud detection.
    Handles missing values, encoding, scaling, and data splitting.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()  # RobustScaler is better for outliers
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical(self, df, categorical_cols=None):
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(df[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown = unique_values - known_values
                    
                    if unknown:
                        # Add unknown categories
                        all_values = list(known_values) + list(unknown)
                        self.label_encoders[col].classes_ = np.array(all_values)
                    
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def remove_outliers(self, df, columns=None, method='IQR'):
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to process
            method: Method for outlier detection ('IQR' or 'Z-score')
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'IQR':
            for col in columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                       (df_clean[col] <= upper_bound)]
        
        return df_clean
    
    def scale_features(self, df, columns=None, fit=True):
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: List of columns to scale
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler.fit(df[columns])
            self.is_fitted = True
        
        df_scaled[columns] = self.scaler.transform(df[columns])
        
        return df_scaled
    
    def preprocess(self, df, target_col='is_fraud', remove_outliers=False):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            remove_outliers: Whether to remove outliers
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical(df_processed)
        
        # Remove outliers if requested
        if remove_outliers:
            df_processed = self.remove_outliers(df_processed)
        
        # Separate features and target
        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
        else:
            X = df_processed
            y = None
        
        # Scale features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = self.scale_features(X, columns=numerical_cols, fit=True)
        
        # Split data if target is available
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, 
                stratify=y if y.nunique() > 1 else None
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, None, None, None
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.encode_categorical(df_processed)
        
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        df_processed = self.scale_features(df_processed, columns=numerical_cols, fit=False)
        
        return df_processed

