"""
Data Augmentation Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataAugmenter:
    """
    Data augmentation for handling imbalanced fraud detection datasets.
    """
    
    def __init__(self, method='smote', sampling_strategy='auto'):
        """
        Initialize data augmenter.
        
        Args:
            method: Augmentation method ('smote', 'adasyn', 'borderline_smote', 
                    'smote_tomek', 'smote_enn', 'undersample')
            sampling_strategy: Sampling strategy (float, dict, or 'auto')
        """
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.sampler = None
        self.scaler = StandardScaler()
        
    def _create_sampler(self):
        """Create the sampling object."""
        if self.method == 'smote':
            return SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == 'adasyn':
            return ADASYN(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == 'borderline_smote':
            return BorderlineSMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == 'smote_tomek':
            return SMOTETomek(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == 'smote_enn':
            return SMOTEENN(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == 'undersample':
            return RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_resample(self, X, y):
        """
        Fit and resample the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Resampled X and y
        """
        self.sampler = self._create_sampler()
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        return X_resampled, y_resampled
    
    def get_class_distribution(self, y):
        """
        Get class distribution.
        
        Args:
            y: Target vector
            
        Returns:
            Dictionary with class distribution
        """
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        total = len(y)
        percentages = {k: (v/total)*100 for k, v in distribution.items()}
        
        return {
            'counts': distribution,
            'percentages': percentages,
            'total': total
        }
    
    def balance_data(self, X, y, target_ratio=0.5):
        """
        Balance data to target ratio.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_ratio: Target ratio for minority class
            
        Returns:
            Balanced X and y
        """
        current_dist = self.get_class_distribution(y)
        minority_class = min(current_dist['counts'], key=current_dist['counts'].get)
        majority_class = max(current_dist['counts'], key=current_dist['counts'].get)
        
        minority_count = current_dist['counts'][minority_class]
        majority_count = current_dist['counts'][majority_class]
        
        # Calculate target minority count
        target_minority = int(majority_count * target_ratio / (1 - target_ratio))
        
        if target_minority > minority_count:
            # Oversample minority class
            sampling_strategy = {minority_class: target_minority}
            self.sampling_strategy = sampling_strategy
            self.method = 'smote'
        else:
            # Undersample majority class
            sampling_strategy = {majority_class: int(minority_count / target_ratio)}
            self.sampling_strategy = sampling_strategy
            self.method = 'undersample'
        
        return self.fit_resample(X, y)


class AdvancedAugmenter:
    """
    Advanced data augmentation with multiple techniques.
    """
    
    def __init__(self):
        """Initialize advanced augmenter."""
        self.augmenters = {
            'smote': DataAugmenter('smote'),
            'adasyn': DataAugmenter('adasyn'),
            'borderline_smote': DataAugmenter('borderline_smote'),
            'smote_tomek': DataAugmenter('smote_tomek')
        }
    
    def augment_with_noise(self, X, y, noise_factor=0.01, n_samples=None):
        """
        Augment data by adding noise to minority class.
        
        Args:
            X: Feature matrix
            y: Target vector
            noise_factor: Noise level
            n_samples: Number of samples to generate
            
        Returns:
            Augmented X and y
        """
        minority_class = 1  # Assuming 1 is fraud
        minority_indices = np.where(y == minority_class)[0]
        minority_X = X[minority_indices]
        
        if n_samples is None:
            n_samples = len(minority_X)
        
        # Generate synthetic samples
        noise = np.random.normal(0, noise_factor, (n_samples, X.shape[1]))
        synthetic_X = minority_X[np.random.choice(len(minority_X), n_samples)] + noise
        synthetic_y = np.ones(n_samples) * minority_class
        
        # Combine
        X_augmented = np.vstack([X, synthetic_X])
        y_augmented = np.hstack([y, synthetic_y])
        
        return X_augmented, y_augmented
    
    def augment_with_rotation(self, X, y, n_samples=None):
        """
        Augment data by rotating features (for 2D+ feature space).
        
        Args:
            X: Feature matrix
            y: Target vector
            n_samples: Number of samples to generate
            
        Returns:
            Augmented X and y
        """
        minority_class = 1
        minority_indices = np.where(y == minority_class)[0]
        minority_X = X[minority_indices]
        
        if n_samples is None:
            n_samples = len(minority_X)
        
        # Simple rotation by swapping features
        synthetic_X = minority_X[np.random.choice(len(minority_X), n_samples)].copy()
        
        # Random feature swaps
        for i in range(n_samples):
            swap_indices = np.random.choice(X.shape[1], 2, replace=False)
            synthetic_X[i, swap_indices] = synthetic_X[i, swap_indices[::-1]]
        
        synthetic_y = np.ones(n_samples) * minority_class
        
        X_augmented = np.vstack([X, synthetic_X])
        y_augmented = np.hstack([y, synthetic_y])
        
        return X_augmented, y_augmented

