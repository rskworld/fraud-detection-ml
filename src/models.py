"""
Machine Learning Models for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    Main fraud detection model class supporting multiple algorithms.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the fraud detection model.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'neural_network', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.models = {}  # For ensemble
        self.is_trained = False
        
    def _create_random_forest(self, n_estimators=100, max_depth=20, 
                              min_samples_split=5, class_weight='balanced'):
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    def _create_xgboost(self, n_estimators=200, max_depth=6, 
                       learning_rate=0.1, scale_pos_weight=1):
        """Create XGBoost model."""
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def _create_neural_network(self, input_dim, layers_config=None):
        """
        Create Neural Network model.
        
        Args:
            input_dim: Input feature dimension
            layers_config: List of layer sizes
        """
        if layers_config is None:
            layers_config = [128, 64, 32]
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(layers_config[0], activation='relu', 
                              input_dim=input_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Hidden layers
        for layer_size in layers_config[1:]:
            model.add(layers.Dense(layer_size, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              hyperparameter_tuning=False):
        """
        Train the fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        """
        if self.model_type == 'random_forest':
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestClassifier(
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                self.model = GridSearchCV(
                    base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
                )
            else:
                # Calculate scale_pos_weight for imbalanced data
                fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
                self.model = self._create_random_forest(scale_pos_weight=fraud_ratio)
            
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgboost':
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                base_model = XGBClassifier(random_state=42, use_label_encoder=False)
                self.model = RandomizedSearchCV(
                    base_model, param_grid, cv=5, scoring='roc_auc', 
                    n_iter=10, n_jobs=-1
                )
            else:
                fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
                self.model = self._create_xgboost(scale_pos_weight=fraud_ratio)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
                
        elif self.model_type == 'neural_network':
            input_dim = X_train.shape[1]
            self.model = self._create_neural_network(input_dim)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            validation_data = (X_val, y_val) if X_val is not None else None
            
            self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
        elif self.model_type == 'ensemble':
            # Train multiple models for ensemble
            fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
            
            # Random Forest
            self.models['rf'] = self._create_random_forest()
            self.models['rf'].fit(X_train, y_train)
            
            # XGBoost
            self.models['xgb'] = self._create_xgboost(scale_pos_weight=fraud_ratio)
            if X_val is not None:
                self.models['xgb'].fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
                self.models['xgb'].fit(X_train, y_train)
            
            # Neural Network
            input_dim = X_train.shape[1]
            self.models['nn'] = self._create_neural_network(input_dim)
            validation_data = (X_val, y_val) if X_val is not None else None
            self.models['nn'].fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=validation_data,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                verbose=0
            )
        
        self.is_trained = True
    
    def predict(self, X):
        """
        Predict fraud probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of fraud probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.model_type == 'ensemble':
            # Average predictions from all models
            predictions = []
            for name, model in self.models.items():
                if name == 'nn':
                    pred = model.predict(X, verbose=0).flatten()
                else:
                    pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        elif self.model_type == 'neural_network':
            return self.model.predict(X, verbose=0).flatten()
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def predict_class(self, X, threshold=0.5):
        """
        Predict fraud class (0 or 1).
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Array of fraud predictions (0 or 1)
        """
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if self.model_type == 'neural_network':
            self.model.save(filepath)
        elif self.model_type == 'ensemble':
            for name, model in self.models.items():
                if name == 'nn':
                    model.save(f"{filepath}_{name}.h5")
                else:
                    joblib.dump(model, f"{filepath}_{name}.pkl")
        else:
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(filepath)
        elif self.model_type == 'ensemble':
            # Load all ensemble models
            self.models['rf'] = joblib.load(f"{filepath}_rf.pkl")
            self.models['xgb'] = joblib.load(f"{filepath}_xgb.pkl")
            self.models['nn'] = keras.models.load_model(f"{filepath}_nn.h5")
        else:
            self.model = joblib.load(filepath)
        
        self.is_trained = True

