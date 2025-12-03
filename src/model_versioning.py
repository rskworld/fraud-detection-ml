"""
Model Versioning and Comparison Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelVersionManager:
    """
    Manage multiple versions of fraud detection models.
    """
    
    def __init__(self, model_dir='data/models/versions/'):
        """
        Initialize model version manager.
        
        Args:
            model_dir: Directory for model versions
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / 'model_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save model metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, model, model_name, version, metrics=None, 
                      description=None, tags=None):
        """
        Register a new model version.
        
        Args:
            model: Trained model
            model_name: Name of the model
            version: Version string (e.g., '1.0.0')
            metrics: Performance metrics dictionary
            description: Model description
            tags: List of tags
        """
        version_dir = self.model_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / 'model.pkl'
        if hasattr(model, 'save'):
            # For TensorFlow/Keras models
            model.save(str(version_dir / 'model.h5'))
        else:
            joblib.dump(model, model_path)
        
        # Create metadata entry
        metadata_entry = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'description': description or '',
            'tags': tags or [],
            'model_path': str(model_path),
            'is_active': False
        }
        
        # Store in metadata
        if model_name not in self.metadata:
            self.metadata[model_name] = {}
        
        self.metadata[model_name][version] = metadata_entry
        self._save_metadata()
        
        return metadata_entry
    
    def get_model_versions(self, model_name):
        """
        Get all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of versions
        """
        return self.metadata.get(model_name, {})
    
    def get_latest_version(self, model_name):
        """
        Get latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version metadata
        """
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        # Sort by created_at
        sorted_versions = sorted(
            versions.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        return sorted_versions[0][1]
    
    def set_active_version(self, model_name, version):
        """
        Set active version of a model.
        
        Args:
            model_name: Name of the model
            version: Version to set as active
        """
        if model_name in self.metadata and version in self.metadata[model_name]:
            # Deactivate all versions
            for v in self.metadata[model_name].values():
                v['is_active'] = False
            
            # Activate specified version
            self.metadata[model_name][version]['is_active'] = True
            self._save_metadata()
        else:
            raise ValueError(f"Model {model_name} version {version} not found")
    
    def load_model(self, model_name, version=None):
        """
        Load a model version.
        
        Args:
            model_name: Name of the model
            version: Version to load (if None, loads active version)
            
        Returns:
            Loaded model
        """
        if version is None:
            # Load active version
            versions = self.get_model_versions(model_name)
            for v, metadata in versions.items():
                if metadata.get('is_active', False):
                    version = v
                    break
            
            if version is None:
                # Load latest
                latest = self.get_latest_version(model_name)
                if latest:
                    version = latest['version']
                else:
                    raise ValueError(f"No versions found for {model_name}")
        
        metadata = self.metadata[model_name][version]
        model_path = Path(metadata['model_path'])
        
        if model_path.suffix == '.h5':
            from tensorflow import keras
            return keras.models.load_model(str(model_path))
        else:
            return joblib.load(model_path)
    
    def compare_versions(self, model_name, versions=None):
        """
        Compare different versions of a model.
        
        Args:
            model_name: Name of the model
            versions: List of versions to compare (if None, compares all)
            
        Returns:
            Comparison DataFrame
        """
        all_versions = self.get_model_versions(model_name)
        
        if versions is None:
            versions = list(all_versions.keys())
        
        comparison_data = []
        for version in versions:
            if version in all_versions:
                metadata = all_versions[version]
                row = {
                    'version': version,
                    'created_at': metadata['created_at'],
                    'is_active': metadata.get('is_active', False),
                    **metadata.get('metrics', {})
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_version(self, model_name, version):
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
        """
        if model_name in self.metadata and version in self.metadata[model_name]:
            metadata = self.metadata[model_name][version]
            
            # Delete model file
            model_path = Path(metadata['model_path'])
            if model_path.exists():
                model_path.unlink()
            
            # Remove from metadata
            del self.metadata[model_name][version]
            self._save_metadata()
        else:
            raise ValueError(f"Model {model_name} version {version} not found")


class ModelComparator:
    """
    Compare multiple models side by side.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.comparison_results = {}
    
    def compare_models(self, models_dict, X_test, y_test, evaluator=None):
        """
        Compare multiple models on test data.
        
        Args:
            models_dict: Dictionary of {name: model}
            X_test: Test features
            y_test: Test labels
            evaluator: ModelEvaluator instance
            
        Returns:
            Comparison DataFrame
        """
        from src.evaluation import ModelEvaluator
        
        if evaluator is None:
            evaluator = ModelEvaluator()
        
        comparison_data = []
        
        for name, model in models_dict.items():
            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = model.predict(X_test)
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Evaluate
            metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            row = {
                'model_name': name,
                **metrics
            }
            comparison_data.append(row)
            
            self.comparison_results[name] = metrics
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, comparison_df, metrics=None, save_path=None):
        """
        Plot model comparison.
        
        Args:
            comparison_df: Comparison DataFrame
            metrics: List of metrics to plot
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            sns.barplot(data=comparison_df, x='model_name', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

