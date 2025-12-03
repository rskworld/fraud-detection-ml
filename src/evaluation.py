"""
Model Evaluation Module for Fraud Detection System
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection.
    Provides multiple metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positive'] = tp
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix', 
                             save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, title='ROC Curve', 
                      save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                   title='Precision-Recall Curve', 
                                   save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curve(self, y_true, y_pred_proba, 
                              title='Calibration Curve', save_path=None):
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def find_optimal_threshold(self, y_true, y_pred_proba, metric='f1'):
        """
        Find optimal threshold based on specified metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
            
        Returns:
            Optimal threshold value
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precision)
        elif metric == 'recall':
            optimal_idx = np.argmax(recall)
        elif metric == 'youden':
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
        else:
            optimal_idx = np.argmax(f1_scores)
        
        return thresholds[optimal_idx]
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None, 
                model_name='Model', save_plots=False, plot_dir='./plots/'):
        """
        Complete evaluation pipeline.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary of evaluation results
        """
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if y_pred_proba is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"Avg Precision: {metrics['average_precision']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positive']}")
        print(f"  True Negatives:  {metrics['true_negative']}")
        print(f"  False Positives: {metrics['false_positive']}")
        print(f"  False Negatives: {metrics['false_negative']}")
        
        # Generate plots
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            
            self.plot_confusion_matrix(
                y_true, y_pred, 
                title=f'{model_name} - Confusion Matrix',
                save_path=f'{plot_dir}{model_name}_confusion_matrix.png'
            )
            
            if y_pred_proba is not None:
                self.plot_roc_curve(
                    y_true, y_pred_proba,
                    title=f'{model_name} - ROC Curve',
                    save_path=f'{plot_dir}{model_name}_roc_curve.png'
                )
                
                self.plot_precision_recall_curve(
                    y_true, y_pred_proba,
                    title=f'{model_name} - Precision-Recall Curve',
                    save_path=f'{plot_dir}{model_name}_pr_curve.png'
                )
                
                self.plot_calibration_curve(
                    y_true, y_pred_proba,
                    title=f'{model_name} - Calibration Curve',
                    save_path=f'{plot_dir}{model_name}_calibration_curve.png'
                )
        
        return metrics
    
    def compare_models(self, results_dict):
        """
        Compare multiple models.
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(results_dict).T
        return comparison_df

