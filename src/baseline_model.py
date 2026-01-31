"""
Baseline Model Module - Logistic Regression
Author: Eva Hallermeier

This module handles training, evaluation, and analysis of the
baseline Logistic Regression model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
import pickle
import json
import time

from . import config


class BaselineModel:
    """
    Logistic Regression baseline model for CVD prediction.
    """
    
    def __init__(self, random_state=config.RANDOM_STATE):
        """Initialize the baseline model."""
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.results = {}
        
    def train(self, X_train, y_train, use_grid_search=True):
        """
        Train the Logistic Regression model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training labels
        use_grid_search : bool
            Whether to use GridSearchCV for hyperparameter tuning
        """
        config.print_subsection_header("Training Logistic Regression Model")
        
        start_time = time.time()
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            print(f"  - CV Folds: {config.CV_FOLDS}")
            print(f"  - Parameter grid: {len(config.LR_PARAM_GRID)} parameters")
            
            # Create base model
            base_model = LogisticRegression(random_state=self.random_state)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                config.LR_PARAM_GRID,
                cv=config.CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"\n✓ Best parameters found:")
            for param, value in self.best_params.items():
                print(f"    {param}: {value}")
            print(f"  Best CV AUC-ROC: {grid_search.best_score_:.4f}")
            
        else:
            print("Training with default parameters...")
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
            self.model.fit(X_train, y_train)
        
        # Extract feature importance (coefficients)
        if hasattr(X_train, 'columns'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': self.model.coef_[0],
                'abs_coefficient': np.abs(self.model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
        
        training_time = time.time() - start_time
        print(f"\n✓ Model trained in {training_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test labels
        threshold : float
            Classification threshold
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        config.print_subsection_header("Evaluating Model Performance")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        results = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred_proba.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # Store ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        results['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Store PR curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        results['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        self.results = results
        
        # Print results
        print(f"\nPerformance Metrics (threshold={threshold}):")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  AUC-ROC:   {results['roc_auc']:.4f}")
        print(f"  Avg Precision: {results['average_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
        
        return results
    
    def find_optimal_threshold(self, X_test, y_test, metric='f1'):
        """
        Find optimal classification threshold.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test labels
        metric : str
            Metric to optimize ('f1', 'recall', 'precision')
            
        Returns:
        --------
        tuple
            (optimal_threshold, optimal_score, all_scores)
        """
        print(f"\nFinding optimal threshold (optimizing {metric})...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_test, y_pred)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        print(f"✓ Optimal threshold: {optimal_threshold:.2f}")
        print(f"  {metric.capitalize()} score: {optimal_score:.4f}")
        
        return optimal_threshold, optimal_score, (thresholds, scores)
    
    def get_feature_importance(self, top_n=10):
        """
        Get top N most important features.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            Top N features with their importance
        """
        if self.feature_importance is None:
            print("Feature importance not available. Train the model first.")
            return None
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath=None):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str or Path, optional
            Path to save the model. If None, uses config.LR_MODEL_FILE
        """
        if filepath is None:
            filepath = config.LR_MODEL_FILE
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    def save_results(self, filepath=None):
        """
        Save evaluation results to JSON file.
        
        Parameters:
        -----------
        filepath : str or Path, optional
            Path to save results. If None, uses config.LR_RESULTS_FILE
        """
        if filepath is None:
            filepath = config.LR_RESULTS_FILE
        
        # Prepare results for JSON serialization
        results_to_save = self.results.copy()
        
        # Add feature importance
        if self.feature_importance is not None:
            results_to_save['feature_importance'] = {
                'features': self.feature_importance['feature'].tolist(),
                'coefficients': self.feature_importance['coefficient'].tolist()
            }
        
        # Add best parameters
        if self.best_params is not None:
            results_to_save['best_params'] = self.best_params
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✓ Results saved to: {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a previously trained model.
        
        Parameters:
        -----------
        filepath : str or Path, optional
            Path to the model file. If None, uses config.LR_MODEL_FILE
        """
        if filepath is None:
            filepath = config.LR_MODEL_FILE
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Model loaded from: {filepath}")
    
    def print_classification_report(self, X_test, y_test, threshold=0.5):
        """
        Print detailed classification report.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test labels
        threshold : float
            Classification threshold
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No CVD', 'CVD'],
                                   digits=4))


def train_and_evaluate_baseline(X_train, X_test, y_train, y_test):
    """
    Convenience function to train and evaluate the baseline model.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series or np.array
        Training and test labels
        
    Returns:
    --------
    BaselineModel
        Trained and evaluated model
    """
    config.print_section_header("BASELINE MODEL: LOGISTIC REGRESSION")
    
    # Initialize model
    model = BaselineModel()
    
    # Train
    model.train(X_train, y_train, use_grid_search=True)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1, _ = model.find_optimal_threshold(X_test, y_test, metric='f1')
    
    # Evaluate at optimal threshold
    print(f"\n{'='*80}")
    print("Re-evaluating at optimal threshold:")
    results_optimal = model.evaluate(X_test, y_test, threshold=optimal_threshold)
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    print(model.get_feature_importance(top_n=10).to_string(index=False))
    
    # Save model and results
    model.save_model()
    model.save_results()
    
    config.print_section_header("BASELINE MODEL COMPLETE")
    
    return model
