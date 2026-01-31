"""
CVD Prediction Project
Author: Eva Hallermeier

A machine learning project for predicting cardiovascular disease
using Logistic Regression (baseline) and XGBoost (improved model).
"""

__version__ = '1.0.0'
__author__ = 'Eva Hallermeier'

from . import config
from .preprocessing import DataPreprocessor, load_processed_data
from .baseline_model import BaselineModel, train_and_evaluate_baseline
from .xgboost_model import ImprovedModel, train_and_evaluate_xgboost
from .visualization import ModelVisualizer, load_and_visualize

__all__ = [
    'config',
    'DataPreprocessor',
    'load_processed_data',
    'BaselineModel',
    'train_and_evaluate_baseline',
    'ImprovedModel',
    'train_and_evaluate_xgboost',
    'ModelVisualizer',
    'load_and_visualize'
]
