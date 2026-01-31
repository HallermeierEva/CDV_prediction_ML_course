"""
Configuration file for CVD Prediction Project
Author: Eva Hallermeier

This file contains all configuration parameters, paths, and constants
used throughout the project.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = RESULTS_DIR / 'models'
METRICS_DIR = RESULTS_DIR / 'metrics'

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
RAW_DATA_FILE = DATA_DIR / 'cardio_dataset.csv'
PROCESSED_X_TRAIN = DATA_DIR / 'X_train.csv'
PROCESSED_X_TEST = DATA_DIR / 'X_test.csv'
PROCESSED_Y_TRAIN = DATA_DIR / 'y_train.csv'
PROCESSED_Y_TEST = DATA_DIR / 'y_test.csv'

# ============================================================================
# MODEL FILES
# ============================================================================
LR_MODEL_FILE = MODELS_DIR / 'logistic_regression_model.pkl'
XGB_MODEL_FILE = MODELS_DIR / 'xgboost_model.pkl'
SCALER_FILE = MODELS_DIR / 'scaler.pkl'

# ============================================================================
# RESULTS FILES
# ============================================================================
LR_RESULTS_FILE = METRICS_DIR / 'logistic_regression_results.json'
XGB_RESULTS_FILE = METRICS_DIR / 'xgboost_results.json'
FEATURE_INFO_FILE = METRICS_DIR / 'feature_info.json'
COMPARISON_FILE = METRICS_DIR / 'model_comparison.json'

# ============================================================================
# RANDOM STATE (for reproducibility)
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# DATA PREPROCESSING PARAMETERS
# ============================================================================
TEST_SIZE = 0.25  # 75% train, 25% test split

# Blood pressure thresholds for data cleaning
BP_THRESHOLDS = {
    'ap_hi_min': 90,
    'ap_hi_max': 200,
    'ap_lo_min': 60,
    'ap_lo_max': 140
}

# Height and weight thresholds (outlier removal)
HEIGHT_THRESHOLDS = {'min': 140, 'max': 210}  # cm
WEIGHT_THRESHOLDS = {'min': 40, 'max': 200}   # kg

# BMI categories
BMI_CATEGORIES = {
    'Underweight': (0, 18.5),
    'Normal': (18.5, 25),
    'Overweight': (25, 30),
    'Obese': (30, 100)
}

# Age groups (in years)
AGE_GROUPS = {
    '30-39': (30, 40),
    '40-49': (40, 50),
    '50-59': (50, 60),
    '60-69': (60, 70)
}

# ============================================================================
# FEATURE INFORMATION
# ============================================================================
FEATURE_NAMES = {
    'age': 'Age (days)',
    'gender': 'Gender',
    'height': 'Height (cm)',
    'weight': 'Weight (kg)',
    'ap_hi': 'Systolic BP',
    'ap_lo': 'Diastolic BP',
    'cholesterol': 'Cholesterol Level',
    'gluc': 'Glucose Level',
    'smoke': 'Smoking',
    'alco': 'Alcohol Consumption',
    'active': 'Physical Activity'
}

# Categorical features
CATEGORICAL_FEATURES = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Numerical features
NUMERICAL_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

# Target variable
TARGET_VARIABLE = 'cardio'

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Logistic Regression Grid Search Parameters
LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000],
    'class_weight': [None, 'balanced']
}

# XGBoost Randomized Search Parameters
XGB_PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# Cross-validation parameters
CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 50  # Number of iterations for RandomizedSearchCV

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Color palette (color-blind friendly)
COLORS = {
    'lr': '#0173B2',        # Blue - Logistic Regression
    'xgb': '#DE8F05',       # Orange - XGBoost
    'baseline': '#CC78BC',  # Purple - Baseline/Random
    'positive': '#029E73',  # Green - Positive class
    'negative': '#D55E00'   # Red - Negative class
}

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE_STANDARD = (10, 6)
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_SQUARE = (8, 8)

# Font sizes
FONT_SIZES = {
    'title': 14,
    'label': 12,
    'tick': 10,
    'legend': 10
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to calculate for both models
METRICS_TO_CALCULATE = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision'
]

# Threshold optimization settings
THRESHOLD_RANGE = (0.1, 0.9)
THRESHOLD_STEP = 0.01

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

# Verbosity level
VERBOSE = 1

# Decimal places for metrics reporting
METRIC_DECIMALS = 4

# Print separators
SEPARATOR = "=" * 80
SHORT_SEPARATOR = "-" * 80

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_display_name(feature_name):
    """Get human-readable display name for a feature."""
    return FEATURE_NAMES.get(feature_name, feature_name)

def print_section_header(title):
    """Print formatted section header."""
    print(f"\n{SEPARATOR}")
    print(f"{title}")
    print(SEPARATOR)

def print_subsection_header(title):
    """Print formatted subsection header."""
    print(f"\n{SHORT_SEPARATOR}")
    print(f"{title}")
    print(SHORT_SEPARATOR)

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that all required directories exist."""
    required_dirs = [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
    
    return True

def check_data_file():
    """Check if the raw data file exists."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {RAW_DATA_FILE}\n"
            f"Please ensure 'cardio_dataset.csv' is in the {DATA_DIR} directory."
        )
    return True

# Run validation when module is imported
if __name__ != '__main__':
    validate_paths()
