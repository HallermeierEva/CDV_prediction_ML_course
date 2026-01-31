"""
Main Execution Script for CVD Prediction Project
Author: Eva Hallermeier

This script runs the complete machine learning pipeline:
1. Data preprocessing
2. Baseline model (Logistic Regression) training and evaluation
3. Improved model (XGBoost) training and evaluation
4. Model comparison and visualization

Usage:
    python main.py
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import config
from src.preprocessing import DataPreprocessor, load_processed_data
from src.baseline_model import train_and_evaluate_baseline
from src.xgboost_model import train_and_evaluate_xgboost
from src.visualization import ModelVisualizer
import json


import warnings
import os

# Suppress specific future warnings and user warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional: Suppress XGBoost logs that aren't warnings
os.environ['XGB_LOGGING_LEVEL'] = '0'


def run_preprocessing():
    """Run the data preprocessing pipeline."""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
    
    return X_train, X_test, y_train, y_test


def run_baseline_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the baseline Logistic Regression model."""
    print("\n" + "="*80)
    print("STEP 2: BASELINE MODEL (LOGISTIC REGRESSION)")
    print("="*80)
    
    lr_model = train_and_evaluate_baseline(X_train, X_test, y_train, y_test)
    
    return lr_model


def run_xgboost_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the improved XGBoost model."""
    print("\n" + "="*80)
    print("STEP 3: IMPROVED MODEL (XGBOOST)")
    print("="*80)
    
    xgb_model = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
    
    return xgb_model


def run_visualization():
    """Generate all visualizations and comparisons."""
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION AND COMPARISON")
    print("="*80)
    
    # Load results
    with open(config.LR_RESULTS_FILE, 'r') as f:
        lr_results = json.load(f)
    
    with open(config.XGB_RESULTS_FILE, 'r') as f:
        xgb_results = json.load(f)
    
    # Create visualizations
    visualizer = ModelVisualizer()
    visualizer.generate_all_visualizations(lr_results, xgb_results)


def print_final_summary():
    """Print final summary of results."""
    print("\n" + "="*80)
    print("PROJECT EXECUTION COMPLETE!")
    print("="*80)
    
    # Load and display comparison
    with open(config.LR_RESULTS_FILE, 'r') as f:
        lr_results = json.load(f)
    
    with open(config.XGB_RESULTS_FILE, 'r') as f:
        xgb_results = json.load(f)
    
    print("\nFINAL MODEL COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Logistic Regression':<20} {'XGBoost':<20} {'Improvement':<15}")
    print("-" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    for metric, name in zip(metrics, metric_names):
        lr_val = lr_results[metric]
        xgb_val = xgb_results[metric]
        improvement = xgb_val - lr_val
        
        print(f"{name:<20} {lr_val:<20.4f} {xgb_val:<20.4f} {improvement:+.4f}")
    
    print("-" * 80)
    
    # Determine winner
    if xgb_results['roc_auc'] > lr_results['roc_auc']:
        winner = "XGBoost"
        improvement = (xgb_results['roc_auc'] - lr_results['roc_auc']) / lr_results['roc_auc'] * 100
    else:
        winner = "Logistic Regression"
        improvement = (lr_results['roc_auc'] - xgb_results['roc_auc']) / xgb_results['roc_auc'] * 100
    
    print(f"\n🏆 BEST MODEL: {winner}")
    print(f"   Improvement in AUC-ROC: {improvement:.2f}%")
    
    print("\n" + "="*80)
    print("OUTPUT FILES GENERATED:")
    print("="*80)
    print(f"  Data:")
    print(f"    - {config.PROCESSED_X_TRAIN}")
    print(f"    - {config.PROCESSED_X_TEST}")
    print(f"    - {config.PROCESSED_Y_TRAIN}")
    print(f"    - {config.PROCESSED_Y_TEST}")
    print(f"\n  Models:")
    print(f"    - {config.LR_MODEL_FILE}")
    print(f"    - {config.XGB_MODEL_FILE}")
    print(f"    - {config.SCALER_FILE}")
    print(f"\n  Results:")
    print(f"    - {config.LR_RESULTS_FILE}")
    print(f"    - {config.XGB_RESULTS_FILE}")
    print(f"    - {config.FEATURE_INFO_FILE}")
    print(f"\n  Visualizations (in {config.FIGURES_DIR}):")
    print(f"    - roc_curve_comparison.png")
    print(f"    - pr_curve_comparison.png")
    print(f"    - confusion_matrices.png")
    print(f"    - feature_importance_comparison.png")
    print(f"    - metrics_comparison.png")
    print(f"    - metrics_comparison_table.csv")
    
    print("\n" + "="*80)
    print("Thank you for using CVD Prediction Project!")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='CVD Prediction Project - Complete ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --skip-preprocess  # Skip preprocessing (use existing data)
  python main.py --visualize-only   # Only generate visualizations
        """
    )
    
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only generate visualizations from existing results'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip hyperparameter tuning (faster but less accurate)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("CVD PREDICTION PROJECT - MACHINE LEARNING PIPELINE")
    print("Author: Eva Hallermeier")
    print("="*80)
    
    try:
        if args.visualize_only:
            # Only run visualization
            run_visualization()
        else:
            # Step 1: Preprocessing
            if args.skip_preprocess:
                print("\nSkipping preprocessing, loading existing data...")
                X_train, X_test, y_train, y_test = load_processed_data()
            else:
                X_train, X_test, y_train, y_test = run_preprocessing()
            
            # Step 2: Baseline Model
            lr_model = run_baseline_model(X_train, X_test, y_train, y_test)
            
            # Step 3: XGBoost Model
            xgb_model = run_xgboost_model(X_train, X_test, y_train, y_test)
            
            # Step 4: Visualization
            run_visualization()
        
        # Print final summary
        print_final_summary()
        
        return 0
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
