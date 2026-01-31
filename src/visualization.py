"""
Visualization Module (Aligned with Notebook)
Author: Eva Hallermeier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import json
from . import config


class ModelVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        self.setup_plotting_style()

    def setup_plotting_style(self):
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10

    def plot_roc_comparison(self, lr_results, xgb_results, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(lr_results['roc_curve']['fpr'], lr_results['roc_curve']['tpr'],
                color=config.COLORS['lr'], lw=2, label=f"LR (AUC={lr_results['roc_auc']:.4f})")
        ax.plot(xgb_results['roc_curve']['fpr'], xgb_results['roc_curve']['tpr'],
                color=config.COLORS['xgb'], lw=2, label=f"XGB (AUC={xgb_results['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], color=config.COLORS['baseline'], linestyle='--')
        ax.set_title('ROC Curve Comparison', fontweight='bold')
        ax.legend(loc='lower right')
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_calibration_curves(self, lr_preds, xgb_preds, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, df, col in [('LR', lr_preds, config.COLORS['lr']), ('XGB', xgb_preds, config.COLORS['xgb'])]:
            prob_true, prob_pred = calibration_curve(df['y_true'], df['y_pred_proba'], n_bins=10)
            ax.plot(prob_pred, prob_true, 's-', color=col, label=name)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title('Calibration Curves', fontweight='bold')
        ax.legend()
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_feature_importance_comparison(self, lr_feat, xgb_feat, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        sns.barplot(x='coefficient', y='feature', data=lr_feat.head(12), ax=ax1, color=config.COLORS['lr'])
        sns.barplot(x='importance', y='feature', data=xgb_feat.head(12), ax=ax2, color=config.COLORS['xgb'])
        ax1.set_title('LR Top Features')
        ax2.set_title('XGB Top Features')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        return fig

    def generate_all_visualizations(self, lr_results, xgb_results, lr_preds=None, xgb_preds=None, lr_feat=None,
                                    xgb_feat=None, output_dir=None):
        """Generates all plots. Uses optional arguments if available for advanced plots."""
        if output_dir is None: output_dir = config.FIGURES_DIR

        # Standard plots (from JSON)
        self.plot_roc_comparison(lr_results, xgb_results, output_dir / 'roc_curve_comparison.png')

        # Advanced plots (from CSVs - only run if data is provided)
        if lr_preds is not None and xgb_preds is not None:
            self.plot_calibration_curves(lr_preds, xgb_preds, output_dir / 'calibration_curves.png')

        if lr_feat is not None and xgb_feat is not None:
            self.plot_feature_importance_comparison(lr_feat, xgb_feat, output_dir / 'feature_importance.png')

        print(f"✓ Visualizations saved to: {output_dir}")


def load_and_visualize():
    """Loads all data sources and runs the visualizer."""
    with open('models/logistic_regression_results.json', 'r') as f:
        lr_res = json.load(f)
    with open('models/xgboost_results.json', 'r') as f:
        xgb_res = json.load(f)

    # Load the CSVs used in the notebook
    lr_preds = pd.read_csv('models/logistic_regression_predictions.csv')
    xgb_preds = pd.read_csv('models/xgboost_predictions.csv')
    lr_feat = pd.read_csv('models/logistic_regression_feature_importance.csv')
    xgb_feat = pd.read_csv('models/xgboost_feature_importance.csv')

    vis = ModelVisualizer()
    vis.generate_all_visualizations(lr_res, xgb_res, lr_preds, xgb_preds, lr_feat, xgb_feat)