# Cardiovascular Disease Prediction Project

**Author:** Eva Hallermeier  
**Project Type:** Academic Machine Learning Project  
**Date:** January 2026

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict cardiovascular disease (CVD) using patient health data. The project compares two models:

1. **Baseline Model:** Logistic Regression
2. **Improved Model:** XGBoost Classifier

The project demonstrates:
- Data preprocessing and feature engineering
- Hyperparameter tuning and model optimization
- Model evaluation and comparison
- Publication-quality visualizations
- Reproducible research practices

## 🎯 Project Objectives

- Build a binary classification model to predict cardiovascular disease
- Compare linear (Logistic Regression) vs. ensemble (XGBoost) approaches
- Identify key risk factors for CVD through feature importance analysis
- Create interpretable and actionable results for clinical decision-making

## 📊 Dataset

**Source:** Cardiovascular Disease Dataset from Kaggle  
**Size:** 70,000 patient records  
**Features:** 12 features including age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, and physical activity  
**Target:** Binary classification (CVD present or absent)

### Dataset Features:
- **Demographic:** Age, Gender, Height, Weight
- **Clinical Measurements:** Systolic BP, Diastolic BP
- **Laboratory Values:** Cholesterol Level, Glucose Level
- **Lifestyle Factors:** Smoking, Alcohol Consumption, Physical Activity

## 🏗️ Project Structure

```
cvd_prediction_project/
│
├── data/                          # Data directory
│   ├── cardio_dataset.csv         # Raw dataset
│   ├── X_train.csv               # Processed training features
│   ├── X_test.csv                # Processed test features
│   ├── y_train.csv               # Training labels
│   └── y_test.csv                # Test labels
│
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration and constants
│   ├── preprocessing.py          # Data preprocessing module
│   ├── baseline_model.py         # Logistic Regression model
│   ├── xgboost_model.py          # XGBoost model
│   └── visualization.py          # Visualization module
│
├── results/                       # Results directory
│   ├── models/                   # Saved models
│   │   ├── logistic_regression_model.pkl
│   │   ├── xgboost_model.pkl
│   │   └── scaler.pkl
│   │
│   ├── metrics/                  # Evaluation metrics
│   │   ├── logistic_regression_results.json
│   │   ├── xgboost_results.json
│   │   └── feature_info.json
│   │
│   └── figures/                  # Visualizations
│       ├── roc_curve_comparison.png
│       ├── pr_curve_comparison.png
│       ├── confusion_matrices.png
│       ├── feature_importance_comparison.png
│       ├── metrics_comparison.png
│       └── metrics_comparison_table.csv
│
├── notebooks/                     # Jupyter notebooks (optional)
│
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Extract the project files:**
   ```bash
   unzip cvd_prediction_project.zip
   cd cvd_prediction_project
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies are:
   - pandas >= 1.3.0
   - numpy >= 1.21.0
   - scikit-learn >= 1.0.0
   - xgboost >= 1.5.0
   - matplotlib >= 3.4.0
   - seaborn >= 0.11.0
   - scipy >= 1.7.0

### Running the Project

#### Complete Pipeline (Recommended)
Run the entire pipeline from data preprocessing to model evaluation:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Train the Logistic Regression baseline model
3. Train the XGBoost improved model
4. Generate all visualizations
5. Save all results and models

**Expected Runtime:** around 2 minutes (depends on your hardware)

#### Alternative Options

**Skip preprocessing (use existing processed data):**
```bash
python main.py --skip-preprocess
```

**Only generate visualizations (requires existing results):**
```bash
python main.py --visualize-only
```

**Quick mode (skip hyperparameter tuning, faster but less accurate):**
```bash
python main.py --quick
```

### Viewing Results

After running the pipeline, you can find:

1. **Visualizations:** Check the `results/figures/` directory for all plots
2. **Model Performance:** See `results/metrics/` for detailed JSON results
3. **Trained Models:** Saved in `results/models/` for future use
4. **Comparison Table:** `results/figures/metrics_comparison_table.csv`

## 📈 Pipeline Details

### 1. Data Preprocessing (`src/preprocessing.py`)

The preprocessing module performs:

- **Data Loading:** Reads the CSV file with proper delimiter handling
- **Data Cleaning:**
  - Removes physiologically impossible blood pressure values
  - Filters unrealistic height and weight measurements
  - Removes ~1,322 records (1.9% of data)
- **Feature Engineering:**
  - Converts age from days to years
  - Calculates BMI (Body Mass Index)
  - Creates age groups and BMI categories
- **Feature Scaling:** StandardScaler for normalization
- **Train/Test Split:** 75% training, 25% testing (stratified)

**Output:** Clean, scaled, and split data ready for modeling

### 2. Baseline Model - Logistic Regression (`src/baseline_model.py`)

**Model:** Logistic Regression with L2 regularization

**Hyperparameter Tuning:**
- Method: GridSearchCV with 5-fold cross-validation
- Parameters tuned: C (regularization), class_weight
- Scoring metric: AUC-ROC

**Key Features:**
- Interpretable coefficients (linear relationships)
- Fast training and prediction
- Provides baseline performance benchmark

**Expected Performance:**
- AUC-ROC: ~0.79
- Accuracy: ~73%
- Recall: ~67%

### 3. Improved Model - XGBoost (`src/xgboost_model.py`)

**Model:** XGBoost Gradient Boosting Classifier

**Hyperparameter Tuning:**
- Method: RandomizedSearchCV (50 iterations, 5-fold CV)
- Parameters tuned: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
- Scoring metric: AUC-ROC

**Key Features:**
- Captures non-linear relationships
- Handles feature interactions
- Built-in regularization
- Feature importance via gain, weight, and cover

**Expected Performance:**
- AUC-ROC: ~0.80
- Accuracy: ~73%
- Recall: ~69%

### 4. Visualization (`src/visualization.py`)

Generates publication-quality figures:

1. **ROC Curve Comparison:** Shows model discrimination ability
2. **Precision-Recall Curve:** Useful for imbalanced datasets
3. **Confusion Matrices:** Side-by-side comparison
4. **Feature Importance:** Top predictive features for each model
5. **Metrics Bar Chart:** Visual comparison of all metrics
6. **Comparison Table:** CSV export for easy reference

**All figures are:**
- 300 DPI resolution (publication-ready)
- Color-blind friendly palette
- Properly labeled and titled
- Saved as PNG files

## 📊 Key Results

### Model Performance Summary

| Metric            | Logistic Regression | XGBoost | Improvement |
|-------------------|--------------------:|--------:|------------:|
| **AUC-ROC**       | 0.7922             | 0.8019  | +0.0097     |
| **Accuracy**      | 0.7288             | 0.7305  | +0.0017     |
| **Precision**     | 0.7564             | 0.7532  | -0.0032     |
| **Recall**        | 0.6686             | 0.6873  | +0.0187     |
| **F1-Score**      | 0.7083             | 0.7175  | +0.0092     |

### Top Risk Factors

**From Logistic Regression:**
1. Age (strongest positive predictor)
2. Weight
3. Systolic Blood Pressure
4. Cholesterol Level

**From XGBoost:**
1. Age
2. Weight
3. Systolic Blood Pressure
4. BMI

### Clinical Insights

- **Age** is the strongest predictor across both models
- **Elevated cholesterol** significantly increases CVD risk
- **Blood pressure** (both systolic and diastolic) is highly predictive
- **BMI/Weight** shows strong association with CVD
- **Lifestyle factors** (smoking, alcohol) have weaker but measurable effects

## 🔬 Methodology

### Model Selection Rationale

**Logistic Regression (Baseline):**
- Simple, interpretable, well-understood
- Provides linear baseline for comparison
- Fast to train and deploy
- Good for understanding feature relationships

**XGBoost (Improved):**
- Handles non-linear relationships
- Robust to outliers
- Captures feature interactions
- State-of-the-art performance on tabular data
- Regularization prevents overfitting

### Evaluation Strategy

**Metrics Used:**
- **AUC-ROC:** Primary metric (threshold-independent)
- **Recall:** Important for medical applications (minimize false negatives)
- **Precision:** Balance against false positives
- **F1-Score:** Harmonic mean of precision and recall
- **Accuracy:** Overall correctness

**Cross-Validation:**
- 5-fold stratified cross-validation
- Maintains class balance in each fold
- Reduces overfitting risk

## 📝 Reproducibility

This project follows best practices for reproducible research:

1. **Fixed Random Seeds:** All random operations use `RANDOM_STATE = 42`
2. **Version Control:** Requirements.txt specifies exact package versions
3. **Modular Code:** Separate modules for each component
4. **Configuration File:** All parameters centralized in `config.py`
5. **Documentation:** Comprehensive docstrings and comments
6. **Saved Artifacts:** Models, scalers, and results are saved

To reproduce the results:
```bash
python main.py
```

Results should be identical (within floating-point precision) on any machine.

## 🎓 Academic Use

This project is suitable for:
- Machine Learning course projects
- Data Science portfolio
- Research on CVD prediction
- Teaching binary classification concepts

**Citation:**
If you use this code, please cite:
```
Hallermeier, E. (2026). Cardiovascular Disease Prediction using 
Logistic Regression and XGBoost. Machine Learning Project.
```

## 🔧 Customization

### Modifying Hyperparameters

Edit `src/config.py` to change:
- Test/train split ratio
- Cross-validation folds
- Hyperparameter search spaces
- Random seed
- File paths

### Adding New Models

1. Create a new module in `src/` (e.g., `random_forest_model.py`)
2. Follow the same structure as `baseline_model.py`
3. Add to `main.py` pipeline
4. Update visualization to include new model

### Custom Visualizations

Modify `src/visualization.py` to add new plots:
- Calibration curves
- Learning curves
- Partial dependence plots
- SHAP values

## 🐛 Troubleshooting

**Issue:** `FileNotFoundError: cardio_dataset.csv not found`
- **Solution:** Ensure `cardio_dataset.csv` is in the `data/` directory

**Issue:** Memory error during training
- **Solution:** Reduce `N_ITER_RANDOM_SEARCH` in `config.py` or use `--quick` mode

**Issue:** ImportError for packages
- **Solution:** Run `pip install -r requirements.txt`

**Issue:** Slow execution
- **Solution:** Use `--skip-preprocess` if you've already run preprocessing
- **Solution:** Use `--quick` to skip hyperparameter tuning

## 📧 Contact

**Author:** Eva Hallermeier  
**Email:** eva.hallermeier@gmail.com

## 📄 License

This project is for academic purposes. Feel free to use and modify for educational use.

## 🙏 Acknowledgments

- Dataset: Cardiovascular Disease Dataset from Kaggle
- Libraries: scikit-learn, XGBoost, pandas, matplotlib, seaborn
- Inspiration: Real-world clinical CVD risk prediction models

---

**Last Updated:** January 2026

For questions or issues, please contact the author or refer to the documentation in the code.
