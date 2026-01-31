import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path

from . import config


class DataPreprocessor:
    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = []
        self.original_shape = None
        self.cleaned_shape = None

    def load_data(self, filepath=None):
        if filepath is None:
            filepath = config.RAW_DATA_FILE
        config.check_data_file()
        df = pd.read_csv(filepath, delimiter=';')
        self.original_shape = df.shape
        return df

    def clean_data(self, df):
        # Matching notebook filters: BP hi<=250, lo<=200, hi>lo; Height 100-250; Weight 30-300
        bp_mask = (df['ap_hi'] > 0) & (df['ap_lo'] > 0) & \
                  (df['ap_hi'] <= 250) & (df['ap_lo'] <= 200) & \
                  (df['ap_hi'] > df['ap_lo'])

        height_mask = (df['height'] >= 100) & (df['height'] <= 250)
        weight_mask = (df['weight'] >= 30) & (df['weight'] <= 300)

        df_clean = df[bp_mask & height_mask & weight_mask].copy()
        self.cleaned_shape = df_clean.shape
        return df_clean

    def engineer_features(self, df):
        df = df.copy()
        # Convert age to years and calculate BMI as per notebook
        df['age_years'] = df['age'] / 365.25
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        return df

    def prepare_features(self, df):
        """Includes One-Hot Encoding for cholesterol and gluc."""
        # Separate features and target, dropping unused original columns
        X = df.drop(columns=['id', 'age', config.TARGET_VARIABLE], errors='ignore')
        y = df[config.TARGET_VARIABLE]

        # Gender normalization (Notebook uses 1/2, convert to binary 0/1)
        X['gender'] = (X['gender'] == 2).astype(int)

        # One-Hot Encoding for categorical levels 1, 2, 3
        X = pd.get_dummies(X, columns=['cholesterol', 'gluc'], prefix=['chol', 'gluc'])

        self.feature_names = X.columns.tolist()
        return X, y

    def split_data(self, X, y, test_size=config.TEST_SIZE):
        return train_test_split(
            X, y, test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

    def scale_features(self, X_train, X_test):
        """Only scales continuous features as specified in the notebook."""
        continuous_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Fit scaler on training data only
        X_train_scaled[continuous_features] = self.scaler.fit_transform(X_train[continuous_features])
        X_test_scaled[continuous_features] = self.scaler.transform(X_test[continuous_features])

        return X_train_scaled, X_test_scaled

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        X_train.to_csv(config.PROCESSED_X_TRAIN, index=False)
        X_test.to_csv(config.PROCESSED_X_TEST, index=False)
        pd.DataFrame(y_train, columns=['cardio']).to_csv(config.PROCESSED_Y_TRAIN, index=False)
        pd.DataFrame(y_test, columns=['cardio']).to_csv(config.PROCESSED_Y_TEST, index=False)

        with open(config.SCALER_FILE, 'wb') as f:
            pickle.dump(self.scaler, f)

        feature_info = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'train_shape': X_train.shape
        }
        with open(config.FEATURE_INFO_FILE, 'w') as f:
            json.dump(feature_info, f, indent=2)

    def run_full_pipeline(self):
        df = self.load_data()
        df_clean = self.clean_data(df)
        df_eng = self.engineer_features(df_clean)
        X, y = self.prepare_features(df_eng)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        self.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test)
        return X_train_scaled, X_test_scaled, y_train, y_test


def load_processed_data():
    """
    Helper function required by main.py and __init__.py
    to reload the CSVs saved during preprocessing.
    """
    X_train = pd.read_csv(config.PROCESSED_X_TRAIN)
    X_test = pd.read_csv(config.PROCESSED_X_TEST)
    y_train = pd.read_csv(config.PROCESSED_Y_TRAIN).values.ravel()
    y_test = pd.read_csv(config.PROCESSED_Y_TEST).values.ravel()
    return X_train, X_test, y_train, y_test