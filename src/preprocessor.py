# src/preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_column_names(self, df):
        """Removes spaces and standardizes column names."""
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    def separate_features_targets(self, df):
        """Separates sensor features and component conditions (multi-label)."""
        condition_cols = ['cooler', 'valve', 'leakage', 'acc', 'stable']
        feature_cols = [col for col in df.columns if col not in condition_cols]
        return df[feature_cols], df[condition_cols]

    def scale_features(self, X):
        """Applies standard scaling to numeric features."""
        return self.scaler.fit_transform(X)
