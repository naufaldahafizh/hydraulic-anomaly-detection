# src/model_trainer.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

class ModelTrainer:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state
        )

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        # IsolationForest: -1 = anomaly, 1 = normal
        return self.model.predict(X)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
