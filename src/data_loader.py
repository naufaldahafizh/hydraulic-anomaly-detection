# src/data_loader.py

import pandas as pd

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self):
        """Loads the dataset as a pandas DataFrame."""
        df = pd.read_csv(self.path)
        return df
