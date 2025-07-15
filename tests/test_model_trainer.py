# tests/test_model_trainer.py

import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_trainer import ModelTrainer

def test_model_trainer_train_and_predict():
    X = np.random.rand(100, 5)  # dummy sensor data
    trainer = ModelTrainer()
    trainer.train(X)
    preds = trainer.predict(X)

    assert len(preds) == 100
    assert set(preds).issubset({-1, 1})  # Hanya boleh -1 atau 1

def test_model_save_and_load(tmp_path):
    X = np.random.rand(10, 3)
    model_path = tmp_path / "test_model.pkl"

    trainer = ModelTrainer()
    trainer.train(X)
    trainer.save_model(str(model_path))

    assert model_path.exists()

    new_trainer = ModelTrainer()
    new_trainer.load_model(str(model_path))
    preds = new_trainer.predict(X)
    assert len(preds) == 10
