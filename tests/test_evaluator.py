# tests/test_evaluator.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.evaluator import Evaluator
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

def test_evaluator_without_ground_truth():
    preds = np.random.choice([-1, 1], size=50)
    evaluator = Evaluator()
    evaluator.evaluate(preds)  # hanya untuk cek tidak error

def test_evaluator_with_ground_truth():
    preds = np.array([1, 1, -1, -1, 1])
    y_true = np.array([0, 0, 1, 1, 0])
    evaluator = Evaluator(y_true)
    evaluator.evaluate(preds)
