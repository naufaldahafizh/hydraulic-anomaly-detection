# src/evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class Evaluator:
    def __init__(self, y_true=None):
        """
        y_true: ground truth anomaly labels (optional), binary (1=anomaly, 0=normal)
        """
        self.y_true = y_true

    def evaluate(self, y_pred):
        """
        y_pred: output dari model (1=normal, -1=anomaly)
        """
        if self.y_true is not None:
            y_pred_bin = np.where(y_pred == -1, 1, 0)  # ubah ke 1=anomaly
            print("Classification Report:\n", classification_report(self.y_true, y_pred_bin))
            cm = confusion_matrix(self.y_true, y_pred_bin)
            self._plot_confusion_matrix(cm)
        else:
            print("No ground truth provided. Only visualizing prediction distribution.")
            self._plot_prediction_distribution(y_pred)

    def _plot_confusion_matrix(self, cm):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def _plot_prediction_distribution(self, y_pred):
        sns.countplot(x=y_pred)
        plt.title("Distribusi Prediksi (1=Normal, -1=Anomali)")
        plt.xlabel("Prediksi")
        plt.ylabel("Jumlah")
        plt.show()
