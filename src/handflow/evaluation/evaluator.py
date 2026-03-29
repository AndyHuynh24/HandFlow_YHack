"""Model evaluation and misclassification analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from handflow.utils.logging import get_logger

logger = get_logger("handflow.evaluation")


class ModelEvaluator:
    """
    Evaluator for HandFlow models.
    """

    def __init__(self, model_path: str, actions: list[str]):
        """
        Initialize evaluator.

        Args:
            model_path: Path to the .keras/.h5 model file.
            actions: List of class names (labels).
        """
        self.model_path = model_path
        self.actions = actions
        self.model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        """Load the trained model."""
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on data.

        Args:
            X: Input features (N, T, D)

        Returns:
            Tuple of (y_pred_classes, y_pred_probs)
        """
        y_pred_probs = self.model.predict(X, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        return y_pred_classes, y_pred_probs

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Calculate extensive metrics.

        Args:
            X: Input features
            y_true: True labels (one-hot or indices)

        Returns:
            Dictionary containing metrics.
        """
        # Convert one-hot to indices if needed
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_cls = np.argmax(y_true, axis=1)
        else:
            y_true_cls = y_true

        y_pred_cls, y_pred_probs = self.predict(X)

        # Basic Metrics
        accuracy = accuracy_score(y_true_cls, y_pred_cls)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_cls, y_pred_cls, average="weighted"
        )
        
        # Per-class logic
        report = classification_report(
            y_true_cls, 
            y_pred_cls, 
            target_names=self.actions, 
            output_dict=True
        )
        
        cm = confusion_matrix(y_true_cls, y_pred_cls)

        return {
            # Aggregated
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # Detail
            "report": report,
            "confusion_matrix": cm,
            # Raw predictions for further analysis
            "y_true": y_true_cls,
            "y_pred": y_pred_cls,
            "y_prob": y_pred_probs
        }

    def get_misclassified_samples(
        self, 

        X: np.ndarray, 
        y_true: np.ndarray,
        paths: np.ndarray | None = None,
    ) -> list[dict]:

        """
        Identify and return all misclassified samples.

        Args:
            X: Input features
            y_true: True labels

        Returns:
            List of dictionaries containing details for each error.
        """
        if y_true.ndim > 1:
            y_true_cls = np.argmax(y_true, axis=1)
        else:
            y_true_cls = y_true

        y_pred_cls, y_pred_probs = self.predict(X)
        
        misclassified_indices = np.where(y_true_cls != y_pred_cls)[0]
        
        samples = []
        for idx in misclassified_indices:
            sample_info = {
                "index": idx,
                "true_label": self.actions[y_true_cls[idx]],
                "pred_label": self.actions[y_pred_cls[idx]],
                "true_idx": y_true_cls[idx],
                "pred_idx": y_pred_cls[idx],
                "confidence": np.max(y_pred_probs[idx]),
                "data": X[idx]
            }
            if paths is not None:
                sample_info["path"] = paths[idx]
            samples.append(sample_info)
            
        return samples
