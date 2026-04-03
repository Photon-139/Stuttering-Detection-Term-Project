import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseModel(ABC):
    """
    Abstract Base Class for all classification models.
    Enforces a common interface for training, prediction, and persistence.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None  # To be initialized by children (e.g. self.model = SVC())
        print(f"[Model: {self.model_name}] Initialized.")

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model. 
        X_val/y_val are optional for models that support early stopping.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Returns binary predictions (0 or 1) as a NumPy array."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Automatically calculates metrics required by Step 3.5.
        Fulfills project requirements for Accuracy, Precision, Recall, and F1.
        """
        if self.model is None and not hasattr(self, 'predict'):
            raise ValueError("Model has not been trained or loaded yet.")
            
        y_pred = self.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1])
        }
        
        print(f"\n--- Evaluation: {self.model_name} ---")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k.capitalize()}: {v:.4f}")
        
        cm = metrics["confusion_matrix"]
        print("\nConfusion Matrix (Binary):")
        print(f"               Predicted: Fluent(0)  Predicted: Stutter(1)")
        print(f"True: Fluent(0)      {cm[0,0]:<15} {cm[0,1]:<15}")
        print(f"True: Stutter(1)     {cm[1,0]:<15} {cm[1,1]:<15}")
        
        return metrics

    @abstractmethod
    def save(self, path):
        """Save the underlying model weights/state to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Restore the model weights/state from disk."""
        pass
