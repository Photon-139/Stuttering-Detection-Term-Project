import joblib
from sklearn.svm import SVC
from .base_model import BaseModel

class LinearSVMModel(BaseModel):
    """
    Support Vector Machine with a strictly LINEAR kernel.
    Represents the simplest 'Margin-Based' baseline for the project.
    """
    def __init__(self, model_name="Linear_SVM", **kwargs):
        super().__init__(model_name)
        # Default parameters
        params = {"kernel": "linear", "C": 1.0, "random_state": 42}
        params.update(kwargs) # Allows tuning to override defaults
        self.model = SVC(**params)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Training with Linear Margin (C={self.model.C})...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")


class KernelSVMModel(BaseModel):
    """
    Support Vector Machine with a FLEXIBLE non-linear kernel.
    Allows for head-to-head comparison between 'rbf', 'poly', and 'sigmoid'.
    """
    def __init__(self, model_name="Kernel_SVM", **kwargs):
        super().__init__(model_name)
        # Defaults for the radial basis function (RBF)
        params = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42}
        params.update(kwargs) # Allows tuning to override defaults
        self.model = SVC(**params)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Training with {self.model.kernel.upper()} Kernel (C={self.model.C})...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")
