import joblib
from sklearn.linear_model import Perceptron
from .base_model import BaseModel

class PerceptronModel(BaseModel):
    """
    Classic Perceptron Algorithm for Stuttering Detection.
    A linear classifier that updates its weights iteratively.
    """
    def __init__(self, model_name="Perceptron_Default", **kwargs):
        super().__init__(model_name)
        # Default to 1000 iterations for convergence on noisy features
        self.model = Perceptron(max_iter=1000, random_state=42, **kwargs)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")
