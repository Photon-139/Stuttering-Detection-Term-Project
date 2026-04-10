import joblib
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticModel(BaseModel):
    """
    Standard Logistic Regression for Stuttering Detection.
    Inherits project-wide evaluation metrics from BaseModel.
    """
    def __init__(self, model_name="Logistic_Default", **kwargs):
        super().__init__(model_name)
        # Defaults for noisy audio data
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)  # Allows tuning to override defaults
        self.model = LogisticRegression(**params)

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
