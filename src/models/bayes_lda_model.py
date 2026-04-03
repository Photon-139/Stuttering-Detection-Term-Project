import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base_model import BaseModel

class LDAModel(BaseModel):
    """
    Linear Discriminant Analysis (Bayesian-Style Classification).
    Calculates the posterior probability using Bayes' Theorem 
    multi-dimensional Gaussian distributions.
    """
    def __init__(self, model_name="LDA_Bayes", solver='svd', **kwargs):
        super().__init__(model_name)
        # SVD solver is the best for high-dimensional feature spaces (768D)
        self.model = LinearDiscriminantAnalysis(solver=solver, **kwargs)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Training Bayesian Probability Map...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")
