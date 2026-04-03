import joblib
from sklearn.naive_bayes import GaussianNB
from .base_model import BaseModel

class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes for Stuttering Detection.
    Assumes features (WavLM Embeddings) follow a normal distribution 
    and are largely independent given the class.
    """
    def __init__(self, model_name="NaiveBayes_Default", **kwargs):
        super().__init__(model_name)
        # GaussianNB is the standard for continuous features like audio embeddings
        self.model = GaussianNB(**kwargs)

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
