import joblib
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel

class KNNModel(BaseModel):
    """
    K-Nearest Neighbors Classifier for Stuttering Detection.
    Assigns classes based on the 'votes' of the nearest neighbors in 
    the 768-dimensional WavLM Feature Space.
    """
    def __init__(self, model_name="KNN_Default", n_neighbors=5, **kwargs):
        super().__init__(model_name)
        # Defaults to 5 neighbors; weights can be 'uniform' or 'distance'
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Stores {len(X_train)} training vectors...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")
