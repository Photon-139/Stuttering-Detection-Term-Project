import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """
    Decision Tree Classifier for Stuttering Detection.
    A single logic-tree that splits the 768D WavLM space using hierarchical rules.
    """
    def __init__(self, model_name="Decision_Tree", max_depth=None, **kwargs):
        super().__init__(model_name)
        # max_depth=None allows the tree to grow until all leaves are pure (Full Complexity)
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42, **kwargs)

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Building Logic Tree (Max Depth: {self.model.max_depth})...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier (Ensemble of Trees).
    Aggregates 'votes' from multiple decision trees to reduce overfitting 
    and provide high-accuracy stuttering detection.
    """
    def __init__(self, model_name="Random_Forest", n_estimators=100, max_depth=None, **kwargs):
        super().__init__(model_name)
        # n_estimators is the number of trees in the forest.
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            n_jobs=-1, # Use all CPU cores for training speed
            **kwargs
        )

    def train(self, X_train, y_train):
        print(f"[{self.model_name}] Planting {self.model.n_estimators} Trees (Max Depth: {self.model.max_depth})...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"[{self.model_name}] Model saved to {file_path}")

    def load(self, file_path):
        self.model = joblib.load(file_path)
        print(f"[{self.model_name}] Model loaded from {file_path}")
