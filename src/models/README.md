# Classification Base Model

### Overview
The `BaseModel` is an Abstract Base Class (ABC) that establishes a uniform interface for all machine learning models within the project. It ensures that diverse model architectures (e.g., SVM, Random Forest, Multi-Layer Perceptrons) adhere to a consistent training and evaluation lifecycle.

### Why it exists
By enforcing a common API, the `BaseModel` allows the high-level application logic to interact with any classification algorithm without requiring model-specific modifications. This architecture facilitates rapid experimentation, simplifies the comparison of different algorithms, and enables automated evaluation against standard performance metrics.

---

### Class Members

| Member | Type | Description |
| :--- | :--- | :--- |
| `model_name` | `str` | A unique identifier for the model instance. |
| `model` | `Any` | The underlying estimator object (e.g., a scikit-learn SVC or a PyTorch module). |

---

### Class Methods

| Method | Purpose | Requirement | Description |
| :--- | :--- | :--- | :--- |
| **`__init__`** | Initialization | Default | Initializes the model identifier and model object storage. |
| **`train`** | Model Learning | Abstract | Mandatory logic for optimizing model parameters given training and validation data. |
| **`predict`** | Inference | Abstract | Mandatory logic for generating binary class predictions (0 or 1). |
| **`evaluate`** | Metric Calculation | Helper | Automatically calculates Accuracy, Precision, Recall, and F1-score. |
| **`save`** | Persistence | Abstract | Logic for exporting the trained model state to the filesystem. |
| **`load`** | Persistence | Abstract | Logic for importing a previously saved model state. |

---

### Model Comparison Logic
The automated `evaluate` method returns a standardized dictionary of results. This allows for centralized comparison scripts that can rank models based on performance without parsing varied output formats from different machine learning libraries.

---

### Implementation Skeleton
To ensure compatibility with the project's orchestration logic, any new classification model should implement the following structure:

```python
from src.models import BaseModel

class TemplateClassifier(BaseModel):
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # [Insert Model-Specific Optimization Logic Here]
        pass

    def predict(self, X):
        # [Insert Inference Logic Here]
        # Required output: 1D NumPy array of binary integers (0, 1)
        pass

    def save(self, path):
        # [Insert Serialization Logic (e.g., joblib.dump or torch.save)]
        pass

    def load(self, path):
        # [Insert Deserialization Logic]
        pass
```

### Required Dependencies
*   `numpy`: For data structure handling.
*   `scikit-learn`: For the underlying evaluation metrics utilized in the `evaluate` method.
