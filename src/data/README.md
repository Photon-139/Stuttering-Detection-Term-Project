# Stuttering Data Manager

### Overview
The `DataManager` class is responsible for the post-extraction pipeline, including data cleansing, scaling, class balancing, and dataset partitioning. It serves as the primary interface for preparing features and labels for machine learning models.

### Why it exists
In stuttering detection tasks, datasets are frequently characterized by significant class imbalances (e.g., a higher frequency of fluent speech compared to disfluent events). The `DataManager` centralizes the logic required to address these imbalances and ensures that data is preprocessed consistently across different modeling experiments. By separating these concerns, the architecture ensures that model training remains reproducible and statistically valid.

---

### Class Members

| Member | Type | Description |
| :--- | :--- | :--- |
| `X` | `np.ndarray` | The complete feature matrix of utterance-level embeddings. |
| `y` | `np.ndarray` | The corresponding binary label vector (0 = Fluent, 1 = Disfluent). |

---

### Class Methods

| Method | Purpose | Description |
| :--- | :--- | :--- |
| **`__init__`** | Initialization | Accepts the complete feature matrix (X) and label vector (y) to be managed internally. |
| **`analyze_distribution`** | Distribution Analysis | Reports the count and percentage of fluent vs. disfluent samples. Fulfills project requirements for data auditing. |
| **`preprocess`** | Scaling and Normalization | Provides standardized methods (StandardScaler, MinMaxScaler, or L2 Normalization) to ensure features are on an appropriate scale. |
| **`balance_data`** | Imbalance Mitigation | Implements strategies such as Oversampling or Undersampling to ensure the model learns from a balanced distribution. |
| **`get_splits`** | Dataset Partitioning | Performs a 3-way stratified split into Training, Validation, and Testing sets. Returns 6 separate NumPy arrays. |

---

### Data Pipeline Logic
The `DataManager` is designed to support the following workflow sequence to ensure data integrity:
1. **Initial Audit**: Analyze the raw distribution using `analyze_distribution`.
2. **Partitioning**: Separate the final "Test" set from the "Training" and "Validation" data using `get_splits` to prevent info leakage.
3. **Balancing**: Apply balancing strategies via `balance_data` **only** to the training subset.
4. **Final Scaling**: Standardize or normalize all subsets consistently using `preprocess`.

---

### Usage Skeleton
The following code demonstrates a standardized pipeline execution using the `DataManager` for binary classification tasks:

```python
from src.data import DataManager

# 1. Initialize with raw features (X) and binary labels (y)
manager = DataManager(features, labels)

# 2. Partition the data (3-way stratified split)
X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(test_size=0.15, val_size=0.15)

# 3. Handle class imbalance (Applied ONLY to the training subset)
# This prevents "leaking" balanced class ratios into your evaluation sets.
X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")

# 4. Preprocess / Scale all subsets consistently
# Use the same 'method' (e.g., "standard") for all subsets
X_train_final = manager.preprocess(X_train_bal, method="standard")
X_val_final = manager.preprocess(X_val, method="standard")
X_test_final = manager.preprocess(X_test, method="standard")
```

### Required Dependencies
*   `numpy`: Primitive data structure management.
*   `scikit-learn`: Core logic for data partitioning, feature scaling, and resampling.
