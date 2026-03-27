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
