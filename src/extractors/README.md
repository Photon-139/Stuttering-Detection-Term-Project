# Audio Feature Extractors

### Overview
The `BaseExtractor` is an Abstract Base Class (ABC) designed to provide a universal interface for speech audio processing. 

### Why it exists
This architecture provides flexibility when working with various speech processing libraries. By utilizing an abstract base class, it ensures that switching between different models (such as WavLM, Whisper, or HuBERT) does not require modifying the downstream application logic. The rest of the project (training, data management, and analysis) remains decoupled from the specific implementation details of the feature extraction library.

---

### Class Members and Methods

| Member | Type | Description |
| :--- | :--- | :--- |
| `TARGET_SR` | `int` | The required audio sample rate (standardized to 16,000 Hz) to ensure consistency across models. |
| `device` | `torch.device` | The hardware device used for processing (automatically detects CUDA/GPU or defaults to CPU). |
| `model_data` | `Any` | Internal storage for the specific model and processor instances. |

| Method | Type | Requirement | Description |
| :--- | :--- | :--- | :--- |
| **`load_model`** | Abstract | **Mandatory** | Component-specific initialization logic (e.g., loading weights from a repository). |
| **`extract_one`** | Abstract | **Mandatory** | The core processing logic. Converts an audio file path into a single 1D NumPy embedding. |
| **`extract_batch`** | Helper | Optional | Processes a list of audio file paths and returns a 2D matrix of embeddings. Includes progress tracking. |
| **`extract_from_dir`**| Helper | Optional | Scans a filesystem directory for audio files and extracts features. Includes an optional limit for testing. |

---

### Implementation Guide
To implement a new feature extractor:
1. Create a new class that inherits from `BaseExtractor`.
2. Implement the `load_model` method to initialize the specific underlying model.
3. Implement the `extract_one` method to handle audio loading and transformation, ensuring the final return value is a standard NumPy array.
