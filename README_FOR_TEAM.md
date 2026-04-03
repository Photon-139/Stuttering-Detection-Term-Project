# Stuttering Detection: Team Analysis Guide

This codebase is a modular framework for comparative stuttering detection using **WavLM Audio Embeddings**. 

## Modular Architecture
Every model in this repository inherits from the **`BaseModel`** class (`src/models/base_model.py`). This ensures they all share the same interface:
*   `model.train(X, y)`: Standardized training call.
*   `model.evaluate(X, y)`: Automated capture of Accuracy, Precision, Recall, F1, and the Confusion Matrix.
*   `model.predict(X)`: Standardized prediction.

## Source Code Organization
*   `src/extractors/`: Contains the **WavLM** logic (WavLMExtractor).
*   `src/data/`: Contains the **DataManager**. It handles the complex work: Master Label Dict lookup, Stratified Splitting, Balancing (Oversampling), and Standard Scaling.
*   `src/models/`: Every AI architecture has its own dedicated file (NN, SVM, Trees, Bayes, etc). 

## Running the Experiment
Use **`main.py`** in the root directory. 
1.  **Configuration**: At the top, you'll find `MODELS_TO_RUN`. 
2.  **Toggle**: To analyze your assigned model, simply uncomment its name in that list.
3.  **Leaderboard**: The script will automatically run the training, print a Pretty Confusion Matrix, and compare your model against the current benchmarks.

## Current Milestone: 70.3% Accuracy
The current champion is the **`DeepNeuralNetwork`** class in `src/models/neural_network_models.py`. 

## Hyperparameter Tuning
All models support `**kwargs`. To tune, instantiate the class with your new parameters:
```python
# Example: Testing a deeper Random Forest
model = RandomForestModel(n_estimators=200, max_depth=20)
```

## Results Persistence
All metrics from the most recent run are automatically exported to **`data/results.json`** for automated report generation.
