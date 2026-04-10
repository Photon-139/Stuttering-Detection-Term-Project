import os
import json
import numpy as np
import pandas as pd
from src.extractors import WavLMExtractor
from src.data import DataManager
from src.models import (
    LogisticModel, 
    PerceptronModel, 
    LinearSVMModel, 
    KernelSVMModel,
    ShallowNeuralNetwork,
    DeepNeuralNetwork,
    RandomForestModel
)

def load_best_params():
    """Helper to load tuned parameters if they exist"""
    config_path = "config/best_linear_params.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def main():
    # 1. Pipeline Configuration
    EXTRACT_LIMIT = 14000
    USE_PCA = True
    RANDOM_SEED = 42
    
    # Path Configuration
    AUDIO_DIR = "Stuttering Events in Podcasts Dataset/clips/stuttering-clips/clips"
    CSV_PATHS = [
        "Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv",
        "Stuttering Events in Podcasts Dataset/fluencybank_labels.csv"
    ]
    FEATURE_DIR = "data/features"
    
    # 2. Extract Features (If needed)
    print("--- [STEP 1: Feature Extraction] ---")
    extractor = WavLMExtractor("microsoft/wavlm-base")
    label_dict = DataManager.generate_label_dict(CSV_PATHS, filter_quality=True)
    
    extractor.extract_from_dir(
        AUDIO_DIR, 
        output_dir=FEATURE_DIR, 
        label_dict=label_dict, 
        limit=EXTRACT_LIMIT, 
        random_sample=True
    )

    # 3. Load and Prepare Data
    print("\n--- [STEP 2: Data Loading & Preprocessing] ---")
    manager = DataManager(None, None)
    X, y = manager.load_from_folders(
        os.path.join(FEATURE_DIR, "fluent"),
        os.path.join(FEATURE_DIR, "disfluent")
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits()
    X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")
    
    # Preprocessing Chain
    X_train_final = manager.preprocess(X_train_bal, method="standard", fit=True)
    X_val_final = manager.preprocess(X_val, fit=False)
    X_test_final = manager.preprocess(X_test, fit=False)
    
    # Optional PCA Denoising
    if USE_PCA:
        print("[PCA] Reducing dimensionality for feature denoising...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        X_train_final = pca.fit_transform(X_train_final)
        X_val_final = pca.transform(X_val_final)
        X_test_final = pca.transform(X_test_final)
        print(f"[PCA] New Feature Dimension: {X_train_final.shape[1]}")

    # 4. Model Selection & Configuration
    print("\n--- [STEP 3: Model Competition] ---")
    best_params = load_best_params()
    
    # Extract specific tuned params (or empty dict if not found)
    log_cfg = best_params.get("logistic_regression", {})
    perc_cfg = best_params.get("perceptron", {})
    
    models = [
        # 1. Linear Baselines
        LogisticModel("Logistic_Regression", **log_cfg),
        PerceptronModel("Perceptron", **perc_cfg),
        
        # 2. Kernel Support Vector Machines
        LinearSVMModel("SVM_Linear", C=1.0),
        KernelSVMModel("SVM_RBF", kernel="rbf", C=10.0, gamma='scale'),
        
        # 3. Neural Architectures
        ShallowNeuralNetwork("Shallow_NN", input_dim=X_train_final.shape[1]),
        # DeepNeuralNetwork("Deep_NN", input_dim=X_train_final.shape[1]),
        
        # 4. Tree Ensembles
        RandomForestModel("Random_Forest", n_estimators=100)
    ]

    # 5. Training & Evaluation Benchmarking
    results = []
    for model in models:
        model.train(X_train_final, y_train_bal)
        metrics = model.evaluate(X_test_final, y_test)
        results.append({
            "model": model.model_name,
            "accuracy": metrics['accuracy'],
            "f1": metrics['f1']
        })

    # 6. Scientific Summary Table
    print("\n" + "="*40)
    print(f"{'MODEL':<20} | {'ACCURACY':<10} | {'F1 SCORE':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['model']:<20} | {res['accuracy']:<10.4f} | {res['f1']:<10.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
