import os
import numpy as np
from src.extractors import WavLMExtractor
from src.data import DataManager
import torch.nn as nn
import torch.optim as optim
from src.models import (
    LogisticModel, PerceptronModel, NaiveBayesModel, 
    KNNModel, ShallowNeuralNetwork, DeepNeuralNetwork,
    LinearSVMModel, KernelSVMModel, LDAModel,
    DecisionTreeModel, RandomForestModel
)

# --- CONFIGURATION (The Experiment Settings) ---
AUDIO_DIR = "Stuttering Events in Podcasts Dataset/clips/stuttering-clips/clips"
CSV_PATHS = [
    "Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv",
    "Stuttering Events in Podcasts Dataset/fluencybank_labels.csv"
]
FEATURE_DIR = "data/features"  # Where individual .npy files live
MODEL_ID = "microsoft/wavlm-base"
RANDOM_SEED = 42

# --- CONTROL FLAGS ---
# 1. CLEAN_START: Set to True to DELETE all existing features on disk before running.
CLEAN_START = False 

# 2. FORCE_EXTRACT: Set to True to re-extract features even if folders already exist.
FORCE_EXTRACT = False

# 3. EXTRACT_LIMIT: Set to an integer (e.g., 500) for a quick test run. 
# Set to None for the full 28k dataset.
EXTRACT_LIMIT = 1000

# 4. MODELS TO RUN: Add/Remove models here for head-to-head comparison
MODELS_TO_RUN = [
    LogisticModel("LogReg_Baseline"),
    NaiveBayesModel("NaiveBayes_Baseline"),
    # KNNModel("KNN_Baseline", n_neighbors=5),
    
    # Lab-Inspired Neural Networks (PyTorch)
    ShallowNeuralNetwork("Shallow_NN", hidden_layer_size=64, 
                          momentum=0.9, activation_fn=nn.Tanh),
    
    DeepNeuralNetwork("Deep_NN", hidden_layer_sizes=[128, 64], 
                        momentum=0.9, activation_fn=nn.Tanh),

    # SVMs
    LinearSVMModel("Linear_SVM"),
    KernelSVMModel("RBF_SVM", kernel_type='rbf'),

    # LDA (General Bayes)
    LDAModel("LDA_Bayes_Final"),

    # Tree-Based Models
    DecisionTreeModel("Decision_Tree_Default", max_depth=10),
    RandomForestModel("Random_Forest_100", n_estimators=100)
]
# ---------------------

def main():
    import shutil
    import json
    print("--- [STUTTERING DETECTION: MAIN PIPELINE] ---")

    # 1. CLEAN SLATE LOGIC
    if CLEAN_START and os.path.exists(FEATURE_DIR):
        print(f"\n[CLEAN START]: Deleting existing features in {FEATURE_DIR}...")
        shutil.rmtree(FEATURE_DIR)

    # 2. INITIALIZE MASTER LABEL LOOKUP
    print(f"\n[1/4] Loading Master Label Switchboard...")
    label_dict = DataManager.generate_label_dict(CSV_PATHS, filter_quality=True)
    
    # 3. FEATURE EXTRACTION (RESUMABLE)
    fluent_dir = os.path.join(FEATURE_DIR, "fluent")
    disfluent_dir = os.path.join(FEATURE_DIR, "disfluent")
    
    # Extraction Trigger
    needs_extraction = FORCE_EXTRACT or \
                       not os.path.exists(fluent_dir) or \
                       not os.path.exists(disfluent_dir) or \
                       (len(os.listdir(fluent_dir)) == 0 and len(os.listdir(disfluent_dir)) == 0)

    if needs_extraction:
        print(f"\n[2/4] Triggering WavLM Extraction...")
        extractor = WavLMExtractor(MODEL_ID)
        
        # Shuffle/Pick top files to process
        all_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
        import random
        random.seed(RANDOM_SEED)
        random.shuffle(all_files)
        
        if EXTRACT_LIMIT:
            print(f"Limiting to first {EXTRACT_LIMIT} clips for this run.")
            all_files = all_files[:EXTRACT_LIMIT]
            
        extractor.extract_batch(all_files, output_dir=FEATURE_DIR, label_dict=label_dict)
    else:
        print(f"\n[2/4] Existing features found in {FEATURE_DIR}. Skipping extraction.")

    # 3. CONSOLIDATED LOADING
    print(f"\n[3/4] Loading distributed data into Training Matrices...")
    manager = DataManager(None, None)
    X, y = manager.load_from_folders(fluent_dir, disfluent_dir)
    print(f"Loaded total of {len(X)} samples.")
    manager.analyze_distribution()

    # 4. PREPARATION (Split -> Balance -> Scale)
    print(f"\n[4/4] Preparing Final Datasets...")
    
    # Stratified Split
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(
        test_size=0.15, val_size=0.15
    )
    
    # Balance
    X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")
    
    # Standardize
    X_train_final = manager.preprocess(X_train_bal, method="standard")
    X_val_final = manager.preprocess(X_val, method="standard")
    X_test_final = manager.preprocess(X_test, method="standard")

    print(f"\n--- [PIPELINE COMPLETE: DATA READY] ---")
    print(f"X_train (Balanced/Scaled): {X_train_final.shape}, y_train: {y_train_bal.shape}")
    print(f"X_val (Scaled):            {X_val_final.shape}, y_val: {y_val.shape}")
    print(f"X_test (Scaled):           {X_test_final.shape}, y_test: {y_test.shape}")
    print(f"Targeting: {MODEL_ID} Embeddings (768 features)")
    print("----------------------------------------\n")

    # 5. MODEL COMPETITION
    all_results = {}

    for model in MODELS_TO_RUN:
        print(f"\n--- [TRAINING: {model.model_name}] ---")
        model.train(X_train_final, y_train_bal)
        
        print(f"\n--- [VALIDATION: {model.model_name}] ---")
        val_m = model.evaluate(X_val_final, y_val)
        
        print(f"\n--- [TEST SET: {model.model_name}] ---")
        test_m = model.evaluate(X_test_final, y_test)

        # Log results for export (convert numpy arrays to lists for JSON)
        all_results[model.model_name] = {
            "validation": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_m.items()},
            "test": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_m.items()}
        }
    
    # Save to JSON
    results_path = os.path.join("data", "results.json")
    os.makedirs("data", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[SUCCESS] Final metrics saved to {results_path}")
    
    print("\n--- [ALL EXPERIMENTS COMPLETED SUCCESSFULLY] ---")

if __name__ == "__main__":
    main()
