import os
import json
import argparse
import numpy as np
from src.data import DataManager
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.decomposition import PCA
from src.models import LogisticModel, PerceptronModel

def tune_models(limit=None, strict=True):
    # Handle the "0" shortcut for full data
    load_limit = None if limit == 0 else limit
    print(f"--- [EXHAUSTIVE TUNING: PCA-ALIGNED] (Limit: {load_limit if load_limit else 'Full Data'}) ---")
    print(f"--- [STRICT MODE: {strict}] ---")
    
    # 1. Setup Data
    FEATURE_DIR = "data/features"
    CSV_PATHS = ["Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv", "Stuttering Events in Podcasts Dataset/fluencybank_labels.csv"]
    
    label_dict = DataManager.generate_label_dict(CSV_PATHS, filter_quality=True, strict=strict)
    manager = DataManager(None, None)
    X, y = manager.load_from_folders(
        os.path.join(FEATURE_DIR, "fluent"),
        os.path.join(FEATURE_DIR, "disfluent"),
        limit=load_limit,
        label_dict=label_dict
    )
    
    # Standard Split to get Natural Validation Data
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(test_size=0.15, val_size=0.15)
    
    # BALANCE only the training set
    X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")
    
    # PREPROCESS (Standard Scaling)
    X_train_final = manager.preprocess(X_train_bal, method="standard", fit=True)
    X_val_final = manager.preprocess(X_val, fit=False)
    
    # 2. PCA DIMENSIONALITY REDUCTION (Aligned with main.py)
    print(f"[PCA] Denoising features (95% Variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_final)
    X_val_pca = pca.transform(X_val_final)
    print(f"[PCA] Reduced from {X_train_final.shape[1]} to {X_train_pca.shape[1]} dimensions.")

    # 3. CREATE PREDEFINED SPLIT (Tune on Natural X_val)
    X_combined = np.vstack((X_train_pca, X_val_pca))
    y_combined = np.hstack((y_train_bal, y_val))
    split_indices = np.hstack((
        np.full(len(y_train_bal), -1), # Always train on these
        np.full(len(y_val), 0)         # Use these for the validation scoring
    ))
    pds = PredefinedSplit(test_fold=split_indices)

    print(f"[Config] Training on {len(y_train_bal)} balanced samples.")
    print(f"[Config] Tuning against {len(y_val)} natural validation samples.")

    # 4. Logistic Regression Grid
    print("\n[LogReg] Tuning...")
    log_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'saga'], 'max_iter': [2000]}
    gs_log = GridSearchCV(LogisticRegression(random_state=42), log_grid, cv=pds, scoring='accuracy', n_jobs=-1)
    gs_log.fit(X_combined, y_combined)
    print(f"Best LogReg: {gs_log.best_params_} (Val Acc: {gs_log.best_score_:.4f})")

    # 5. Perceptron Grid
    print("\n[Perceptron] Tuning...")
    perc_grid = {
        'eta0': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'penalty': ['l2', 'l1', None],
        'alpha': [1e-4, 1e-3, 0.01],
        'max_iter': [2000]
    }
    gs_perc = GridSearchCV(Perceptron(random_state=42), perc_grid, cv=pds, scoring='accuracy', n_jobs=-1)
    gs_perc.fit(X_combined, y_combined)
    print(f"Best Perceptron: {gs_perc.best_params_} (Val Acc: {gs_perc.best_score_:.4f})")

    # 6. Save Results
    results = {
        "logistic_regression": gs_log.best_params_,
        "perceptron": gs_perc.best_params_
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/best_linear_params.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[Success] PCA-Optimized parameters saved to config/best_linear_params.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Research Hypertuning Script")
    parser.add_argument("--limit", type=int, default=0, help="Number of samples (use 0 for full data)")
    parser.add_argument("--strict", type=int, default=1, help="Use strict labels (1=True, 0=False)")
    args = parser.parse_args()
    tune_models(limit=args.limit, strict=bool(args.strict))
