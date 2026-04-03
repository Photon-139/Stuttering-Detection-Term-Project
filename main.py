import os
import numpy as np
from src.extractors import WavLMExtractor
from src.data import DataManager
from src.models import LogisticModel

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
CLEAN_START = True 

# 2. FORCE_EXTRACT: Set to True to re-extract features even if folders already exist.
FORCE_EXTRACT = False

# 3. EXTRACT_LIMIT: Set to an integer (e.g., 500) for a quick test run. 
# Set to None for the full 28k dataset.
EXTRACT_LIMIT = 1000
# ---------------------

def main():
    import shutil
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
    
    # Stratified Split (70% Train, 15% Val, 15% Test)
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(
        test_size=0.15, val_size=0.15
    )
    
    # Balance (Oversample the minority class in the training set ONLY)
    X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")
    
    # Standardize (StandardScaler - based on training data only)
    X_train_final = manager.preprocess(X_train_bal, method="standard")
    X_val_final = manager.preprocess(X_val, method="standard")
    X_test_final = manager.preprocess(X_test, method="standard")

    # --- READY FOR MODELING ---
    print(f"\n--- [PIPELINE COMPLETE: DATA READY] ---")
    print(f"X_train (Balanced/Scaled): {X_train_final.shape}, y_train: {y_train_bal.shape}")
    print(f"X_val (Scaled):            {X_val_final.shape}, y_val: {y_val.shape}")
    print(f"X_test (Scaled):           {X_test_final.shape}, y_test: {y_test.shape}")
    print(f"Targeting: {MODEL_ID} Embeddings (768 features)")
    print("----------------------------------------\n")

    # 5. MODEL TRAINING & EVALUATION
    print(f"\n[5/5] Training Logistic Regression Model...")
    model = LogisticModel("LogReg_Experiment_1")
    model.train(X_train_final, y_train_bal)
    
    # Evaluate on Validation Set
    print("\n--- VALIDATION RESULTS ---")
    model.evaluate(X_val_final, y_val)
    
    # Final Evaluate on Test Set
    print("\n--- FINAL TEST RESULTS ---")
    model.evaluate(X_test_final, y_test)
    
    print("\n--- [ALL STEPS COMPLETED SUCCESSFULLY] ---")
    
if __name__ == "__main__":
    main()
