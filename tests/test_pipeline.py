import numpy as np
import os
import sys
import argparse

# Ensure we can import from src/
sys.path.append(os.getcwd())

from src.extractors import WavLMExtractor
from src.data import DataManager

def run_test(audio_dir, csv_path, model_id, sample_dir):
    # 0. Path Validation
    abs_audio_dir = os.path.abspath(audio_dir)
    abs_csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(abs_audio_dir):
        print(f"\n[CRITICAL ERROR]: Could not find the audio directory at '{abs_audio_dir}'")
        sys.exit(1)
        
    if not os.path.exists(abs_csv_path):
        print(f"\n[CRITICAL ERROR]: Could not find the CSV labels at '{abs_csv_path}'")
        sys.exit(1)

    # Ensure sample folder exists
    os.makedirs(sample_dir, exist_ok=True)

    print("\n--- [STARTING REAL-WORLD PIPELINE & STORAGE TEST] ---")

    # Step 1: EXTRACT (50 files)
    print(f"\n[1/6] Extracting 50 real samples from {abs_audio_dir}...")
    extractor = WavLMExtractor(model_id)
    features = extractor.extract_from_dir(
        abs_audio_dir, limit=50, save_path=os.path.join(sample_dir, "raw_features.npy")
    )
    
    # We need the paths to find the labels in the CSV
    audio_paths = [os.path.join(abs_audio_dir, f) for f in os.listdir(abs_audio_dir) if f.lower().endswith('.wav')]
    audio_paths.sort()
    audio_paths = audio_paths[:50]

    # Step 2: LOAD REAL LABELS FROM CSV
    print(f"\n[2/6] Correlating features with labels from {abs_csv_path}...")
    manager = DataManager(features, None) # Initialize without labels first
    labels = manager.load_labels_from_csv(abs_csv_path, audio_paths)
    
    # Update manager with real labels
    manager.y = labels
    manager.analyze_distribution()

    # Step 3: SPLIT (Train/Val/Test)
    print("\n[3/5] Performing 3-Way Stratified Split...")
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(test_size=0.15, val_size=0.15)
    print(f"Split results -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 4: BALANCE & SCALE
    print("\n[4/5] Applying Balancing (Oversampling) and Scaling (StandardScaler)...")
    X_train_bal, y_train_bal = manager.balance_data(X_train, y_train, strategy="oversample")
    X_train_final = manager.preprocess(X_train_bal, method="standard")
    print(f"Final training samples after balancing: {len(X_train_bal)}")

    # Step 5: SAVE PROCESSED NPY
    print("\n[5/5] Saving processed NumPy arrays to sample_data/...")
    X_TRAIN_FILE = os.path.join(sample_dir, "X_train_final.npy")
    np.save(X_TRAIN_FILE, X_train_final)

    # FINAL VERIFICATION
    print(f"\n--- [FINAL STORAGE & STATS VERIFICATION] ---")
    raw_npy = os.path.join(sample_dir, "raw_features.npy")
    if os.path.exists(raw_npy):
        raw_data = np.load(raw_npy)
        print(f"[SUCCESS]: [raw_features.npy] exists ({os.path.getsize(raw_npy)} bytes)")
        print(f"           Mean value: {raw_data.mean():.6f}")

    if os.path.exists(X_TRAIN_FILE):
        final_data = np.load(X_TRAIN_FILE)
        print(f"[SUCCESS]: [X_train_final.npy] exists ({os.path.getsize(X_TRAIN_FILE)} bytes)")
        print(f"           Final Mean (should be near 0.0): {final_data.mean():.6f}")
        print(f"           Final Std  (should be near 1.0): {final_data.std():.6f}")

    print("\n--- [TEST PIPELINE COMPLETED SUCCESSFULLY] ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Test for Stuttering Detection Pipeline")
    parser.add_argument("--audio_dir", type=str, 
                        default="Stuttering Events in Podcasts Dataset/clips/stuttering-clips/clips",
                        help="Relative or absolute path to the audio clips directory")
    parser.add_argument("--csv_path", type=str, 
                        default="Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv",
                        help="Path to the SEP-28k_labels.csv file")
    parser.add_argument("--model_id", type=str, 
                        default="microsoft/wavlm-base",
                        help="HuggingFace Model ID for the feature extractor")
    parser.add_argument("--sample_dir", type=str, 
                        default="sample_data",
                        help="Output directory for generated .npy files")
    
    args = parser.parse_args()
    run_test(args.audio_dir, args.csv_path, args.model_id, args.sample_dir)
