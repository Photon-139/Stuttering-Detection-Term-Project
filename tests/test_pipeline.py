import numpy as np
import os
import sys
import argparse
import shutil
import random

# Ensure we can import from src/
sys.path.append(os.getcwd())

from src.extractors import WavLMExtractor
from src.data import DataManager

def run_test(audio_dir, sep_csv, fb_csv, sample_dir):
    # --- CONFIGURATION VARIABLE ---
    NUM_SAMPLES = 500 
    # ------------------------------

    # 0. Path Validation
    abs_audio_dir = os.path.abspath(audio_dir)
    abs_sep_csv = os.path.abspath(sep_csv)
    abs_fb_csv = os.path.abspath(fb_csv)
    
    for p in [abs_audio_dir, abs_sep_csv, abs_fb_csv]:
        if not os.path.exists(p):
            print(f"\n[CRITICAL ERROR]: Could not find path: {p}")
            sys.exit(1)

    # Prepare specific feature directories
    feat_dir = os.path.join(sample_dir, "features")
    if os.path.exists(feat_dir):
        shutil.rmtree(feat_dir) # Start fresh for test
    os.makedirs(feat_dir, exist_ok=True)

    print("\n--- [STARTING DISTRIBUTED PIPELINE TEST (500 SAMPLES)] ---")

    # Step 1: GENERATE MASTER LABEL DICT (One-time load)
    print(f"\n[1/5] Building Master Label Switchboard from CSVs...")
    label_dict = DataManager.generate_label_dict([abs_sep_csv, abs_fb_csv], filter_quality=True)
    print(f"Master Dictionary loaded with {len(label_dict)} items.")

    # Get and Shuffle files to ensure we see stutters!
    random.seed(42) # Reproducible
    all_files = [f for f in os.listdir(abs_audio_dir) if f.lower().endswith('.wav')]
    random.shuffle(all_files)
    test_files = [os.path.join(abs_audio_dir, f) for f in all_files[:NUM_SAMPLES]]

    # Step 2: INDIVIDUAL EXTRACTION & SORTING
    print(f"\n[2/5] Extracting {NUM_SAMPLES} samples into individual files...")
    extractor = WavLMExtractor("microsoft/wavlm-base")
    # Using extract_batch directly since we already found the files
    extractor.extract_batch(
        test_files, 
        output_dir=feat_dir, 
        label_dict=label_dict, 
        log_path=os.path.join(sample_dir, "failed_files.log")
    )

    # Step 3: DISTRIBUTED LOADING
    print(f"\n[3/5] Loading from sorted folders using DataManager...")
    manager = DataManager(None, None)
    fluent_dir = os.path.join(feat_dir, "fluent")
    disfluent_dir = os.path.join(feat_dir, "disfluent")
    
    X, y = manager.load_from_folders(fluent_dir, disfluent_dir)
    print(f"Loaded {len(X)} features back into memory.")
    manager.analyze_distribution()

    # Step 4: PROCESS (Split & Scaling)
    # Note: With 200 samples, 3-way split should be stable
    print(f"\n[4/5] Normalizing and Splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = manager.get_splits(test_size=0.15, val_size=0.15)
    X_train_scaled = manager.preprocess(X_train, method="standard")

    # Step 5: VERIFICATION
    print(f"\n--- [FINAL DISTRIBUTED VERIFICATION] ---")
    fluent_files = len(os.listdir(fluent_dir))
    disfluent_files = len(os.listdir(disfluent_dir))
    
    print(f"✅ FOLDER [fluent/]: {fluent_files} individual files.")
    print(f"✅ FOLDER [disfluent/]: {disfluent_files} individual files.")
    print(f"✅ STATS: Scaled Mean={X_train_scaled.mean():.6f}, Std={X_train_scaled.std():.6f}")

    print("\n--- [DISTRIBUTED TEST COMPLETED SUCCESSFULLY] ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Distributed Pipeline Test")
    parser.add_argument("--audio_dir", type=str, 
                        default="Stuttering Events in Podcasts Dataset/clips/stuttering-clips/clips",
                        help="Path to audio clips")
    parser.add_argument("--sep_csv", type=str, 
                        default="Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv",
                        help="Path to SEP-28k labels")
    parser.add_argument("--fb_csv", type=str, 
                        default="Stuttering Events in Podcasts Dataset/fluencybank_labels.csv",
                        help="Path to FluencyBank labels")
    parser.add_argument("--sample_dir", type=str, 
                        default="sample_data",
                        help="Sample data output root")
    
    args = parser.parse_args()
    run_test(args.audio_dir, args.sep_csv, args.fb_csv, args.sample_dir)
