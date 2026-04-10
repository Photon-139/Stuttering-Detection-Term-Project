import os
import pandas as pd

def cleanup():
    FEATURE_DIR = 'data/features'
    CSV_PATHS = [
        'Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv',
        'Stuttering Events in Podcasts Dataset/fluencybank_labels.csv'
    ]

    print("\n--- [TEAM CLEANUP TOOL: REMOVING NOISE & ORPHANS] ---")
    
    # 1. Load and identify high-quality human speech only
    if not all(os.path.exists(p) for p in CSV_PATHS):
        print("Error: Could not find CSV metadata. Please run this script from the root of the project directory.")
        return

    print("Analyzing metadata for quality filters...")
    df = pd.concat([pd.read_csv(p) for p in CSV_PATHS])
    df.columns = [c.strip() for c in df.columns] # Clean column headers

    # 1. Apply standard project quality filters
    filtered_df = df[
        (df['Music'] < 1) & 
        (df['NoSpeech'] < 1) & 
        (df['PoorAudioQuality'] < 1)
    ]
    
    # 2. Robust Key Generation (handles leading spaces in CSV data)
    def make_key(row):
        return f"{str(row['Show']).strip()}_{str(row['EpId']).strip()}_{str(row['ClipId']).strip()}"
    
    valid_keys = set(filtered_df.apply(make_key, axis=1))
    print(f"Total verified speech entries: {len(valid_keys)}")

    # 3. Scan and Remove Invalid .npy files
    deleted = 0
    remaining = 0
    
    if not os.path.exists(FEATURE_DIR):
        print(f"No feature directory found at {FEATURE_DIR}. Extraction hasn't run yet.")
        return

    for sub in ['fluent', 'disfluent']:
        path = os.path.join(FEATURE_DIR, sub)
        if not os.path.exists(path): continue
        
        print(f"Scanning {sub} folder...")
        for f in os.listdir(path):
            if not f.endswith('.npy'): continue
            key = f.replace('.npy', '')
            
            if key not in valid_keys:
                os.remove(os.path.join(path, f))
                deleted += 1
            else:
                remaining += 1

    print("\nCleanup Complete!")
    print(f" -> Noisy/Orphan files removed: {deleted}")
    print(f" -> Valid samples remaining:     {remaining}")
    print("Your local data is now synchronized with the latest quality filters.\n")

if __name__ == "__main__":
    cleanup()
