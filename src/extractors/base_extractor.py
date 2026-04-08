import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseExtractor(ABC):
    """
    Abstract Base Class for all audio feature extractors.
    Final output is always a NumPy array (CDF - Common Data Format).
    """
    TARGET_SR = 16000  # Standard for most speech AI models (WavLM, Whisper, etc.)

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Initializing on {self.device}...")
        self.model_data = self.load_model()

    @abstractmethod
    def load_model(self):
        """
        Must return a dictionary (or object) containing model and processor.
        Executed once during initialization.
        """
        pass

    @abstractmethod
    def extract_one(self, audio_path):
        """
        Loads a single audio file and returns a 1D NumPy embedding.
        Input: str (path)
        Output: np.ndarray (shape: [embedding_dim])
        """
        pass

    def extract_batch(self, audio_paths, output_dir=None, label_dict=None, log_path=None):
        """
        Processes a list of audio files with a progress bar.
        Individual Save (New): If output_dir is provided, saves one .npy per audio file.
        Automatic Sorting: If label_dict is provided, sorts into fluent/disfluent subfolders.
        """
        X = []
        failed_paths = []
        # Building the 'Existing Features' cache (Optimized Pre-Scan)
        existing_filenames = set()
        if output_dir:
            for sub in ["fluent", "disfluent"]:
                sub_path = os.path.join(output_dir, sub)
                os.makedirs(sub_path, exist_ok=True) # Ensure folders exist
                if os.path.exists(sub_path):
                    # Cache filenames without extension for fast O(1) lookup
                    existing_filenames.update({
                        os.path.splitext(f)[0] for f in os.listdir(sub_path) 
                        if f.endswith('.npy')
                    })
            print(f"[{self.__class__.__name__}] Found {len(existing_filenames)} pre-existing features. Skipping redundant work.")

        stats = {"fluent": 0, "disfluent": 0, "noisy": 0, "existing": len(existing_filenames)}
        
        for path in tqdm(audio_paths, desc="Batch Extraction"):
            filename = os.path.splitext(os.path.basename(path))[0]
            
            # 0. Optimized Resumable Check
            if filename in existing_filenames:
                continue
            
            # 1. Validation & Quality Check
            if label_dict:
                if filename not in label_dict:
                    stats["noisy"] += 1
                    continue
            
            # 2. Actual Extraction
            emb = self.extract_one(path)
            if emb is not None:
                if output_dir:
                    subfolder = ""
                    label = 0
                    if label_dict:
                        label = label_dict[filename]
                        subfolder = "disfluent" if label == 1 else "fluent"
                    
                    if label == 0: stats["fluent"] += 1
                    else: stats["disfluent"] += 1

                    target_path = os.path.join(output_dir, subfolder, f"{filename}.npy")
                    np.save(target_path, emb)
                    existing_filenames.add(filename)
                else:
                    X.append(emb)
            else:
                failed_paths.append(path)
            
        print(f"\n[{self.__class__.__name__}] Extraction Summary:")
        print(f" - New Fluent: {stats['fluent']}")
        print(f" - New Disfluent: {stats['disfluent']}")
        print(f" - Skipped (Noise/No-Meta): {stats['noisy']}")
        print(f" - Skipped (On Disk): {stats['existing']}")
        if failed_paths:
            print(f" - Failed Extractions: {len(failed_paths)}")
            if log_path:
                with open(log_path, "w") as f:
                    f.write("\n".join(failed_paths))
            
        return np.array(X) if X else None

    def extract_from_dir(self, directory_path, output_dir=None, label_dict=None, 
                         limit=None, log_path=None, random_sample=False, seed=42):
        """
        Scans a folder and extracts features for all audio files.
        'output_dir': Directory where individual .npy files will be saved.
        'label_dict': Dictionary {filename: label} for sorting.
        'random_sample': If True, picks 'limit' files randomly instead of sequentially.
        """
        all_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        
        if limit and limit > 0:
            if random_sample:
                import random
                random.seed(seed)
                random.shuffle(all_files)
                print(f"[{self.__class__.__name__}] Randomly sampling {limit} audio files from across the entire dataset...")
            else:
                all_files = sorted(all_files)
                print(f"[{self.__class__.__name__}] Limiting to the first {limit} audio files.")
            
            all_files = all_files[:limit]
            
        return self.extract_batch(all_files, output_dir=output_dir, label_dict=label_dict, log_path=log_path)
