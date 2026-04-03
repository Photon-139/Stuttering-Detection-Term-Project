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
        print(f"[{self.__class__.__name__}] Processing {len(audio_paths)} samples...")
        
        # Prepare directories if saving individually
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if label_dict:
                os.makedirs(os.path.join(output_dir, "fluent"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "disfluent"), exist_ok=True)

        for path in tqdm(audio_paths, desc="Batch Extraction"):
            emb = self.extract_one(path)
            if emb is not None:
                # 1. Save individually if output_dir is set
                if output_dir:
                    filename = os.path.splitext(os.path.basename(path))[0]
                    
                    # Determine subfolder based on label
                    subfolder = ""
                    if label_dict:
                        label = label_dict.get(filename, 0)
                        subfolder = "disfluent" if label == 1 else "fluent"
                    
                    target_path = os.path.join(output_dir, subfolder, f"{filename}.npy")
                    np.save(target_path, emb)
                else:
                    X.append(emb) # Keep in memory for smaller batches
            else:
                failed_paths.append(path)
            
        if log_path and failed_paths:
            with open(log_path, "w") as f:
                f.write("\n".join(failed_paths))
            print(f"[{self.__class__.__name__}] Logged {len(failed_paths)} failed files to {log_path}")
            
        return np.array(X) if X else None

    def extract_from_dir(self, directory_path, output_dir=None, label_dict=None, limit=None, log_path=None):
        """
        Scans a folder and extracts features for all audio files.
        'output_dir': Directory where individual .npy files will be saved.
        'label_dict': Dictionary {filename: label} for sorting.
        """
        all_files = [
            os.path.join(directory_path, f) 
            for f in sorted(os.listdir(directory_path)) 
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        
        if limit and limit > 0:
            print(f"[{self.__class__.__name__}] Limiting to the first {limit} audio files.")
            all_files = all_files[:limit]
            
        return self.extract_batch(all_files, output_dir=output_dir, label_dict=label_dict, log_path=log_path)
