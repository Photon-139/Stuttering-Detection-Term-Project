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

    def extract_batch(self, audio_paths, save_path=None, log_path=None):
        """
        Processes a list of audio files with a progress bar and returns a 2D matrix.
        Returns: np.ndarray (shape: [num_samples, embedding_dim])
        """
        X = []
        failed_paths = []
        print(f"[{self.__class__.__name__}] Extracting batch of {len(audio_paths)} samples...")
        
        for path in tqdm(audio_paths, desc="Batch Extraction"):
            emb = self.extract_one(path)
            if emb is not None:
                X.append(emb)
            else:
                failed_paths.append(path)
            
        X = np.array(X)
        if save_path:
            np.save(save_path, X)
            print(f"[{self.__class__.__name__}] Saved features to {save_path}")
            
        if log_path and failed_paths:
            with open(log_path, "w") as f:
                f.write("\n".join(failed_paths))
            print(f"[{self.__class__.__name__}] Logged {len(failed_paths)} failed files to {log_path}")
            
        return X

    def extract_from_dir(self, directory_path, limit=None, save_path=None, log_path=None):
        """
        Convenience method to scan a folder and extract features for all audio files.
        'limit': Optional int to only process the first 'n' files for fast testing.
        """
        all_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        all_files.sort()
        
        if limit and limit > 0:
            print(f"[{self.__class__.__name__}] Limiting to the first {limit} audio files.")
            all_files = all_files[:limit]
            
        return self.extract_batch(all_files, save_path=save_path, log_path=log_path)
