import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import resample
from sklearn.decomposition import PCA

class DataManager:
    """
    Handles preprocessing, class imbalance, and data splitting.
    Follows project requirements (Steps 3.2 and 3.3).
    """
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
        self.scaler = None # Scientific State: Avoids Data Leakage
        self.pca_model = None

    def analyze_distribution(self, y_subset=None):
        """Specifically fulfils Step 3.3 'Analyze the distribution'"""
        target = y_subset if y_subset is not None else self.y
        if target is None: return
        fluent_count = np.sum(target == 0)
        disfluent_count = np.sum(target == 1)
        total = len(target)
        print(f"--- Data Distribution ---")
        print(f"Fluent (0): {fluent_count} ({fluent_count/total:.1%})")
        print(f"Disfluent (1): {disfluent_count} ({disfluent_count/total:.1%})")
        print(f"Total: {total}")

    def preprocess(self, X_input, method="standard", fit=True):
        """
        FLEXIBLE scaling for Step 3.2 Data Preparation. 
        State-Aware: 'fit=True' learns statistics from training data. 
        'fit=False' transforms test data using those exact same statistics (No Leakage).
        """
        if fit or self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "l2":
                self.scaler = Normalizer(norm='l2')
            else:
                raise ValueError("Unknown method. Use 'standard', 'minmax', or 'l2'.")
            
            return self.scaler.fit_transform(X_input)
        else:
            # Re-uses statistics from the training set for valid scientific testing
            return self.scaler.transform(X_input)

    def reduce_dimensions(self, X_input, n_components=0.95, fit=True):
        """
        Step 3.2 Enhancement (Optional): PCA Dimensionality Reduction.
        Keeps 'n_components' of the variance (e.g., 0.95).
        Fits ONLY on training data to prevent leakage.
        """
        if fit or self.pca_model is None:
            self.pca_model = PCA(n_components=n_components, random_state=42)
            X_reduced = self.pca_model.fit_transform(X_input)
            print(f"[DataManager] PCA: Reduced features to {self.pca_model.n_components_} components (Variance={n_components})")
            return X_reduced
        else:
            return self.pca_model.transform(X_input)

    def balance_data(self, X_train, y_train, strategy="oversample"):
        """
        SAFE BALANCING: Only for training data to avoid leaking into validation/test.
        Fulfills Step 3.3 'Handle class imbalance'.
        """
        if strategy == "none":
            return X_train, y_train

        X_fluent = X_train[y_train == 0]
        X_disfluent = X_train[y_train == 1]
        
        # Safety Check: If one class is missing, we cannot balance.
        if len(X_fluent) == 0 or len(X_disfluent) == 0:
            print(f"[{self.__class__.__name__}] Warning: Training subset only contains one class. Skipping balance.")
            return X_train, y_train

        if strategy == "oversample":
            # Duplicate minority class
            dis_upsampled = resample(X_disfluent, replace=True, 
                                     n_samples=len(X_fluent), random_state=42)
            X_bal = np.vstack((X_fluent, dis_upsampled))
            y_bal = np.hstack((np.zeros(len(X_fluent)), np.ones(len(X_fluent))))
            
        elif strategy == "undersample":
            # Shrink majority class
            flu_downsampled = resample(X_fluent, replace=False, 
                                       n_samples=len(X_disfluent), random_state=42)
            X_bal = np.vstack((flu_downsampled, X_disfluent))
            y_bal = np.hstack((np.zeros(len(X_disfluent)), np.ones(len(X_disfluent))))
        else:
            raise ValueError("Unknown strategy. Use 'oversample', 'undersample', or 'none'.")
            
        return X_bal, y_bal

    def get_splits(self, test_size=0.15, val_size=0.15):
        """
        THREE-WAY STRATIFIED SPLIT.
        Ensures Train/Val/Test all have the same class distribution.
        """
        # 1. Pull out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # 2. Split the remainder into train and validation
        # Adjust val_size relative to the remaining temp size
        adj_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=adj_val_size, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_from_folders(self, fluent_dir, disfluent_dir, limit=None, random_seed=42, label_dict=None):
        """
        Loads individual .npy files from class-specific folders.
        OPTIMIZED: Subsets file lists before reading from disk to save I/O time.
        NEW: Optional label_dict allows filtering out ambiguous samples (Strict Mode).
        """
        import random
        
        # 1. Gather all file paths first (just names, no reading yet)
        f_files = [os.path.join(fluent_dir, f) for f in os.listdir(fluent_dir) if f.endswith('.npy')]
        d_files = [os.path.join(disfluent_dir, f) for f in os.listdir(disfluent_dir) if f.endswith('.npy')]
        
        # 2. If a strict label_dict is provided, filter the lists immediately
        if label_dict:
            initial_count = len(f_files) + len(d_files)
            f_files = [f for f in f_files if os.path.splitext(os.path.basename(f))[0] in label_dict]
            d_files = [f for f in d_files if os.path.splitext(os.path.basename(f))[0] in label_dict]
            print(f"[DataManager] Strict Filtering: Kept {len(f_files)+len(d_files)} high-agreement samples (Discarded {initial_count - (len(f_files)+len(d_files))} ambiguous samples).")

        # 3. If a limit is set, we shuffle and pick only what we need BEFORE loading
        if limit:
            random.seed(random_seed)
            # Try to keep classes balanced in the raw load if possible
            half_limit = limit // 2
            
            # Ensure we don't request more than available
            n_f = min(half_limit, len(f_files))
            n_d = min(half_limit, len(d_files))
            
            # If one class is short, take more from the other to reach total limit
            if n_f < half_limit: n_d = min(limit - n_f, len(d_files))
            if n_d < half_limit: n_f = min(limit - n_d, len(f_files))
            
            f_files = random.sample(f_files, n_f)
            d_files = random.sample(d_files, n_d)
            print(f"[DataManager] Smart Load: Pre-selected {len(f_files)} fluent and {len(d_files)} disfluent files.")

        # 4. Now perform the expensive I/O only on the selected subset
        X_fluent = [np.load(f) for f in f_files]
        X_disfluent = [np.load(f) for f in d_files]
        
        if not X_fluent and not X_disfluent:
            raise ValueError("No .npy files found in the specified directories.")

        X = np.vstack(X_fluent + X_disfluent)
        y = np.hstack((np.zeros(len(X_fluent)), np.ones(len(X_disfluent))))
        
        self.X, self.y = X, y
        return X, y

    @staticmethod
    def generate_label_dict(csv_paths, filter_quality=True, strict=True):
        """
        Creates a 'Master Lookup Table' from multiple CSVs.
        Automatically filters out poor audio, music, and non-speech clips.
        NEW: 'strict=True' filters for high-agreement samples only (NoStutteredWords 0 or 3).
        """
        import pandas as pd
        master_df = pd.concat([pd.read_csv(p) for p in csv_paths])
        
        # 1. Quality Filtering (Eliminate Garbage)
        if filter_quality:
            initial_count = len(master_df)
            master_df = master_df[
                (master_df['Music'] < 1) & 
                (master_df['NoSpeech'] < 1) & 
                (master_df['PoorAudioQuality'] < 1)
            ]
            print(f"[DataManager] Quality Filter: Removed {initial_count - len(master_df)} low-quality samples.")
        
        # 2. Strict Agreement Filtering (High Impact Enhancement)
        if strict:
            # Keeps only samples where annotators unanimously agree:
            # 0 = NO annotators agreed it was stutter-free (Strong Stutter)
            # 3 = ALL annotators agreed it was stutter-free (Strong Fluent)
            pre_strict_count = len(master_df)
            master_df = master_df[master_df['NoStutteredWords'].isin([0, 3])]
            print(f"[DataManager] Strict Filter: Kept {len(master_df)} high-agreement samples (Removed {pre_strict_count - len(master_df)} ambiguous samples).")

        # 3. Binary Target Generation
        # Standard Research Rule: (NoStutteredWords < 2) => Stutter(1)
        # With strict mode, this becomes: 0 => 1 (Stutter), 3 => 0 (Fluent)
        master_df['target'] = (master_df['NoStutteredWords'] < 2).astype(int)
        
        # 4. Create Search Key
        master_df['key'] = master_df['Show'].astype(str) + '_' + \
                          master_df['EpId'].astype(str) + '_' + \
                          master_df['ClipId'].astype(str)
                          
        return master_df.set_index('key')['target'].to_dict()
