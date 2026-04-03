import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import resample

class DataManager:
    """
    Handles preprocessing, class imbalance, and data splitting.
    Follows project requirements (Steps 3.2 and 3.3).
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def analyze_distribution(self, y_subset=None):
        """Specifically fulfils Step 3.3 'Analyze the distribution'"""
        target = y_subset if y_subset is not None else self.y
        fluent_count = np.sum(target == 0)
        disfluent_count = np.sum(target == 1)
        total = len(target)
        print(f"--- Data Distribution ---")
        print(f"Fluent (0): {fluent_count} ({fluent_count/total:.1%})")
        print(f"Disfluent (1): {disfluent_count} ({disfluent_count/total:.1%})")
        print(f"Total: {total}")

    def preprocess(self, X_input, method="standard"):
        """FLEXIBLE scaling for Step 3.2 Data Preparation."""
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "l2":
            scaler = Normalizer(norm='l2')
        else:
            raise ValueError("Unknown method. Use 'standard', 'minmax', or 'l2'.")
        
        return scaler.fit_transform(X_input)

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

    def load_from_folders(self, fluent_dir, disfluent_dir):
        """
        Loads individual .npy files from class-specific folders.
        This fulfills the TA's 'Distributed Data' requirement.
        """
        X_fluent = []
        for f in os.listdir(fluent_dir):
            if f.endswith('.npy'):
                X_fluent.append(np.load(os.path.join(fluent_dir, f)))
        
        X_disfluent = []
        for f in os.listdir(disfluent_dir):
            if f.endswith('.npy'):
                X_disfluent.append(np.load(os.path.join(disfluent_dir, f)))
        
        X = np.vstack(X_fluent + X_disfluent)
        y = np.hstack((np.zeros(len(X_fluent)), np.ones(len(X_disfluent))))
        
        self.X, self.y = X, y
        return X, y

    @staticmethod
    def generate_label_dict(csv_paths, filter_quality=True):
        """
        Creates a 'Master Lookup Table' from multiple CSVs.
        Automatically filters out poor audio, music, and non-speech clips.
        """
        import pandas as pd
        master_df = pd.concat([pd.read_csv(p) for p in csv_paths])
        
        # 1. Quality Filtering (Eliminate Garbage)
        if filter_quality:
            # Drop rows where annotators heard strictly Music or NoSpeech
            # Or where quality was marked as poor
            initial_count = len(master_df)
            master_df = master_df[
                (master_df['Music'] < 1) & 
                (master_df['NoSpeech'] < 1) & 
                (master_df['PoorAudioQuality'] < 1)
            ]
            print(f"[DataManager] Quality Filter: Removed {initial_count - len(master_df)} low-quality samples.")
        
        # 2. Binary Target Generation
        # Standard Research Rule: (NoStutteredWords < 2) => Stutter(1)
        master_df['target'] = (master_df['NoStutteredWords'] < 2).astype(int)
        
        # 3. Create Search Key
        master_df['key'] = master_df['Show'].astype(str) + '_' + \
                          master_df['EpId'].astype(str) + '_' + \
                          master_df['ClipId'].astype(str)
                          
        return master_df.set_index('key')['target'].to_dict()
