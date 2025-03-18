####################
# Required Modules #
####################

# Generic/Built-in
from typing import *

# Libs
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torch.nn as nn


class CryptoDataset(Dataset):
    def __init__(self, csv_file: str, seq_length: int = 5):
        """
        Args:
            csv_file (str): Path to the CSV file.
            seq_length (int, optional): Number of time steps in each input sequence. Defaults to 5.
        """
        # Values for Kaggle Crypto dataset
        self.seq_list = seq_length
        self.feature_cols = ['Open', 'High', 'Low', 'Close']
        
        # Load CSV file into a DataFrame
        self.df = pd.read_csv(csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date']) # Datetime conversion
        self.df = self.df.sort_values(['Crypto', 'Date']) # Sorted by crypto type and then by date
        
        # Map each crypto to an integer (categorical index)
        self.crypto_list = self.df['Crypto'].unique().tolist()
        self.crypto_to_idx = {crypto: idx for idx, crypto in enumerate(self.crypto_list)}
        # {'BTC': 0, 'ETH': 1, 'LTC': 2, 'XRP': 3}
        
        # Construct sequences (each sequence comes from a single crypto)
        self.sequences = []  # Will store tuples: (sequence_features, target, crypto_index)
        
        # Generate sequences for each crypto
        # TODO: add a stride parameter, save indexes instead (avoid redundant copying)
        for crypto, group in self.df.groupby('Crypto'):
            group = group.reset_index(drop=True)
            # Create sequences: using a sliding window over the group's rows
            for i in range(len(group) - seq_length):
                # Input: sequence of features for seq_length consecutive days
                seq_features = group.loc[i:i+seq_length-1, self.feature_cols].values.astype(np.float32)
                # Target: next day’s 'Close' price (many-to-one prediction)
                target = group.loc[i+seq_length, 'Close']
                self.sequences.append((seq_features, np.float32(target), self.crypto_to_idx[crypto]))
                
        # TODO: instead of lists, use numpy arrays
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target, crypto_idx = self.sequences[idx]
        # Convert numpy arrays to PyTorch tensors
        seq = torch.tensor(seq, dtype=torch.float32) # Shape: (seq_length, num_features)
        target = torch.tensor(target, dtype=torch.float32) # Scalar target
        crypto_idx = torch.tensor(crypto_idx, dtype=torch.long) # Categorical index for crypto
        return seq, target, crypto_idx


class Normalizer:
    def __init__(self, training_dataset: Optional[CryptoDataset | Subset] = None):
        # Normalization statistics
        self.mean = 0
        self.std = 0
        
        if training_dataset:
            self.fit(training_dataset)
        
    def fit(self, training_dataset: CryptoDataset):
        """
        We compute normalization statistics across all sequences from training dataset, including duplicate time steps 
        from overlapping windows, to match the data distribution seen by the model during training.

        Args:
            training_dataset (CryptoDataset): Training dataset to compute normalization statistics from. 
        """
        
        # Compute normalization statistics from given training dataset
        all_features = [
            training_dataset[i][0] # X of shape (seq_length, num_features)
            for i in range(len(training_dataset))
        ]

        # Stack all sequences along the time dimension
        all_features = torch.cat(all_features, dim=0)  # Shape: (total_time_steps, num_features)
        
        # Compute mean and std per feature
        self.mean = torch.mean(all_features, dim=0)  # Shape: (num_features,)
        self.std = torch.std(all_features, dim=0)  # Shape: (num_features,)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization to input tensor."""
        # x is of shape (x - self.mean) / self.std
        x_norm = (x - self.mean) / (self.std + 1e-8)
        return x_norm