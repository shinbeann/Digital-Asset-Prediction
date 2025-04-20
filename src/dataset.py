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
    def __init__(self, csv_file: str, seq_length: int = 14, stride: int = 1):
        """
        Cryptocurrency dataset with lazy loading.
        
        Args:
            csv_file (str): Path to the CSV file.
            seq_length (int, optional): Size of the input sequence (number of time steps). Defaults to 14 (two weeks).
            stride (int, optional): Number of steps between consecutive sequences. Defaults to 1.
        """
        self.seq_length = seq_length
        self.stride = stride
        
        # Load CSV file into a DataFrame
        self.df = pd.read_csv(csv_file) 
        self.df['date'] = pd.to_datetime(self.df['date']) # Datetime conversion
        self.df = self.df.sort_values(['symbol', 'date']) # Should already be sorted but just in case
        self.feature_cols = [col for col in self.df.columns if col not in ['date', 'symbol']]
        self.num_features = len(self.feature_cols)
        self.target_col = 'close' # Closing price of next time step
        
        # Precompute all valid sequence start indices (for lazy loading)
        self.sequence_indices = []
        
        for crypto, group in self.df.groupby('symbol'):
            # Create sequences: using a sliding window over the group's rows
            for i in range(0, len(group) - self.seq_length, self.stride):
                # Window contains both the input sequence (seq_length) and the next time step (for the target)
                # Hence the window size is (seq_length + 1) 
                self.sequence_indices.append(group.index[i])
                
        self.sequence_indices = np.array(self.sequence_indices, dtype=np.int64)
        
        # Symbol to index map (symbol NOT used for prediction)
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.df['symbol'].unique())}
        self.idx_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_idx.items()}
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.seq_length
        
        # Get input sequence features
        sequence_features = self.df.iloc[start_idx: end_idx][self.feature_cols].values
        # Get target: next time step's closing price
        target = self.df.iloc[end_idx][self.target_col]
        # Get crypto type of the sequence (NOT used for prediction)
        crypto_symbol = self.df.iloc[start_idx]['symbol']
        symbol_idx = self.symbol_to_idx[crypto_symbol]
        
        X = torch.tensor(sequence_features, dtype=torch.float32) # Shape: (seq_length, num_features)
        y = torch.tensor(target, dtype=torch.float32) # Scalar target
        symbol_idx = torch.tensor(symbol_idx, dtype=torch.long)
        
        return X, y, symbol_idx
    
    
class Normalizer:
    """Normalization utility for standardizing input features."""
    def __init__(self, training_dataset: Optional[CryptoDataset] = None):
        """
        Constructs a `Normalizer` instance.
        """
        self.mean = None
        self.std = None
        
        if training_dataset:
            self.fit(training_dataset)
        
    def to(self, device: torch.device):
        """Move normalization statistics to the specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization to input tensor."""
        # Move normalization statistics to same device as input
        if (x.device != self.mean.device) or (x.device != self.std.device):
            self.to(x.device)
        
        x_norm = (x - self.mean) / (self.std + 1e-8)
        return x_norm
    
    def fit(self, training_dataset: CryptoDataset | Subset) -> Dict[str, torch.Tensor]:
        """
        Compute the mean and standard deviation of the input training dataset for future normalization. Returns the two
        values and also updates its corresponding attributes.

        Args:
            training_dataset (CryotoDataset | Subset): Training dataset to fit to (compute normalization statistics).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing computed mean and standard deviation values.
        """
        # Compute normalization statistics from given training dataset
        all_features = [
            training_dataset[i][0] # X of shape (sequence_size, num_features)
            for i in range(len(training_dataset))
        ]
        
        # Stack all sequences along the time dimension
        all_features = torch.cat(all_features, dim=0)  # Shape: (total_time_steps, num_features)
        
        # Compute mean and std per feature
        self.mean = torch.mean(all_features, dim=0)  # Shape: (num_features,)
        self.std = torch.std(all_features, dim=0) # Shape: (num_features,)
        return {'mean': self.mean, 'std': self.std}