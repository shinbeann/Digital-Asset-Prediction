####################
# Required Modules #
####################

# Generic/Built-in
from typing import *
from abc import ABC, abstractmethod
import math

# Libs
import numpy as np
import torch
import torch.nn as nn


class CryptoBaseModel(nn.Module, ABC):
    """Abstract base class for crypto prediction models"""
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob if num_layers > 1 else 0.0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class CryptoGRU(CryptoBaseModel):
    """GRU-based cryptocurrency price predictor"""
    def __init__(
        self, 
        input_size: int = 11, 
        hidden_size: int = 64, 
        num_layers: int = 1, 
        dropout_prob: float = 0.1
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0
        )
        
        # Fully connected layer to map output hidden state to predicted closing price
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        gru_out, _ = self.gru(x)
        # Take last time step's output (final hidden state)
        last_out = gru_out[:, -1, :]
        
        # Map the GRU's output to a single prediction value (closing price of next time step)
        pred = self.fc(last_out) # Shape: (batch_size, 1)
        pred = pred.squeeze(-1) # MSELoss expects shape (batch_size,)
        return pred
    

class CryptoLSTM(CryptoBaseModel):
    """LSTM-based cryptocurrency price predictor"""
    def __init__(
        self, 
        input_size: int = 11, 
        hidden_size: int = 64, 
        num_layers: int = 1, 
        dropout_prob: float = 0.1
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0
        )
        
        # Fully connected layer to map output hidden state to predicted closing price
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step's output (final hidden state)
        last_out = lstm_out[:, -1, :]
        
        # Map the LSTM's output to a single prediction value (closing price of next time step)
        pred = self.fc(last_out) # Shape: (batch_size, 1)
        pred = pred.squeeze(-1) # MSELoss expects shape (batch_size,)
        return pred


class CryptoTransformer(CryptoBaseModel):
    """Transformer-based (encoder-only) cryptocurrency price predictor"""
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64, # i.e. d_model
        num_layers: int = 2,
        dropout_prob: float = 0.1,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        max_sequence_length: int = 500
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_sequence_length = max_sequence_length
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoder to inject sequence order information
        # Needed as self-attention mechanisms of Transformers are inherently permutation-invariant
        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=self.hidden_size, 
            dropout=self.dropout_prob, 
            max_len=self.max_sequence_length
        )
        
        # Transformer encoder for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Maps refined embedding of last input time step to a single prediction value (closing price of next time step)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        
        # Embed input features
        x = self.embedding(x) # (batch_size, seq_length, hidden_size)
        
        # Add positional encoding
        x = self.positional_encoder(x)
        
        # Transformer processing (obtain refined, contextualized embedding vectors for each time step)
        transformer_out = self.transformer_encoder(x) # (batch_size, seq_length, hidden_size)
        
        # Use only the last time step's output (i.e. embedding of last token)
        last_out = transformer_out[:, -1, :] # (batch_size, hidden_size)
        
        # Map the Encoder's last output to a single prediction value (closing price of next time step)
        pred = self.fc(last_out) # Shape: (batch_size, 1)
        pred = pred.squeeze(-1) # MSELoss expects shape (batch_size,)
        return pred

    
class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds positional encodings to input sequences to inject information about token positions. 
    This class uses sinusoidal functions (sine and cosine) to generate encodings, following the original Transformer 
    architecture. The encodings are added to the input embeddings, and dropout is applied for regularization.
    
    Code taken from an 
    [official PyTorch tutorial](https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html)
    and adapted such that batch dimension comes first (batch_first = True).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Transpose pe to [seq_len, 1, d_model] then add to x
        x = x + self.pe[:x.size(1)].transpose(0, 1)  # [1, seq_len, d_model]
        return self.dropout(x)