####################
# Required Modules #
####################

# Generic/Built-in
from typing import *

# Libs
import numpy as np
import torch
import torch.nn as nn


class CryptoGRU(nn.Module):
    def __init__(
        self, 
        input_size: int = 4, 
        embed_dim: int = 4, 
        hidden_size: int = 64, 
        num_layers: int = 1, 
        num_crypto: int = 4,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            input_size (int, optional): Number of input features.. Defaults to 4.
            embed_dim (int, optional): Dimension of the crypto category embedding. Defaults to 4.
            hidden_size (int, optional): Hidden size for the GRU. Defaults to 64.
            num_layers (int, optional): Number of GRU layers. Defaults to 1.
            num_crypto (int, optional): Number of distinct cryptocurrencies. Defaults to 4.
            device (Optional[torch.device]): Device to run the model on (CPU or GPU).
        """
        super(CryptoGRU, self).__init__()
        self.embed_dim = embed_dim
        # Embedding layer for the crypto type (e.g. "BTC", "ETH", etc.).
        self.crypto_embedding = nn.Embedding(num_crypto, embed_dim)
        
        # Since we are concatenating the embedding to every time step,
        # the effective input size to the GRU becomes: original features + embed_dim.
        self.gru = nn.GRU(
            input_size + embed_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        
        # Fully connected layer to map GRU output to the predicted price.
        self.fc = nn.Linear(hidden_size, 1)
        
        # Move model to device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")
        
    def forward(self, x: torch.Tensor, crypto_type: torch.Tensor):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_length, input_size).
            crypto_type (torch.Tensor): Tensor of crypto indices of shape (batch_size,).

        Returns:
            _type_: Predicted next-day closing price (batch_size,).
        """
        batch_size, seq_length, _ = x.size() # i.e. unpack input tensor shape
        
        # Get the embedding for each sample in the batch (shape: (batch_size, embed_dim))
        crypto_embed = self.crypto_embedding(crypto_type)
        
        # Expand the embedding to be concatenated with each time step in the sequence:
        # from (batch_size, embed_dim) to (batch_size, seq_length, embed_dim)
        crypto_embed = crypto_embed.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Concatenate the original input with the crypto embedding along the feature dimension.
        x_cat = torch.cat([x, crypto_embed], dim=2)  # New shape: (batch_size, seq_length, input_size + embed_dim)
        
        # Process the concatenated sequence with the GRU.
        gru_out, _ = self.gru(x_cat)
        # Use the output of the last time step.
        last_output = gru_out[:, -1, :]
        
        # Map the GRU's output to a single prediction value (closing price of next time step).
        output = self.fc(last_output) # Shape: (batch_size, 1)
        output = output.squeeze(-1) # PyTorch MSELoss expects shape (batch_size,)
        return output