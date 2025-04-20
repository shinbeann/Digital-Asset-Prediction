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
import torch.nn.functional as F


##############
# Base Class #
##############

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


##############
# RNN Models #
##############

class CryptoGRU(CryptoBaseModel):
    """GRU-based cryptocurrency price predictor"""
    def __init__(
        self, 
        input_size: int = 11, 
        hidden_size: int = 64, 
        num_layers: int = 3,
        dropout_prob: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Fully connected layer to map output hidden state to predicted closing price
        self.fc_input_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(self.fc_input_size, 1)
        
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
        num_layers: int = 2, 
        dropout_prob: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
         # Fully connected layer to map output hidden state to predicted closing price
        self.fc_input_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(self.fc_input_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step's output (final hidden state)
        last_out = lstm_out[:, -1, :]
        
        # Map the LSTM's output to a single prediction value (closing price of next time step)
        pred = self.fc(last_out) # Shape: (batch_size, 1)
        pred = pred.squeeze(-1) # MSELoss expects shape (batch_size,)
        return pred


##########################
# Base Transformer Model #
##########################

class CryptoTransformer(CryptoBaseModel):
    """Transformer-based (encoder-only) cryptocurrency price predictor"""
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64, # i.e. d_model
        num_layers: int = 4,
        dropout_prob: float = 0.1,
        num_heads: int = 4,
        dim_feedforward: int = 256, # i.e. d_ff
        max_sequence_length: int = 500
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_sequence_length = max_sequence_length
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Add learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
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
        
        # Prepend CLS token to each input sequence (learns to capture aggregate sequence-level information)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1) # (batch_size, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1) # (batch_size, seq_len+1, hidden_size)
        
        # Add positional encoding
        x = self.positional_encoder(x)
        
        # Transformer processing (obtain refined, contextualized embedding vectors for each time step)
        x = self.transformer_encoder(x) # (batch_size, seq_length, hidden_size)
        
        # Use the CLS token as input to linear processing layer
        cls_out = x[:, 0, :] # (batch_size, hidden_size)
        
        # Map the Encoder's last output to a single prediction value (closing price of next time step)
        pred = self.fc(cls_out) # Shape: (batch_size, 1)
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


##################
# Informer Model #
##################

class CryptoInformer(CryptoBaseModel):
    """
    `CryptoInformer` is an encoder-only transformer model for cryptocurrency price prediction that implements 
    innovations from the 'Informer' model architecture as detailed by 
    Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting".

    The paper can be found [here](https://arxiv.org/abs/2012.07436).
    """
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64, # i.e. d_model
        num_layers: int = 4,
        dropout_prob: float = 0.1,
        num_heads: int = 4,
        dim_feedforward: int = 256, # i.e. d_ff
        probsparse_sampling_factor: int = 5,
        distil: bool = False, # Whether to use distilling between layers
        max_sequence_length: int = 500
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob)
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_sequence_length = max_sequence_length
        self.probsparse_sampling_factor = probsparse_sampling_factor
        self.distil = distil
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Add learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Positional encoder to inject sequence order information
        # Needed as self-attention mechanisms of Transformers are inherently permutation-invariant
        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=self.hidden_size, 
            dropout=self.dropout_prob, 
            max_len=self.max_sequence_length
        )
        
        # Informer Encoder for prob-sparse self-attention
        self.attn = ProbAttention(
            mask_flag=False, 
            factor=self.probsparse_sampling_factor, 
            attention_dropout=self.dropout_prob
        )
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                attention=AttentionLayer(
                    attention=self.attn,
                    d_model=self.hidden_size,
                    n_heads=self.num_heads
                ),
                d_model=self.hidden_size,
                d_ff=self.dim_feedforward,
                dropout=self.dropout_prob,
                activation="relu"
            )
            for _ in range(num_layers)
        ])
        
        # Optional convolutional layers for sequence distillation (effectively reduces length)
        self.distil_layers = nn.ModuleList([
            ConvLayer(c_in=self.hidden_size)
            for _ in range(num_layers - 1) # Distillation happens between attention layers, so one less distil layer
        ]) if self.distil else None
        
        # Maps refined embedding of last input time step to a single prediction value (closing price of next time step)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        
        # Embed input features
        x = self.embedding(x) # (batch_size, seq_length, hidden_size)
        
        # Prepend CLS token to each input sequence (learns to capture aggregate sequence-level information)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1) # (batch_size, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1) # (batch_size, seq_len+1, hidden_size)
        
        # Add positional encoding
        x = self.positional_encoder(x)
        
        # Process through encoder layers (obtain refined, contextualized embedding vectors for each time step)
        for i, encoder_layer in enumerate(self.encoder_layers):
            # Self-attention + FFN
            x, attn = encoder_layer(x, attn_mask=attn_mask)
            
            # Distilling between layers
            if self.distil and i < len(self.encoder_layers) - 1: # for all but last encoder layer
                x = self.distil_layers[i](x)
        
        # Use the CLS token as input to linear processing layer
        cls_out = x[:, 0, :] # (batch_size, hidden_size)
        
        # Map the Encoder's last output to a single prediction value (closing price of next time step)
        pred = self.fc(cls_out) # Shape: (batch_size, 1)
        pred = pred.squeeze(-1) # MSELoss expects shape (batch_size,)
        return pred
    
"""
The code below for the encoder components (`EncoderLayer`, `ProbAttention`, `AttentionLayer`, and `ConvLayer`) were 
taken DIRECTLY from the Informer paper's official [github repository](https://github.com/zhouhaoyi/Informer2020/).
"""

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn
    

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbAttention.ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn
    
    class ProbMask():
        def __init__(self, B, H, L, index, scores, device="cpu"):
            _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
            _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
            indicator = _mask_ex[torch.arange(B)[:, None, None],
                                torch.arange(H)[None, :, None],
                                index, :].to(device)
            self._mask = indicator.view(scores.shape).to(device)
        
        @property
        def mask(self):
            return self._mask

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
