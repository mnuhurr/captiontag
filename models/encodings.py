import math
import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_steps: int):
        super().__init__()

        position = torch.arange(max_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_steps, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume x.shape is (batch, t, d_model)
        return x + self.pe[:, :x.size(1)]


class TimeFrequencyEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_steps: int, n_freq_patches: int, dropout: float = 0.0, time_freq_pos_offset: int = 1):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        time_pos = torch.arange(max_steps) // n_freq_patches
        offset = time_pos[-1] + time_freq_pos_offset
        freq_pos = (torch.arange(max_steps) % n_freq_patches) + offset

        self.last_pos = torch.max(freq_pos)

        te = self.make_encoding(time_pos, d_model)
        fe = self.make_encoding(freq_pos, d_model)
        self.register_buffer('te', te)
        self.register_buffer('fe', fe)

    def forward(self, x: torch.Tensor):
        x = x + self.te[0, :x.size(1)]
        x = x + self.fe[0, :x.size(1)]
        return self.dropout(x)

    @staticmethod
    def make_encoding(position: torch.Tensor, d_model: int):
        position = position.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, len(position), d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


