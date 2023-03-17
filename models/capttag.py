
import torch
import torch.nn.functional as F

from .encodings import PositionalEncoding
from .transformer import TransformerEncoder

from typing import Optional



class CaptionTagger(torch.nn.Module):
    def __init__(self, 
                 n_labels: int,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 d_ff: Optional[int] = None,
                 max_sequence_length: int = 512,
                 dropout: float = 0.0):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_steps=max_sequence_length)

        self.caption_encoder = TransformerEncoder(
            d_model=d_model,
            d_ff=d_ff if d_ff is not None else 4 * d_model,
            n_heads=n_heads,
            n_layers=n_layers)
        
        self.label_embedding = torch.nn.Embedding(n_labels, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.layernorm = torch.nn.LayerNorm(d_model)
        self.tag = torch.nn.Linear(d_model, n_labels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x, _ = self.caption_encoder(x, mask=mask)

        x = self.dropout(x)
        x = self.layernorm(x)
        x = self.tag(x[:, 0, :])
        
        # add sigmoid if needed 

        return x
