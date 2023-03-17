
import math
import torch
import torch.nn.functional as F
from .encodings import PositionalEncoding

from typing import Optional, Tuple, Dict, List



class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ff_ln = torch.nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # check mask
        if mask is not None and mask.is_floating_point():
            mask = mask != 0

        # 1. self attention
        x = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out

        # 2. ff network
        x = x + self.ff(self.ff_ln(x))

        return x, attn_weights.detach()



class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff: int, n_heads: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_weights = []

        if mask is not None and torch.is_floating_point(mask):
            mask = mask != 0

        for layer in self.layers:
            x, w = layer(x, mask=mask)
            attn_weights.append(w)

        return x, attn_weights


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, max_sequence_len: int, causal: bool = True):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(d_model)

        self.cross_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_ln = torch.nn.LayerNorm(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ff_ln = torch.nn.LayerNorm(d_model)

        # causal/lookback mask
        self.causal = causal
        if causal:
            mask = torch.empty(max_sequence_len, max_sequence_len).fill_(-float('inf')).triu(1)
            self.register_buffer('causal_mask', mask.to(torch.bool), persistent=False)

    def forward(self,
                x: torch.Tensor,
                xa: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                xa_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # 1. self attention
        x = self.attn_ln(x)
        
        # 2. causal mask
        causal_mask = self.causal_mask[:x.size(1), :x.size(1)] if self.causal else None
        attn_out, self_attn_weights = self.attn(x, x, x, key_padding_mask=mask, attn_mask=causal_mask)
        self_attn_weights = self_attn_weights.detach()
        x = x + attn_out

        # 3. optional cross attention
        cross_attn_weights = None
        if xa is not None:
            x = self.cross_attn_ln(x)
            attn_out, cross_attn_weights = self.cross_attn(query=x, key=xa, value=xa, key_padding_mask=xa_mask)
            x = x + attn_out
            cross_attn_weights = cross_attn_weights.detach()

        # 4. ff network
        x = x + self.ff(self.ff_ln(x))

        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff: int, n_heads: int, max_sequence_len: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, max_sequence_len) for _ in range(n_layers)])

    def forward(self,
                x: torch.Tensor,
                xa: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                xa_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        # do not collect layers into a list
        attn_weights = []
        x_attn_weights = []
        for layer in self.layers:
            x, w, xw = layer(x, xa, mask=mask, xa_mask=xa_mask)
            attn_weights.append(w)
            x_attn_weights.append(xw)

        return x, attn_weights, x_attn_weights


class TextDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, max_sequence_length: int, d_model: int, n_layers: int, n_heads: int, d_ff: Optional[int] = None): 
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length)

        self.decoder = TransformerDecoder(
            d_model=d_model, 
            n_layers=n_layers, 
            d_ff=d_ff if d_ff is not None else 4 * d_model,
            n_heads=n_heads,
            max_sequence_len=max_sequence_length)

        self.layernorm = torch.nn.LayerNorm(d_model)
       
    def forward(self, tokens: torch.Tensor, xa: Optional[torch.Tensor] = None, token_mask: Optional[torch.Tensor] = None, xa_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x = tokens, xa = encoder context
        x = self.token_embedding(tokens)
        x = self.positional_encoding(x)

        if token_mask is None:
            token_mask = torch.zeros_like(tokens, dtype=torch.bool)
        elif token_mask.is_floating_point():
            token_mask = token_mask.to(torch.bool)

        x, _, _ = self.decoder(x=x, xa=xa, mask=token_mask, xa_mask=xa_mask)

        x = self.layernorm(x)
        logits = x @ torch.transpose(self.token_embedding.weight, 0, 1)

        return logits


