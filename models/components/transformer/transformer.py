import torch
import torch.nn as nn
from ..attention.attention import MultiHeadAttention


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        return self.layers(input)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffb = FeedForwardBlock(dim, hidden_dim or dim*4, dropout=dropout)

    def forward(self, input):
        x = self.norm1(input)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.ffb(x)
        return x

class Transformer(nn.Module):
    def __init__(Self, num_layers, dim, num_heads=8, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(input)
            return x