import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.h_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.linear = nn.Linear(dim, dim)
        self.scale = num_heads ** -0.5



    def forward(self, x):
        B, N, embed = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.h_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # for each B, num_heads, N, h_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        attn_output = attn @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, N, embed)

        output = self.linear(attn_output)
        output = self.dropout(output)

        return output
