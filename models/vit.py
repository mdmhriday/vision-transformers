import torch
import torch.nn as nn
from .components.patch_embedding.patch_embedding_vit import PatchEmbedding
from .components.transformer.transformer import Transformer

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size, 
        patch_size, 
        in_channels, 
        num_classes, 
        embed_dim=768, 
        num_heads=8, 
        num_layers=12, 
        hidden_dim=None, 
        dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer = Transformer(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.cls_token = self.patch_embed.class_token
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.transformer(x)
        cls_token = x[:, 0]
        output = self.fc(cls_token)

        return output