import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    ):
        super().__init__()
        self.embedding_layer = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, embed_dim))

    
    def forward(self, input):
        x = self.embedding_layer(input).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed
        return x
