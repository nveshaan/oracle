import torch
import torch.nn as nn
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class TrendlineEmbedder(nn.Module):
    """
    Embeds different trendlines into one single vector.
    """
    def __init__(self, input_channels, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.Tanh(),
            nn.Conv2d(32, 128, kernel_size=(1, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(3840, hidden_size),
            nn.Tanh(),
        )
    
    def forward(self, y):
        y_emb = self.embedder(y)
        return y_emb
    

class UNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()
        self.norm_final = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*output_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.linear(x)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.activation(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, nodes, input_channels=7, hidden_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = nodes
        self.hidden_size = hidden_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TrendlineEmbedder(input_channels, hidden_size)

        self.blocks = nn.ModuleList([
            UNetBlock(nodes[i], nodes[i+1], hidden_size) for i in range(len(nodes)-1)
        ])

    def forward(self, x, t, y):
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        return x