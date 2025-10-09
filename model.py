import torch
import torch.nn as nn
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), stride=(5, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1920, hidden_size),
            nn.Tanh(),
        )
    
    def forward(self, y):
        y_emb = self.embedder(y)
        return y_emb
    

class UNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)