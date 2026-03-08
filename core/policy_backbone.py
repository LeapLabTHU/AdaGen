import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class TimestepEmbedder(nn.Module):
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
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class AdaLN(nn.Module):
    def __init__(self, x_dim, c_dim):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * x_dim, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x)
        if x.ndim == 3:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return x * (1 + scale) + shift


class PolicyBackbone(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, hidden_size=512):
        super().__init__()

        self.x_embedder = nn.Linear(in_channels, hidden_size) if in_channels != hidden_size else None
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.adaLN1 = AdaLN(x_dim=hidden_size, c_dim=hidden_size)
        self.dw_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1, groups=hidden_size)
        nn.init.xavier_uniform_(self.dw_conv.weight)
        if self.dw_conv.bias is not None:
            nn.init.zeros_(self.dw_conv.bias)

        self.pw_conv = nn.Conv2d(hidden_size, 128, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.pw_conv.weight)
        if self.pw_conv.bias is not None:
            nn.init.zeros_(self.pw_conv.bias)

        self.linear1 = nn.Linear(64 * 128, 1024)
        self.adaLN2 = AdaLN(x_dim=1024, c_dim=hidden_size)

        self.linear2 = nn.Linear(1024, 512)
        self.adaLN3 = AdaLN(x_dim=512, c_dim=hidden_size)

        self.linear3 = nn.Linear(512, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        def _ada_ln_init(module):
            if isinstance(module, AdaLN):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)

        self.apply(_ada_ln_init)
        nn.init.constant_(self.linear3.weight, 0)
        nn.init.constant_(self.linear3.bias, 0)

    def forward(self, state):
        x = state["feat"]
        t = state["timestep"]

        if self.x_embedder is not None:
            x = self.x_embedder(x)

        c = self.t_embedder(t)

        x = self.adaLN1(x, c)
        x = x.permute(0, 2, 1).reshape(x.size(0), -1, 16, 16).contiguous()

        x = F.gelu(self.dw_conv(x))
        x = F.gelu(self.pw_conv(x))

        x = x.view(x.size(0), -1)

        x = F.gelu(self.adaLN2(self.linear1(x), c))
        x = F.gelu(self.adaLN3(self.linear2(x), c))
        return self.linear3(x)
