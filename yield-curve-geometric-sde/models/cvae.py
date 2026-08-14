"""Conditional VAE for curve + LevelScript (or other) context."""

import torch
import torch.nn as nn


class CVAE(nn.Module):
    """CVAE with condition vector (e.g. LevelScript level)."""

    def __init__(self, n_tenors: int, cond_dim: int, latent_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_tenors + cond_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tenors),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar
