"""Standard VAE scaffold for yield-curve manifold learning."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Basic VAE.

    Shapes:
    - x: [batch, n_tenors]
    - z: [batch, latent_dim]
    - x_hat: [batch, n_tenors]
    """

    def __init__(self, n_tenors: int, latent_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        # TODO: improve architecture depth/regularization.
        self.encoder = nn.Sequential(
            nn.Linear(n_tenors, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tenors),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
