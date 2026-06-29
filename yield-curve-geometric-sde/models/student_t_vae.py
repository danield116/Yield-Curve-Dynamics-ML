"""Student-t VAE/CVAE scaffold for heavy-tailed yield moves."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cvae import CVAE


class StudentTCVAE(CVAE):
    """Extends CVAE with Student-t decoder likelihood parameters."""

    def __init__(self, n_tenors: int, cond_dim: int, latent_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__(n_tenors=n_tenors, cond_dim=cond_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.log_scale_head = nn.Linear(hidden_dim, n_tenors)
        self.raw_nu = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_hat = self.decode(z, c)
        log_scale = self.log_scale_head(h)
        return x_hat, mu, logvar, log_scale

    def degrees_of_freedom(self):
        """Ensure nu > 2 for finite variance."""
        return F.softplus(self.raw_nu) + 2.01

    def student_t_nll(self, x, x_hat, log_scale):
        """Closed-form Student-t negative log-likelihood (per-element mean)."""
        nu = self.degrees_of_freedom()
        scale = torch.exp(log_scale).clamp_min(1e-8)
        z = (x - x_hat) / scale
        t_term = 0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
        const = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * math.pi)
        )
        nll = const - torch.log(scale) + t_term
        return nll.mean()
