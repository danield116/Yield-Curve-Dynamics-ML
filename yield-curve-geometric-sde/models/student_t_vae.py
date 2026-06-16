"""Student-t VAE/CVAE scaffold for heavy-tailed yield moves."""

import torch
import torch.nn as nn

from models.cvae import CVAE


class StudentTCVAE(CVAE):
    """Extends CVAE with Student-t decoder likelihood parameters.

    TODO:
    - Learn degrees-of-freedom `nu` globally or per tenor.
    - Implement Student-t negative log-likelihood.
    """

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

    def student_t_nll(self, x, x_hat, log_scale):
        """Placeholder Student-t NLL.

        Shapes:
        - x, x_hat, log_scale: [B, N_tenors]
        """
        # TODO: implement exact closed-form Student-t NLL.
        scale = torch.exp(log_scale)
        residual = (x - x_hat) / (scale + 1e-8)
        return (residual**2).mean()
