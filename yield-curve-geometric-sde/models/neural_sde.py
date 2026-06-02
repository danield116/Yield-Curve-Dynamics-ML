"""Neural SDE scaffold over latent states."""

import torch
import torch.nn as nn


class LatentNeuralSDE(nn.Module):
    """Latent dynamics:
    dz_t = mu_P(z_t) dt + sigma(z_t) dW_t
    mu_Q(z_t) = mu_P(z_t) - sigma(z_t) lambda(z_t)
    """

    def __init__(self, latent_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mu_p = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.lambda_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def drift_p(self, z: torch.Tensor) -> torch.Tensor:
        return self.mu_p(z)

    def diffusion(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: enforce positive diffusion and/or matrix structure.
        return torch.nn.functional.softplus(self.sigma_net(z))

    def drift_q(self, z: torch.Tensor) -> torch.Tensor:
        mu_p = self.drift_p(z)
        sigma = self.diffusion(z)
        lam = self.lambda_net(z)
        return mu_p - sigma * lam

    def euler_maruyama(self, z0: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
        """Simulate paths under P-measure (placeholder).

        Input:
        - z0: [B, latent_dim]
        Output:
        - paths: [B, n_steps + 1, latent_dim]
        """
        z = z0
        out = [z]
        for _ in range(n_steps):
            dW = torch.randn_like(z) * (dt**0.5)
            z = z + self.drift_p(z) * dt + self.diffusion(z) * dW
            out.append(z)
        return torch.stack(out, dim=1)
