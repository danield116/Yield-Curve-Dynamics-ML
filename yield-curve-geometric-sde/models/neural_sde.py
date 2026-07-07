"""Neural SDE over latent states with optional latent-history conditioning."""

import torch
import torch.nn as nn


class LatentNeuralSDE(nn.Module):
    """Latent dynamics with history-conditioned drift/diffusion.

    Networks take the flattened last `history_steps` latent vectors.
    The state being advanced remains the current latent z_t in R^{latent_dim}.
    """

    def __init__(self, latent_dim: int = 3, hidden_dim: int = 64, history_steps: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.history_steps = max(1, int(history_steps))
        input_dim = latent_dim * self.history_steps

        self.mu_p = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.lambda_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def drift_p(self, sde_input: torch.Tensor) -> torch.Tensor:
        return self.mu_p(sde_input)

    def diffusion(self, sde_input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.sigma_net(sde_input))

    def drift_q(self, sde_input: torch.Tensor) -> torch.Tensor:
        mu_p = self.drift_p(sde_input)
        sigma = self.diffusion(sde_input)
        lam = self.lambda_net(sde_input)
        return mu_p - sigma * lam

    def euler_maruyama(self, z_hist: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
        """Simulate paths under P-measure from a latent history window."""
        z_window = z_hist
        z = z_window[:, -1, :]
        out = [z]
        for _ in range(n_steps):
            sde_input = self._build_input(z_window)
            dW = torch.randn_like(z) * (dt**0.5)
            z = z + self.drift_p(sde_input) * dt + self.diffusion(sde_input) * dW
            z_window = torch.cat([z_window[:, 1:, :], z.unsqueeze(1)], dim=1)
            out.append(z)
        return torch.stack(out, dim=1)

    def _build_input(self, z_hist: torch.Tensor) -> torch.Tensor:
        batch, t_len, dim = z_hist.shape
        k = min(self.history_steps, t_len)
        chunk = z_hist[:, -k:, :]
        if k < self.history_steps:
            pad = torch.zeros(batch, self.history_steps - k, dim, device=z_hist.device, dtype=z_hist.dtype)
            chunk = torch.cat([pad, chunk], dim=1)
        return chunk.reshape(batch, self.history_steps * dim)
