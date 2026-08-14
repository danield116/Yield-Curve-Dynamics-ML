"""SDE integration helpers."""

import torch


def euler_maruyama_step(z: torch.Tensor, drift: torch.Tensor, diffusion: torch.Tensor, dt: float) -> torch.Tensor:
    """Single Euler-Maruyama step for latent state update."""
    dW = torch.randn_like(z) * (dt**0.5)
    return z + drift * dt + diffusion * dW
