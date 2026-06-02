"""No-arbitrage penalty scaffold (diagnostics + PDE residual placeholders)."""

import torch

from constraints.bond_math import yield_to_discount


def discount_monotonicity_penalty(y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Penalize violations of non-increasing discount curve over maturity."""
    p = yield_to_discount(y, tau)  # [B, N]
    diffs = p[:, 1:] - p[:, :-1]
    return torch.relu(diffs).mean()


def forward_smoothness_penalty(forward: torch.Tensor) -> torch.Tensor:
    """Penalize rough forward curves as a soft regularizer."""
    second_diff = forward[:, 2:] - 2 * forward[:, 1:-1] + forward[:, :-2]
    return (second_diff**2).mean()


def pde_residual_placeholder(
    p: torch.Tensor,
    grad_z_p: torch.Tensor,
    hess_z_p: torch.Tensor,
    mu_q: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    dP_dtau: torch.Tensor,
) -> torch.Tensor:
    """Placeholder for:
    R_arb = -dP/dtau + grad_z(P)^T mu_Q + 0.5 Tr[sigma sigma^T Hess_z(P)] - rP
    """
    # TODO: implement full trace term with consistent latent diffusion matrix shape.
    residual = -dP_dtau + (grad_z_p * mu_q).sum(dim=-1) - r * p
    return (residual**2).mean()
