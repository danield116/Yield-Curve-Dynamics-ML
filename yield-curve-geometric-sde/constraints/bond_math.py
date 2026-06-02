"""Bond/yield utility functions for no-arbitrage diagnostics."""

import torch


def yield_to_discount(y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """P(t,T) = exp(-y(t,T) * T).

    Shapes:
    - y:   [B, N_tenors]
    - tau: [N_tenors]
    - P:   [B, N_tenors]
    """
    return torch.exp(-y * tau)


def discount_to_instant_forward(discount: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Approximate instantaneous forward curve from discount function."""
    # TODO: use numerically stable derivative wrt maturity.
    log_p = torch.log(discount.clamp_min(1e-10))
    d_logp = torch.gradient(log_p, spacing=(tau,), dim=-1)[0]
    return -d_logp
