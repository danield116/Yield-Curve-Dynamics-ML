"""Bond/yield utility functions for no-arbitrage diagnostics and PDE terms."""

import torch


# MVP tenor grid in years: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
DEFAULT_TENOR_YEARS = [1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]


def default_tau_grid(device=None, dtype=torch.float32):
    """Return fixed maturity grid tensor [N_tenors]."""
    return torch.tensor(DEFAULT_TENOR_YEARS, device=device, dtype=dtype)


def _broadcast_tau(tau, batch_size, device, dtype):
    """Ensure tau has shape [B, N_tenors]."""
    if not torch.is_tensor(tau):
        tau = torch.tensor(tau, device=device, dtype=dtype)
    if tau.ndim == 1:
        tau = tau.unsqueeze(0).expand(batch_size, -1)
    return tau


def yield_to_discount(y, tau):
    """Zero-coupon price from yields: P(t,T) = exp(-y(t,T) * tau).

    Shapes:
    - y:   [B, N_tenors]
    - tau: [N_tenors] or [B, N_tenors]
    - P:   [B, N_tenors]
    """
    tau = _broadcast_tau(tau, y.shape[0], y.device, y.dtype)
    return torch.exp(-y * tau)


def discount_to_instant_forward(discount, tau):
    """Approximate instantaneous forward curve from discount function.

    f(t,T) = -d(log P)/dT, implemented with stable discrete differences.
    """
    tau = _broadcast_tau(tau, discount.shape[0], discount.device, discount.dtype)
    log_p = torch.log(discount.clamp_min(1e-10))

    if tau.shape[1] < 2:
        return torch.zeros_like(discount)

    # Central difference in maturity for interior points.
    d_logp = (log_p[:, 2:] - log_p[:, :-2]) / (tau[:, 2:] - tau[:, :-2]).clamp_min(1e-8)
    left = (log_p[:, 1:2] - log_p[:, 0:1]) / (tau[:, 1:2] - tau[:, 0:1]).clamp_min(1e-8)
    right = (log_p[:, -1:] - log_p[:, -2:-1]) / (tau[:, -1:] - tau[:, -2:-1]).clamp_min(1e-8)
    d_logp_full = torch.cat([left, d_logp, right], dim=1)
    return -d_logp_full


def short_rate_from_curve(y, short_index=0):
    """Proxy short rate from the shortest tenor yield.

    Shapes:
    - y: [B, N_tenors]
    - r: [B, 1]
    """
    return y[:, short_index : short_index + 1]


def bond_price_from_decoder(z, tau, decoder):
    """Bond prices implied by decoder yields at latent state z.

    Shapes:
    - z:   [B, latent_dim]
    - tau: [N_tenors] or [B, N_tenors]
    - P:   [B, N_tenors]
    """
    y = decoder(z)
    return yield_to_discount(y, tau)
