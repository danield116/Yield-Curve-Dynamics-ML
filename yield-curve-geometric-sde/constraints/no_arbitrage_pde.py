"""No-arbitrage diagnostics and PDE residual penalties."""

import torch

from constraints.bond_math import (
    bond_price_from_decoder,
    default_tau_grid,
    discount_to_instant_forward,
    short_rate_from_curve,
    yield_to_discount,
)


def discount_monotonicity_penalty(y, tau):
    """Penalize violations of non-increasing discount factors over maturity."""
    p = yield_to_discount(y, tau)
    diffs = p[:, 1:] - p[:, :-1]
    return torch.relu(diffs).mean()


def forward_smoothness_penalty(forward):
    """Penalize rough forward curves via second-difference energy."""
    if forward.shape[1] < 3:
        return torch.tensor(0.0, device=forward.device, dtype=forward.dtype)
    second_diff = forward[:, 2:] - 2 * forward[:, 1:-1] + forward[:, :-2]
    return (second_diff**2).mean()


def arbitrage_diagnostic_loss(y, tau):
    """Cheap no-arbitrage diagnostics combined into one scalar loss."""
    p = yield_to_discount(y, tau)
    f = discount_to_instant_forward(p, tau)
    mono = discount_monotonicity_penalty(y, tau)
    smooth = forward_smoothness_penalty(f)
    return mono + smooth


def compute_dP_dtau(y, p, tau):
    """dP/dtau = -y * P for P = exp(-y * tau)."""
    tau = tau if tau.ndim == 2 else tau.unsqueeze(0).expand_as(y)
    return -y * p


def compute_grad_z_p(z, tau, decoder):
    """dP/dz, shape [B, N_tenors, latent_dim]."""
    z = z.detach().requires_grad_(True)
    p = bond_price_from_decoder(z, tau, decoder)
    batch_size, n_tenors = p.shape
    latent_dim = z.shape[1]
    grad_z_p = torch.zeros(batch_size, n_tenors, latent_dim, device=z.device, dtype=z.dtype)

    for tenor_idx in range(n_tenors):
        grad_z = torch.autograd.grad(
            p[:, tenor_idx].sum(),
            z,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_z_p[:, tenor_idx, :] = grad_z

    return grad_z_p


def compute_hessian_trace_term(z, tau, decoder, sigma, tenor_indices=None):
    """0.5 * Tr[sigma sigma^T Hess_z(P)] with diagonal diffusion; optional tenor subset."""
    if tenor_indices is None:
        tenor_indices = [0]

    z = z.detach().requires_grad_(True)
    p = bond_price_from_decoder(z, tau, decoder)
    batch_size = z.shape[0]
    latent_dim = z.shape[1]
    trace_term = torch.zeros(batch_size, p.shape[1], device=z.device, dtype=z.dtype)

    for tenor_idx in tenor_indices:
        grad_z = torch.autograd.grad(
            p[:, tenor_idx].sum(),
            z,
            create_graph=True,
            retain_graph=True,
        )[0]

        diag_hess = torch.zeros(batch_size, latent_dim, device=z.device, dtype=z.dtype)
        for dim_idx in range(latent_dim):
            h_ij = torch.autograd.grad(
                grad_z[:, dim_idx].sum(),
                z,
                retain_graph=True,
                create_graph=True,
            )[0][:, dim_idx]
            diag_hess[:, dim_idx] = h_ij

        # Diagonal diffusion approximation: sum_j sigma_j^2 * d2P/dz_j^2
        trace_term[:, tenor_idx] = 0.5 * torch.sum((sigma**2) * diag_hess, dim=1)

    return trace_term


def pde_residual(
    z,
    tau,
    decoder,
    mu_q,
    sigma,
    r=None,
    short_index=0,
    include_hessian=False,
    hessian_tenor_indices=None,
):
    """Compute no-arbitrage PDE residual per tenor.

    R_arb = -dP/dtau + grad_z(P)^T mu_Q + 0.5 Tr[sigma sigma^T Hess_z(P)] - rP
    """
    y = decoder(z)
    p = yield_to_discount(y, tau)
    dP_dtau = compute_dP_dtau(y, p, tau)
    grad_z_p = compute_grad_z_p(z, tau, decoder)

    if r is None:
        r = short_rate_from_curve(y, short_index=short_index)

    drift_term = torch.einsum("bnd,bd->bn", grad_z_p, mu_q)
    residual = -dP_dtau + drift_term - r * p

    if include_hessian:
        trace_term = compute_hessian_trace_term(
            z,
            tau,
            decoder,
            sigma,
            tenor_indices=hessian_tenor_indices,
        )
        residual = residual + trace_term

    return residual


def pde_penalty_loss(
    z,
    decoder,
    mu_q,
    sigma,
    tau=None,
    r=None,
    short_index=0,
    include_hessian=False,
    hessian_tenor_indices=None,
):
    """Mean squared PDE residual."""
    if tau is None:
        tau = default_tau_grid(device=z.device, dtype=z.dtype)

    residual = pde_residual(
        z=z,
        tau=tau,
        decoder=decoder,
        mu_q=mu_q,
        sigma=sigma,
        r=r,
        short_index=short_index,
        include_hessian=include_hessian,
        hessian_tenor_indices=hessian_tenor_indices,
    )
    return (residual**2).mean()


def total_constraint_loss(
    y,
    z,
    decoder,
    encoder=None,
    mu_q=None,
    sigma=None,
    tau=None,
    lambda_pde=0.1,
    lambda_diag=0.1,
    lambda_jac=0.1,
    use_pde=True,
    use_diag=True,
    use_jacobian=False,
    projection_method="reencode",
    include_hessian=False,
    encode_fn=None,
):
    """Composite constraint loss for Stage B ablations."""
    if tau is None:
        tau = default_tau_grid(device=y.device, dtype=y.dtype)

    loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)

    if use_diag:
        loss = loss + lambda_diag * arbitrage_diagnostic_loss(y, tau)

    if use_pde:
        loss = loss + lambda_pde * pde_penalty_loss(
            z=z,
            decoder=decoder,
            mu_q=mu_q,
            sigma=sigma,
            tau=tau,
            include_hessian=include_hessian,
        )

    if use_jacobian:
        from constraints.jacobian_projection import manifold_projection_loss

        loss = loss + lambda_jac * manifold_projection_loss(
            y_pred=y,
            z=z,
            encode_fn=encode_fn,
            decode_fn=decoder,
            encoder=encoder,
            method=projection_method,
        )

    return loss
