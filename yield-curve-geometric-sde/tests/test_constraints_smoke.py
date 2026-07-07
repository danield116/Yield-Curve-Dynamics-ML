"""Smoke tests for constraint module implementations."""

import torch

from constraints.bond_math import default_tau_grid, yield_to_discount
from constraints.jacobian_projection import decoder_jacobian, manifold_projection_loss
from constraints.no_arbitrage_pde import arbitrage_diagnostic_loss, pde_penalty_loss
from models.neural_sde import LatentNeuralSDE
from models.vae import VAE


def test_bond_math_shapes():
    y = torch.randn(4, 11)
    tau = default_tau_grid()
    p = yield_to_discount(y, tau)
    assert p.shape == (4, 11)


def test_pde_and_jacobian_runs():
    torch.manual_seed(0)
    model = VAE(n_tenors=11, latent_dim=3)
    sde = LatentNeuralSDE(latent_dim=3, history_steps=1)
    sde_input = torch.randn(4, 3)

    z = torch.randn(4, 3)
    y = model.decode(z)
    tau = default_tau_grid()
    mu_q = sde.drift_q(sde_input)
    sigma = sde.diffusion(sde_input)

    diag_loss = arbitrage_diagnostic_loss(y, tau)
    pde_loss = pde_penalty_loss(z, model.decode, mu_q, sigma, tau=tau, include_hessian=False)
    jac = decoder_jacobian(z, model.decode)
    proj_loss = manifold_projection_loss(y, z, model, model.decode, method="reencode")

    assert diag_loss.ndim == 0
    assert pde_loss.ndim == 0
    assert jac.shape == (4, 11, 3)
    assert proj_loss.ndim == 0
