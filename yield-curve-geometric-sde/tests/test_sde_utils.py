"""Tests for latent-history SDE utilities."""

import torch

from models.neural_sde import LatentNeuralSDE
from training.sde_utils import build_sde_input, roll_latent_forecast


def test_build_sde_input_shape():
    z_hist = torch.randn(4, 21, 3)
    out = build_sde_input(z_hist, history_steps=5)
    assert out.shape == (4, 15)


def test_roll_latent_forecast_runs():
    sde = LatentNeuralSDE(latent_dim=3, history_steps=5)
    z_hist = torch.randn(4, 21, 3)
    z_final = roll_latent_forecast(sde, z_hist, horizon=3, dt=1.0, stochastic=False)
    assert z_final.shape == (4, 3)
