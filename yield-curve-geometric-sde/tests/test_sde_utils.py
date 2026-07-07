"""Tests for latent-history SDE utilities."""

import torch

from models.neural_sde import LatentNeuralSDE
from training.sde_utils import build_sde_input, persistence_residual_curve_forecast, roll_latent_forecast


def test_build_sde_input_shape():
    z_hist = torch.randn(4, 21, 3)
    out = build_sde_input(z_hist, history_steps=5)
    assert out.shape == (4, 15)


def test_persistence_residual_matches_last_curve():
    def decode_fn(z):
        return z[:, :2]

    z_hist = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    y_hist = torch.tensor([[[0.5, 0.6], [0.7, 0.8]]])
    z_pred = z_hist[:, -1, :]
    out = persistence_residual_curve_forecast(decode_fn, z_hist, y_hist, z_pred)
    assert torch.allclose(out, y_hist[:, -1, :])


def test_roll_latent_forecast_runs():
    sde = LatentNeuralSDE(latent_dim=3, history_steps=5)
    z_hist = torch.randn(4, 21, 3)
    z_final = roll_latent_forecast(sde, z_hist, horizon=3, dt=1.0, stochastic=False)
    assert z_final.shape == (4, 3)
