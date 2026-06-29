"""Tests for LevelScript-aware manifold helpers."""

import torch

from models.student_t_vae import StudentTCVAE
from training.manifold_ops import linearized_curve_forecast, make_manifold_ops


def test_levelscript_encode_decode_roundtrip():
    torch.manual_seed(0)
    model = StudentTCVAE(n_tenors=11, cond_dim=1, latent_dim=3)
    y = torch.randn(4, 11)
    level = y[:, 3:4]
    encode_fn, decode_fn = make_manifold_ops(model, level, use_levelscript=True, level_tenor_index=3)

    z = encode_fn(y)
    y_hat = decode_fn(z)
    assert z.shape == (4, 3)
    assert y_hat.shape == (4, 11)


def test_linearized_forecast_runs():
    torch.manual_seed(0)
    model = StudentTCVAE(n_tenors=11, cond_dim=1, latent_dim=3)
    level = torch.zeros(4, 1)
    _, decode_fn = make_manifold_ops(model, level, use_levelscript=True, level_tenor_index=3)
    z_t = torch.randn(4, 3)
    z_pred = z_t + 0.1
    y_lin = linearized_curve_forecast(decode_fn, z_t, z_pred)
    assert y_lin.shape == (4, 11)
