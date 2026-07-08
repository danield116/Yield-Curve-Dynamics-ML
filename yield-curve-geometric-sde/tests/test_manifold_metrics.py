"""Tests for manifold consistency metrics."""

import torch

from evaluation.manifold_metrics import (
    manifold_correction_gain,
    manifold_off_manifold_rmse,
    tangent_move_residual_rmse,
)


def _encode_fn(y):
    return y[:, :1]


def _decode_fn(z):
    return torch.cat([z, z], dim=1)


def test_manifold_off_manifold_rmse_zero_on_manifold():
    y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert manifold_off_manifold_rmse(y, _encode_fn, _decode_fn) == 0.0


def test_manifold_correction_gain_negative_when_pred_closer():
    # y_pred is on-manifold (gain uses e_pred - e_persist); persistence is off-manifold.
    y_pred = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    y_persist = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    gain = manifold_correction_gain(y_pred, y_persist, _encode_fn, _decode_fn)
    assert gain < 0.0


def test_tangent_move_residual_rmse_small_for_linear_decoder():
    # Linear decoder -> the decoded move lies exactly in the tangent space -> ~0 residual.
    z_last = torch.tensor([[0.5], [1.0]])
    z_pred = torch.tensor([[0.7], [1.2]])
    residual = tangent_move_residual_rmse(z_last, z_pred, _decode_fn)
    assert residual < 1e-4
