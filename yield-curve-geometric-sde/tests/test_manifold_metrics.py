"""Tests for manifold consistency metrics."""

import torch

from evaluation.manifold_metrics import (
    manifold_delta_off_manifold_rmse,
    manifold_off_manifold_rmse,
)


def test_manifold_off_manifold_rmse_zero_on_manifold():
    def encode_fn(y):
        return y[:, :1]

    def decode_fn(z):
        return torch.cat([z, z], dim=1)

    y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert manifold_off_manifold_rmse(y, encode_fn, decode_fn) == 0.0


def test_manifold_delta_off_manifold_rmse_zero_on_manifold():
    def encode_fn(y):
        return y[:, :1]

    def decode_fn(z):
        return torch.cat([z, z], dim=1)

    y_prev = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert manifold_delta_off_manifold_rmse(y_prev, y, encode_fn, decode_fn) == 0.0
