"""Manifold consistency metrics for decoded yield curves."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def manifold_off_manifold_rmse(
    y_curves: torch.Tensor,
    encode_fn,
    decode_fn,
) -> float:
    """RMSE between curves and their re-encode/decode manifold projection."""
    z = encode_fn(y_curves)
    y_proj = decode_fn(z)
    mse = torch.mean((y_curves - y_proj) ** 2)
    return float(math.sqrt(mse.item()))
