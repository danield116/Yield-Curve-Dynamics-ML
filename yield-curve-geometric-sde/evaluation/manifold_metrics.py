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


@torch.no_grad()
def manifold_delta_off_manifold_rmse(
    y_prev: torch.Tensor,
    y_curves: torch.Tensor,
    encode_fn,
    decode_fn,
) -> float:
    """RMSE between forecast delta and its manifold-projected delta.

    This focuses on the forecast move, which is where the Jacobian constraint acts.
    """
    z = encode_fn(y_curves)
    y_proj = decode_fn(z)
    delta = y_curves - y_prev
    delta_proj = y_proj - y_prev
    mse = torch.mean((delta - delta_proj) ** 2)
    return float(math.sqrt(mse.item()))
