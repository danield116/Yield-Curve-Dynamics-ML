"""Manifold consistency metrics for decoded yield curves."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def _off_manifold_rmse(y_curves: torch.Tensor, encode_fn, decode_fn) -> torch.Tensor:
    """Mean-squared distance between curves and their re-encode/decode projection."""
    z = encode_fn(y_curves)
    y_proj = decode_fn(z)
    return torch.mean((y_curves - y_proj) ** 2)


@torch.no_grad()
def manifold_off_manifold_rmse(
    y_curves: torch.Tensor,
    encode_fn,
    decode_fn,
) -> float:
    """RMSE ||y - D(E(y))||. Persistence-residual forecasts are dominated by the last observed offset; compare ablations with `manifold_correction_gain`."""
    return float(math.sqrt(_off_manifold_rmse(y_curves, encode_fn, decode_fn).item()))


@torch.no_grad()
def manifold_correction_gain(
    y_pred: torch.Tensor,
    y_persist: torch.Tensor,
    encode_fn,
    decode_fn,
) -> float:
    """off_manifold(y_pred) - off_manifold(y_persist). Negative = closer to the manifold than persistence."""
    e_pred = torch.sqrt(_off_manifold_rmse(y_pred, encode_fn, decode_fn))
    e_persist = torch.sqrt(_off_manifold_rmse(y_persist, encode_fn, decode_fn))
    return float((e_pred - e_persist).item())


def tangent_move_residual_rmse(
    z_last: torch.Tensor,
    z_pred: torch.Tensor,
    decode_fn,
    eps: float = 1e-5,
) -> float:
    """RMSE of dy = D(z_pred)-D(z_last) after removing the tangent component at z_last."""
    from constraints.jacobian_projection import decoder_jacobian, project_delta_to_tangent

    with torch.enable_grad():
        y0 = decode_fn(z_last)
        y1 = decode_fn(z_pred)
        delta_y = (y1 - y0).detach()
        jac = decoder_jacobian(z_last, decode_fn)

    delta_tan = project_delta_to_tangent(jac, delta_y, eps=eps)
    mse = torch.mean((delta_y - delta_tan) ** 2)
    return float(math.sqrt(mse.item()))
