"""Manifold consistency metrics for decoded yield curves.

These metrics isolate the *dynamics* contribution to manifold geometry, rather than
being dominated by Stage A's static reconstruction gap. This is what makes the
Jacobian constraint's effect visible (see `manifold_correction_gain`).
"""

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
    """RMSE between curves and their re-encode/decode manifold projection.

    NOTE: For persistence-residual forecasts this is dominated by the (constant,
    model-independent) off-manifold offset of the last observed curve, so absolute
    values look nearly identical across ablations. Use `manifold_correction_gain`
    to expose the model-specific contribution.
    """
    return float(math.sqrt(_off_manifold_rmse(y_curves, encode_fn, decode_fn).item()))


@torch.no_grad()
def manifold_correction_gain(
    y_pred: torch.Tensor,
    y_persist: torch.Tensor,
    encode_fn,
    decode_fn,
) -> float:
    """Net manifold correction the forecast adds over the raw persistence anchor.

    gain = ||y_pred - D(E(y_pred))|| - ||y_persist - D(E(y_persist))||

    `y_persist` (last observed curve) is identical across all Stage B ablations, so
    subtracting its off-manifold error removes the dominating constant baseline and
    isolates what the latent dynamics contribute. Negative = the forecast is pulled
    *closer* to the decoder manifold than naive persistence (Jacobian should be the
    most negative).
    """
    e_pred = torch.sqrt(_off_manifold_rmse(y_pred, encode_fn, decode_fn))
    e_persist = torch.sqrt(_off_manifold_rmse(y_persist, encode_fn, decode_fn))
    return float((e_pred - e_persist).item())


def tangent_move_residual_rmse(
    z_last: torch.Tensor,
    z_pred: torch.Tensor,
    decode_fn,
    eps: float = 1e-5,
) -> float:
    """Off-tangent component of the decoded latent move at z_last.

    Decodes the forecast move dy = D(z_pred) - D(z_last), projects it onto the decoder
    tangent space span(J(z_last)), and returns the residual RMSE. Measures how much of
    the predicted curve move leaves the manifold's local tangent plane.
    """
    from constraints.jacobian_projection import decoder_jacobian, project_delta_to_tangent

    with torch.enable_grad():
        y0 = decode_fn(z_last)
        y1 = decode_fn(z_pred)
        delta_y = (y1 - y0).detach()
        jac = decoder_jacobian(z_last, decode_fn)

    delta_tan = project_delta_to_tangent(jac, delta_y, eps=eps)
    mse = torch.mean((delta_y - delta_tan) ** 2)
    return float(math.sqrt(mse.item()))
