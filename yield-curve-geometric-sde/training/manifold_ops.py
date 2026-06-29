"""LevelScript-aware encode/decode helpers for Stage B constraints and evaluation."""

from __future__ import annotations

import torch


def split_shape_level(y_full: torch.Tensor, level_tenor_index: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Split full curve into shape and level (scaled units)."""
    level = y_full[:, level_tenor_index : level_tenor_index + 1]
    shape = y_full - level
    return shape, level


def make_full_curve_decoder(stage_a, level, use_levelscript):
    """Return decode(z) -> full_curve in scaled units."""

    def decode_full_curve(z):
        if use_levelscript:
            return stage_a.decode(z, level) + level
        return stage_a.decode(z)

    return decode_full_curve


def make_full_curve_encoder(stage_a, level, use_levelscript, level_tenor_index: int = 3):
    """Return encode(full_curve) -> latent mean mu."""

    def encode_full_curve(y_full):
        if use_levelscript:
            if level is not None:
                shape = y_full - level
                cond = level
            else:
                shape, cond = split_shape_level(y_full, level_tenor_index)
            mu, _ = stage_a.encode(shape, cond)
        else:
            mu, _ = stage_a.encode(y_full)
        return mu

    return encode_full_curve


def make_manifold_ops(stage_a, level, use_levelscript, level_tenor_index: int = 3):
    """Return (encode_full, decode_full) callables for constraint modules."""
    decode_fn = make_full_curve_decoder(stage_a, level, use_levelscript)
    encode_fn = make_full_curve_encoder(stage_a, level, use_levelscript, level_tenor_index)
    return encode_fn, decode_fn


def linearized_curve_forecast(decode_fn, z_t, z_pred):
    """First-order curve forecast that can leave the decoder manifold.

    y_hat ≈ D(z_t) + J(z_t) (z_pred - z_t)
    """
    z_t = z_t.detach().requires_grad_(True)
    y_on = decode_fn(z_t)
    batch_size, n_tenors = y_on.shape
    latent_dim = z_t.shape[1]
    delta_z = (z_pred - z_t).detach()

    jac = torch.zeros(batch_size, n_tenors, latent_dim, device=z_t.device, dtype=z_t.dtype)
    for tenor_idx in range(n_tenors):
        grad_z = torch.autograd.grad(
            y_on[:, tenor_idx].sum(),
            z_t,
            create_graph=True,
            retain_graph=True,
        )[0]
        jac[:, tenor_idx, :] = grad_z

    delta_y = torch.einsum("bnd,bd->bn", jac, delta_z)
    return y_on + delta_y
