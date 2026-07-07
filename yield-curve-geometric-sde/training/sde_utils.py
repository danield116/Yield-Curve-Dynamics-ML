"""Helpers for latent-history SDE inputs and multi-step rollouts."""

from __future__ import annotations

import torch

from models.neural_sde import LatentNeuralSDE


def build_sde_input(z_hist: torch.Tensor, history_steps: int) -> torch.Tensor:
    """Flatten the last `history_steps` latent vectors for SDE networks.

    z_hist: [B, T, latent_dim] -> [B, history_steps * latent_dim]
    """
    if z_hist.ndim != 3:
        raise ValueError(f"Expected z_hist [B, T, D], got shape {tuple(z_hist.shape)}")

    batch, t_len, dim = z_hist.shape
    k = min(int(history_steps), t_len)
    chunk = z_hist[:, -k:, :]
    if k < history_steps:
        pad = torch.zeros(batch, history_steps - k, dim, device=z_hist.device, dtype=z_hist.dtype)
        chunk = torch.cat([pad, chunk], dim=1)
    return chunk.reshape(batch, history_steps * dim)


def persistence_residual_curve_forecast(
    decode_fn,
    z_hist: torch.Tensor,
    y_hist: torch.Tensor,
    z_pred: torch.Tensor,
) -> torch.Tensor:
    """Forecast = last observed curve + decoded latent change.

    With zero drift (z_pred == z_last), this exactly equals persistence in yield space.
    """
    z_last = z_hist[:, -1, :]
    y_persist = y_hist[:, -1, :]
    return y_persist + decode_fn(z_pred) - decode_fn(z_last).detach()


def build_latent_neural_sde(config) -> LatentNeuralSDE:
    """Construct SDE module from YAML config."""
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    return LatentNeuralSDE(
        latent_dim=int(model_cfg.get("latent_dim", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        history_steps=int(training_cfg.get("latent_history_steps", 5)),
    )


def roll_latent_forecast(
    sde: LatentNeuralSDE,
    z_hist: torch.Tensor,
    horizon: int,
    dt: float,
    *,
    stochastic: bool = False,
) -> torch.Tensor:
    """Roll out latent path and return final latent z_{t+horizon}."""
    z_window = z_hist
    z = z_window[:, -1, :]
    history_steps = sde.history_steps

    for _ in range(horizon):
        sde_input = build_sde_input(z_window, history_steps)
        drift = sde.drift_p(sde_input)
        if stochastic:
            sigma = sde.diffusion(sde_input)
            z = z + drift * dt + sigma * torch.randn_like(z) * (dt**0.5)
        else:
            z = z + drift * dt
        z_window = torch.cat([z_window[:, 1:, :], z.unsqueeze(1)], dim=1)

    return z
