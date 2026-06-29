"""Arbitrage diagnostics for generated yield/discount curves."""

import numpy as np

from constraints.bond_math import DEFAULT_TENOR_YEARS


def discount_monotonicity_violations(discount_curves: np.ndarray) -> float:
    """Fraction of maturity steps violating monotonicity."""
    diffs = discount_curves[:, 1:] - discount_curves[:, :-1]
    return float((diffs > 0.0).mean())


def forward_smoothness_score(forward_curves: np.ndarray) -> float:
    """Lower is smoother; based on second differences."""
    if forward_curves.shape[1] < 3:
        return 0.0
    second_diff = forward_curves[:, 2:] - 2 * forward_curves[:, 1:-1] + forward_curves[:, :-2]
    return float((second_diff**2).mean())


def scenario_stability_score(generated_curves: np.ndarray) -> float:
    """Penalize unrealistic path explosions (max abs yield move)."""
    if generated_curves.shape[0] < 2:
        return float(np.nan_to_num(np.abs(generated_curves).max(), nan=1e6))
    step_moves = np.abs(np.diff(generated_curves, axis=0)).max()
    level = np.abs(generated_curves).max()
    return float(max(step_moves, level))


def yields_to_discount_np(yields: np.ndarray, tau: np.ndarray | None = None) -> np.ndarray:
    tau = np.asarray(DEFAULT_TENOR_YEARS if tau is None else tau, dtype=np.float64)
    return np.exp(-yields * tau[None, :])


def discount_to_forward_np(discount: np.ndarray, tau: np.ndarray | None = None) -> np.ndarray:
    tau = np.asarray(DEFAULT_TENOR_YEARS if tau is None else tau, dtype=np.float64)
    log_p = np.log(np.clip(discount, 1e-10, None))
    if tau.shape[0] < 2:
        return np.zeros_like(discount)
    d_logp = (log_p[:, 2:] - log_p[:, :-2]) / np.clip(tau[2:] - tau[:-2], 1e-8, None)[None, :]
    left = (log_p[:, 1:2] - log_p[:, 0:1]) / np.clip(tau[1:2] - tau[0:1], 1e-8, None)[None, :]
    right = (log_p[:, -1:] - log_p[:, -2:-1]) / np.clip(tau[-1:] - tau[-2:-1], 1e-8, None)[None, :]
    return -np.concatenate([left, d_logp, right], axis=1)


def curve_arbitrage_metrics(yields: np.ndarray) -> dict:
    """Compute no-arbitrage diagnostics from yield curves [T, N_tenors]."""
    discount = yields_to_discount_np(yields)
    forwards = discount_to_forward_np(discount)
    return {
        "discount_monotonicity_violations": discount_monotonicity_violations(discount),
        "forward_smoothness": forward_smoothness_score(forwards),
        "scenario_stability": scenario_stability_score(yields),
    }
