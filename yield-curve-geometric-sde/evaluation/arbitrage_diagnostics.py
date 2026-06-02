"""Arbitrage diagnostics for generated yield/discount curves."""

import numpy as np


def discount_monotonicity_violations(discount_curves: np.ndarray) -> float:
    """Fraction of maturity steps violating monotonicity."""
    # discount_curves shape: [T, N_tenors]
    diffs = discount_curves[:, 1:] - discount_curves[:, :-1]
    violations = (diffs > 0.0).mean()
    return float(violations)


def forward_smoothness_score(forward_curves: np.ndarray) -> float:
    """Lower is smoother; based on second differences."""
    second_diff = forward_curves[:, 2:] - 2 * forward_curves[:, 1:-1] + forward_curves[:, :-2]
    return float((second_diff**2).mean())


def scenario_stability_score(generated_curves: np.ndarray) -> float:
    """Heuristic score for unrealistic path explosions."""
    # TODO: replace heuristic with calibrated threshold metrics.
    return float(np.nan_to_num(np.abs(generated_curves).mean(), nan=1e6))
