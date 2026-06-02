"""Visualization placeholders for yield curves and latent paths."""

import numpy as np


def plot_curve_snapshot(curve: np.ndarray, tenors: list[str]) -> None:
    """TODO: matplotlib figure for one curve snapshot."""
    _ = (curve, tenors)


def plot_forecast_fan(forecast_paths: np.ndarray, tenors: list[str]) -> None:
    """TODO: uncertainty fan chart from simulated paths."""
    _ = (forecast_paths, tenors)
