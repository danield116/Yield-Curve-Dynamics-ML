"""Forecast/reconstruction metrics for yield curves."""

import numpy as np


def rmse_by_tenor(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return RMSE per tenor, shape [N_tenors]."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def mae_by_tenor(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return MAE per tenor, shape [N_tenors]."""
    return np.abs(y_true - y_pred).mean(axis=0)


def mean_curve_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Aggregate RMSE over full curve and time."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def segment_errors(y_true: np.ndarray, y_pred: np.ndarray, segment_slices: dict) -> dict:
    """Compute short-end / belly / long-end error summaries."""
    # TODO: define default slices by tenor index mapping.
    out = {}
    for name, sl in segment_slices.items():
        out[name] = float(np.sqrt(((y_true[:, sl] - y_pred[:, sl]) ** 2).mean()))
    return out
