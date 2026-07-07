"""Random-walk / persistence baseline: next curve equals last observed curve."""

from __future__ import annotations

import numpy as np


def rolling_forecast_persistence(
    train_curves: np.ndarray,
    test_curves: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict y_{t+h} = y_t where y_t is the last curve in the lookback window."""
    _ = train_curves
    preds = []
    truths = []
    max_start = test_curves.shape[0] - lookback - horizon
    if max_start <= 0:
        raise ValueError("Not enough test data for persistence rolling forecast.")

    for start in range(max_start):
        y_pred = test_curves[start + lookback - 1]
        y_true = test_curves[start + lookback + horizon - 1]
        preds.append(y_pred)
        truths.append(y_true)

    return np.asarray(truths), np.asarray(preds)
