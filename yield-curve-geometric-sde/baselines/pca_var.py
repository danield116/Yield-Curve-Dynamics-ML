"""PCA and PCA+VAR baseline."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR


def fit_pca(train_curves: np.ndarray, n_components: int = 3) -> dict:
    """Fit PCA for reconstruction baseline."""
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(train_curves)
    return {
        "pca": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "train_scores": scores,
    }


def reconstruct_with_pca(curves: np.ndarray, pca_state: dict) -> np.ndarray:
    """Reconstruct curves from PCA latent factors."""
    pca: PCA = pca_state["pca"]
    return pca.inverse_transform(pca.transform(curves))


def fit_var_on_pca_scores(train_scores: np.ndarray, lags: int = 1) -> dict:
    """Fit VAR on PCA scores for forecasting baseline."""
    model = VAR(train_scores)
    fitted = model.fit(maxlags=lags, ic=None, trend="c")
    return {"var": fitted, "lags": fitted.k_ar}


def forecast_pca_var(var_state: dict, last_scores: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast latent PCA scores recursively."""
    fitted = var_state["var"]
    start = last_scores[-fitted.k_ar :]
    forecast = fitted.forecast(start, steps=horizon)
    return forecast


def decode_scores_to_curves(scores: np.ndarray, pca_state: dict) -> np.ndarray:
    pca: PCA = pca_state["pca"]
    return pca.inverse_transform(scores)


def rolling_forecast_pca_var(
    train_curves: np.ndarray,
    test_curves: np.ndarray,
    lookback: int,
    horizon: int,
    n_components: int = 3,
    var_lags: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling H-step curve forecasts on the test split."""
    pca_state = fit_pca(train_curves, n_components=n_components)
    train_scores = pca_state["train_scores"]
    var_state = fit_var_on_pca_scores(train_scores, lags=var_lags)

    preds = []
    truths = []
    max_start = test_curves.shape[0] - lookback - horizon
    if max_start <= 0:
        raise ValueError("Not enough test data for PCA+VAR rolling forecast.")

    history_scores = train_scores.copy()
    for start in range(max_start):
        window = test_curves[start : start + lookback]
        window_scores = pca_state["pca"].transform(window)
        history_scores = np.vstack([history_scores, window_scores])

        var_state = fit_var_on_pca_scores(history_scores[-500:], lags=var_lags)
        score_forecast = forecast_pca_var(var_state, history_scores, horizon=horizon)
        y_pred = decode_scores_to_curves(score_forecast[-1:], pca_state)[0]
        y_true = test_curves[start + lookback + horizon - 1]

        preds.append(y_pred)
        truths.append(y_true)

    return np.asarray(truths), np.asarray(preds)
