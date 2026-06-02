"""PCA and PCA+VAR baseline scaffold."""

import numpy as np


def fit_pca(train_curves: np.ndarray, n_components: int = 3) -> dict:
    """Fit PCA for reconstruction baseline.

    train_curves shape: [T_train, N_tenors]
    """
    # TODO: use sklearn PCA and return components/mean/explained variance.
    return {"components": None, "mean": train_curves.mean(axis=0, keepdims=True)}


def reconstruct_with_pca(curves: np.ndarray, pca_state: dict) -> np.ndarray:
    """Reconstruct curves from PCA latent factors."""
    # TODO: implement transform + inverse_transform.
    return curves


def fit_var_on_pca_scores(train_scores: np.ndarray, lags: int = 1) -> dict:
    """Fit VAR on PCA scores for forecasting baseline."""
    # TODO: integrate statsmodels VAR.
    return {"lags": lags}


def forecast_pca_var(var_state: dict, last_scores: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast latent PCA scores and decode to curve space."""
    # TODO: implement recursive VAR forecast.
    return np.repeat(last_scores[None, :], horizon, axis=0)
