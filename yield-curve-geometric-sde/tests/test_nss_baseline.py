"""Smoke tests for NSS baseline."""

import numpy as np

from baselines.nelson_siegel import (
    fit_nss_factors,
    fit_nss_hyperparameters,
    nss_loadings,
    reconstruct_nss,
    rolling_forecast_nss,
)
from constraints.bond_math import DEFAULT_TENOR_YEARS


def test_nss_loadings_shape():
    loadings = nss_loadings(np.array(DEFAULT_TENOR_YEARS), tau1=2.0, tau2=10.0)
    assert loadings.shape == (len(DEFAULT_TENOR_YEARS), 4)


def test_nss_fit_reconstruct_roundtrip():
    rng = np.random.default_rng(0)
    maturities = np.array(DEFAULT_TENOR_YEARS)
    loadings = nss_loadings(maturities, tau1=1.0, tau2=5.0)
    true_beta = rng.normal(size=(20, 4))
    curves = true_beta @ loadings.T
    beta_hat = fit_nss_factors(curves, maturities=maturities, tau1=1.0, tau2=5.0)
    recon = reconstruct_nss(beta_hat, maturities=maturities, tau1=1.0, tau2=5.0)
    assert recon.shape == curves.shape
    assert np.allclose(recon, curves, atol=1e-5)


def test_nss_hyperparameter_search_runs():
    rng = np.random.default_rng(1)
    maturities = np.array(DEFAULT_TENOR_YEARS)
    loadings = nss_loadings(maturities, tau1=2.0, tau2=10.0)
    curves = rng.normal(size=(4,)) @ loadings.T + rng.normal(scale=0.01, size=(50, 4)) @ loadings.T
    tau1, tau2 = fit_nss_hyperparameters(curves, maturities=maturities)
    assert tau2 > tau1


def test_nss_rolling_forecast_runs():
    rng = np.random.default_rng(2)
    maturities = np.array(DEFAULT_TENOR_YEARS)
    loadings = nss_loadings(maturities, tau1=2.0, tau2=10.0)
    beta = rng.normal(size=(300, 4))
    curves = beta @ loadings.T
    train, test = curves[:200], curves[200:]
    y_true, y_pred = rolling_forecast_nss(train, test, lookback=21, horizon=1, tau1=2.0, tau2=10.0)
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 2
