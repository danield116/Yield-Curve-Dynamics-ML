"""Nelson-Siegel-Svensson (NSS) Diebold-Li style baseline.

NSS extends Nelson-Siegel with a second curvature term (Svensson, 1994):

    y(m) = b0 + b1*L1(m,t1) + b2*L2(m,t1) + b3*L3(m,t2)

where L1 = 1, L2 = (1-exp(-m/t1))/(m/t1), L3 = L2(m,t1) - exp(-m/t1) for the
first curvature block, and the Svensson term uses L4 = L2(m,t2) - exp(-m/t2).

We fix (t1, t2) via grid search on the training panel, then estimate factor
time series by OLS each day and fit a VAR for rolling forecasts.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.api import VAR

from constraints.bond_math import DEFAULT_TENOR_YEARS

DEFAULT_TAU1_GRID = (0.5, 1.0, 2.0, 3.0)
DEFAULT_TAU2_GRID = (3.0, 5.0, 7.0, 10.0, 15.0, 20.0)


def _eta1(x: np.ndarray) -> np.ndarray:
    """(1 - exp(-x)) / x with limit 1 as x -> 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    mask = x > 1e-8
    out[mask] = (1.0 - np.exp(-x[mask])) / x[mask]
    return out


def _eta2(x: np.ndarray) -> np.ndarray:
    """Curvature loading: eta1(x) - exp(-x)."""
    return _eta1(x) - np.exp(-x)


def nss_loadings(
    maturities: np.ndarray,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """NSS factor loadings matrix [N_tenors, 4]."""
    m = np.asarray(maturities, dtype=np.float64)
    x1 = m / max(float(tau1), 1e-8)
    x2 = m / max(float(tau2), 1e-8)
    return np.column_stack(
        [
            np.ones_like(m),
            _eta1(x1),
            _eta2(x1),
            _eta2(x2),
        ]
    )


def reconstruct_nss(
    beta: np.ndarray,
    maturities: np.ndarray | None = None,
    tau1: float = 2.0,
    tau2: float = 10.0,
) -> np.ndarray:
    """Map NSS factors to yields. beta shape [..., 4]."""
    maturities = np.asarray(DEFAULT_TENOR_YEARS if maturities is None else maturities, dtype=np.float64)
    loadings = nss_loadings(maturities, tau1=tau1, tau2=tau2)
    if beta.ndim == 1:
        return loadings @ beta
    return beta @ loadings.T


def fit_nss_factors(
    curves: np.ndarray,
    maturities: np.ndarray | None = None,
    tau1: float = 2.0,
    tau2: float = 10.0,
) -> np.ndarray:
    """OLS NSS factor time series, shape [T, 4]."""
    maturities = np.asarray(DEFAULT_TENOR_YEARS if maturities is None else maturities, dtype=np.float64)
    loadings = nss_loadings(maturities, tau1=tau1, tau2=tau2)
    beta_t, _, _, _ = np.linalg.lstsq(loadings, curves.T, rcond=None)
    return beta_t.T


def _panel_ssr(curves: np.ndarray, maturities: np.ndarray, tau1: float, tau2: float) -> float:
    beta_t = fit_nss_factors(curves, maturities=maturities, tau1=tau1, tau2=tau2)
    recon = reconstruct_nss(beta_t, maturities=maturities, tau1=tau1, tau2=tau2)
    return float(((curves - recon) ** 2).sum())


def fit_nss_hyperparameters(
    curves: np.ndarray,
    maturities: np.ndarray | None = None,
    tau1_grid: tuple[float, ...] | None = None,
    tau2_grid: tuple[float, ...] | None = None,
    max_fit_rows: int = 800,
) -> tuple[float, float]:
    """Grid-search (tau1, tau2) minimizing in-sample SSR on the training panel."""
    maturities = np.asarray(DEFAULT_TENOR_YEARS if maturities is None else maturities, dtype=np.float64)
    tau1_grid = tau1_grid or DEFAULT_TAU1_GRID
    tau2_grid = tau2_grid or DEFAULT_TAU2_GRID

    fit_curves = curves
    if curves.shape[0] > max_fit_rows:
        idx = np.linspace(0, curves.shape[0] - 1, max_fit_rows, dtype=int)
        fit_curves = curves[idx]

    best_ssr = np.inf
    best_pair = (2.0, 10.0)
    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue
            ssr = _panel_ssr(fit_curves, maturities, tau1, tau2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_pair = (float(tau1), float(tau2))
    return best_pair


def fit_nss_var(
    curves: np.ndarray,
    maturities: np.ndarray | None = None,
    tau1: float | None = None,
    tau2: float | None = None,
    var_lags: int = 1,
) -> dict:
    """Fit NSS factors and VAR dynamics on training curves."""
    if tau1 is None or tau2 is None:
        tau1, tau2 = fit_nss_hyperparameters(curves, maturities=maturities)

    beta_t = fit_nss_factors(curves, maturities=maturities, tau1=tau1, tau2=tau2)
    var = VAR(beta_t).fit(maxlags=var_lags, ic=None, trend="c")
    return {
        "beta_t": beta_t,
        "tau1": tau1,
        "tau2": tau2,
        "var": var,
        "maturities": maturities,
    }


def forecast_nss(state: dict, last_beta: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast NSS factors with VAR and decode to a yield curve."""
    var = state["var"]
    start = last_beta[-var.k_ar :]
    beta_path = var.forecast(start, steps=horizon)
    return reconstruct_nss(
        beta_path[-1],
        maturities=state.get("maturities"),
        tau1=state["tau1"],
        tau2=state["tau2"],
    )


def rolling_forecast_nss(
    train_curves: np.ndarray,
    test_curves: np.ndarray,
    lookback: int,
    horizon: int,
    tau1: float | None = None,
    tau2: float | None = None,
    refit_taus_every: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling H-step NSS+VAR forecasts on the test split."""
    maturities = np.asarray(DEFAULT_TENOR_YEARS, dtype=np.float64)
    if tau1 is None or tau2 is None:
        tau1, tau2 = fit_nss_hyperparameters(train_curves, maturities=maturities)

    state = fit_nss_var(train_curves, maturities=maturities, tau1=tau1, tau2=tau2)
    beta_history = state["beta_t"]

    preds = []
    truths = []
    max_start = test_curves.shape[0] - lookback - horizon
    if max_start <= 0:
        raise ValueError("Not enough test data for NSS rolling forecast.")

    for start in range(max_start):
        if refit_taus_every and start > 0 and start % refit_taus_every == 0:
            expanded = np.vstack([train_curves, test_curves[: start + lookback]])
            tau1, tau2 = fit_nss_hyperparameters(expanded, maturities=maturities)

        window = test_curves[start : start + lookback]
        window_beta = fit_nss_factors(window, maturities=maturities, tau1=tau1, tau2=tau2)
        beta_history = np.vstack([beta_history, window_beta])

        var = VAR(beta_history[-500:]).fit(maxlags=1, ic=None, trend="c")
        y_pred = forecast_nss(
            {"var": var, "tau1": tau1, "tau2": tau2, "maturities": maturities},
            beta_history,
            horizon=horizon,
        )
        y_true = test_curves[start + lookback + horizon - 1]
        preds.append(y_pred)
        truths.append(y_true)

    return np.asarray(truths), np.asarray(preds)


# Backward-compatible aliases (3-factor Nelson-Siegel names).
nelson_siegel_loadings = nss_loadings
fit_ns_factors = fit_nss_factors
reconstruct_ns = reconstruct_nss
fit_diebold_li = fit_nss_var
forecast_diebold_li = forecast_nss
rolling_forecast_diebold_li = rolling_forecast_nss
