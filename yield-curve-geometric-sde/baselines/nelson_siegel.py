"""Nelson-Siegel / Diebold-Li baseline scaffold."""

import numpy as np


def nelson_siegel_loadings(maturities: np.ndarray, tau: float) -> np.ndarray:
    """Compute NS factor loadings.

    maturities shape: [N_tenors]
    returns loadings shape: [N_tenors, 3]
    """
    # TODO: implement stable formulas for small tau*maturity.
    x = maturities / tau
    l1 = np.ones_like(x)
    l2 = (1.0 - np.exp(-x)) / (x + 1e-8)
    l3 = l2 - np.exp(-x)
    return np.column_stack([l1, l2, l3])


def fit_diebold_li(curves: np.ndarray, maturities: np.ndarray) -> dict:
    """Fit time series of NS factors and optional VAR on factors."""
    # TODO: OLS each date for factors, then fit AR/VAR dynamics.
    return {"beta_t": None, "tau": 1.0}
