"""Ornstein-Uhlenbeck model of the spread, for s-score signals.

Why I added this. The v1 z-score divides the spread by a short rolling
standard deviation, and I documented two failure modes of that choice: a
one-day spike inflates the denominator so z normalizes without any price
reversion, and a rolling mean chases the dislocation it is supposed to
measure. Both problems come from estimating the equilibrium distribution
with a window that is short relative to the half-life.

The fix (Avellaneda and Lee, 2010) is to fit the spread's actual dynamics
once per formation window and score dislocations against the *stationary*
distribution of that process. I model the spread as a discretized OU
process, which in discrete time is an AR(1):

    X_t = a + phi * X_{t-1} + eps_t,    eps ~ N(0, sigma_resid^2)

From the AR(1) fit I recover:
    m         = a / (1 - phi)                  equilibrium level
    sigma_eq  = sigma_resid / sqrt(1 - phi^2)  stationary std deviation
    kappa     = -ln(phi)                       mean-reversion speed (per day)
    half_life = ln(2) / kappa

The s-score is s_t = (X_t - m) / sigma_eq. Because m and sigma_eq are
fitted on the FORMATION window only and frozen for the trading window,
the score cannot be contaminated by the dislocation being traded, and
"s = 2" means the same thing on day one of the trading window as on day
126: two stationary standard deviations from equilibrium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ou(spread: pd.Series) -> dict | None:
    """Fit the AR(1)/OU parameters on a formation-window spread.

    Returns None when the fit implies no stable mean reversion
    (phi outside (0, 1)), which tells the caller to skip the pair.
    """
    x = spread.dropna()
    if len(x) < 60:
        return None
    lag = x.shift(1).iloc[1:]
    cur = x.iloc[1:]
    fit = sm.OLS(cur, sm.add_constant(lag)).fit()
    a, phi = float(fit.params.iloc[0]), float(fit.params.iloc[1])
    if not (0.0 < phi < 1.0):
        return None
    resid_var = float(fit.resid.var())
    m = a / (1.0 - phi)
    sigma_eq = float(np.sqrt(resid_var / (1.0 - phi**2)))
    kappa = float(-np.log(phi))
    return {
        "m": m,
        "sigma_eq": sigma_eq,
        "kappa": kappa,
        "half_life": float(np.log(2) / kappa),
        "phi": phi,
    }


def s_score(spread: pd.Series, ou: dict) -> pd.Series:
    """Dislocation in stationary standard deviations from equilibrium."""
    return (spread - ou["m"]) / ou["sigma_eq"]
