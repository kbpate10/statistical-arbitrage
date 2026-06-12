"""Pair discovery via ADF and Engle-Granger cointegration tests.

I run this on TRAIN data only, persist the results to
results/selected_pairs.json, and never recompute them on validation or
test data. Re-selecting pairs on later data would be look-ahead bias at
the portfolio level: I would be keeping exactly the pairs that happened
to stay cointegrated.

My reasoning for each step:

* Step 1 requires each price series to be I(1): the ADF test must FAIL to
  reject a unit root (p > 0.05). If a price series were stationary on its
  own I could trade it by itself, and the Engle-Granger framework, which
  looks for a stationary combination of two non-stationary series, would
  not apply to it.
* Step 2 regresses y on x (OLS with intercept) and tests the residuals
  for stationarity. A raw `adfuller` on the residuals uses the wrong
  critical values, because OLS has already chosen alpha and beta to make
  those residuals look as stationary as possible. The proper test is
  Engle-Granger with MacKinnon critical values, available as
  `statsmodels.tsa.stattools.coint`. I gate on the `coint` p-value and
  report the raw residual ADF p-value alongside it so the (always
  flattering) difference is visible.
* Step 3 fits an AR(1) on residual changes to get the mean-reversion
  speed lambda, then half-life = -ln(2)/ln(1+lambda). I keep only
  half-lives in [5, 60] days: below 5 the reversion is mostly over before
  my next-day fill, above 60 capital sits too long and the relationship
  has time to break.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from . import config


def adf_pvalue(series: pd.Series) -> float:
    """ADF test p-value (constant, automatic lag selection by AIC)."""
    series = series.dropna()
    return adfuller(series, regression="c", autolag="AIC")[1]


def engle_granger(y: pd.Series, x: pd.Series,
                  with_naive_adf: bool = True) -> dict:
    """OLS y = alpha + beta*x; return fit, residuals, and stationarity tests.

    `with_naive_adf=False` skips the pedagogical residual-ADF p-value. The
    walk-forward engine scans tens of thousands of pairs and only the
    MacKinnon `coint` p-value gates anything, so I avoid paying for a
    number nobody uses there.
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    resid = y - alpha - beta * x
    out = {
        "alpha": float(alpha),
        "beta": float(beta),
        "resid": resid,
        # Correct EG test (MacKinnon critical values):
        "coint_pvalue": float(coint(y, x)[1]),
    }
    # Naive residual ADF, reported for comparison only:
    out["resid_adf_pvalue"] = (float(adf_pvalue(resid))
                               if with_naive_adf else float("nan"))
    return out


def half_life(resid: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1) fit on residual changes.

    delta_resid_t = lambda * resid_{t-1} + noise. Mean reversion requires
    lambda < 0; half_life = -ln(2)/ln(1+lambda). Returns inf if lambda >= 0.
    """
    resid = resid.dropna()
    lagged = resid.shift(1).dropna()
    delta = resid.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")
    lam = float(sm.OLS(delta, lagged).fit().params.iloc[0])
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + lam))


def test_pair(y: pd.Series, x: pd.Series,
              adf_cache: dict | None = None,
              with_naive_adf: bool = True) -> dict | None:
    """Full pipeline for one ordered pair on formation data. None = rejected.

    `adf_cache` maps ticker -> precomputed ADF p-value on this window. The
    ADF test depends on one series only, so computing it once per ticker
    instead of twice per pair cuts the scan cost roughly in half on a
    wide universe. Results are identical either way.
    """
    y, x = y.dropna().align(x.dropna(), join="inner")

    # Step 1: both series must be non-stationary (unit root not rejected).
    if adf_cache is not None:
        p_y, p_x = adf_cache[y.name], adf_cache[x.name]
    else:
        p_y, p_x = adf_pvalue(y), adf_pvalue(x)
    if p_y <= config.ADF_NONSTATIONARY_P or p_x <= config.ADF_NONSTATIONARY_P:
        return None

    # Step 2: Engle-Granger cointegration.
    eg = engle_granger(y, x, with_naive_adf=with_naive_adf)
    if eg["coint_pvalue"] >= config.COINT_P_THRESHOLD:
        return None

    # Step 3: tradeable half-life.
    hl = half_life(eg["resid"])
    if not (config.HALF_LIFE_MIN <= hl <= config.HALF_LIFE_MAX):
        return None

    return {
        "y": y.name, "x": x.name,
        "alpha": eg["alpha"], "beta": eg["beta"],
        "coint_pvalue": eg["coint_pvalue"],
        "resid_adf_pvalue": eg["resid_adf_pvalue"],
        "adf_p_y": p_y, "adf_p_x": p_x,
        "half_life": hl,
        # Train residual variance: calibrates the Kalman observation noise
        # R to the pair's actual equilibrium dispersion. A generic tiny R
        # makes the filter absorb prices into the state and leaves a
        # white-noise "spread" with no tradeable mean reversion.
        "resid_var": float(eg["resid"].var()),
    }


def find_pairs(train_prices: pd.DataFrame,
               sectors: dict[str, list[str]] | None = None,
               with_naive_adf: bool = True) -> pd.DataFrame:
    """Scan every same-sector pair (both orderings, keep the better one).

    `train_prices` must be the formation slice (v1: the train split; v2:
    one walk-forward formation window). Callers are responsible for never
    passing later data here. `sectors` defaults to the v1 universe so v1
    behavior is unchanged; the walk-forward engine passes SECTORS_V2.

    I precompute the per-ticker ADF p-values once per call and gate
    ineligible tickers before any pair regression runs: the step-1 filter
    depends only on the individual series, so testing it inside every
    pair combination would repeat the same test up to 25 times per ticker.
    """
    if sectors is None:
        sectors = config.SECTORS
    available = set(train_prices.columns)

    results = []
    for sector, tickers in sectors.items():
        tickers = [t for t in tickers if t in available]
        adf_cache = {t: adf_pvalue(train_prices[t]) for t in tickers}
        eligible = [t for t in tickers
                    if adf_cache[t] > config.ADF_NONSTATIONARY_P]
        for a, b in itertools.combinations(eligible, 2):
            candidates = [
                test_pair(train_prices[a], train_prices[b],
                          adf_cache=adf_cache, with_naive_adf=with_naive_adf),
                test_pair(train_prices[b], train_prices[a],
                          adf_cache=adf_cache, with_naive_adf=with_naive_adf),
            ]
            candidates = [c for c in candidates if c is not None]
            if not candidates:
                continue
            best = min(candidates, key=lambda c: c["coint_pvalue"])
            best["sector"] = sector
            results.append(best)

    if not results:
        return pd.DataFrame()
    df = (pd.DataFrame(results)
          .sort_values(["coint_pvalue", "half_life"])
          .reset_index(drop=True))
    return df


def select_pairs(train_prices: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Run (or load cached) pair selection; keep the top N pairs."""
    if config.SELECTED_PAIRS_JSON.exists() and not force:
        return pd.read_json(config.SELECTED_PAIRS_JSON)
    df = find_pairs(train_prices).head(config.MAX_SELECTED_PAIRS)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(config.SELECTED_PAIRS_JSON, orient="records", indent=2)
    return df


if __name__ == "__main__":
    from . import universe

    px = universe.download_universe()
    train = universe.get_split(px, "train")[universe.stock_columns(px)]
    pairs = select_pairs(train, force=True)
    if pairs.empty:
        print("No cointegrated pairs found.")
    else:
        cols = ["sector", "y", "x", "beta", "coint_pvalue",
                "resid_adf_pvalue", "half_life"]
        print(pairs[cols].to_string(
            float_format=lambda v: f"{v:.4f}"))
