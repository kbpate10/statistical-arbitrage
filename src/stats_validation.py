"""Significance statistics: is the measured Sharpe real or selection noise?

The problem these solve. I tuned over a grid of configurations and kept
the best one. The maximum of N noisy Sharpe estimates is biased upward
even when every configuration has zero true edge, so reporting the
winner's Sharpe without a correction overstates the evidence. These are
the standard corrections (Bailey and Lopez de Prado):

* Probabilistic Sharpe Ratio (PSR): the probability that the true Sharpe
  exceeds a benchmark, given the estimate, the sample length, and the
  non-normality of returns (skew and kurtosis enter the standard error).
* Deflated Sharpe Ratio (DSR): a PSR whose benchmark is not zero but the
  Sharpe one would EXPECT the best of N trials to show by luck alone,
  estimated from the variance of Sharpes across the trials actually run.
  I count every configuration evaluated during tuning and store the
  count, so N is the real number, not a guess.
* Moving-block bootstrap CI: a confidence interval for the Sharpe that
  preserves short-range autocorrelation by resampling blocks of the daily
  return series instead of single days.

All Sharpe arithmetic happens in per-day units; only the displayed values
are annualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

EULER_GAMMA = 0.5772156649015329


def daily_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std()) if r.std() > 0 else np.nan


def probabilistic_sharpe(returns: pd.Series, sr_benchmark_daily: float = 0.0
                         ) -> float:
    """P(true SR > benchmark | observed returns), non-normality adjusted."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan
    sr = daily_sharpe(r)
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0       # pandas kurtosis is excess
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark_daily) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr_daily: float) -> float:
    """E[max of n_trials] of zero-true-Sharpe estimates with this variance.

    This is the luck benchmark: the best of N tries looks this good even
    when nothing works.
    """
    if n_trials <= 1 or var_sr_daily <= 0:
        return 0.0
    sd = np.sqrt(var_sr_daily)
    return float(sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                       + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(returns: pd.Series, trial_sharpes_annual: list[float]
                    ) -> dict:
    """DSR of `returns` given the annualized Sharpes of all trials run."""
    ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
    trials_daily = np.array(trial_sharpes_annual) / ann
    var_sr = float(np.var(trials_daily, ddof=1)) if len(trials_daily) > 1 else 0.0
    sr0 = expected_max_sharpe(len(trials_daily), var_sr)
    return {
        "n_trials": len(trials_daily),
        "expected_max_sharpe_annual": sr0 * ann,
        "psr": probabilistic_sharpe(returns, 0.0),
        "dsr": probabilistic_sharpe(returns, sr0),
    }


def block_bootstrap_sharpe_ci(returns: pd.Series, block: int = 20,
                              n_boot: int = 2000, ci: float = 0.90,
                              seed: int = config.HMM_RANDOM_STATE) -> dict:
    """Moving-block bootstrap CI for the ANNUALIZED Sharpe."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 2 * block:
        return {"lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.concatenate([r[s:s + block] for s in starts[i]])[:n]
        sd = sample.std()
        sharpes[i] = ann * sample.mean() / sd if sd > 0 else np.nan
    alpha = (1 - ci) / 2
    return {"lo": float(np.nanquantile(sharpes, alpha)),
            "hi": float(np.nanquantile(sharpes, 1 - alpha))}


def full_report(equity: pd.Series, trial_sharpes_annual: list[float]) -> dict:
    """All significance stats for one equity curve."""
    rets = equity.pct_change().dropna()
    ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
    out = {"sharpe_annual": daily_sharpe(rets) * ann if len(rets) else np.nan}
    out.update(deflated_sharpe(rets, trial_sharpes_annual))
    out["bootstrap_90ci"] = block_bootstrap_sharpe_ci(rets)
    return out
