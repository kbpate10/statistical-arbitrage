"""Cointegration breakdown detection: the kill-switch.

Why this exists. The two largest v1 test-year losses came from pairs
whose train-period relationship physically broke while the system kept
trading them. Cointegration established on a formation window is a
hypothesis, not a property, and I want an online monitor that withdraws
the hypothesis when the evidence turns against it. Both layers below are
forward-only: the flag at day t uses data up to day t, never after.

Layer 1, slow guard: rolling re-test of the Engle-Granger premise. Every
BREAK_ADF_EVERY trading days I re-run the ADF test on the trailing
BREAK_ADF_WINDOW days of the live spread. If the p-value exceeds
BREAK_ADF_P, the spread no longer looks stationary and I block NEW
entries (existing positions may still exit normally). I use a loose 0.20
threshold rather than the 0.05 selection gate because a freshly selected
pair hovers near 0.05 by construction and a hard gate would whipsaw.

Layer 2, fast detector: Bayesian online changepoint detection (Adams and
MacKay, 2007) on standardized daily spread changes. BOCPD maintains a
posterior over the "run length", the number of days since the last
change in the generating process, using a Normal-Inverse-Gamma conjugate
model whose predictive distribution is a Student-t. When the posterior
mass concentrates on short run lengths, the process that generates the
spread changed recently. That is precisely what a cointegration break
looks like from inside: the spread's drift/variance regime shifts. On
trigger I force-exit the position (exit reason "break") and retire the
pair for the rest of the cycle.

Why BOCPD and not a simpler CUSUM: BOCPD is exact Bayesian inference
under its model, has no tuning beyond a hazard rate and priors with
natural scales (I standardize the input so the priors are generic), and
produces a probability I can also feed to the meta-labeling features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from . import config
from .cointegration import adf_pvalue


# ---------------------------------------------------------------------------
# Layer 2: BOCPD (Adams-MacKay) with Normal-Inverse-Gamma / Student-t
# ---------------------------------------------------------------------------
def bocpd_short_run_prob(x: np.ndarray,
                         hazard: float = config.BOCPD_HAZARD,
                         short_run: int = config.BOCPD_SHORT_RUN,
                         max_run: int = config.BOCPD_MAX_RUN) -> np.ndarray:
    """P(run length <= short_run | data up to t) for each t.

    `x` should be standardized (mean ~0, std ~1 on the formation window)
    so the unit-scale priors are appropriate. The recursion is the
    standard one: at each step, every run-length hypothesis either grows
    by one (no changepoint, prob 1-hazard) or resets to zero (changepoint,
    prob hazard), weighted by how well that hypothesis predicted today's
    observation under its Student-t predictive.
    """
    # Normal-Inverse-Gamma priors on (mean, variance) of the changes.
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    T = len(x)
    out = np.zeros(T)

    # Run-length posterior and per-run posterior parameters. Index r =
    # run length. I truncate at max_run for speed; mass beyond that is
    # negligible at hazard 1/100.
    R = np.array([1.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    for t in range(T):
        # Student-t predictive for each run-length hypothesis.
        df = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        z2 = (x[t] - mu) ** 2 / scale2
        # log pdf of standardized t with df degrees of freedom
        log_pred = (gammaln((df + 1) / 2) - gammaln(df / 2)
                    - 0.5 * np.log(np.pi * df * scale2)
                    - (df + 1) / 2 * np.log1p(z2 / df))
        pred = np.exp(log_pred - log_pred.max())

        growth = R * pred * (1.0 - hazard)
        cp = float((R * pred).sum() * hazard)
        R_new = np.concatenate([[cp], growth])
        R_new /= R_new.sum()

        # Posterior parameter update: prepend the prior (run length 0),
        # then update every surviving hypothesis with x[t].
        mu_new = np.concatenate([[mu0], (kappa * mu + x[t]) / (kappa + 1)])
        kappa_new = np.concatenate([[kappa0], kappa + 1.0])
        alpha_new = np.concatenate([[alpha0], alpha + 0.5])
        beta_new = np.concatenate(
            [[beta0], beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))])

        if len(R_new) > max_run:
            R_new = R_new[:max_run]
            R_new /= R_new.sum()
            mu_new, kappa_new = mu_new[:max_run], kappa_new[:max_run]
            alpha_new, beta_new = alpha_new[:max_run], beta_new[:max_run]

        R, mu, kappa, alpha, beta = R_new, mu_new, kappa_new, alpha_new, beta_new
        upto = min(short_run + 1, len(R))
        out[t] = float(R[:upto].sum())

    return out


# ---------------------------------------------------------------------------
# Kill-switch: precomputed per-pair daily flags, queried by the event loop
# ---------------------------------------------------------------------------
class KillSwitch:
    """Per-pair forward-only break flags over a trading window.

    For each pair I precompute a daily flag series ("exit", "block", or
    None). Precomputing is safe because every value at day t derives only
    from spread data up to day t; the event loop just looks the value up.
    """

    def __init__(self):
        self.flags: dict[str, pd.Series] = {}
        self.short_run_prob: dict[str, pd.Series] = {}

    def add_pair(self, pair: str, spread: pd.Series,
                 formation_end, trading_dates: pd.DatetimeIndex) -> None:
        spread = spread.dropna()

        # Layer 2: BOCPD on standardized daily changes over the full
        # history (formation warm-up + trading), standardized by the
        # FORMATION std so the trading window cannot rescale itself.
        chg = spread.diff().dropna()
        form_std = chg.loc[:formation_end].std()
        if not np.isfinite(form_std) or form_std == 0:
            self.flags[pair] = pd.Series(None, index=trading_dates,
                                         dtype=object)
            return
        p_short = pd.Series(
            bocpd_short_run_prob((chg / form_std).to_numpy()),
            index=chg.index)
        self.short_run_prob[pair] = p_short.reindex(trading_dates).ffill()

        # Layer 1: trailing ADF re-test on the trading dates, every
        # BREAK_ADF_EVERY days, forward-filled between re-tests.
        adf_flags = pd.Series(np.nan, index=trading_dates)
        for i in range(0, len(trading_dates), config.BREAK_ADF_EVERY):
            d = trading_dates[i]
            window = spread.loc[:d].tail(config.BREAK_ADF_WINDOW)
            if len(window) >= config.BREAK_ADF_WINDOW // 2:
                adf_flags.iloc[i] = adf_pvalue(window)
        adf_flags = adf_flags.ffill()

        flags = pd.Series(None, index=trading_dates, dtype=object)
        flags[adf_flags > config.BREAK_ADF_P] = "block"
        # Persistence rule: act only after BOCPD_PERSIST consecutive alarm
        # days, which separates genuine process changes from one-day
        # outliers (see config for the synthetic-data calibration).
        alarm = self.short_run_prob[pair] > config.BOCPD_TRIGGER
        consec = alarm.groupby((~alarm).cumsum()).cumsum()
        tripped = consec >= config.BOCPD_PERSIST
        if tripped.any():
            first = tripped.idxmax()
            flags.loc[first:] = "block"   # retired for the cycle
            flags.loc[first] = "exit"
        self.flags[pair] = flags

    def __call__(self, pair: str, date) -> str | None:
        s = self.flags.get(pair)
        if s is None or date not in s.index:
            return None
        v = s.loc[date]
        return v if isinstance(v, str) else None
