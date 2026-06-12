"""Z-score entry/exit signals on the Kalman spread.

How I construct the z-score without look-ahead: I use an expanding-window
mean and std for the first ZSCORE_EXPANDING_UNTIL observations (with at
least ZSCORE_EXPANDING_MIN observations before producing a value), then a
rolling window. Both estimators at time t use only spread values up to
and including t, never future data. The expanding start exists because a
rolling std over the first handful of observations is so noisy that it
produces meaningless z-values of 5 or 10 on day three.

Rules
-----
  entry:  z < -ENTRY_Z  -> long spread (long y, short beta*x)
          z > +ENTRY_Z  -> short spread (short y, long beta*x)
  exit:   z crosses 0 (mean reversion complete)
  stop:   z beyond +-STOP_Z against the position
  gate:   new entries only while the HMM filtered state is mean-reverting.
          Exits are always allowed, because blocking a risk-reducing
          trade can never be correct.

The signal frame is advisory: the backtester owns position state and
executes decisions at the NEXT bar's price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _trailing_std(spread: pd.Series,
                  window: int = config.ZSCORE_WINDOW) -> pd.Series:
    """Expanding std early, rolling std thereafter (trailing only)."""
    exp_std = spread.expanding(config.ZSCORE_EXPANDING_MIN).std()
    roll_std = spread.rolling(window).std()
    cutover = max(config.ZSCORE_EXPANDING_UNTIL, window)
    use_expanding = np.arange(len(spread)) < cutover
    return pd.Series(np.where(use_expanding, exp_std, roll_std),
                     index=spread.index)


def zscore(spread: pd.Series, center: str = "rolling",
           window: int = config.ZSCORE_WINDOW) -> pd.Series:
    """Trailing z-score: expanding window early, rolling thereafter.

    center="rolling": classic (spread - trailing_mean) / trailing_std.
    center="zero":    spread / trailing_std. Appropriate for the Kalman
                      residual spread, whose equilibrium is zero by
                      construction (alpha absorbs the level); a short
                      rolling mean would chase dislocations and make z
                      revert without the spread actually reverting.

    `window` should be long relative to the spread's half-life: a short
    window makes the rolling std spike-sensitive, so a one-day dislocation
    inflates the denominator and z normalizes without any price reversion.
    Both center and window are tuned on the VALIDATION set.
    """
    std = _trailing_std(spread, window)
    if center == "zero":
        mean = 0.0
    else:
        exp_mean = spread.expanding(config.ZSCORE_EXPANDING_MIN).mean()
        roll_mean = spread.rolling(window).mean()
        cutover = max(config.ZSCORE_EXPANDING_UNTIL, window)
        use_expanding = np.arange(len(spread)) < cutover
        mean = pd.Series(np.where(use_expanding, exp_mean, roll_mean),
                         index=spread.index)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def build_signal_frame(kf: pd.DataFrame, regimes: pd.Series,
                       center: str = "rolling",
                       window: int = config.ZSCORE_WINDOW,
                       ou_params: dict | None = None,
                       p_mr: pd.Series | None = None) -> pd.DataFrame:
    """Combine Kalman output + regime gate into one frame for the backtester.

    Columns: spread, beta, alpha, z, regime_ok, spread_std (for sizing).
    `spread_std` is the same dispersion the signal saw, so position sizing
    and the signal always agree about scale.

    center="ou" (v2): the score is the Avellaneda-Lee s-score against the
    OU parameters fitted on the FORMATION window (passed via `ou_params`),
    and the sizing dispersion is the frozen stationary sigma_eq. This
    makes the score immune to the v1 failure where the very spike being
    traded inflates a short rolling std. The caller guarantees ou_params
    came from formation data only.

    `p_mr` optionally carries the filtered P(mean-reverting) so the
    meta-labeling layer can use the probability rather than the binary
    gate; it does not affect trading rules.
    """
    if center == "ou":
        if ou_params is None:
            raise ValueError("center='ou' requires ou_params")
        z = (kf["spread"] - ou_params["m"]) / ou_params["sigma_eq"]
        spread_std = pd.Series(ou_params["sigma_eq"], index=kf.index)
    else:
        z = zscore(kf["spread"], center=center, window=window)
        spread_std = _trailing_std(kf["spread"], window)

    frame = pd.DataFrame({
        "spread": kf["spread"],
        "beta": kf["beta"],
        "alpha": kf["alpha"],
        "z": z,
        "spread_std": spread_std,
        "regime_ok": regimes.reindex(kf.index, fill_value=False).astype(bool),
    })
    if p_mr is not None:
        frame["p_mr"] = p_mr.reindex(kf.index).ffill().fillna(0.5)
    # Kalman warm-up: signals are not tradeable until the filter has settled.
    frame.iloc[:config.KALMAN_WARMUP,
               frame.columns.get_loc("z")] = np.nan
    return frame


def entry_signal(row: pd.Series) -> int:
    """+1 long spread, -1 short spread, 0 no entry. Regime-gated."""
    if not row["regime_ok"] or np.isnan(row["z"]):
        return 0
    if row["z"] < -config.ENTRY_Z:
        return 1
    if row["z"] > config.ENTRY_Z:
        return -1
    return 0


def exit_signal(z: float, direction: int) -> str | None:
    """'signal' on mean-reversion exit, 'stop' on stop-loss, else None.

    direction: +1 = long spread (entered at z < -2, profits as z rises),
               -1 = short spread (entered at z > +2, profits as z falls).
    """
    if np.isnan(z):
        return None
    if direction == 1:
        if z >= config.EXIT_Z:
            return "signal"
        if z < -config.STOP_Z:
            return "stop"
    else:
        if z <= config.EXIT_Z:
            return "signal"
        if z > config.STOP_Z:
            return "stop"
    return None
