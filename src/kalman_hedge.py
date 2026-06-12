"""Dynamic hedge ratio via Kalman filter.

State-space model
-----------------
    observation:  y_t = beta_t * x_t + alpha_t + eps_t,   eps ~ N(0, R)
    transition:   [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t,
                  w ~ N(0, Q)   (random walk)

Noise calibration (this matters more than anything else in this project)
-------------------------------------------------------------------------
* I calibrate R per pair to the variance of the train OLS residual, which
  is the natural dispersion of the pair around its equilibrium. My first
  version used the common textbook value R = 1e-3 on dollar prices. That
  told the filter the model fits to within a few cents, so the state
  absorbed every dislocation and the leftover "spread" degenerated into
  white measurement noise with no tradeable reversion. The diagnostic
  that exposed it: the spread half-life collapsed to roughly zero days
  while the OLS residual half-life was 25 days.
* I scale Q dimensionally:  Q = delta/(1-delta) * R * diag(1/s_x^2, 1),
  where s_x is the scale of x measured on the warm-up window. Beta
  multiplies x, so an unscaled beta random walk on an $800 stock moves
  the prediction by dollars per day and lets the state track dislocations
  through beta no matter how small delta is. With this scaling, delta has
  the same meaning for every pair: the dimensionless fraction of
  observation variance the state may absorb per day.

I implemented the filter twice, once with pykalman and once as a manual
numpy loop, and `self_test()` asserts they agree to numerical tolerance.
The manual loop shows exactly what the filter computes; pykalman protects
me from algebra mistakes.

No look-ahead bias
------------------
* I use FILTERED estimates only, never the Kalman smoother. The smoother
  conditions every day's state on the whole sample, including the future,
  which disqualifies it for trading.
* The tradeable spread at time t is the regression residual under the
  CURRENT filtered state:  spread_t = y_t - beta_{t|t} * x_t - alpha_{t|t}.
  beta_{t|t} conditions only on observations up to and including day t,
  so I could compute this at day t's close in live trading with no future
  data.
* I also return the one-step-ahead innovation
  e_t = y_t - beta_{t-1|t-1}*x_t - alpha_{t-1|t-1} for diagnostics, but I
  do not trade it: a well-calibrated filter's innovations are close to
  white noise, so their "mean reversion" is one-day noise that
  transaction costs destroy. The residual spread reverts on the
  cointegration half-life timescale, which is what the strategy needs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Manual numpy implementation
# ---------------------------------------------------------------------------
def kalman_filter_manual(y: pd.Series, x: pd.Series,
                         delta: float = 1e-4,
                         obs_var: float = config.KALMAN_OBS_VAR_FALLBACK) -> pd.DataFrame:
    """Predict/update loop over [beta, alpha] with random-walk dynamics.

    Returns a DataFrame indexed like y with columns:
      beta, alpha:    filtered state AFTER seeing day t (beta_{t|t})
      spread:         residual y_t - beta_{t|t}*x_t - alpha_{t|t} (tradeable)
      innovation:     one-step forecast error (diagnostics only)
      innovation_var: innovation variance S_t
    """
    y_v = y.to_numpy(dtype=float)
    x_v = x.to_numpy(dtype=float)
    n = len(y_v)

    R = obs_var                               # observation noise
    # Dimensional Q scaling: x-scale from the warm-up window only, so no
    # look-ahead (trading starts after KALMAN_WARMUP anyway).
    s_x = np.abs(x_v[:config.KALMAN_WARMUP]).mean()
    Q = (delta / (1.0 - delta)) * R * np.diag([1.0 / s_x**2, 1.0])

    state = np.zeros(2)                       # [beta, alpha], diffuse start
    P = np.eye(2) * 1.0                       # state covariance

    betas = np.empty(n)
    alphas = np.empty(n)
    spreads = np.empty(n)
    innovations = np.empty(n)
    innovation_vars = np.empty(n)

    for t in range(n):
        # Predict (random walk: state unchanged, covariance grows). At t=0
        # the initial covariance IS the prior (pykalman convention: the
        # first transition happens between t=0 and t=1).
        if t > 0:
            P = P + Q

        H = np.array([x_v[t], 1.0])           # observation vector

        # Innovation against the PRIOR state (diagnostics).
        e = y_v[t] - H @ state
        S = H @ P @ H + R                     # innovation variance
        innovations[t] = e
        innovation_vars[t] = S

        # Update with day t's observation.
        K = P @ H / S                         # Kalman gain (2,)
        state = state + K * e
        P = P - np.outer(K, H) @ P

        betas[t] = state[0]
        alphas[t] = state[1]
        # Tradeable spread: residual under the day-t filtered state.
        spreads[t] = y_v[t] - state[0] * x_v[t] - state[1]

    return pd.DataFrame(
        {"beta": betas, "alpha": alphas, "spread": spreads,
         "innovation": innovations, "innovation_var": innovation_vars},
        index=y.index,
    )


# ---------------------------------------------------------------------------
# pykalman implementation
# ---------------------------------------------------------------------------
def kalman_filter_pykalman(y: pd.Series, x: pd.Series,
                           delta: float = 1e-4,
                           obs_var: float = config.KALMAN_OBS_VAR_FALLBACK) -> pd.DataFrame:
    """Same model via pykalman.KalmanFilter (filtered estimates only)."""
    from pykalman import KalmanFilter

    n = len(y)
    x_v0 = x.to_numpy(dtype=float)
    obs_mat = np.stack([x_v0, np.ones(n)], axis=1)[:, np.newaxis, :]  # (n,1,2)

    s_x = np.abs(x_v0[:config.KALMAN_WARMUP]).mean()
    Q = (delta / (1.0 - delta)) * obs_var * np.diag([1.0 / s_x**2, 1.0])

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=obs_var,
        transition_covariance=Q,
    )
    means, _ = kf.filter(y.to_numpy(dtype=float))  # filtered, not smoothed

    betas = means[:, 0]
    alphas = means[:, 1]
    y_v = y.to_numpy(dtype=float)
    x_v = x.to_numpy(dtype=float)
    # Tradeable spread: residual under the day-t filtered state.
    spread = y_v - betas * x_v - alphas
    # Innovation (diagnostics): prior state vs today's observation. For
    # t=0 the prior is the initial mean (zeros).
    prior_beta = np.concatenate([[0.0], betas[:-1]])
    prior_alpha = np.concatenate([[0.0], alphas[:-1]])
    innovation = y_v - prior_beta * x_v - prior_alpha
    return pd.DataFrame(
        {"beta": betas, "alpha": alphas, "spread": spread,
         "innovation": innovation},
        index=y.index,
    )


def self_test(y: pd.Series, x: pd.Series, delta: float = 1e-4,
              obs_var: float = config.KALMAN_OBS_VAR_FALLBACK) -> float:
    """Assert manual and pykalman implementations agree; return max |diff|."""
    a = kalman_filter_manual(y, x, delta, obs_var)
    b = kalman_filter_pykalman(y, x, delta, obs_var)
    diff = max(
        np.abs(a["beta"] - b["beta"]).max(),
        np.abs(a["alpha"] - b["alpha"]).max(),
        np.abs(a["spread"] - b["spread"]).max(),
        np.abs(a["innovation"] - b["innovation"]).max(),
    )
    assert diff < 1e-6, f"implementations disagree: max diff {diff:.2e}"
    return float(diff)


# ---------------------------------------------------------------------------
# Delta tuning (validation set only)
# ---------------------------------------------------------------------------
def _quick_signal_sharpe(spread: pd.Series) -> float:
    """Cheap proxy for spread quality: Sharpe of a naive +-2/0 z-score rule.

    Uses trailing-window z-scores and next-bar positions (position taken at
    t applies to the spread change from t to t+1), so the proxy itself is
    free of look-ahead bias. This is only used to RANK deltas on the
    validation set, not to report performance.
    """
    mean = spread.rolling(config.ZSCORE_WINDOW).mean()
    std = spread.rolling(config.ZSCORE_WINDOW).std()
    z = (spread - mean) / std

    pos = pd.Series(0.0, index=spread.index)
    current = 0.0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pos.iloc[t] = current
            continue
        if current == 0.0:
            if zt < -config.ENTRY_Z:
                current = 1.0
            elif zt > config.ENTRY_Z:
                current = -1.0
        elif current == 1.0 and zt >= 0:
            current = 0.0
        elif current == -1.0 and zt <= 0:
            current = 0.0
        pos.iloc[t] = current

    pnl = pos.shift(1) * spread.diff()        # enter next bar
    pnl = pnl.dropna()
    if pnl.std() == 0 or len(pnl) < 20:
        return -np.inf
    return float(np.sqrt(config.TRADING_DAYS_PER_YEAR) * pnl.mean() / pnl.std())


def tune_delta(prices: pd.DataFrame, pairs: pd.DataFrame,
               splits: dict, force: bool = False) -> float:
    """Pick delta from KALMAN_DELTA_GRID by a validation-set quality proxy.

    The filter is run on contiguous train+validation history (filtered
    estimates only use the past), and the quality metric is computed on the
    validation segment alone. Test data is never touched.

    NOTE: this standalone proxy tuner is kept for demonstration; the
    production pipeline tunes delta JOINTLY with the z-score parameters by
    running the actual validation backtest (backtest.tune_signal_params),
    which is the true objective. Results are stored under 'delta_proxy' so
    they never clobber the jointly tuned 'delta'.
    """
    params = (json.loads(config.TUNED_PARAMS_JSON.read_text())
              if config.TUNED_PARAMS_JSON.exists() else {})
    if "delta_proxy" in params and not force:
        return params["delta_proxy"]

    val_start, val_end = splits["validation"]
    history = prices.loc[:val_end]            # train + validation, no test

    scores = {}
    for delta in config.KALMAN_DELTA_GRID:
        sharpes = []
        for _, p in pairs.iterrows():
            kf = kalman_filter_manual(history[p["y"]], history[p["x"]], delta,
                                      obs_var=p["resid_var"])
            val_spread = kf["spread"].loc[val_start:val_end]
            sharpes.append(_quick_signal_sharpe(val_spread))
        scores[delta] = float(np.mean(sharpes))
        print(f"[kalman] delta={delta:.0e}  mean validation Sharpe proxy "
              f"= {scores[delta]:+.3f}")

    best = max(scores, key=scores.get)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    params["delta_proxy"] = best
    params["delta_proxy_scores"] = {f"{k:.0e}": v for k, v in scores.items()}
    config.TUNED_PARAMS_JSON.write_text(json.dumps(params, indent=2))
    return best


# ---------------------------------------------------------------------------
# Diagnostics plot
# ---------------------------------------------------------------------------
def plot_static_vs_kalman(y: pd.Series, x: pd.Series, static_beta: float,
                          delta: float,
                          obs_var: float = config.KALMAN_OBS_VAR_FALLBACK,
                          out_path=None):
    """Static OLS hedge ratio vs Kalman dynamic beta over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kf = kalman_filter_manual(y, x, delta, obs_var)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(kf.index, kf["beta"], label=f"Kalman beta (delta={delta:.0e})",
            lw=1.2)
    ax.axhline(static_beta, color="crimson", ls="--",
               label=f"static OLS beta = {static_beta:.3f}")
    ax.set_title(f"Hedge ratio: {y.name} vs {x.name}")
    ax.set_ylabel("beta")
    ax.legend()
    fig.tight_layout()
    if out_path is None:
        out_path = config.RESULTS_DIR / f"hedge_ratio_{y.name}_{x.name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from . import cointegration, universe

    px = universe.download_universe()
    splits = universe.load_splits()
    stocks = px[universe.stock_columns(px)]
    train = universe.get_split(stocks, "train")
    pairs = cointegration.select_pairs(train)

    top = pairs.iloc[0]
    diff = self_test(train[top["y"]], train[top["x"]],
                     obs_var=top["resid_var"])
    print(f"[kalman] self-test passed: manual vs pykalman max diff {diff:.2e}")

    delta = tune_delta(stocks, pairs, splits, force=True)
    print(f"[kalman] tuned delta = {delta:.0e}")

    path = plot_static_vs_kalman(
        stocks[top["y"]].loc[:splits["validation"][1]],
        stocks[top["x"]].loc[:splits["validation"][1]],
        static_beta=top["beta"], delta=delta, obs_var=top["resid_var"])
    print(f"[kalman] plot saved: {path}")
