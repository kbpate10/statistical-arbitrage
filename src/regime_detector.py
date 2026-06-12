"""HMM regime classification: mean-reverting vs trending.

Pairs trading assumes mean reversion, and mean reversion is conditional.
During stress periods correlations break down and spreads trend, so I
gate new entries on a regime model. I fit a 2-state Gaussian HMM on
TRAIN-period features only. The features, all computed with trailing
windows so the value at t uses data up to and including t:

  * 20d rolling volatility of the Kalman spread
  * 5d and 20d rolling correlation between the pair's daily returns
  * VIX level
  * 20d momentum of the spread (spread_t - spread_{t-20})

After fitting, I label the states by their feature means: the state with
lower spread volatility, lower absolute spread momentum, and lower VIX is
the MEAN-REVERTING regime; the other is TRENDING. The backtester only
opens new trades while the model classifies the current day as
mean-reverting.

No look-ahead bias
------------------
The Viterbi path (`model.predict` over a full series) conditions every
day's state on the WHOLE sample, including the future, so I never use it
for backtesting decisions. Instead I compute the forward algorithm's
filtered state probabilities P(state_t | observations_{1..t}) with an
explicit recursion below. The regime flag at day t therefore uses only
data available at day t. Feature standardization uses a scaler fit on
train only, for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from . import config

MEAN_REVERTING = 1
TRENDING = 0


def build_features(y: pd.Series, x: pd.Series, spread: pd.Series,
                   vix: pd.Series) -> pd.DataFrame:
    """Trailing-window regime features for one pair. NaN rows dropped."""
    ret_y = y.pct_change()
    ret_x = x.pct_change()
    feats = pd.DataFrame({
        "spread_vol": spread.rolling(config.REGIME_VOL_WINDOW).std(),
        "corr_5d": ret_y.rolling(config.REGIME_CORR_SHORT).corr(ret_x),
        "corr_20d": ret_y.rolling(config.REGIME_CORR_LONG).corr(ret_x),
        "vix": vix,
        "spread_mom": spread.diff(config.REGIME_MOM_WINDOW),
    })
    return feats.dropna()


class RegimeDetector:
    """Fit on train features; classify any later period via forward filter."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=config.HMM_N_STATES,
            covariance_type="full",
            n_iter=config.HMM_N_ITER,
            random_state=config.HMM_RANDOM_STATE,
        )
        self.state_labels: dict[int, str] = {}
        self._mr_state: int | None = None

    # -- fitting ------------------------------------------------------------
    def fit(self, train_features: pd.DataFrame) -> "RegimeDetector":
        X = self.scaler.fit_transform(train_features.to_numpy())
        self.model.fit(X)
        self._label_states(train_features.columns)
        return self

    def _label_states(self, feature_names) -> None:
        """Mean-reverting = lower spread vol + lower |momentum| (+ lower VIX
        as tie-breaker). Score each state and label the calmer one."""
        means = pd.DataFrame(self.model.means_, columns=feature_names)
        score = (means["spread_vol"].rank()
                 + means["spread_mom"].abs().rank()
                 + means["vix"].rank())
        self._mr_state = int(score.idxmin())
        self.state_labels = {
            s: ("mean_reverting" if s == self._mr_state else "trending")
            for s in range(config.HMM_N_STATES)
        }

    # -- inference ----------------------------------------------------------
    def filtered_regimes(self, features: pd.DataFrame) -> pd.Series:
        """Boolean Series: True where the FILTERED state is mean-reverting.

        Runs the forward algorithm recursion explicitly:
            alpha_t  proportional to  b(o_t) * (A' alpha_{t-1})
        so P(state_t | o_{1..t}) at each t uses only observations up to t.
        """
        X = self.scaler.transform(features.to_numpy())
        # Per-observation Gaussian log-likelihoods under each state.
        log_b = self.model._compute_log_likelihood(X)   # (T, n_states)
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))

        A = self.model.transmat_
        pi = self.model.startprob_

        T, S = b.shape
        alpha = np.empty((T, S))
        a = pi * b[0]
        alpha[0] = a / a.sum()
        for t in range(1, T):
            a = b[t] * (alpha[t - 1] @ A)
            alpha[t] = a / a.sum()

        states = alpha.argmax(axis=1)
        return pd.Series(states == self._mr_state, index=features.index,
                         name="mean_reverting")

    def filtered_probabilities(self, features: pd.DataFrame) -> pd.Series:
        """P(mean-reverting | data up to t), for charting."""
        X = self.scaler.transform(features.to_numpy())
        log_b = self.model._compute_log_likelihood(X)
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        A, pi = self.model.transmat_, self.model.startprob_
        T, S = b.shape
        alpha = np.empty((T, S))
        a = pi * b[0]
        alpha[0] = a / a.sum()
        for t in range(1, T):
            a = b[t] * (alpha[t - 1] @ A)
            alpha[t] = a / a.sum()
        return pd.Series(alpha[:, self._mr_state], index=features.index,
                         name="p_mean_reverting")


def plot_regimes(spread: pd.Series, regimes: pd.Series, pair_name: str,
                 out_path=None):
    """Color-coded regime chart (plotly): spread with mean-reverting bands."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values,
                             mode="lines", name="spread",
                             line=dict(width=1.2)))
    # Shade contiguous mean-reverting stretches.
    in_block = False
    start = None
    blocks = []
    for ts, mr in regimes.items():
        if mr and not in_block:
            in_block, start = True, ts
        elif not mr and in_block:
            blocks.append((start, ts))
            in_block = False
    if in_block:
        blocks.append((start, regimes.index[-1]))
    for s, e in blocks:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(46,160,67,0.15)",
                      line_width=0)
    fig.update_layout(
        title=(f"{pair_name}: spread with mean-reverting regime shaded "
               f"(green = trades allowed)"),
        height=420, template="plotly_white",
    )
    if out_path is None:
        out_path = config.RESULTS_DIR / f"regimes_{pair_name}.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


if __name__ == "__main__":
    import json

    from . import cointegration, kalman_hedge, universe

    px = universe.download_universe()
    splits = universe.load_splits()
    stocks = px[universe.stock_columns(px)]
    vix = px[config.VIX_TICKER]
    train_end = splits["train"][1]

    pairs = cointegration.select_pairs(universe.get_split(stocks, "train"))
    top = pairs.iloc[0]
    delta = (json.loads(config.TUNED_PARAMS_JSON.read_text()).get("delta", 1e-4)
             if config.TUNED_PARAMS_JSON.exists() else 1e-4)

    y, x = stocks[top["y"]], stocks[top["x"]]
    kf = kalman_hedge.kalman_filter_manual(y, x, delta,
                                           obs_var=top["resid_var"])
    feats = build_features(y, x, kf["spread"], vix)

    det = RegimeDetector().fit(feats.loc[:train_end])
    regimes = det.filtered_regimes(feats)

    frac_mr = regimes.mean()
    print(f"[regime] pair {top['y']}/{top['x']}: "
          f"{frac_mr:.1%} of days classified mean-reverting")
    for s, lbl in det.state_labels.items():
        m = pd.Series(det.model.means_[s], index=feats.columns)
        print(f"  state {s} ({lbl}): "
              + ", ".join(f"{k}={v:+.2f}" for k, v in m.items()))
    path = plot_regimes(kf["spread"], regimes, f"{top['y']}_{top['x']}")
    print(f"[regime] chart saved: {path}")
